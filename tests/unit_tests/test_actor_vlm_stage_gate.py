# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import fcntl
import inspect
import os
import queue
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.storage.replay import PriorityStore
from rlinf.profiling import ProfileStage
from rlinf.scheduler.stage_gate import InterProcessStageGate
from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor


def _gate(gate_id, synchronize, record_wait=None, *, priority_waiter=False):
    return InterProcessStageGate(
        gate_id,
        node_rank=0,
        accelerator_rank=0,
        wait_stage=ProfileStage.ACTOR_WAIT_ROLLOUT_VLM,
        synchronize=synchronize,
        record_wait=record_wait,
        priority_waiter=priority_waiter,
    )


def _buffer_aware_actor(*, version=4, staleness_threshold=2, store_size=1):
    actor = AsyncPPOEmbodiedFSDPActor.__new__(AsyncPPOEmbodiedFSDPActor)
    actor.version = version
    actor.cfg = SimpleNamespace(
        algorithm={"staleness_threshold": staleness_threshold}
    )
    actor._recv_queue = queue.Queue()
    actor._recv_queue_metadata_lock = threading.Lock()
    actor._recv_version_counts = Counter()
    actor.rollout_store_size = store_size
    actor.rollout_store = PriorityStore(maxsize=store_size)
    actor._actor_vlm_stage_gate = object()
    return actor


def test_stage_gate_serializes_separate_holders():
    synchronized = []
    waits = []
    actor_gate = _gate("serialize-test", lambda: synchronized.append("actor"))
    rollout_gate = _gate(
        "serialize-test",
        lambda: synchronized.append("rollout"),
        record_wait=waits.append,
        priority_waiter=True,
    )
    attempted = threading.Event()
    acquired = threading.Event()

    def acquire_rollout_gate():
        attempted.set()
        with rollout_gate.hold():
            acquired.set()

    try:
        with actor_gate.hold():
            thread = threading.Thread(target=acquire_rollout_gate)
            thread.start()
            assert attempted.wait(timeout=1)
            assert not acquired.wait(timeout=0.05)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert acquired.is_set()
        assert synchronized == ["actor", "rollout"]
        assert len(waits) == 1
        assert waits[0] >= 0.05
    finally:
        actor_gate.close()
        rollout_gate.close()


def test_stage_gate_releases_lock_when_stage_fails():
    first_gate = _gate("exception-test", lambda: None)
    second_gate = _gate("exception-test", lambda: None)
    try:
        with pytest.raises(RuntimeError, match="stage failed"):
            with first_gate.hold():
                raise RuntimeError("stage failed")
        with second_gate.hold():
            pass
    finally:
        first_gate.close()
        second_gate.close()


def test_waiting_priority_holder_cannot_be_overtaken():
    order = []
    first_actor_gate = _gate("priority-test", lambda: None)
    second_actor_gate = _gate("priority-test", lambda: None)
    rollout_gate = _gate(
        "priority-test",
        lambda: None,
        priority_waiter=True,
    )

    def hold(gate, name):
        with gate.hold():
            order.append(name)

    try:
        with first_actor_gate.hold():
            rollout_thread = threading.Thread(
                target=hold,
                args=(rollout_gate, "rollout"),
            )
            rollout_thread.start()

            probe_fd = os.open(rollout_gate.priority_path, os.O_RDWR)
            try:
                deadline = time.monotonic() + 1
                while True:
                    try:
                        fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        break
                    fcntl.flock(probe_fd, fcntl.LOCK_UN)
                    assert time.monotonic() < deadline
                    time.sleep(0.001)
            finally:
                os.close(probe_fd)

            actor_thread = threading.Thread(
                target=hold,
                args=(second_actor_gate, "actor"),
            )
            actor_thread.start()

        rollout_thread.join(timeout=1)
        actor_thread.join(timeout=1)
        assert not rollout_thread.is_alive()
        assert not actor_thread.is_alive()
        assert order == ["rollout", "actor"]
    finally:
        first_actor_gate.close()
        second_actor_gate.close()
        rollout_gate.close()


def test_actor_forward_and_backward_gates_are_independently_configurable():
    forward_source = inspect.getsource(
        AsyncPPOEmbodiedFSDPActor.forward_and_loss_micro_batch
    )
    training_source = inspect.getsource(AsyncPPOEmbodiedFSDPActor.run_training)

    assert "_actor_vlm_stage_gate.hold" not in forward_source
    assert training_source.count("_actor_vlm_stage_gate.hold") == 2
    assert 'self.cfg.runner.get("actor_vlm_gate_forward", False)' in training_source
    assert 'self.cfg.runner.get("actor_vlm_gate_backward", True)' in training_source


def test_fresh_inventory_is_evaluated_for_next_actor_version():
    actor = _buffer_aware_actor(version=4, staleness_threshold=2)

    # Next actor version is 5, so version 3 remains eligible while version 2 does not.
    actor._recv_version_counts[2.0] = 1
    assert not actor._has_fresh_next_batch()

    actor._recv_version_counts[3.0] = 1
    assert actor._has_fresh_next_batch()


def test_inventory_includes_reusable_store_and_requires_full_next_batch():
    actor = _buffer_aware_actor(store_size=2)
    actor.rollout_store.add((3.0, 3.0), object())

    assert not actor._has_fresh_next_batch()
    actor._recv_version_counts[3.0] = 1
    assert actor._has_fresh_next_batch()


def test_no_sync_backward_uses_local_inventory():
    actor = _buffer_aware_actor()

    assert not actor._should_gate_actor_stage(require_rank_consensus=False)
    actor._recv_version_counts[3.0] = 1
    assert actor._should_gate_actor_stage(require_rank_consensus=False)


def test_sync_backward_requires_inventory_on_every_actor_rank(monkeypatch):
    actor = _buffer_aware_actor()
    actor._recv_version_counts[3.0] = 1

    class Decision:
        value = 1

        def item(self):
            return self.value

    decision = Decision()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: decision)

    def all_reduce(value, *, op):
        assert op == torch.distributed.ReduceOp.MIN
        value.value = 0

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    assert not actor._should_gate_actor_stage(require_rank_consensus=True)
