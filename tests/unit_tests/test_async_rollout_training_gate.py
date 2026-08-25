# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


class _Result:
    def __init__(self, events, name, value=None, error=None):
        self._events = events
        self._name = name
        self._value = value
        self._error = error

    def wait(self):
        self._events.append(f"{self._name}.wait")
        if self._error is not None:
            raise self._error
        return self._value


class _Rollout:
    def __init__(self, events):
        self._events = events

    def pause_inference_at_boundary(self):
        self._events.append("pause")
        return _Result(self._events, "pause")

    def resume_inference(self):
        self._events.append("resume")
        return _Result(self._events, "resume")

    def configure_actor_vlm_stage_gate(self, gate_id):
        self._events.append(("rollout.configure_gate", gate_id))
        return _Result(self._events, "rollout.configure_gate")


class _Actor:
    def __init__(self, events, error=None):
        self._events = events
        self._error = error

    def run_training(self):
        self._events.append("train")
        return _Result(
            self._events,
            "train",
            value={"loss": 1.0},
            error=self._error,
        )

    def configure_actor_vlm_stage_gate(self, gate_id):
        self._events.append(("actor.configure_gate", gate_id))
        return _Result(self._events, "actor.configure_gate")


def _runner(events, *, error=None):
    runner = AsyncPPOEmbodiedRunner.__new__(AsyncPPOEmbodiedRunner)
    runner.pause_rollout_during_actor_training = True
    runner.rollout = _Rollout(events)
    runner.actor = _Actor(events, error=error)

    @contextmanager
    def timer(name):
        events.append(f"timer.{name}.enter")
        try:
            yield
        finally:
            events.append(f"timer.{name}.exit")

    runner.timer = timer
    return runner


def test_rollout_pause_blocks_next_inference_until_resume():
    async def exercise():
        synchronize_calls = []
        worker = SimpleNamespace(
            _inference_pause_requested=False,
            _inference_resume_event=asyncio.Event(),
            torch_platform=SimpleNamespace(
                synchronize=lambda: synchronize_calls.append("synchronize")
            ),
        )
        worker._inference_resume_event.set()

        await AsyncMultiStepRolloutWorker.pause_inference_at_boundary(worker)
        permission = asyncio.create_task(
            AsyncMultiStepRolloutWorker._wait_for_inference_permission(worker)
        )
        await asyncio.sleep(0)
        assert not permission.done()

        await AsyncMultiStepRolloutWorker.resume_inference(worker)
        await asyncio.wait_for(permission, timeout=1)
        assert synchronize_calls == ["synchronize"]

    asyncio.run(exercise())


def test_runner_pauses_rollout_around_actor_training():
    events = []
    runner = _runner(events)

    _, metrics = runner._run_actor_training()

    assert metrics == {"loss": 1.0}
    assert events == [
        "timer.pause_rollout_inference.enter",
        "pause",
        "pause.wait",
        "timer.pause_rollout_inference.exit",
        "timer.actor_training.enter",
        "train",
        "train.wait",
        "timer.actor_training.exit",
        "timer.resume_rollout_inference.enter",
        "resume",
        "resume.wait",
        "timer.resume_rollout_inference.exit",
    ]


def test_runner_resumes_rollout_when_actor_training_fails():
    events = []
    runner = _runner(events, error=RuntimeError("training failed"))

    with pytest.raises(RuntimeError, match="training failed"):
        runner._run_actor_training()

    assert "resume" in events
    assert events.index("resume") > events.index("train.wait")


def test_runner_configures_same_stage_gate_for_actor_and_rollout():
    events = []
    runner = _runner(events)
    runner.actor_vlm_stage_gate = True

    runner._configure_actor_vlm_stage_gate()

    actor_event = next(event for event in events if isinstance(event, tuple))
    rollout_event = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0].startswith("rollout")
    )
    assert actor_event[0] == "actor.configure_gate"
    assert actor_event[1] == rollout_event[1]
