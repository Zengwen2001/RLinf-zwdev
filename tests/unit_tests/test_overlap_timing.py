# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import time
from types import SimpleNamespace

from toolkits.overlap_profiler.timing import run_replay_at


class _Adapter:
    stage = SimpleNamespace(value="actor.forward")

    @staticmethod
    def synchronize():
        pass

    @staticmethod
    def run_once():
        time.sleep(0.001)
        return None

    @staticmethod
    def work_signature():
        return {"shape": [8, 4]}

    @staticmethod
    def work_units():
        return {"samples": 8}


def test_run_replay_at_uses_absolute_start_and_returns_fixed_work(monkeypatch):
    monkeypatch.setattr(
        "toolkits.overlap_profiler.timing._current_device", lambda: None
    )
    worker = SimpleNamespace(
        _rank=0,
        _cluster_node_rank=0,
        _overlap_replay_role="actor",
        _overlap_replay_adapter=_Adapter(),
    )
    requested_start = time.monotonic_ns() + 2_000_000

    result = run_replay_at(worker, requested_start, "pair:0:A_alone")

    assert result["ok"] is True
    assert result["start_ns"] >= requested_start
    assert result["duration_ms"] >= 1.0
    assert result["work_units"] == {"samples": 8}
