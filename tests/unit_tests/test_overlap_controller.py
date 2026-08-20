# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from omegaconf import OmegaConf

from toolkits.overlap_profiler.controller import (
    PairwiseController,
    run_configured_matrix,
)


class _Pending:
    def __init__(self, future):
        self.future = future

    def wait(self):
        return [self.future.result()]


class _Endpoint:
    executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self, role, duration_ms):
        self.role = role
        self.duration_ms = duration_ms
        self.stage = None

    def prepare(self, fixture_dir, stage):
        self.stage = stage
        return [{"ok": True, "role": self.role, "rank": 0, "node_rank": 0}]

    def prepare_iteration(self):
        return [{"ok": True, "role": self.role, "rank": 0}]

    def dispatch_run(self, start_ns, trial_id):
        def run():
            while time.monotonic_ns() < start_ns:
                time.sleep(0.0001)
            actual_start = time.monotonic_ns()
            time.sleep(self.duration_ms / 1000)
            end = time.monotonic_ns()
            return {
                "ok": True,
                "trial_id": trial_id,
                "role": self.role,
                "rank": 0,
                "node_rank": 0,
                "device_id": "cpu",
                "stage": self.stage,
                "requested_start_ns": start_ns,
                "start_ns": actual_start,
                "end_ns": end,
                "duration_ms": (end - actual_start) / 1e6,
                "cuda_duration_ms": None,
                "peak_memory_bytes": 0,
                "work_signature": {"stage": self.stage, "rank": 0},
            }

        return _Pending(self.executor.submit(run))

    def reset(self):
        return [{"ok": True, "role": self.role, "rank": 0}]

    def close(self):
        return [{"ok": True, "role": self.role, "rank": 0}]


def test_controller_runs_alone_and_concurrent_trials(tmp_path):
    controller = PairwiseController(
        {"actor": _Endpoint("actor", 15), "rollout": _Endpoint("rollout", 10)},
        fixture_dir=str(tmp_path),
        output_dir=str(tmp_path / "run"),
        warmup=0,
        repetitions=2,
        start_lead_ms=5,
        max_start_skew_ms=5,
    )
    samples = controller.run(
        [
            {
                "id": "test_pair",
                "a": {"role": "actor", "stage": "actor.forward"},
                "b": {"role": "rollout", "stage": "rollout.vlm_prefix"},
            }
        ]
    )

    assert len(samples) == 2
    assert all(sample["accepted"] for sample in samples)
    assert all(sample["benefit_ms"] > 0 for sample in samples)
    assert len((tmp_path / "run" / "raw_trials.jsonl").read_text().splitlines()) == 6
    assert (tmp_path / "run" / "experiment.json").exists()


def test_controller_supports_intentional_gpu_alignment_delay(tmp_path):
    controller = PairwiseController(
        {"actor": _Endpoint("actor", 30), "rollout": _Endpoint("rollout", 10)},
        fixture_dir=str(tmp_path),
        output_dir=str(tmp_path / "run"),
        warmup=0,
        repetitions=1,
        start_lead_ms=5,
        max_start_skew_ms=5,
    )

    sample = controller.run(
        [
            {
                "id": "gpu_aligned",
                "alignment_mode": "gpu_compute",
                "a": {"role": "actor", "stage": "actor.forward"},
                "b": {
                    "role": "rollout",
                    "stage": "rollout.vlm_prefix",
                    "start_delay_ms": 10,
                },
            }
        ]
    )[0]

    assert sample["accepted"] is True
    assert sample["alignment_mode"] == "gpu_compute"
    assert sample["b_start_delay_ms"] == 10
    assert sample["scheduled_start_offset_ms"] == pytest.approx(10, abs=2)


def test_controller_rejects_multiple_nodes(tmp_path):
    actor = _Endpoint("actor", 1)
    rollout = _Endpoint("rollout", 1)

    def prepare_other_node(fixture_dir, stage):
        rollout.stage = stage
        return [{"ok": True, "role": "rollout", "rank": 0, "node_rank": 1}]

    rollout.prepare = prepare_other_node
    controller = PairwiseController(
        {"actor": actor, "rollout": rollout},
        fixture_dir=str(tmp_path),
        output_dir=str(tmp_path / "run"),
        warmup=0,
        repetitions=1,
    )

    with pytest.raises(ValueError, match="one physical node"):
        controller.run(
            [
                {
                    "id": "p",
                    "a": {"role": "actor", "stage": "a"},
                    "b": {"role": "rollout", "stage": "b"},
                }
            ]
        )


def test_configured_matrix_wraps_run_in_profile_window(tmp_path, monkeypatch):
    pairs_path = tmp_path / "pairs.yaml"
    pairs_path.write_text("pairs: []\n")
    events = []

    class _Immediate:
        @staticmethod
        def wait():
            return []

    class _Group:
        def __init__(self, role):
            self.role = role

        def start_profile(self, step):
            events.append((self.role, "start", step))
            return _Immediate()

        def stop_profile(self):
            events.append((self.role, "stop"))
            return _Immediate()

    class _Controller:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def run(pairs):
            assert pairs == []
            events.append(("controller", "run"))

    monkeypatch.setattr(
        "toolkits.overlap_profiler.controller.PairwiseController", _Controller
    )
    cfg = OmegaConf.create(
        {
            "cluster": {"profiling": {"enabled": True, "steps": [0]}},
            "runner": {
                "overlap_profile": {
                    "enabled": True,
                    "fixture_dir": str(tmp_path / "fixtures"),
                    "pairs_file": str(pairs_path),
                    "output_dir": str(tmp_path / "output"),
                    "profile_workers": True,
                }
            },
        }
    )

    assert run_configured_matrix(cfg, _Group("actor"), _Group("rollout"))
    assert events == [
        ("actor", "start", 0),
        ("rollout", "start", 0),
        ("controller", "run"),
        ("actor", "stop"),
        ("rollout", "stop"),
    ]
