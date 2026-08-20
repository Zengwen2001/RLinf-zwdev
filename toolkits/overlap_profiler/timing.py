# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Worker-side timing and absolute-start helpers for replay trials."""

from __future__ import annotations

import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .adapters import ActorFSDPReplayAdapter, OpenPIReplayAdapter


def _find_fixture(fixture_dir: str, role: str, kind: str, rank: int) -> Path:
    pattern = f"{role}-{kind}-rank{rank:04d}-*.pt"
    matches = sorted(Path(fixture_dir).glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one fixture matching {pattern!r} in {fixture_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def install_actor_replay(worker: Any, fixture_dir: str, stage: str) -> dict[str, Any]:
    _close_existing_adapter(worker)
    fixture = _find_fixture(fixture_dir, "actor", "actor_microbatch", worker._rank)
    worker._overlap_replay_adapter = ActorFSDPReplayAdapter(worker, str(fixture))
    worker._overlap_replay_role = "actor"
    worker._overlap_replay_adapter.prepare(stage)
    return _prepared_result(worker, fixture, stage)


def install_rollout_replay(worker: Any, fixture_dir: str, stage: str) -> dict[str, Any]:
    _close_existing_adapter(worker)
    fixture = _find_fixture(fixture_dir, "rollout", "rollout_observation", worker._rank)
    worker._overlap_replay_adapter = OpenPIReplayAdapter(worker.hf_model, str(fixture))
    worker._overlap_replay_role = "rollout"
    worker._overlap_replay_adapter.prepare(stage)
    return _prepared_result(worker, fixture, stage)


def _prepared_result(worker: Any, fixture: Path, stage: str) -> dict[str, Any]:
    return {
        "ok": True,
        "role": worker._overlap_replay_role,
        "rank": int(worker._rank),
        "node_rank": int(worker._cluster_node_rank),
        "stage": stage,
        "fixture": str(fixture.resolve()),
        "work_signature": worker._overlap_replay_adapter.work_signature(),
    }


def prepare_replay_iteration(worker: Any) -> dict[str, Any]:
    try:
        adapter = _get_adapter(worker)
        adapter.prepare_iteration()
        adapter.synchronize()
        return {
            "ok": True,
            "role": worker._overlap_replay_role,
            "rank": int(worker._rank),
        }
    except Exception as error:
        return _error_result(worker, error)


def run_replay_at(
    worker: Any, start_at_monotonic_ns: int, trial_id: str
) -> dict[str, Any]:
    adapter = _get_adapter(worker)
    role = worker._overlap_replay_role
    rank = int(worker._rank)
    node_rank = int(worker._cluster_node_rank)
    cpu_start_ns = None
    try:
        adapter.synchronize()
        device = _current_device()
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)
        _wait_until(int(start_at_monotonic_ns))

        start_event = (
            torch.cuda.Event(enable_timing=True) if device is not None else None
        )
        end_event = torch.cuda.Event(enable_timing=True) if device is not None else None
        cpu_start_ns = time.monotonic_ns()
        if start_event is not None:
            start_event.record()
        nvtx_range = (
            torch.cuda.nvtx.range(f"OVERLAP/{trial_id}/{role}/{adapter.stage.value}")
            if device is not None
            else nullcontext()
        )
        with nvtx_range:
            output = adapter.run_once()
        if end_event is not None:
            end_event.record()
        adapter.synchronize()
        cpu_end_ns = time.monotonic_ns()
        del output

        cuda_duration_ms = (
            float(start_event.elapsed_time(end_event))
            if start_event is not None
            else None
        )
        peak_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device is not None else 0
        )
        return {
            "ok": True,
            "trial_id": trial_id,
            "role": role,
            "rank": rank,
            "node_rank": node_rank,
            "device_id": str(device) if device is not None else "cpu",
            "stage": adapter.stage.value,
            "requested_start_ns": int(start_at_monotonic_ns),
            "start_ns": cpu_start_ns,
            "end_ns": cpu_end_ns,
            "duration_ms": (cpu_end_ns - cpu_start_ns) / 1e6,
            "cuda_duration_ms": cuda_duration_ms,
            "peak_memory_bytes": peak_memory_bytes,
            "work_signature": adapter.work_signature(),
            "work_units": adapter.work_units(),
        }
    except Exception as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        result = _error_result(worker, error)
        result.update(
            {
                "trial_id": trial_id,
                "requested_start_ns": int(start_at_monotonic_ns),
                "start_ns": cpu_start_ns,
                "end_ns": time.monotonic_ns(),
            }
        )
        return result


def reset_replay(worker: Any) -> dict[str, Any]:
    try:
        adapter = _get_adapter(worker)
        adapter.reset_between_trials()
        return {
            "ok": True,
            "role": worker._overlap_replay_role,
            "rank": int(worker._rank),
        }
    except Exception as error:
        return _error_result(worker, error)


def close_replay(worker: Any) -> dict[str, Any]:
    try:
        _close_existing_adapter(worker)
        return {
            "ok": True,
            "role": getattr(worker, "_overlap_replay_role", "unknown"),
            "rank": int(worker._rank),
        }
    except Exception as error:
        return _error_result(worker, error)


def _close_existing_adapter(worker: Any) -> None:
    adapter = getattr(worker, "_overlap_replay_adapter", None)
    if adapter is not None:
        adapter.close()
    worker._overlap_replay_adapter = None


def _get_adapter(worker: Any):
    adapter = getattr(worker, "_overlap_replay_adapter", None)
    if adapter is None:
        raise RuntimeError("prepare_overlap_replay() must be called before replay")
    return adapter


def _current_device() -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    return torch.device("cuda", torch.cuda.current_device())


def _wait_until(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 2_000_000:
            time.sleep((remaining_ns - 1_000_000) / 1e9)


def _error_result(worker: Any, error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "role": getattr(worker, "_overlap_replay_role", "unknown"),
        "rank": int(getattr(worker, "_rank", -1)),
        "node_rank": int(getattr(worker, "_cluster_node_rank", -1)),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }
