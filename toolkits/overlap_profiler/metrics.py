# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pairwise overlap metrics and interval statistics."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from statistics import median
from typing import Any

import numpy as np


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def interval_concurrency(
    intervals: Iterable[tuple[int, int]], window: tuple[int, int]
) -> tuple[float, int]:
    window_start, window_end = window
    events: list[tuple[int, int]] = []
    for start, end in intervals:
        start = max(int(start), window_start)
        end = min(int(end), window_end)
        if end > start:
            events.extend(((start, 1), (end, -1)))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    previous = window_start
    concurrent_ns = 0
    maximum = 0
    for timestamp, delta in events:
        if active >= 2:
            concurrent_ns += max(0, timestamp - previous)
        active += delta
        maximum = max(maximum, active)
        previous = timestamp
    wall = max(0, window_end - window_start)
    return (concurrent_ns / wall if wall else 0.0), maximum


def summarize_role_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"ok": False, "errors": ["missing_results"]}
    errors = [
        f"rank{item.get('rank', '?')}:{item.get('error_type', 'error')}"
        for item in results
        if not item.get("ok", False)
    ]
    if errors:
        return {"ok": False, "errors": errors}
    starts = [int(item["start_ns"]) for item in results]
    ends = [int(item["end_ns"]) for item in results]
    durations = [float(item["duration_ms"]) for item in results]
    cuda_durations = [
        float(item["cuda_duration_ms"])
        for item in results
        if item.get("cuda_duration_ms") is not None
    ]
    signatures = {
        int(item["rank"]): json.dumps(
            item["work_signature"], sort_keys=True, separators=(",", ":")
        )
        for item in results
    }
    work_units: dict[str, float] = {}
    for item in results:
        for name, value in item.get("work_units", {}).items():
            work_units[name] = work_units.get(name, 0.0) + float(value)
    duration_ms = max(durations)
    return {
        "ok": True,
        "start_ns": min(starts),
        "end_ns": max(ends),
        "duration_ms": duration_ms,
        "cuda_duration_ms": max(cuda_durations) if cuda_durations else None,
        "start_skew_ms": (max(starts) - min(starts)) / 1e6,
        "peak_memory_bytes": max(
            int(item.get("peak_memory_bytes", 0)) for item in results
        ),
        "signatures": signatures,
        "ranks": len(results),
        "work_units": work_units,
        "throughput": {
            name: value / (duration_ms / 1000.0)
            for name, value in work_units.items()
            if duration_ms > 0
        },
    }


def compute_pair_sample(
    *,
    pair_id: str,
    repetition: int,
    stage_a: str,
    stage_b: str,
    a_alone: list[dict[str, Any]],
    b_alone: list[dict[str, Any]],
    concurrent: list[dict[str, Any]],
    alignment_mode: str = "stage_entry",
    concurrent_start_delays_ms: dict[str, float] | None = None,
    max_start_skew_ms: float = 2.0,
    min_overlap_coverage: float = 0.8,
) -> dict[str, Any]:
    a_role = a_alone[0].get("role", "actor") if a_alone else "actor"
    b_role = b_alone[0].get("role", "rollout") if b_alone else "rollout"
    concurrent_a = [item for item in concurrent if item.get("role") == a_role]
    concurrent_b = [item for item in concurrent if item.get("role") == b_role]
    summaries = {
        "a_alone": summarize_role_results(a_alone),
        "b_alone": summarize_role_results(b_alone),
        "a_concurrent": summarize_role_results(concurrent_a),
        "b_concurrent": summarize_role_results(concurrent_b),
    }
    reasons: list[str] = []
    for name, summary in summaries.items():
        if not summary.get("ok", False):
            reasons.extend(f"{name}:{error}" for error in summary.get("errors", []))

    result: dict[str, Any] = {
        "pair_id": pair_id,
        "repetition": int(repetition),
        "stage_a": stage_a,
        "stage_b": stage_b,
        "role_a": a_role,
        "role_b": b_role,
    }
    if reasons:
        result.update({"accepted": False, "rejection_reasons": reasons})
        return result

    a = summaries["a_alone"]
    b = summaries["b_alone"]
    ac = summaries["a_concurrent"]
    bc = summaries["b_concurrent"]
    if a["signatures"] != ac["signatures"]:
        reasons.append("work_signature_a_mismatch")
    if b["signatures"] != bc["signatures"]:
        reasons.append("work_signature_b_mismatch")

    if all(item.get("requested_start_ns") is not None for item in concurrent):
        quality_starts_ns = [
            int(item["start_ns"]) - int(item["requested_start_ns"])
            for item in concurrent
        ]
    else:
        quality_starts_ns = [int(item["start_ns"]) for item in concurrent]
    start_skew_ms = (max(quality_starts_ns) - min(quality_starts_ns)) / 1e6
    scheduled_start_offset_ms = (bc["start_ns"] - ac["start_ns"]) / 1e6
    delays = concurrent_start_delays_ms or {a_role: 0.0, b_role: 0.0}
    overlap_ns = max(
        0,
        min(ac["end_ns"], bc["end_ns"]) - max(ac["start_ns"], bc["start_ns"]),
    )
    shorter_ns = min(ac["end_ns"] - ac["start_ns"], bc["end_ns"] - bc["start_ns"])
    coverage = overlap_ns / shorter_ns if shorter_ns > 0 else 0.0
    if start_skew_ms > max_start_skew_ms:
        reasons.append("start_skew_exceeded")
    if coverage < min_overlap_coverage:
        reasons.append("overlap_coverage_below_threshold")

    makespan_ms = (
        max(ac["end_ns"], bc["end_ns"]) - min(ac["start_ns"], bc["start_ns"])
    ) / 1e6
    serial_ms = a["duration_ms"] + b["duration_ms"]
    benefit_ms = serial_ms - makespan_ms
    result.update(
        {
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "a_alone_ms": a["duration_ms"],
            "b_alone_ms": b["duration_ms"],
            "a_concurrent_ms": ac["duration_ms"],
            "b_concurrent_ms": bc["duration_ms"],
            "a_slowdown": ac["duration_ms"] / a["duration_ms"],
            "b_slowdown": bc["duration_ms"] / b["duration_ms"],
            "concurrent_makespan_ms": makespan_ms,
            "overlap_ms": overlap_ns / 1e6,
            "overlap_coverage": coverage,
            "start_skew_ms": start_skew_ms,
            "alignment_mode": alignment_mode,
            "a_start_delay_ms": float(delays.get(a_role, 0.0)),
            "b_start_delay_ms": float(delays.get(b_role, 0.0)),
            "scheduled_start_offset_ms": scheduled_start_offset_ms,
            "benefit_ms": benefit_ms,
            "benefit_norm": benefit_ms / serial_ms if serial_ms else 0.0,
            "interference_ms": makespan_ms - max(a["duration_ms"], b["duration_ms"]),
            "peak_memory_bytes": max(ac["peak_memory_bytes"], bc["peak_memory_bytes"]),
            "a_alone_work_units": a["work_units"],
            "b_alone_work_units": b["work_units"],
            "a_alone_throughput": a["throughput"],
            "b_alone_throughput": b["throughput"],
            "a_concurrent_throughput": ac["throughput"],
            "b_concurrent_throughput": bc["throughput"],
        }
    )
    return result


def distribution_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = float(np.mean(array))
    if array.size == 1:
        ci_low = ci_high = float(array[0])
    else:
        standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
        ci_low, ci_high = mean - 1.96 * standard_error, mean + 1.96 * standard_error
    return {
        "count": int(array.size),
        "mean": mean,
        "p10": float(np.percentile(array, 10)),
        "p50": float(median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def summarize_pairs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["pair_id"], []).append(sample)
    rows = []
    metric_names = (
        "a_slowdown",
        "b_slowdown",
        "benefit_ms",
        "benefit_norm",
        "interference_ms",
        "overlap_coverage",
        "start_skew_ms",
        "scheduled_start_offset_ms",
    )
    for pair_id, pair_samples in sorted(grouped.items()):
        accepted = [sample for sample in pair_samples if sample.get("accepted")]
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "stage_a": pair_samples[0]["stage_a"],
            "stage_b": pair_samples[0]["stage_b"],
            "alignment_mode": pair_samples[0].get("alignment_mode", "stage_entry"),
            "a_start_delay_ms": pair_samples[0].get("a_start_delay_ms", 0.0),
            "b_start_delay_ms": pair_samples[0].get("b_start_delay_ms", 0.0),
            "total_samples": len(pair_samples),
            "accepted_samples": len(accepted),
        }
        for metric in metric_names:
            summary = distribution_summary(
                float(sample[metric]) for sample in accepted if metric in sample
            )
            for statistic, value in summary.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return rows
