# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest

from toolkits.overlap_profiler.metrics import (
    compute_pair_sample,
    interval_concurrency,
    interval_union_ns,
    summarize_pairs,
)


def _result(role, rank, start, end, signature=None, work_units=None):
    return {
        "ok": True,
        "role": role,
        "rank": rank,
        "start_ns": start,
        "end_ns": end,
        "duration_ms": (end - start) / 1e6,
        "cuda_duration_ms": None,
        "peak_memory_bytes": 100,
        "work_signature": signature or {"role": role, "rank": rank},
        "work_units": work_units or {"samples": 10},
    }


def test_interval_union_and_concurrency_do_not_double_count():
    intervals = [(10, 40), (20, 50), (70, 80)]
    assert interval_union_ns(intervals) == 50
    ratio, maximum = interval_concurrency(intervals, (0, 100))
    assert ratio == pytest.approx(0.2)
    assert maximum == 2


def test_pair_metrics_use_actual_intervals():
    sample = compute_pair_sample(
        pair_id="p",
        repetition=0,
        stage_a="actor.forward",
        stage_b="rollout.vlm_prefix",
        a_alone=[_result("actor", 0, 0, 100_000_000)],
        b_alone=[_result("rollout", 0, 0, 60_000_000)],
        concurrent=[
            _result("actor", 0, 1_000_000_000, 1_120_000_000),
            _result("rollout", 0, 1_001_000_000, 1_081_000_000),
        ],
    )

    assert sample["accepted"] is True
    assert sample["a_slowdown"] == pytest.approx(1.2)
    assert sample["b_slowdown"] == pytest.approx(80 / 60)
    assert sample["concurrent_makespan_ms"] == pytest.approx(120)
    assert sample["benefit_ms"] == pytest.approx(40)
    assert sample["interference_ms"] == pytest.approx(20)
    assert sample["overlap_coverage"] == pytest.approx(1.0)
    assert sample["start_skew_ms"] == pytest.approx(1.0)
    assert sample["a_alone_work_units"] == {"samples": 10.0}
    assert sample["a_alone_throughput"]["samples"] == pytest.approx(100.0)


def test_pair_rejects_signature_and_coverage_mismatch():
    sample = compute_pair_sample(
        pair_id="p",
        repetition=0,
        stage_a="a",
        stage_b="b",
        a_alone=[_result("actor", 0, 0, 10, {"shape": 1})],
        b_alone=[_result("rollout", 0, 0, 10)],
        concurrent=[
            _result("actor", 0, 100, 110, {"shape": 2}),
            _result("rollout", 0, 109, 119),
        ],
        max_start_skew_ms=0.000001,
        min_overlap_coverage=0.8,
    )

    assert sample["accepted"] is False
    assert "work_signature_a_mismatch" in sample["rejection_reasons"]
    assert "start_skew_exceeded" in sample["rejection_reasons"]
    assert "overlap_coverage_below_threshold" in sample["rejection_reasons"]


def test_pair_summary_excludes_rejected_samples():
    accepted = {
        "pair_id": "p",
        "stage_a": "a",
        "stage_b": "b",
        "accepted": True,
        "a_slowdown": 1.1,
        "b_slowdown": 1.2,
        "benefit_ms": 3.0,
        "benefit_norm": 0.2,
        "interference_ms": 1.0,
        "overlap_coverage": 0.9,
        "start_skew_ms": 0.2,
    }
    rejected = {**accepted, "accepted": False, "benefit_ms": -100.0}
    summary = summarize_pairs([accepted, rejected])[0]
    assert summary["accepted_samples"] == 1
    assert summary["benefit_ms_p50"] == pytest.approx(3.0)
