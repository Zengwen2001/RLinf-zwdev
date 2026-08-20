# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import sqlite3

import pytest

from toolkits.overlap_profiler.nsys_parser import (
    UnsupportedNsightSchema,
    parse_nsys_sqlite,
)
from toolkits.overlap_profiler.report import generate_report


def _create_sqlite(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT,
                                 textId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER, end INTEGER, streamId INTEGER, deviceId INTEGER,
            demangledName INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER,
            correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER,
            correlationId INTEGER);
        CREATE TABLE META_DATA_EXPORT(name TEXT, value TEXT);
        """
    )
    connection.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [
            (1, "gemm"),
            (2, "ncclAllReduce"),
            (3, "cudaLaunchKernel"),
            (4, "cudaDeviceSynchronize"),
        ],
    )
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
        (0, 100_000_000, "ACTOR/FORWARD", None, 7),
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
        [
            (10_000_000, 40_000_000, 1, 0, 1, 101),
            (20_000_000, 50_000_000, 2, 0, 2, 102),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (5_000_000, 6_000_000, 7, 3, 101),
            (15_000_000, 16_000_000, 7, 3, 102),
            (25_000_000, 26_000_000, 7, 3, 103),
            (60_000_000, 70_000_000, 7, 4, None),
        ],
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?)",
        (25_000_000, 35_000_000, 4096, 1, 103),
    )
    connection.execute(
        "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
        ("EXPORT_PRODUCT_VERSION", "2026.4.1"),
    )
    connection.commit()
    connection.close()


def test_parser_derives_stage_interval_metrics(tmp_path):
    path = tmp_path / "rlinf_profile_ActorGroup_1_123.sqlite"
    _create_sqlite(path)
    row = parse_nsys_sqlite(path)[0]

    assert row["role"] == "actor"
    assert row["rank"] == 1
    assert row["source_version"] == "2026.4.1"
    assert row["stage_wall_ms"] == pytest.approx(100)
    assert row["kernel_sum_ms"] == pytest.approx(60)
    assert row["gpu_active_union_ms"] == pytest.approx(40)
    assert row["gpu_idle_inside_stage_ms"] == pytest.approx(60)
    assert row["gpu_idle_inside_compute_envelope_ms"] == pytest.approx(0)
    assert row["gpu_compute_envelope_ms"] == pytest.approx(40)
    assert row["gpu_first_activity_offset_ms"] == pytest.approx(10)
    assert row["gpu_last_activity_offset_ms"] == pytest.approx(50)
    assert row["kernel_attribution"] == "cuda_launch_correlation"
    assert row["stream_concurrency_ratio"] == pytest.approx(0.5)
    assert row["max_concurrent_kernels"] == 2
    assert row["cpu_launch_gap_p50_ms"] == pytest.approx(10)
    assert row["cuda_sync_blocking_ms"] == pytest.approx(10)
    assert row["cuda_device_sync_ms"] == pytest.approx(10)
    assert row["cuda_stream_sync_ms"] == pytest.approx(0)
    assert row["memcpy_h2d_bytes"] == 4096
    assert row["nccl_ms"] == pytest.approx(30)
    assert row["resource_sm_active"] == "unavailable"
    assert row["nccl_collective_metadata"] == "unavailable"


def test_parser_attributes_async_kernel_tail_to_launching_stage(tmp_path):
    path = tmp_path / "rlinf_profile_ActorGroup_0_123.sqlite"
    _create_sqlite(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        (90_000_000, 91_000_000, 7, 3, 104),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
        (120_000_000, 150_000_000, 1, 0, 1, 104),
    )
    connection.commit()
    connection.close()

    row = parse_nsys_sqlite(path)[0]

    assert row["stage_wall_ms"] == pytest.approx(100)
    assert row["gpu_last_activity_offset_ms"] == pytest.approx(150)
    assert row["gpu_compute_envelope_ms"] == pytest.approx(140)
    assert row["kernel_sum_ms"] == pytest.approx(90)


def test_parser_reports_missing_required_tables(tmp_path):
    path = tmp_path / "empty.sqlite"
    sqlite3.connect(path).close()
    with pytest.raises(UnsupportedNsightSchema, match="missing required tables"):
        parse_nsys_sqlite(path)


def test_parser_extracts_pair_trial_identity(tmp_path):
    path = tmp_path / "rlinf_profile_RolloutGroup_0_1.sqlite"
    _create_sqlite(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
        (
            0,
            100_000_000,
            "OVERLAP/forward__vlm:3:concurrent/rollout/rollout.vlm_prefix",
            None,
            7,
        ),
    )
    connection.commit()
    connection.close()

    row = next(
        item for item in parse_nsys_sqlite(path) if item["stage"].startswith("OVERLAP/")
    )

    assert row["pair_id"] == "forward__vlm"
    assert row["repetition"] == 3
    assert row["trial_kind"] == "concurrent"


def test_report_writes_csv_json_and_zoomable_html(tmp_path):
    run_dir = tmp_path / "run"
    sqlite_dir = run_dir / "sqlite"
    sqlite_dir.mkdir(parents=True)
    _create_sqlite(sqlite_dir / "rlinf_profile_ActorGroup_0_1.sqlite")
    sample = {
        "pair_id": "p",
        "repetition": 0,
        "stage_a": "actor.forward",
        "stage_b": "rollout.vlm_prefix",
        "accepted": True,
        "a_slowdown": 1.1,
        "b_slowdown": 1.2,
        "benefit_ms": 4.0,
        "benefit_norm": 0.2,
        "interference_ms": 1.0,
        "overlap_coverage": 0.9,
        "start_skew_ms": 0.2,
    }
    (run_dir / "pair_samples.jsonl").write_text(json.dumps(sample) + "\n")

    outputs = generate_report(run_dir)

    assert all(path.exists() for path in outputs.values())
    report = outputs["report"].read_text()
    assert "Plotly.newPlot" in report
    assert "rangeslider" in report
    assert "Net benefit" in report
