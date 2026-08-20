# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Nsight Systems SQLite export and stage-level timing extraction."""

from __future__ import annotations

import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import interval_concurrency, interval_union_ns

STAGE_PREFIXES = ("ACTOR/", "ROLLOUT/", "ENV/", "WEIGHT_SYNC/", "OVERLAP/")
KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
RUNTIME_TABLE = "CUPTI_ACTIVITY_KIND_RUNTIME"
MEMCPY_TABLE = "CUPTI_ACTIVITY_KIND_MEMCPY"


class UnsupportedNsightSchema(RuntimeError):
    pass


def export_nsys_report(
    report_path: str | Path, sqlite_path: str | Path | None = None
) -> Path:
    report = Path(report_path)
    output = (
        Path(sqlite_path) if sqlite_path is not None else report.with_suffix(".sqlite")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "--output",
            str(output),
            str(report),
        ],
        check=True,
    )
    return output


def parse_nsys_sqlite(path: str | Path) -> list[dict[str, Any]]:
    sqlite_path = Path(path)
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        tables = _table_names(connection)
        required = {"NVTX_EVENTS", "StringIds", KERNEL_TABLE}
        missing = required - tables
        if missing:
            raise UnsupportedNsightSchema(
                f"Nsight SQLite {sqlite_path} is missing required tables: "
                f"{sorted(missing)}; available tables: {sorted(tables)}"
            )
        strings = {
            int(row["id"]): row["value"]
            for row in connection.execute("SELECT id, value FROM StringIds")
        }
        kernels = _load_kernels(connection, strings)
        runtimes = _load_runtime(connection, strings) if RUNTIME_TABLE in tables else []
        memcpys = _load_memcpy(connection) if MEMCPY_TABLE in tables else []
        stages = _load_stages(connection, strings)
        runtimes_by_stage = _associate_intervals(stages, runtimes)
        kernels_by_stage, kernel_attribution = _associate_launched_work(
            stages, runtimes_by_stage, kernels
        )
        memcpys_by_stage, memcpy_attribution = _associate_launched_work(
            stages, runtimes_by_stage, memcpys
        )
        role, rank = _infer_report_identity(sqlite_path.name)
        source_version = _metadata_value(
            connection, tables, "META_DATA_EXPORT", "EXPORT_PRODUCT_VERSION"
        )
        return [
            _stage_metrics(
                stage,
                kernels=kernels_by_stage[index],
                runtimes=runtimes_by_stage[index],
                memcpys=memcpys_by_stage[index],
                source_file=str(sqlite_path.resolve()),
                role=role,
                rank=rank,
                source_version=source_version,
                kernel_attribution=kernel_attribution,
                memcpy_attribution=memcpy_attribution,
            )
            for index, stage in enumerate(stages)
        ]
    finally:
        connection.close()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _metadata_value(
    connection: sqlite3.Connection,
    tables: set[str],
    table: str,
    name: str,
) -> str | None:
    if table not in tables:
        return None
    row = connection.execute(
        f"SELECT value FROM {table} WHERE name = ? LIMIT 1", (name,)
    ).fetchone()
    return str(row[0]) if row is not None else None


def _load_stages(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    stages = []
    for row in connection.execute(
        "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND end > start"
    ):
        text = row["text"] or strings.get(row["textId"])
        if text and text.startswith(STAGE_PREFIXES):
            stages.append(
                {
                    "stage": text,
                    "start_ns": int(row["start"]),
                    "end_ns": int(row["end"]),
                    "global_tid": row["globalTid"],
                }
            )
    return stages


def _load_kernels(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    has_correlation = _has_column(connection, KERNEL_TABLE, "correlationId")
    correlation = "correlationId" if has_correlation else "NULL AS correlationId"
    return [
        {
            "start_ns": int(row["start"]),
            "end_ns": int(row["end"]),
            "stream_id": int(row["streamId"]),
            "device_id": int(row["deviceId"]),
            "name": strings.get(row["demangledName"], str(row["demangledName"])),
            "correlation_id": row["correlationId"],
        }
        for row in connection.execute(
            f"SELECT start, end, streamId, deviceId, demangledName, {correlation} "
            f"FROM {KERNEL_TABLE}"
        )
    ]


def _load_runtime(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    has_correlation = _has_column(connection, RUNTIME_TABLE, "correlationId")
    correlation = "runtime.correlationId" if has_correlation else "NULL"
    return [
        {
            "start_ns": int(row["start"]),
            "end_ns": int(row["end"]),
            "global_tid": row["globalTid"],
            "name": strings.get(row["nameId"], str(row["nameId"])),
            "correlation_id": row["correlationId"],
        }
        for row in connection.execute(
            f"SELECT runtime.start, runtime.end, runtime.globalTid, runtime.nameId, "
            f"{correlation} AS correlationId "
            f"FROM {RUNTIME_TABLE} AS runtime "
            "JOIN StringIds AS strings ON strings.id = runtime.nameId "
            "WHERE strings.value LIKE '%Launch%' "
            "OR strings.value LIKE '%Synchronize%' "
            "OR strings.value LIKE '%WaitEvent%' "
            "OR strings.value LIKE '%Memcpy%'"
        )
    ]


def _load_memcpy(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    has_correlation = _has_column(connection, MEMCPY_TABLE, "correlationId")
    correlation = "correlationId" if has_correlation else "NULL AS correlationId"
    return [
        {
            "start_ns": int(row["start"]),
            "end_ns": int(row["end"]),
            "bytes": int(row["bytes"]),
            "copy_kind": int(row["copyKind"]),
            "correlation_id": row["correlationId"],
        }
        for row in connection.execute(
            f"SELECT start, end, bytes, copyKind, {correlation} FROM {MEMCPY_TABLE}"
        )
    ]


def _stage_metrics(
    stage: dict[str, Any],
    *,
    kernels: list[dict[str, Any]],
    runtimes: list[dict[str, Any]],
    memcpys: list[dict[str, Any]],
    source_file: str,
    role: str | None,
    rank: int | None,
    source_version: str | None,
    kernel_attribution: str,
    memcpy_attribution: str,
) -> dict[str, Any]:
    start, end = stage["start_ns"], stage["end_ns"]
    stage_kernels = kernels
    kernel_intervals = [
        (item["start_ns"], item["end_ns"]) for item in stage_kernels
    ]
    first_gpu_ns = min((item["start_ns"] for item in stage_kernels), default=None)
    last_gpu_ns = max((item["end_ns"] for item in stage_kernels), default=None)
    gpu_window = (
        (first_gpu_ns, last_gpu_ns)
        if first_gpu_ns is not None and last_gpu_ns is not None
        else (start, start)
    )
    active_ns = interval_union_ns(kernel_intervals)
    stage_active_ns = interval_union_ns(
        [
            (max(start, interval_start), min(end, interval_end))
            for interval_start, interval_end in kernel_intervals
        ]
    )
    concurrency_ratio, max_concurrent = interval_concurrency(
        kernel_intervals, gpu_window
    )
    stage_runtimes = [
        item
        for item in runtimes
        if (stage["global_tid"] is None or item["global_tid"] == stage["global_tid"])
    ]
    launch_starts = sorted(
        item["start_ns"]
        for item in stage_runtimes
        if "Launch" in item["name"] or "launch" in item["name"]
    )
    launch_gaps_ms = np.diff(launch_starts) / 1e6 if len(launch_starts) > 1 else []
    explicit_sync_ns = sum(
        min(end, item["end_ns"]) - max(start, item["start_ns"])
        for item in stage_runtimes
        if any(token in item["name"] for token in ("Synchronize", "WaitEvent"))
    )
    device_sync_ns = _runtime_duration_ns(
        stage_runtimes, start, end, "DeviceSynchronize"
    )
    stream_sync_ns = _runtime_duration_ns(
        stage_runtimes, start, end, "StreamSynchronize"
    )
    event_wait_ns = _runtime_duration_ns(
        stage_runtimes, start, end, "EventSynchronize", "WaitEvent"
    )
    blocking_memcpy_ns = sum(
        min(end, item["end_ns"]) - max(start, item["start_ns"])
        for item in stage_runtimes
        if "Memcpy" in item["name"] and "Async" not in item["name"]
    )
    sync_runtime_ns = explicit_sync_ns + blocking_memcpy_ns
    stage_memcpys = memcpys
    h2d = [item for item in stage_memcpys if item["copy_kind"] in {1, 11}]
    d2h = [item for item in stage_memcpys if item["copy_kind"] in {2, 12}]
    nccl_intervals = [
        (item["start_ns"], item["end_ns"])
        for item in stage_kernels
        if "nccl" in item["name"].lower()
    ]
    wall_ns = end - start
    gpu_envelope_ns = max(0, gpu_window[1] - gpu_window[0])
    result = {
        "source_file": source_file,
        "source_backend": "nsight_systems",
        "source_version": source_version,
        "role": role,
        "rank": rank,
        "stage": stage["stage"],
        "start_ns": start,
        "end_ns": end,
        "stage_wall_ms": wall_ns / 1e6,
        "kernel_sum_ms": sum(interval[1] - interval[0] for interval in kernel_intervals)
        / 1e6,
        "gpu_active_union_ms": active_ns / 1e6,
        "gpu_idle_inside_stage_ms": max(0, wall_ns - stage_active_ns) / 1e6,
        "gpu_idle_inside_compute_envelope_ms": max(
            0, gpu_envelope_ns - active_ns
        )
        / 1e6,
        "gpu_compute_envelope_ms": gpu_envelope_ns / 1e6,
        "gpu_first_activity_offset_ms": (
            (first_gpu_ns - start) / 1e6 if first_gpu_ns is not None else None
        ),
        "gpu_last_activity_offset_ms": (
            (last_gpu_ns - start) / 1e6 if last_gpu_ns is not None else None
        ),
        "gpu_end_ns": last_gpu_ns,
        "kernel_attribution": kernel_attribution,
        "memcpy_attribution": memcpy_attribution,
        "stream_concurrency_ratio": concurrency_ratio,
        "max_concurrent_kernels": max_concurrent,
        "cpu_launch_gap_p50_ms": float(np.percentile(launch_gaps_ms, 50))
        if len(launch_gaps_ms)
        else None,
        "cpu_launch_gap_p95_ms": float(np.percentile(launch_gaps_ms, 95))
        if len(launch_gaps_ms)
        else None,
        "cuda_sync_blocking_ms": sync_runtime_ns / 1e6,
        "cuda_device_sync_ms": device_sync_ns / 1e6,
        "cuda_stream_sync_ms": stream_sync_ns / 1e6,
        "cuda_event_wait_ms": event_wait_ns / 1e6,
        "blocking_memcpy_sync_ms": blocking_memcpy_ns / 1e6,
        "memcpy_h2d_ms": _duration_sum_ms(
            h2d, start, end, clip=memcpy_attribution != "cuda_launch_correlation"
        ),
        "memcpy_h2d_bytes": sum(item["bytes"] for item in h2d),
        "memcpy_d2h_ms": _duration_sum_ms(
            d2h, start, end, clip=memcpy_attribution != "cuda_launch_correlation"
        ),
        "memcpy_d2h_bytes": sum(item["bytes"] for item in d2h),
        "nccl_ms": interval_union_ns(nccl_intervals) / 1e6,
        "nccl_collective_metadata": "unavailable",
        "nccl_metadata_reason": (
            "Nsight kernel names do not reliably provide collective payload bytes, "
            "algBW, or busBW; use NCCL trace or explicit collective metadata"
        ),
        "kernel_count": len(stage_kernels),
        "resource_sm_active": "unavailable",
        "resource_tensor_active": "unavailable",
        "resource_dram_active": "unavailable",
        "resource_nvlink_traffic": "unavailable",
        "resource_pcie_traffic": "unavailable",
        "resource_unavailable_reason": (
            "GPU Metrics samples are not decoded by this timeline parser; "
            "use Nsight GPU Metrics output for SM/Tensor/DRAM activity"
        ),
    }
    result.update(_overlap_trial_identity(stage["stage"]))
    return result


def _runtime_duration_ns(
    items: list[dict[str, Any]], start: int, end: int, *tokens: str
) -> int:
    return sum(
        min(end, item["end_ns"]) - max(start, item["start_ns"])
        for item in items
        if any(token in item["name"] for token in tokens)
    )


def _overlap_trial_identity(stage: str) -> dict[str, Any]:
    if not stage.startswith("OVERLAP/"):
        return {
            "pair_id": None,
            "repetition": None,
            "trial_kind": None,
            "trial_id": None,
        }
    parts = stage.split("/", 3)
    if len(parts) != 4:
        return {
            "pair_id": None,
            "repetition": None,
            "trial_kind": None,
            "trial_id": None,
        }
    trial_id = parts[1]
    trial_parts = trial_id.rsplit(":", 2)
    if len(trial_parts) != 3:
        return {
            "pair_id": None,
            "repetition": None,
            "trial_kind": None,
            "trial_id": trial_id,
        }
    pair_id, repetition, trial_kind = trial_parts
    try:
        repetition_value = int(repetition)
    except ValueError:
        repetition_value = None
    return {
        "pair_id": pair_id,
        "repetition": repetition_value,
        "trial_kind": trial_kind,
        "trial_id": trial_id,
    }


def _associate_intervals(
    stages: list[dict[str, Any]], items: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Associate overlapping intervals in one sweep instead of per-stage scans."""
    associated = [[] for _ in stages]
    stage_order = sorted(
        range(len(stages)), key=lambda index: stages[index]["start_ns"]
    )
    item_order = sorted(items, key=lambda item: item["start_ns"])
    active: list[int] = []
    stage_cursor = 0

    for item in item_order:
        item_start = item["start_ns"]
        item_end = item["end_ns"]
        active = [index for index in active if stages[index]["end_ns"] > item_start]
        while (
            stage_cursor < len(stage_order)
            and stages[stage_order[stage_cursor]]["start_ns"] < item_end
        ):
            active.append(stage_order[stage_cursor])
            stage_cursor += 1
        for index in active:
            stage = stages[index]
            if stage["start_ns"] < item_end and stage["end_ns"] > item_start:
                associated[index].append(item)
    return associated


def _associate_launched_work(
    stages: list[dict[str, Any]],
    runtimes_by_stage: list[list[dict[str, Any]]],
    work: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], str]:
    if work and all(item.get("correlation_id") is not None for item in work):
        by_correlation: dict[int, list[dict[str, Any]]] = {}
        for item in work:
            by_correlation.setdefault(int(item["correlation_id"]), []).append(item)
        associated = []
        for runtimes in runtimes_by_stage:
            correlation_ids = {
                int(item["correlation_id"])
                for item in runtimes
                if item.get("correlation_id") is not None
            }
            associated.append(
                [
                    item
                    for correlation_id in correlation_ids
                    for item in by_correlation.get(correlation_id, [])
                ]
            )
        return associated, "cuda_launch_correlation"
    return _associate_intervals(stages, work), "interval_overlap_fallback"


def _has_column(
    connection: sqlite3.Connection, table: str, column: str
) -> bool:
    return any(row[1] == column for row in connection.execute(f"PRAGMA table_info({table})"))


def _duration_sum_ms(
    items: list[dict[str, Any]], start: int, end: int, *, clip: bool
) -> float:
    if not clip:
        return sum(item["end_ns"] - item["start_ns"] for item in items) / 1e6
    return (
        sum(
            max(
                0,
                min(end, item["end_ns"]) - max(start, item["start_ns"]),
            )
            for item in items
        )
        / 1e6
    )


def _infer_report_identity(name: str) -> tuple[str | None, int | None]:
    role_match = re.search(r"_(ActorGroup|RolloutGroup|EnvGroup)_", name)
    role = role_match.group(1).removesuffix("Group").lower() if role_match else None
    rank_match = re.search(r"_(?:ActorGroup|RolloutGroup|EnvGroup)_(\d+)_", name)
    return role, int(rank_match.group(1)) if rank_match else None
