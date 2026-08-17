#!/usr/bin/env python3
"""Record CPU/GPU utilization and render an interactive HTML report.

The tool can either monitor the current machine until interrupted, or wrap a
training command and stop automatically when that command exits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


GPU_QUERY_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
]


@dataclass
class GpuSample:
    index: int
    name: str
    util_pct: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record system CPU/GPU utilization and generate a zoomable HTML report. "
            "Put the command to run after '--'."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory. Defaults to ./hardware_util_<timestamp>.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds when no wrapped command is provided.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU indices to record. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip nvidia-smi GPU sampling.",
    )
    parser.add_argument(
        "--title",
        default="Hardware Utilization",
        help="Title shown in the HTML report.",
    )
    parser.add_argument(
        "--metrics-log",
        default=None,
        help=(
            "Path to RLinf metrics.log for post-step-1 averages. Defaults to "
            "<output-dir>/../metrics.log when that file exists."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional command to run. Prefix it with '--'.",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"hardware_util_{stamp}")


def parse_gpu_ids(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            ids.add(int(part))
    return ids


def to_float(raw: str) -> float | None:
    raw = raw.strip()
    if raw == "" or raw.upper() in {"N/A", "NA", "NAN", "[N/A]"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def read_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as file:
            line = file.readline()
    except OSError:
        return None
    fields = line.split()
    if not fields or fields[0] != "cpu":
        return None
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def cpu_util_pct(prev: tuple[int, int] | None, curr: tuple[int, int] | None) -> float | None:
    if prev is None or curr is None:
        return None
    total_delta = curr[0] - prev[0]
    idle_delta = curr[1] - prev[1]
    if total_delta <= 0:
        return None
    return 100.0 * (1.0 - idle_delta / total_delta)


def query_gpus(selected_ids: set[int] | None) -> list[GpuSample]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []

    samples = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < len(GPU_QUERY_FIELDS):
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue
        if selected_ids is not None and index not in selected_ids:
            continue
        samples.append(
            GpuSample(
                index=index,
                name=row[1].strip(),
                util_pct=to_float(row[2]),
            )
        )
    return samples


def flatten_sample(
    elapsed_s: float,
    cpu_pct: float | None,
    gpu_samples: Iterable[GpuSample],
) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": now_iso(),
        "elapsed_s": round(elapsed_s, 3),
        "cpu_util_pct": cpu_pct,
    }
    for gpu in gpu_samples:
        prefix = f"gpu{gpu.index}"
        row[f"{prefix}_name"] = gpu.name
        row[f"{prefix}_util_pct"] = gpu.util_pct
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen = set()
    preferred = ["timestamp", "elapsed_s", "cpu_util_pct"]
    for key in preferred:
        seen.add(key)
        fields.append(key)
    for row in rows:
        for key in sorted(row):
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric_series(rows: list[dict[str, object]], key: str) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
        else:
            values.append(None)
    return values


def count_completed_steps(metrics_log: Path | None) -> int:
    if metrics_log is None:
        return 0
    try:
        text = metrics_log.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    steps = {int(match) for match in re.findall(r"Global Step:\s*(\d+)\s*/", text)}
    return max(steps) if steps else 0


def resolve_metrics_log(output_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path:
        return Path(raw_path)
    candidate = output_dir.parent / "metrics.log"
    return candidate if candidate.exists() or output_dir.name == "hardware_util" else None


def mean_numeric_after(
    rows: list[dict[str, object]], key: str, start_elapsed_s: float
) -> float | None:
    values = []
    for row in rows:
        elapsed = row.get("elapsed_s")
        value = row.get(key)
        if (
            isinstance(elapsed, (int, float))
            and float(elapsed) >= start_elapsed_s
            and isinstance(value, (int, float))
        ):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def compute_post_warmup_averages(
    rows: list[dict[str, object]],
    start_elapsed_s: float | None,
    completed_steps: int,
) -> dict[str, object]:
    if start_elapsed_s is None or completed_steps <= 1:
        return {"enabled": False, "reason": "requires more than one completed step"}

    util_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if (
                key == "cpu_util_pct"
                or (key.startswith("gpu") and key.endswith("_util_pct"))
            )
            and isinstance(value, (int, float))
        }
    )
    means = {}
    for key in util_keys:
        mean_value = mean_numeric_after(rows, key, start_elapsed_s)
        if mean_value is not None:
            means[key] = mean_value
    if not means:
        return {"enabled": False, "reason": "no utilization samples after first step"}
    return {
        "enabled": True,
        "start_elapsed_s": start_elapsed_s,
        "completed_steps": completed_steps,
        "means": means,
    }


def render_html(path: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    elapsed = [row["elapsed_s"] for row in rows]
    keys = sorted({key for row in rows for key in row if key not in {"timestamp", "elapsed_s"}})

    cpu_keys = [key for key in keys if key == "cpu_util_pct"]
    gpu_keys = [key for key in keys if key.startswith("gpu") and key.endswith("_util_pct")]
    average_data = metadata.get("average_util_after_first_step", {})
    average_enabled = isinstance(average_data, dict) and bool(average_data.get("enabled"))
    average_start = (
        float(average_data.get("start_elapsed_s", 0.0)) if average_enabled else None
    )
    average_means = average_data.get("means", {}) if average_enabled else {}

    def make_trace(key: str) -> dict[str, object]:
        return {
            "x": elapsed,
            "y": numeric_series(rows, key),
            "mode": "lines",
            "name": key,
            "connectgaps": False,
            "hovertemplate": "%{x:.1f}s<br>%{y:.2f}<extra>" + key + "</extra>",
        }

    def make_average_trace(key: str) -> dict[str, object] | None:
        if not average_enabled or average_start is None or not elapsed:
            return None
        if not isinstance(average_means, dict) or key not in average_means:
            return None
        value = average_means[key]
        if not isinstance(value, (int, float)):
            return None
        return {
            "x": [average_start, elapsed[-1]],
            "y": [float(value), float(value)],
            "mode": "lines",
            "name": f"{key} avg after step 1",
            "line": {"dash": "dash", "width": 2},
            "hovertemplate": "%{x:.1f}s<br>avg=%{y:.2f}<extra>"
            + key
            + " avg</extra>",
        }

    def make_traces(keys_for_chart: list[str]) -> list[dict[str, object]]:
        traces = [make_trace(key) for key in keys_for_chart]
        for key in keys_for_chart:
            avg_trace = make_average_trace(key)
            if avg_trace is not None:
                traces.append(avg_trace)
        return traces

    figures = [
        {
            "id": "cpu",
            "title": "CPU Utilization (%)",
            "ytitle": "percent",
            "traces": make_traces(cpu_keys),
        },
        {
            "id": "gpu",
            "title": "GPU Utilization (%)",
            "ytitle": "percent",
            "traces": make_traces(gpu_keys),
        },
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape_html(str(metadata.get("title", "Hardware Utilization")))}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      background: #f6f8fb;
      color: #172033;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 20px 28px 14px;
      background: #ffffff;
      border-bottom: 1px solid #d8dee9;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 6px 18px;
      font-size: 13px;
      color: #41516a;
    }}
    main {{
      padding: 16px 24px 28px;
    }}
    .chart {{
      height: 360px;
      margin: 0 0 18px;
      background: #ffffff;
      border: 1px solid #d8dee9;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escape_html(str(metadata.get("title", "Hardware Utilization")))}</h1>
    <div class="meta">
      <div>samples: {len(rows)}</div>
      <div>interval: {metadata.get("interval_s")}s</div>
      <div>started: {escape_html(str(metadata.get("started_at", "")))}</div>
      <div>ended: {escape_html(str(metadata.get("ended_at", "")))}</div>
      <div>csv: {escape_html(str(metadata.get("csv", "")))}</div>
      <div>command: {escape_html(str(metadata.get("command", "")))}</div>
      <div>post-step-1 average: {escape_html(format_average_status(average_data))}</div>
    </div>
  </header>
  <main>
    <div id="cpu" class="chart"></div>
    <div id="gpu" class="chart"></div>
  </main>
  <script>
    const figures = {json.dumps(figures)};
    const layoutBase = {{
      margin: {{l: 56, r: 20, t: 42, b: 45}},
      hovermode: "x unified",
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      legend: {{orientation: "h", y: -0.24}},
      xaxis: {{title: "elapsed seconds", rangeslider: {{visible: true}}}},
    }};
    const config = {{
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    }};
    for (const fig of figures) {{
      const layout = {{
        ...layoutBase,
        title: {{text: fig.title, x: 0.02, xanchor: "left"}},
        yaxis: {{title: fig.ytitle, rangemode: "tozero"}},
      }};
      Plotly.newPlot(fig.id, fig.traces, layout, config);
    }}
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_average_status(average_data: object) -> str:
    if not isinstance(average_data, dict) or not average_data.get("enabled"):
        return "disabled"
    start_elapsed_s = average_data.get("start_elapsed_s")
    completed_steps = average_data.get("completed_steps")
    return f"enabled from {float(start_elapsed_s):.1f}s, completed_steps={completed_steps}"


def start_command(command: list[str] | None) -> subprocess.Popen | None:
    if not command:
        return None
    return subprocess.Popen(command)


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def save_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "samples.csv"
    html_path = output_dir / "report.html"
    metadata_path = output_dir / "metadata.json"
    metadata["csv"] = str(csv_path)
    metadata["html"] = str(html_path)
    write_csv(csv_path, rows)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    render_html(html_path, rows, metadata)
    return csv_path, html_path, metadata_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    metrics_log = resolve_metrics_log(output_dir, args.metrics_log)
    selected_gpu_ids = parse_gpu_ids(args.gpu_ids)
    rows: list[dict[str, object]] = []
    stop_requested = False
    process = start_command(args.command)
    completed_steps = 0
    post_warmup_start_s: float | None = None

    def handle_signal(signum: int, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_wall = time.time()
    start_mono = time.monotonic()
    prev_cpu = read_cpu_times()
    metadata: dict[str, object] = {
        "title": args.title,
        "started_at": now_iso(),
        "interval_s": args.interval,
        "duration_s": args.duration,
        "command": " ".join(args.command) if args.command else "",
        "output_dir": str(output_dir),
        "metrics_log": str(metrics_log) if metrics_log is not None else None,
        "gpu_ids": sorted(selected_gpu_ids) if selected_gpu_ids is not None else "all",
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
    }

    print(f"[hardware-monitor] writing outputs under {output_dir}", flush=True)
    if process is not None:
        print(f"[hardware-monitor] started command pid={process.pid}", flush=True)

    try:
        while True:
            elapsed = time.monotonic() - start_mono
            curr_cpu = read_cpu_times()
            cpu_pct = cpu_util_pct(prev_cpu, curr_cpu)
            prev_cpu = curr_cpu
            gpu_samples = [] if args.no_gpu else query_gpus(selected_gpu_ids)
            rows.append(flatten_sample(elapsed, cpu_pct, gpu_samples))

            latest_completed_steps = count_completed_steps(metrics_log)
            if latest_completed_steps > completed_steps:
                completed_steps = latest_completed_steps
            if post_warmup_start_s is None and completed_steps >= 1:
                post_warmup_start_s = elapsed

            if process is not None and process.poll() is not None:
                break
            if args.duration is not None and elapsed >= args.duration:
                break
            if stop_requested:
                break

            next_tick = start_wall + len(rows) * args.interval
            sleep_s = max(0.0, next_tick - time.time())
            time.sleep(sleep_s)
    finally:
        if args.duration is not None and process is not None and process.poll() is None:
            terminate_process(process)
        metadata["ended_at"] = now_iso()
        metadata["samples"] = len(rows)
        metadata["exit_code"] = process.poll() if process is not None else None
        metadata["average_util_after_first_step"] = compute_post_warmup_averages(
            rows,
            post_warmup_start_s,
            completed_steps,
        )
        csv_path, html_path, metadata_path = save_outputs(output_dir, rows, metadata)
        print(f"[hardware-monitor] csv: {csv_path}", flush=True)
        print(f"[hardware-monitor] html: {html_path}", flush=True)
        print(f"[hardware-monitor] metadata: {metadata_path}", flush=True)

    if process is not None:
        return int(process.poll() or 0)
    return 130 if stop_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
