"""
Plot timeline JSONL traces as a Gantt chart (one row per component/rank lane).
"""
from __future__ import annotations

import os
import sys

# Running `python .../rlinf/utils/plot_timeline.py` puts this directory first on sys.path,
# so `import logging` resolves to rlinf/utils/logging.py and breaks PIL/matplotlib.
if sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import json
from collections import defaultdict
from glob import glob

# Default palette for known components; others cycle from tab10
_COMPONENT_COLORS: dict[str, str] = {
    "env": "#27ae60",
    "env_micro": "#58d68d",
    "rollout": "#2980b9",
    "rollout_preprocess": "#85c1e9",
    "rollout_vlm": "#2471a3",
    "rollout_ae": "#8e44ad",
    "rollout_ae_detail": "#bb8fce",
    "rollout_postprocess": "#d2b4de",
    "kv_transfer": "#16a085",
    "kv_wait": "#16a085",
    "kv_queue": "#76d7c4",
    "kv_unpack": "#48c9b0",
    "actor": "#d35400",
}

_COMPONENT_ORDER: dict[str, int] = {
    "env": 0,
    "env_micro": 1,
    "rollout": 2,
    "rollout_preprocess": 3,
    "rollout_vlm": 4,
    "kv_transfer": 5,
    "kv_wait": 5,
    "kv_queue": 6,
    "kv_unpack": 7,
    "rollout_ae_detail": 8,
    "rollout_ae": 9,
    "rollout_postprocess": 10,
    "actor": 11,
}


def _load_events(timeline_dir: str) -> list[dict]:
    events: list[dict] = []
    pattern = os.path.join(timeline_dir, "*.jsonl")
    for path in sorted(glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t0, t1 = float(rec["t0"]), float(rec["t1"])
                if t1 < t0:
                    t0, t1 = t1, t0
                rec["t0"], rec["t1"] = t0, t1
                events.append(rec)
    return events


def _lane_key(rec: dict) -> tuple[str, int]:
    return str(rec["component"]), int(rec["rank"])


def _component_sort_key(component: str) -> tuple[int, str]:
    return (_COMPONENT_ORDER.get(component, 999), component)


def _lane_sort_key(lane: tuple[str, int]) -> tuple[int, str, int]:
    component, rank = lane
    order, name = _component_sort_key(component)
    return (order, name, rank)


def plot_timeline(
    timeline_dir: str,
    output_path: str | None = None,
    fig_width: float = 14.0,
    lane_height: float = 0.65,
    dpi: int = 150,
) -> str:
    """
    Draw a Gantt chart: x = time (relative to first event), y = component/rank lane.

    Returns path to saved figure.
    """
    return plot_timeline_png(
        timeline_dir,
        output_path=output_path,
        fig_width=fig_width,
        lane_height=lane_height,
        dpi=dpi,
    )


def plot_timeline_png(
    timeline_dir: str,
    output_path: str | None = None,
    fig_width: float = 14.0,
    lane_height: float = 0.65,
    dpi: int = 150,
) -> str:
    """Matplotlib PNG output."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    events = _load_events(timeline_dir)
    if not events:
        raise ValueError(f"No events found under {timeline_dir!r}")

    t_min = min(e["t0"] for e in events)
    t_max = max(e["t1"] for e in events)

    lanes = sorted({_lane_key(e) for e in events}, key=_lane_sort_key)
    lane_index = {lk: i for i, lk in enumerate(lanes)}

    # Color by component name
    components = sorted({lk[0] for lk in lanes}, key=_component_sort_key)
    tab = plt.get_cmap("tab10")
    comp_color: dict[str, str] = dict(_COMPONENT_COLORS)
    for i, c in enumerate(components):
        if c not in comp_color:
            comp_color[c] = tab(i % tab.N)

    by_lane: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for e in events:
        by_lane[_lane_key(e)].append(e)
    for lk in by_lane:
        by_lane[lk].sort(key=lambda r: r["t0"])

    nlanes = len(lanes)
    fig_h = max(4.0, 1.2 + nlanes * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    bar_h = min(lane_height, 0.85)
    for e in events:
        lk = _lane_key(e)
        y = lane_index[lk]
        x0 = e["t0"] - t_min
        w = e["t1"] - e["t0"]
        color = comp_color[lk[0]]
        ax.barh(
            y,
            w,
            left=x0,
            height=bar_h,
            color=color,
            alpha=0.85,
            linewidth=0,
            align="center",
        )

    yticklabels = [f"{c}/r{r}" for c, r in lanes]
    ax.set_yticks(range(nlanes))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time from trace start (s)")
    ax.set_title(
        f"Timeline Gantt · {len(events)} events · "
        f"span {t_max - t_min:.2f}s · {os.path.basename(timeline_dir.rstrip(os.sep))}"
    )
    ax.set_xlim(0, max(t_max - t_min, 1e-6))
    ax.set_ylim(-0.6, nlanes - 0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    handles = [
        mpatches.Patch(color=comp_color[c], label=c, alpha=0.85)
        for c in components
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(timeline_dir, "timeline_gantt.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_timeline_html(
    timeline_dir: str,
    output_path: str | None = None,
    *,
    lane_height: float = 0.55,
    width_px: int = 1200,
) -> str:
    """
    Interactive HTML output (pan/zoom + hover) using Plotly.
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotly is required for --format html. Install it with: pip install plotly"
        ) from e

    events = _load_events(timeline_dir)
    if not events:
        raise ValueError(f"No events found under {timeline_dir!r}")

    t_min = min(e["t0"] for e in events)
    t_max = max(e["t1"] for e in events)

    lanes = sorted({_lane_key(e) for e in events}, key=_lane_sort_key)
    lane_labels = [f"{c}/r{r}" for c, r in lanes]
    lane_label_by_key = {lk: f"{lk[0]}/r{lk[1]}" for lk in lanes}

    # Color by component
    components = sorted({lk[0] for lk in lanes}, key=_component_sort_key)
    comp_color: dict[str, str] = dict(_COMPONENT_COLORS)
    # Plotly's default qualitative palette
    plotly_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for i, c in enumerate(components):
        if c not in comp_color:
            comp_color[c] = plotly_palette[i % len(plotly_palette)]

    # Group events by component into separate traces (legend toggles)
    events_by_component: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        events_by_component[str(e["component"])].append(e)

    fig = go.Figure()
    for comp in components:
        comp_events = events_by_component.get(comp, [])
        if not comp_events:
            continue
        y = [lane_label_by_key[_lane_key(e)] for e in comp_events]
        base = [e["t0"] - t_min for e in comp_events]
        dur = [e["t1"] - e["t0"] for e in comp_events]
        tag = [str(e.get("tag", "")) for e in comp_events]
        step = [e.get("global_step", None) for e in comp_events]

        fig.add_trace(
            go.Bar(
                name=comp,
                orientation="h",
                y=y,
                x=dur,
                base=base,
                marker=dict(color=comp_color[comp]),
                opacity=0.85,
                customdata=list(zip(tag, dur, base, step)),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "component=%{fullData.name}<br>"
                    "tag=%{customdata[0]}<br>"
                    "t0=%{customdata[2]:.3f}s<br>"
                    "dt=%{customdata[1]:.3f}s<br>"
                    "global_step=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="overlay",
        bargap=0.15,
        height=max(320, int(120 + len(lanes) * 55)),
        width=width_px,
        title=(
            f"Timeline Gantt · {len(events)} events · "
            f"span {t_max - t_min:.2f}s · {os.path.basename(timeline_dir.rstrip(os.sep))}"
        ),
        xaxis=dict(title="Time from trace start (s)", rangeslider=dict(visible=True)),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=lane_labels,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=120, r=20, t=60, b=40),
    )

    # make bars slightly thinner by adjusting marker line / gaps; Plotly doesn't have direct lane height
    fig.update_traces(marker_line_width=0)

    if output_path is None:
        output_path = os.path.join(timeline_dir, "timeline_gantt.html")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timeline JSONL as a Gantt chart.")
    parser.add_argument(
        "timeline_dir",
        nargs="?",
        default="/mnt/public/zengwen/RLinf/logs/20260331-03:46:32-behavior_ppo_openpi_pi05/timeline",
        help="Directory containing *_rank*.jsonl timeline files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default depends on --format)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "html"],
        default="html",
        help="Output format: png (static) or html (interactive zoom/pan)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Alias for --format html",
    )
    parser.add_argument("--width", type=float, default=14.0, help="Figure width in inches")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    args = parser.parse_args()

    out_format = "html" if args.interactive else args.format
    if out_format == "html":
        out = plot_timeline_html(args.timeline_dir, output_path=args.output)
    else:
        out = plot_timeline_png(
            args.timeline_dir, output_path=args.output, fig_width=args.width, dpi=args.dpi
        )
    print(out)


if __name__ == "__main__":
    main()
