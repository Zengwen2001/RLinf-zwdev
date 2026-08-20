# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""CSV/JSON and interactive HTML reporting for overlap experiments."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .metrics import summarize_pairs
from .nsys_parser import export_nsys_report, parse_nsys_sqlite


def generate_report(
    run_dir: str | Path, nsys_dir: str | Path | None = None
) -> dict[str, Path]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    pair_samples = _read_jsonl(root / "pair_samples.jsonl")
    stage_samples = []
    sqlite_dir = root / "sqlite"
    source_dir = Path(nsys_dir) if nsys_dir is not None else root / "raw"
    reports = sorted(source_dir.rglob("*.nsys-rep")) if source_dir.exists() else []
    if reports:
        sqlite_dir.mkdir(parents=True, exist_ok=True)
        for index, report in enumerate(reports):
            output = sqlite_dir / f"{index:04d}_{report.stem}.sqlite"
            export_nsys_report(report, output)
    if sqlite_dir.exists():
        for sqlite_path in sorted(sqlite_dir.glob("*.sqlite")):
            stage_samples.extend(parse_nsys_sqlite(sqlite_path))
    compatibility = summarize_pairs(pair_samples)
    resource_samples = [_resource_row(row) for row in stage_samples]

    outputs = {
        "pair_samples": root / "pair_samples.csv",
        "stage_samples": root / "stage_samples.csv",
        "resource_samples": root / "resource_samples.csv",
        "compatibility_matrix": root / "compatibility_matrix.csv",
        "compatibility_json": root / "compatibility_matrix.json",
        "report": root / "report.html",
    }
    _write_csv(outputs["pair_samples"], pair_samples)
    _write_csv(outputs["stage_samples"], stage_samples)
    _write_csv(outputs["resource_samples"], resource_samples)
    _write_csv(outputs["compatibility_matrix"], compatibility)
    outputs["compatibility_json"].write_text(
        json.dumps(compatibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["report"].write_text(
        _render_html(pair_samples, stage_samples, compatibility), encoding="utf-8"
    )
    return outputs


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _resource_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_file",
        "source_version",
        "pair_id",
        "repetition",
        "trial_kind",
        "trial_id",
        "role",
        "rank",
        "stage",
        "start_ns",
        "end_ns",
        "resource_sm_active",
        "resource_tensor_active",
        "resource_dram_active",
        "resource_nvlink_traffic",
        "resource_pcie_traffic",
        "resource_unavailable_reason",
    )
    result = {key: row.get(key) for key in keys}
    result["source_backend"] = row.get("source_backend", "nsight_systems")
    result["sample_start_ns"] = row.get("start_ns")
    result["sample_end_ns"] = row.get("end_ns")
    return result


def _render_html(
    pair_samples: list[dict[str, Any]],
    stage_samples: list[dict[str, Any]],
    compatibility: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "pairSamples": pair_samples,
            "stageSamples": stage_samples,
            "compatibility": compatibility,
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")
    rejected = [sample for sample in pair_samples if not sample.get("accepted", False)]
    rejected_rows = (
        "".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('pair_id')))}</td>"
            f"<td>{row.get('repetition')}</td>"
            f"<td>{html.escape(', '.join(row.get('rejection_reasons', [])))}</td>"
            "</tr>"
            for row in rejected
        )
        or '<tr><td colspan="3">None</td></tr>'
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLinf overlap profile</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{margin:0;font:14px system-ui,sans-serif;color:#1c2430;background:#f7f8fa}}
header{{padding:20px 28px;background:#17202b;color:white}} h1{{font-size:22px;margin:0}}
main{{padding:20px 28px;display:grid;gap:20px}} section{{background:white;border:1px solid #dce1e7;padding:16px}}
.plot{{height:430px}} table{{border-collapse:collapse;width:100%}} th,td{{padding:7px;border-bottom:1px solid #e4e7eb;text-align:left}}
</style></head><body><header><h1>RLinf overlap compatibility</h1></header><main>
<section><h2>Net benefit</h2><div id="benefit" class="plot"></div></section>
<section><h2>Slowdown</h2><div id="slowdown" class="plot"></div></section>
<section><h2>Sample acceptance</h2><div id="quality" class="plot"></div></section>
<section><h2>Stage GPU activity</h2><div id="activity" class="plot"></div></section>
<section><h2>Nsight stage timeline</h2><div id="timeline" class="plot"></div></section>
<section><h2>Rejected samples</h2><table><thead><tr><th>Pair</th><th>Repetition</th><th>Reason</th></tr></thead><tbody>{rejected_rows}</tbody></table></section>
</main><script>const DATA={payload};
const labels=DATA.compatibility.map(x=>x.pair_id);
const benefits=DATA.compatibility.map(x=>x.benefit_ms_p50);
Plotly.newPlot('benefit',[{{type:'bar',x:labels,y:benefits,marker:{{color:benefits.map(v=>v>=0?'#218c74':'#c44536')}}}}],{{yaxis:{{title:'Benefit p50 (ms)'}},xaxis:{{tickangle:-25}},margin:{{b:160}},dragmode:'zoom'}},{{responsive:true}});
Plotly.newPlot('slowdown',[{{type:'heatmap',x:['A slowdown','B slowdown'],y:labels,z:DATA.compatibility.map(x=>[x.a_slowdown_p50,x.b_slowdown_p50]),zmid:1,colorscale:'RdYlGn',reversescale:true,colorbar:{{title:'x'}}}}],{{margin:{{l:260}},dragmode:'zoom'}},{{responsive:true}});
const accepted=DATA.pairSamples.filter(x=>x.accepted);
Plotly.newPlot('quality',[{{type:'scatter',mode:'markers',x:accepted.map(x=>x.start_skew_ms),y:accepted.map(x=>x.overlap_coverage),text:accepted.map(x=>x.pair_id),marker:{{size:9,color:accepted.map(x=>x.benefit_ms),colorscale:'RdYlGn',colorbar:{{title:'Benefit ms'}}}},hovertemplate:'%{{text}}<br>skew=%{{x:.3f}} ms<br>coverage=%{{y:.3f}}<extra></extra>'}}],{{xaxis:{{title:'Start skew (ms)'}},yaxis:{{title:'Overlap coverage',range:[0,1.05]}},dragmode:'zoom'}},{{responsive:true}});
const stageLabels=DATA.stageSamples.map(x=>`${{x.role??'unknown'}}:${{x.rank??'?'}} ${{x.stage}}`);
Plotly.newPlot('activity',[{{type:'bar',name:'GPU active union',x:stageLabels,y:DATA.stageSamples.map(x=>x.gpu_active_union_ms)}},{{type:'bar',name:'GPU idle in compute envelope',x:stageLabels,y:DATA.stageSamples.map(x=>x.gpu_idle_inside_compute_envelope_ms)}}],{{barmode:'stack',yaxis:{{title:'GPU compute envelope (ms)'}},xaxis:{{tickangle:-30,rangeslider:{{visible:true}}}},margin:{{b:180}},dragmode:'zoom'}},{{responsive:true}});
const byFile={{}}; for(const s of DATA.stageSamples){{(byFile[s.source_file]??=[]).push(s)}}
const traces=[]; for(const [file,rows] of Object.entries(byFile)){{const base=Math.min(...rows.map(x=>x.start_ns)); traces.push({{type:'bar',orientation:'h',name:file.split('/').pop(),y:rows.map(x=>x.stage),x:rows.map(x=>x.stage_wall_ms),base:rows.map(x=>(x.start_ns-base)/1e6),customdata:rows.map(x=>[x.gpu_active_union_ms,x.gpu_compute_envelope_ms,x.gpu_first_activity_offset_ms,x.gpu_last_activity_offset_ms,x.nccl_ms]),hovertemplate:'%{{y}}<br>CPU range=%{{x:.3f}} ms<br>GPU active=%{{customdata[0]:.3f}} ms<br>GPU envelope=%{{customdata[1]:.3f}} ms<br>first GPU offset=%{{customdata[2]:.3f}} ms<br>last GPU offset=%{{customdata[3]:.3f}} ms<br>NCCL=%{{customdata[4]:.3f}} ms<extra></extra>'}})}}
Plotly.newPlot('timeline',traces,{{barmode:'overlay',xaxis:{{title:'Relative time (ms)',rangeslider:{{visible:true}}}},margin:{{l:220}},dragmode:'zoom'}},{{responsive:true}});
</script></body></html>"""
