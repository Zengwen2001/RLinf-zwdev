# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Command-line analysis entry points for overlap profiling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .nsys_parser import export_nsys_report, parse_nsys_sqlite
from .report import generate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m toolkits.overlap_profiler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parse_parser = subparsers.add_parser("parse-nsys")
    parse_parser.add_argument("input", type=Path)
    parse_parser.add_argument("--output", type=Path)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    report_parser.add_argument(
        "--nsys-dir",
        type=Path,
        help="Directory searched recursively for .nsys-rep files",
    )
    args = parser.parse_args(argv)

    if args.command == "parse-nsys":
        sqlite_path = (
            export_nsys_report(args.input)
            if args.input.suffix == ".nsys-rep"
            else args.input
        )
        rows = parse_nsys_sqlite(sqlite_path)
        output = args.output or sqlite_path.with_name("stage_samples.csv")
        _write_rows(output, rows)
        print(output)
        return 0
    outputs = generate_report(args.run_dir, args.nsys_dir)
    print(outputs["report"])
    return 0


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
