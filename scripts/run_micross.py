#!/usr/bin/env python3
"""Run MicroSS v2 preprocessing, temporal graph extraction, training and RCA."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("micross_root", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/micross_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/micross_v2")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--source-timezone", default="UTC")
    args = parser.parse_args()
    if min(args.epochs, args.history, args.batch_size) < 1:
        parser.error("epochs, history and batch-size must be positive")
    root = args.micross_root.expanduser().resolve()
    data = args.data_dir.expanduser().resolve()

    def run(script, *arguments):
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)

    if not args.skip_prepare:
        run("prepare_micross.py", root, "--output", data, "--source-timezone", args.source_timezone)
    import json

    schema = json.loads((data / "feature_schema.json").read_text())
    if schema.get("source_timezone", "UTC") != args.source_timezone:
        raise ValueError("source timezone differs from cached data; regenerate preprocessing")
    if schema.get("version") != 2:
        raise ValueError("pipeline requires v2 data; regenerate preprocessing outputs")
    if not args.skip_graph:
        run(
            "build_micross_graph.py",
            root / "trace/trace_split.zip",
            "--training-data-dir",
            data,
            "--output-dir",
            data,
            "--history",
            args.history,
            "--work-dir",
            args.work_dir,
        )
    run(
        "train_micross.py",
        "--data-dir",
        data,
        "--graph",
        data / "service_adjacency_1m.npz",
        "--output-dir",
        args.output_dir,
        "--epochs",
        args.epochs,
        "--history",
        args.history,
        "--batch-size",
        args.batch_size,
        "--device",
        args.device,
        "--rca-method",
        "masking",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
