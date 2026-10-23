#!/usr/bin/env python3
"""Read-only inventory for CSV/JSON files in a downloaded dataset directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

MAX_SAMPLE_ROWS = 5


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    hints = {
        "timestamp": ("time", "timestamp", "date", "datetime"),
        "service_or_instance": ("service", "instance", "pod", "node", "container"),
        "fault_or_label": ("fault", "failure", "anomaly", "label", "root", "inject"),
        "trace_or_span": ("trace", "span", "parent", "caller", "callee"),
        "metric": ("metric", "cpu", "memory", "latency", "response", "usage"),
        "log_or_event": ("log", "event", "message", "level", "severity"),
    }
    lowered = {col: col.lower() for col in columns}
    return {
        category: [
            original
            for original, name in lowered.items()
            if any(token in name for token in tokens)
        ]
        for category, tokens in hints.items()
    }


def inspect_csv(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            samples = []
            row_count = 0
            for row in reader:
                row_count += 1
                if len(samples) < MAX_SAMPLE_ROWS:
                    samples.append(row)
        return {
            "columns": columns,
            "row_count": row_count,
            "column_hints": classify_columns(columns),
            "sample_rows": samples,
        }
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Directory containing extracted dataset files")
    args = parser.parse_args()
    root = args.dataset_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    extensions = Counter(path.suffix.lower() or "[no extension]" for path in files)
    report: dict[str, object] = {
        "root": str(root),
        "file_count": len(files),
        "extensions": dict(sorted(extensions.items())),
        "files": [],
    }
    entries = []
    for path in files:
        item: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            item["csv"] = inspect_csv(path)
        elif path.suffix.lower() == ".json":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                item["json_top_level_type"] = type(data).__name__
                if isinstance(data, dict):
                    item["json_top_level_keys"] = list(data)[:50]
                elif isinstance(data, list):
                    item["json_item_count"] = len(data)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                item["json_error"] = str(exc)
        entries.append(item)
    report["files"] = entries
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
