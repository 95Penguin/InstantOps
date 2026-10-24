#!/usr/bin/env python3
"""Extract a service call graph from GAIA MicroSS trace parent/child spans."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from constants import SERVICES  # noqa: E402
from data.micross import MicrossWindowDataset  # noqa: E402
from data.splits import chronological_split  # noqa: E402

TRACE_MEMBER_PREFIX = "trace/trace_table_"
TRACE_MONTH_SUFFIX = "_2021-07.csv"
BATCH_SIZE = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_archive", type=Path, help="Original trace_split.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/micross")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for the temporary SQLite index (needs free space for the index)",
    )
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        help="Processed CSV directory; derives the exact first 70%% training-window range",
    )
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument(
        "--start-time",
        help="Inclusive UTC ISO timestamp for a manually specified training interval",
    )
    parser.add_argument(
        "--end-time",
        help="Exclusive UTC ISO timestamp for a manually specified training interval",
    )
    parser.add_argument("--source-timezone", default="UTC")
    args = parser.parse_args()
    if not args.trace_archive.is_file():
        parser.error(f"trace archive does not exist: {args.trace_archive}")
    if args.history < 1:
        parser.error("--history must be positive")
    if args.training_data_dir and (args.start_time or args.end_time):
        parser.error("use either --training-data-dir or the manual time bounds")
    if bool(args.start_time) != bool(args.end_time):
        parser.error("--start-time and --end-time must be supplied together")
    if not args.training_data_dir and not args.start_time:
        parser.error(
            "provide --training-data-dir to derive the training range, "
            "or provide both manual time bounds"
        )
    if args.start_time:
        try:
            args.start_epoch = parse_iso_epoch(args.start_time)
            args.end_epoch = parse_iso_epoch(args.end_time)
        except ValueError as exc:
            parser.error(str(exc))
        if args.end_epoch <= args.start_epoch:
            parser.error("--end-time must be later than --start-time")
    else:
        args.start_epoch = args.end_epoch = None
    return args


def parse_iso_epoch(value: str) -> int:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return int(timestamp.timestamp())


def parse_trace_epoch(value: str, source_timezone: str = "UTC") -> int | None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo(source_timezone))
        return int(timestamp.timestamp())
    except ValueError:
        return None


def list_trace_members(seven_zip: str, archive: Path) -> list[str]:
    result = subprocess.run(
        [seven_zip, "l", "-slt", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    members = [
        line.removeprefix("Path = ")
        for line in result.stdout.splitlines()
        if line.startswith(f"Path = {TRACE_MEMBER_PREFIX}") and line.endswith(TRACE_MONTH_SUFFIX)
    ]
    if not members:
        raise RuntimeError(f"No July 2021 trace tables found in {archive}")
    return sorted(members)


def open_member(
    seven_zip: str, archive: Path, member: str
) -> tuple[subprocess.Popen[bytes], TextIO]:
    process = subprocess.Popen(
        [seven_zip, "x", "-so", str(archive), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("7-Zip did not provide a readable output stream")
    stream = io.TextIOWrapper(process.stdout, encoding="utf-8-sig", newline="")
    return process, stream


def create_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        PRAGMA cache_size = -65536;
        CREATE TABLE spans (
            trace_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            service TEXT NOT NULL,
            start_epoch INTEGER NOT NULL,
            PRIMARY KEY (trace_id, span_id)
        ) WITHOUT ROWID;
        CREATE TABLE links (
            trace_id TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            child_span_id TEXT NOT NULL,
            child_service TEXT NOT NULL,
            observed_epoch INTEGER NOT NULL,
            PRIMARY KEY (trace_id, parent_id, child_span_id)
        ) WITHOUT ROWID;
        """
    )


def load_span_index(
    connection: sqlite3.Connection,
    seven_zip: str,
    archive: Path,
    start_epoch: int | None,
    end_epoch: int | None,
    source_timezone: str = "UTC",
) -> tuple[int, int]:
    members = list_trace_members(seven_zip, archive)
    span_batch: list[tuple[str, str, str, int]] = []
    link_batch: list[tuple[str, str, str, str, int]] = []
    spans_seen = 0
    links_seen = 0
    valid_services = set(SERVICES)

    for member in members:
        process, stream = open_member(seven_zip, archive, member)
        try:
            reader = csv.DictReader(stream)
            required = {"trace_id", "span_id", "parent_id", "service_name", "start_time"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Trace columns missing in {member}; expected {sorted(required)}")
            for row in reader:
                trace_id = (row.get("trace_id") or "").strip()
                span_id = (row.get("span_id") or "").strip()
                service = (row.get("service_name") or "").strip()
                parent_id = (row.get("parent_id") or "").strip()
                start_epoch_for_span = parse_trace_epoch(
                    row.get("start_time") or "", source_timezone
                )
                observed_epoch = parse_trace_epoch(row.get("end_time") or "", source_timezone)
                if (
                    not trace_id
                    or not span_id
                    or service not in valid_services
                    or start_epoch_for_span is None
                    or observed_epoch is None
                ):
                    continue
                # Keep all span-to-service mappings so training-period child
                # spans can still resolve parents that started before the window.
                span_batch.append((trace_id, span_id, service, start_epoch_for_span))
                if parent_id and parent_id not in {"0", "null", "None"}:
                    link_batch.append((trace_id, parent_id, span_id, service, observed_epoch))
                if len(span_batch) >= BATCH_SIZE:
                    connection.executemany(
                        "INSERT OR IGNORE INTO spans VALUES (?, ?, ?, ?)", span_batch
                    )
                    spans_seen += len(span_batch)
                    span_batch.clear()
                if len(link_batch) >= BATCH_SIZE:
                    connection.executemany(
                        "INSERT OR IGNORE INTO links VALUES (?, ?, ?, ?, ?)", link_batch
                    )
                    links_seen += len(link_batch)
                    link_batch.clear()
            connection.executemany("INSERT OR IGNORE INTO spans VALUES (?, ?, ?, ?)", span_batch)
            connection.executemany("INSERT OR IGNORE INTO links VALUES (?, ?, ?, ?, ?)", link_batch)
            spans_seen += len(span_batch)
            links_seen += len(link_batch)
            span_batch.clear()
            link_batch.clear()
        finally:
            stream.close()
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"7-Zip failed while reading {member}")
        connection.commit()
        print(f"Indexed {member}", file=sys.stderr)
    return spans_seen, links_seen


def derive_training_interval(data_dir: Path, history: int) -> tuple[int, int]:
    dataset = MicrossWindowDataset(
        data_dir / "service_features_1m.csv",
        data_dir / "next_minute_labels.csv",
        adjacency=np.eye(len(SERVICES), dtype=np.float32),
        history=history,
    )
    sample_indices = dataset.sample_indices
    count = len(sample_indices)
    if count < 3:
        raise ValueError(f"need at least 3 usable windows to split; found {count}")
    train_ends, _, _ = chronological_split(sample_indices)
    first_train_window = train_ends[0] - history + 1
    last_train_window = train_ends[-1]
    start_epoch = dataset.timestamp_epochs[first_train_window]
    end_epoch = dataset.timestamp_epochs[last_train_window] + dataset.bucket_seconds
    return start_epoch, end_epoch


def aggregate_edges(
    connection: sqlite3.Connection, start_epoch: int, end_epoch: int
) -> list[tuple[str, str, int]]:
    connection.execute("CREATE INDEX idx_links_parent ON links(trace_id, parent_id)")
    connection.commit()
    query = """
        SELECT parent.service, child.child_service, COUNT(*) AS span_count
        FROM links AS child
        JOIN spans AS parent
          ON parent.trace_id = child.trace_id
         AND parent.span_id = child.parent_id
        WHERE parent.service != child.child_service
          AND child.observed_epoch >= ? AND child.observed_epoch < ?
          AND parent.start_epoch <= child.observed_epoch
        GROUP BY parent.service, child.child_service
        ORDER BY parent.service, child.child_service
    """
    return [
        (source, destination, count)
        for source, destination, count in connection.execute(query, (start_epoch, end_epoch))
    ]


def write_graph(output_dir: Path, edges: list[tuple[str, str, int]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "service_call_edges.csv"
    adjacency_path = output_dir / "service_adjacency.npy"
    with edge_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_service", "destination_service", "span_count"])
        writer.writerows(edges)

    indices = {service: index for index, service in enumerate(SERVICES)}
    adjacency = np.zeros((len(SERVICES), len(SERVICES)), dtype=np.float32)
    for source, destination, _ in edges:
        # GCN uses symmetric degree normalization, so retain call direction in
        # the edge CSV but use an undirected binary adjacency for message passing.
        adjacency[indices[source], indices[destination]] = 1.0
        adjacency[indices[destination], indices[source]] = 1.0
    np.save(adjacency_path, adjacency)
    return edge_path, adjacency_path


def write_dynamic_graph(connection, data_dir: Path, output_dir: Path, history: int):
    """Graph at minute t uses completed child spans in t only, never t+1."""
    dataset = MicrossWindowDataset(
        data_dir / "service_features_1m.csv",
        data_dir / "next_minute_labels.csv",
        np.eye(len(SERVICES)),
        history=history,
    )
    time_indices = {epoch: index for index, epoch in enumerate(dataset.timestamp_epochs)}
    services = {service: index for index, service in enumerate(SERVICES)}
    adjacency = np.zeros((len(time_indices), len(SERVICES), len(SERVICES)), dtype=np.float32)
    query = """SELECT (child.observed_epoch / 60) * 60, parent.service, child.child_service, COUNT(*)
               FROM links child JOIN spans parent ON parent.trace_id=child.trace_id AND parent.span_id=child.parent_id
               WHERE parent.service != child.child_service AND parent.start_epoch <= child.observed_epoch
               GROUP BY 1,2,3"""
    resolved = 0
    for epoch, source, target, count in connection.execute(query):
        if epoch not in time_indices:
            continue
        i, j, k = time_indices[epoch], services[source], services[target]
        adjacency[i, j, k] += count
        adjacency[i, k, j] += count
        resolved += count
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "service_adjacency_1m.npz",
        timestamps=np.array(dataset.timestamp_epochs),
        services=np.array(SERVICES),
        adjacency=adjacency,
    )
    candidate_links = connection.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    matched = connection.execute(
        "SELECT COUNT(*) FROM links child JOIN spans parent ON parent.trace_id=child.trace_id AND parent.span_id=child.parent_id"
    ).fetchone()[0]
    report = {
        "resolved_parent_links_all_periods": matched,
        "unresolved_parent_links_all_periods": candidate_links - matched,
        "parent_resolution_fraction": matched / candidate_links if candidate_links else None,
        "candidate_parent_links_all_periods": candidate_links,
        "resolved_cross_service_calls_in_panel": resolved,
        "minutes_with_edges": int(np.count_nonzero(adjacency.sum(axis=(1, 2)))),
        "panel_minutes": len(time_indices),
        "edge_definition": "symmetric sum of completed child-span counts per minute",
    }
    (output_dir / "graph_quality.json").write_text(json.dumps(report, indent=2))
    if resolved == 0:
        raise ValueError("no cross-service edges resolved in feature panel")


def main() -> int:
    args = parse_args()
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    if seven_zip is None:
        print("error: install 7-Zip first (macOS: brew install sevenzip)", file=sys.stderr)
        return 2
    if args.training_data_dir:
        args.training_data_dir = args.training_data_dir.expanduser().resolve()
        schema_path = args.training_data_dir / "feature_schema.json"
        if schema_path.exists():
            args.source_timezone = json.loads(schema_path.read_text()).get("source_timezone", "UTC")
        try:
            args.start_epoch, args.end_epoch = derive_training_interval(
                args.training_data_dir.expanduser().resolve(), args.history
            )
        except (OSError, ValueError) as exc:
            print(f"error: could not derive the training interval: {exc}", file=sys.stderr)
            return 2
    print(
        "Graph evidence interval (UTC, end exclusive): "
        f"{datetime.fromtimestamp(args.start_epoch, timezone.utc).isoformat()} to "
        f"{datetime.fromtimestamp(args.end_epoch, timezone.utc).isoformat()}",
        file=sys.stderr,
    )
    work_dir = (args.work_dir or args.output_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    descriptor, database_name = tempfile.mkstemp(
        prefix="micross-trace-graph-", suffix=".sqlite3", dir=work_dir
    )
    os.close(descriptor)
    database_path = Path(database_name)
    try:
        with sqlite3.connect(database_path) as connection:
            create_index(connection)
            spans_seen, links_seen = load_span_index(
                connection,
                seven_zip,
                args.trace_archive.expanduser().resolve(),
                args.start_epoch,
                args.end_epoch,
                args.source_timezone,
            )
            edges = aggregate_edges(connection, args.start_epoch, args.end_epoch)
            if args.training_data_dir:
                write_dynamic_graph(
                    connection, args.training_data_dir, args.output_dir, args.history
                )
        edge_path, adjacency_path = write_graph(args.output_dir.expanduser().resolve(), edges)
    except (OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        database_path.unlink(missing_ok=True)

    print(f"Indexed span records: {spans_seen:,}; parent links: {links_seen:,}")
    print(f"Directed service edges: {len(edges)}")
    print(f"Wrote edge counts: {edge_path}")
    print(f"Wrote symmetric binary adjacency: {adjacency_path}")
    if not edges:
        print(
            "warning: no cross-service parent/child relationships were resolved",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
