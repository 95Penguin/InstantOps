#!/usr/bin/env python3
"""Build a minute-aligned, service-level MicroSS feature table for InstantOps.

The script streams metric and trace archives; it does not expand the large trace
archive to disk. GAIA run-log injection starts are used as proxy next-minute
labels. They are not equivalent to the application-level SLA labels in the
InstantOps paper.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

SERVICES = (
    "dbservice1",
    "dbservice2",
    "logservice1",
    "logservice2",
    "mobservice1",
    "mobservice2",
    "redisservice1",
    "redisservice2",
    "webservice1",
    "webservice2",
)
SERVICE_PATTERN = re.compile(r"^(" + "|".join(SERVICES) + r")_(.+)$")
DATE_RANGE_PATTERN = re.compile(r"_(20\d{2}-\d{2}-\d{2})_(20\d{2}-\d{2}-\d{2})\.csv$")
INJECTION_PATTERN = re.compile(
    r"\[(?P<kind>cpu_anomalies|memory_anomalies)\].*?"
    r"start at (?P<start>20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r" and lasts (?P<duration>[0-9.]+) seconds",
    re.IGNORECASE,
)

METRIC_FAMILIES = {
    "cpu": ("cpu",),
    "memory": ("memory", "mem_"),
    "network": ("network",),
    "disk": ("diskio", "disk_", "filesystem"),
}


@dataclass(frozen=True)
class Injection:
    service: str
    kind: str
    started_at: datetime
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gaia_root",
        type=Path,
        help="GAIA checkout, MicroSS directory, or directory containing MicroSS/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/micross"),
        help="Output directory (default: data/processed/micross)",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=60,
        help="Alignment bucket size in seconds (default: 60)",
    )
    parser.add_argument(
        "--trace-archive",
        type=Path,
        help="Original multipart trace archive, usually trace/trace_split.zip",
    )
    parser.add_argument(
        "--metric-archive",
        type=Path,
        help="Combined metric archive, usually metric/metric_full.zip",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Directory containing run_table_YYYY-MM.csv files",
    )
    args = parser.parse_args()
    if args.window_seconds != 60:
        parser.error("only 60-second windows are supported because labels target the next minute")
    return args


def locate_micross(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = (root, root / "MicroSS", root / "GAIA-DataSet" / "MicroSS")
    for candidate in candidates:
        if (candidate / "run").is_dir() and (candidate / "metric").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find MicroSS run/ and metric/ directories under {root}"
    )


def resolve_input(path: Path | None, default: Path) -> Path:
    result = (path or default).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def minute_epoch(timestamp: int, bucket_seconds: int) -> int:
    return timestamp - timestamp % bucket_seconds


def metric_family(metric_name: str) -> str:
    lowered = metric_name.lower()
    for family, tokens in METRIC_FAMILIES.items():
        if any(token in lowered for token in tokens):
            return family
    return "other"


def metric_member_identity(member: str) -> tuple[str, str] | None:
    """Return (service, metric-name) for a GAIA per-series CSV member."""
    filename = Path(member).name
    date_match = DATE_RANGE_PATTERN.search(filename)
    if not date_match:
        return None
    prefix = filename[: date_match.start()]
    service_match = SERVICE_PATTERN.match(prefix)
    if not service_match:
        return None
    service, remainder = service_match.groups()
    # Metric filenames place the container IP between service and metric name.
    metric_parts = remainder.split("_", 1)
    if len(metric_parts) != 2:
        return None
    return service, metric_parts[1]


def aggregate_metrics(
    archive_path: Path, bucket_seconds: int
) -> tuple[dict[tuple[str, int, str], tuple[float, int]], int]:
    """Average each series within a bucket, then average series per family."""
    totals: dict[tuple[str, int, str], list[float | int]] = defaultdict(
        lambda: [0.0, 0]
    )
    files_read = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith("metric/")
            and name.endswith(".csv")
            and "2021-07-" in name
        ]
        for member in members:
            identity = metric_member_identity(member)
            if identity is None:
                continue
            service, metric_name = identity
            family = metric_family(metric_name)
            per_bucket: dict[int, list[float | int]] = defaultdict(lambda: [0.0, 0])
            try:
                with archive.open(member) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                    reader = csv.DictReader(text)
                    if not reader.fieldnames or not {"timestamp", "value"}.issubset(
                        reader.fieldnames
                    ):
                        continue
                    for row in reader:
                        try:
                            timestamp = int(row["timestamp"]) // 1000
                            value = float(row["value"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if not math.isfinite(value):
                            continue
                        bucket = minute_epoch(timestamp, bucket_seconds)
                        aggregate = per_bucket[bucket]
                        aggregate[0] = float(aggregate[0]) + value
                        aggregate[1] = int(aggregate[1]) + 1
            except (OSError, UnicodeError, zipfile.BadZipFile, csv.Error) as exc:
                print(f"warning: skipped metric member {member}: {exc}", file=sys.stderr)
                continue
            for bucket, (value_sum, sample_count) in per_bucket.items():
                if sample_count:
                    aggregate = totals[(service, bucket, family)]
                    aggregate[0] = float(aggregate[0]) + float(value_sum) / int(sample_count)
                    aggregate[1] = int(aggregate[1]) + 1
            files_read += 1
            if files_read % 100 == 0:
                print(f"Read {files_read} metric series...", file=sys.stderr)
    return {
        key: (float(value[0]) / int(value[1]), int(value[1]))
        for key, value in totals.items()
        if value[1]
    }, files_read


def open_trace_member(
    seven_zip: str, archive_path: Path, member: str
) -> tuple[subprocess.Popen[bytes], TextIO]:
    process = subprocess.Popen(
        [seven_zip, "x", "-so", str(archive_path), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("7-Zip did not provide a readable output stream")
    return process, io.TextIOWrapper(process.stdout, encoding="utf-8-sig", newline="")


def trace_members(archive_path: Path, seven_zip: str) -> list[str]:
    result = subprocess.run(
        [seven_zip, "l", "-slt", str(archive_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    members = []
    for line in result.stdout.splitlines():
        if line.startswith("Path = trace/trace_table_") and line.endswith("_2021-07.csv"):
            members.append(line.removeprefix("Path = "))
    return sorted(members)


def parse_trace_timestamp(value: str) -> int | None:
    value = value.strip()
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(datetime.strptime(value, pattern).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def aggregate_traces(
    archive_path: Path, bucket_seconds: int
) -> tuple[dict[tuple[str, int], dict[str, float]], Counter[str]]:
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    if seven_zip is None:
        raise FileNotFoundError(
            "7-Zip executable not found. Install it with `brew install sevenzip`; "
            "use the original trace_split.zip with its .z01, .z02, ... volumes."
        )
    members = trace_members(archive_path, seven_zip)
    if not members:
        raise RuntimeError(f"No July trace tables found in {archive_path}")

    # service/bucket -> [span count, 5xx count, duration sum, duration count]
    totals: dict[tuple[str, int], list[float]] = defaultdict(lambda: [0, 0, 0.0, 0])
    member_counts: Counter[str] = Counter()
    for member in members:
        process, stream = open_trace_member(seven_zip, archive_path, member)
        try:
            reader = csv.DictReader(stream)
            for row in reader:
                service = (row.get("service_name") or "").strip()
                if service not in SERVICES:
                    continue
                timestamp = parse_trace_timestamp(row.get("start_time") or row.get("timestamp") or "")
                if timestamp is None:
                    continue
                bucket = minute_epoch(timestamp, bucket_seconds)
                aggregate = totals[(service, bucket)]
                aggregate[0] += 1
                status = (row.get("status_code") or "").strip()
                if status.startswith("5"):
                    aggregate[1] += 1
                try:
                    started = datetime.strptime(
                        row.get("start_time", "").strip(), "%Y-%m-%d %H:%M:%S.%f"
                    )
                    ended = datetime.strptime(
                        row.get("end_time", "").strip(), "%Y-%m-%d %H:%M:%S.%f"
                    )
                    duration_ms = max(0.0, (ended - started).total_seconds() * 1000)
                    aggregate[2] += duration_ms
                    aggregate[3] += 1
                except (TypeError, ValueError):
                    pass
                member_counts[service] += 1
        finally:
            stream.close()
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"7-Zip failed while streaming {member}")
        print(f"Read trace table: {member}", file=sys.stderr)

    result = {}
    for key, (span_count, errors, duration_sum, duration_count) in totals.items():
        result[key] = {
            "trace_span_count": span_count,
            "trace_5xx_count": errors,
            "trace_5xx_rate": errors / span_count if span_count else 0.0,
            "trace_duration_mean_ms": duration_sum / duration_count if duration_count else 0.0,
        }
    return result, member_counts


def parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    return None


def load_injections(run_dir: Path) -> list[Injection]:
    unique: dict[tuple[str, str, datetime, float], Injection] = {}
    for path in sorted(run_dir.glob("run_table_2021-07*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"service", "message"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                print(f"warning: unexpected run-log columns in {path}", file=sys.stderr)
                continue
            for row in reader:
                service = (row.get("service") or "").strip()
                if service not in SERVICES:
                    continue
                match = INJECTION_PATTERN.search(row.get("message") or "")
                if not match:
                    continue
                started_at = parse_datetime(match.group("start"))
                if started_at is None:
                    continue
                kind = match.group("kind").lower()
                duration = float(match.group("duration"))
                key = (service, kind, started_at, duration)
                unique[key] = Injection(service, kind, started_at, duration)
    return sorted(unique.values(), key=lambda event: (event.started_at, event.service))


def write_outputs(
    output_dir: Path,
    bucket_seconds: int,
    metrics: dict[tuple[str, int, str], tuple[float, int]],
    traces: dict[tuple[str, int], dict[str, float]],
    injections: list[Injection],
) -> tuple[int, int]:
    injection_starts: dict[tuple[str, int], set[str]] = defaultdict(set)
    for event in injections:
        epoch = int(event.started_at.replace(tzinfo=timezone.utc).timestamp())
        bucket = minute_epoch(epoch, bucket_seconds)
        injection_starts[(event.service, bucket)].add(event.kind)

    def observed_buckets() -> Iterator[int]:
        yield from (bucket for _, bucket, _ in metrics)
        yield from (bucket for _, bucket in traces)
        yield from (bucket for _, bucket in injection_starts)

    first_bucket = min(observed_buckets(), default=None)
    last_bucket = max(observed_buckets(), default=None)
    if first_bucket is None or last_bucket is None:
        raise RuntimeError("No July metric, trace, or injection observations were found")
    bucket_count = (last_bucket - first_bucket) // bucket_seconds + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "service_features_1m.csv"
    label_path = output_dir / "next_minute_labels.csv"
    metric_families = tuple(METRIC_FAMILIES) + ("other",)

    feature_fields = [
        "timestamp_utc",
        "service",
        "metric_cpu_mean",
        "metric_memory_mean",
        "metric_network_mean",
        "metric_disk_mean",
        "metric_other_mean",
        "metric_cpu_series_count",
        "metric_memory_series_count",
        "metric_network_series_count",
        "metric_disk_series_count",
        "metric_other_series_count",
        "trace_span_count",
        "trace_5xx_count",
        "trace_5xx_rate",
        "trace_duration_mean_ms",
    ]
    label_fields = ["input_minute_utc", "target_minute_utc", "service", "fault_next_minute", "fault_types"]
    feature_rows = 0
    label_rows = 0
    with feature_path.open("w", encoding="utf-8", newline="") as feature_handle, label_path.open(
        "w", encoding="utf-8", newline=""
    ) as label_handle:
        feature_writer = csv.DictWriter(feature_handle, fieldnames=feature_fields)
        label_writer = csv.DictWriter(label_handle, fieldnames=label_fields)
        feature_writer.writeheader()
        label_writer.writeheader()
        for bucket_index in range(bucket_count):
            bucket = first_bucket + bucket_index * bucket_seconds
            stamp = datetime.fromtimestamp(bucket, timezone.utc).isoformat()
            for service in SERVICES:
                row: dict[str, str | int | float] = {
                    "timestamp_utc": stamp,
                    "service": service,
                    "trace_span_count": 0,
                    "trace_5xx_count": 0,
                    "trace_5xx_rate": 0.0,
                    "trace_duration_mean_ms": 0.0,
                }
                for family in metric_families:
                    value = metrics.get((service, bucket, family))
                    row[f"metric_{family}_mean"] = value[0] if value else 0.0
                    row[f"metric_{family}_series_count"] = value[1] if value else 0
                row.update(traces.get((service, bucket), {}))
                feature_writer.writerow(row)
                feature_rows += 1
                target_bucket = bucket + bucket_seconds
                kinds = sorted(injection_starts.get((service, target_bucket), set()))
                label_writer.writerow(
                    {
                        "input_minute_utc": stamp,
                        "target_minute_utc": datetime.fromtimestamp(
                            target_bucket, timezone.utc
                        ).isoformat(),
                        "service": service,
                        "fault_next_minute": int(bool(kinds)),
                        "fault_types": ";".join(kinds),
                    }
                )
                label_rows += 1
    return feature_rows, label_rows


def main() -> int:
    args = parse_args()
    try:
        root = locate_micross(args.gaia_root)
        metric_archive = resolve_input(args.metric_archive, root / "metric" / "metric_full.zip")
        trace_archive = resolve_input(args.trace_archive, root / "trace" / "trace_split.zip")
        run_dir = (args.run_dir or root / "run").expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)

        print(f"MicroSS root: {root}", file=sys.stderr)
        print(f"Reading metrics from {metric_archive}", file=sys.stderr)
        metrics, metric_files = aggregate_metrics(metric_archive, args.window_seconds)
        print(f"Read {metric_files} metric series; aligned keys: {len(metrics)}", file=sys.stderr)
        print(f"Reading traces from original multipart archive {trace_archive}", file=sys.stderr)
        traces, trace_counts = aggregate_traces(trace_archive, args.window_seconds)
        print(f"Trace spans by service: {dict(trace_counts)}", file=sys.stderr)
        injections = load_injections(run_dir)
        print(f"Unique injection events: {len(injections)}", file=sys.stderr)
        feature_rows, label_rows = write_outputs(
            args.output.expanduser().resolve(),
            args.window_seconds,
            metrics,
            traces,
            injections,
        )
    except (OSError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {feature_rows:,} service-minute rows to {args.output / 'service_features_1m.csv'}")
    print(f"Wrote {label_rows:,} next-minute labels to {args.output / 'next_minute_labels.csv'}")
    print("Label source: GAIA run-log injection starts (proxy labels, not paper SLA labels).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
