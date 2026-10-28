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
import json
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
from zoneinfo import ZoneInfo

from constants import SERVICES
from data.micross_schema import COUNTERS, FEATURES, LOG_FEATURES, METRICS
from data.paths import MICROSS_PROCESSED_DIR, MICROSS_RAW_DIR, resolve_micross_root
from data.preflight import check_micross_inputs

SERVICE_PATTERN = re.compile(r"^(" + "|".join(SERVICES) + r")_(.+)$")
DATE_RANGE_PATTERN = re.compile(r"_(20\d{2}-\d{2}-\d{2})_(20\d{2}-\d{2}-\d{2})\.csv$")
INJECTION_PATTERN = re.compile(
    r"\[(?P<kind>cpu_anomalies|memory_anomalies)\].*?"
    r"start at (?P<start>20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r" and lasts (?P<duration>[0-9.]+) seconds",
    re.IGNORECASE,
)


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
        nargs="?",
        default=MICROSS_RAW_DIR,
        help="MicroSS directory or parent (default: <project>/data/MicroSS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MICROSS_PROCESSED_DIR,
        help="Output directory (default: <project>/data/processed/micross_v2)",
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
    parser.add_argument(
        "--business-archive", type=Path, help="Original multipart business/business_split.zip"
    )
    parser.add_argument(
        "--source-timezone",
        default="UTC",
        help="Timezone of naive trace/business/run timestamps; metrics epoch is unchanged",
    )
    parser.add_argument(
        "--check-inputs-only", action="store_true",
        help="Check dependencies and archive directories without scanning telemetry",
    )
    args = parser.parse_args()
    ZoneInfo(args.source_timezone)
    if args.window_seconds != 60:
        parser.error("only 60-second windows are supported because labels target the next minute")
    return args


def locate_micross(root: Path) -> Path:
    return resolve_micross_root(root)


def resolve_input(path: Path | None, default: Path) -> Path:
    result = (path or default).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def minute_epoch(timestamp: int, bucket_seconds: int) -> int:
    return timestamp - timestamp % bucket_seconds


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


def aggregate_metrics(archive_path: Path, bucket_seconds: int, quality: list | None = None):
    """Keep separate metric channels; convert network counters to causal rates."""
    totals = defaultdict(lambda: [0.0, 0])
    files_read = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            identity = metric_member_identity(member)
            if identity is None or "2021-07-" not in member:
                continue
            service, metric_name = identity
            if metric_name not in METRICS:
                continue
            previous = None
            count = invalid = zero = resets = 0
            minimum, maximum = float("inf"), float("-inf")
            with archive.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                if not reader.fieldnames or not {"timestamp", "value"}.issubset(reader.fieldnames):
                    raise ValueError(f"metric columns missing: {member}")
                for row in reader:
                    try:
                        timestamp = int(row["timestamp"]) // 1000
                        value = float(row["value"])
                        if not math.isfinite(value):
                            raise ValueError("nonfinite")
                    except (ValueError, TypeError, KeyError):
                        invalid += 1
                        continue
                    if metric_name in COUNTERS:
                        current = (timestamp, value)
                        if previous is None:
                            previous = current
                            continue
                        elapsed = timestamp - previous[0]
                        delta = value - previous[1]
                        if elapsed <= 0:
                            invalid += 1
                            continue
                        previous = current
                        if delta < 0:
                            resets += 1
                            continue
                        value = delta / elapsed
                    bucket = minute_epoch(timestamp, bucket_seconds)
                    aggregate = totals[(service, bucket, metric_name)]
                    aggregate[0] += value
                    aggregate[1] += 1
                    count += 1
                    zero += int(value == 0)
                    minimum, maximum = min(minimum, value), max(maximum, value)
            files_read += 1
            if quality is not None:
                quality.append(
                    {
                        "member": member,
                        "service": service,
                        "metric": metric_name,
                        "valid": count,
                        "invalid": invalid,
                        "counter_resets": resets,
                        "zero": zero,
                        "min": minimum if count else None,
                        "max": maximum if count else None,
                        "constant": bool(count and minimum == maximum),
                    }
                )
            print(f"Read metric {files_read}: {member}", file=sys.stderr)
    if not totals:
        raise ValueError("no selected resource metrics found")
    return {key: (values[0] / values[1], values[1]) for key, values in totals.items()}, files_read


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


def parse_trace_timestamp(value: str, source_timezone: str = "UTC") -> int | None:
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    return int(parsed.replace(tzinfo=ZoneInfo(source_timezone)).timestamp())


def aggregate_traces(
    archive_path: Path, bucket_seconds: int, source_timezone: str = "UTC"
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
                timestamp = parse_trace_timestamp(
                    row.get("end_time") or row.get("timestamp") or "", source_timezone
                )
                if timestamp is None:
                    continue
                bucket = minute_epoch(timestamp, bucket_seconds)
                aggregate = totals[(service, bucket)]
                aggregate[0] += 1
                status = (row.get("status_code") or "").strip()
                if status.startswith("5"):
                    aggregate[1] += 1
                try:
                    started = parse_datetime((row.get("start_time") or "").strip())
                    ended = parse_datetime((row.get("end_time") or "").strip())
                    if started is None or ended is None or ended < started:
                        raise ValueError("invalid span duration")
                    duration_ms = (ended - started).total_seconds() * 1000
                    aggregate[2] += duration_ms
                    aggregate[3] += 1
                except (TypeError, ValueError):
                    pass
                member_counts[service] += 1
        finally:
            failed = sys.exc_info()[0] is not None
            if failed and process.poll() is None:
                process.terminate()
            stream.close()
            return_code = process.wait()
            if return_code and not failed:
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


LOCAL_ISO_TIME = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(?:[.][0-9]{1,6})?"
)


def parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    # The canonical source format has a much faster ISO parser. Keep the
    # legacy strptime fallback for previously accepted non-padded values.
    if LOCAL_ISO_TIME.fullmatch(value):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    return None


def load_injections(run_dir: Path, source_timezone: str = "UTC") -> list[Injection]:
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
                started_at = started_at.replace(tzinfo=ZoneInfo(source_timezone))
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
    logs: dict | None = None,
) -> tuple[int, int]:
    injection_starts: dict[tuple[str, int], set[str]] = defaultdict(set)
    for event in injections:
        started = event.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        epoch = int(started.timestamp())
        bucket = minute_epoch(epoch, bucket_seconds)
        injection_starts[(event.service, bucket)].add(event.kind)

    def observed_buckets() -> Iterator[int]:
        yield from (bucket for _, bucket, _ in metrics)
        yield from (bucket for _, bucket in traces)
        # Injection records provide targets, never extend telemetry coverage.
        # Business logs are aligned onto the existing resource/trace panel.

    first_bucket = min(observed_buckets(), default=None)
    last_bucket = max(observed_buckets(), default=None)
    if first_bucket is None or last_bucket is None:
        raise RuntimeError("No July metric, trace, or injection observations were found")
    bucket_count = (last_bucket - first_bucket) // bucket_seconds + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "service_features_1m.csv"
    label_path = output_dir / "next_minute_labels.csv"

    feature_fields = ["timestamp_utc", "service", "telemetry_valid", *FEATURES]
    label_fields = [
        "input_minute_utc",
        "target_minute_utc",
        "service",
        "fault_next_minute",
        "fault_types",
    ]
    feature_rows = 0
    label_rows = 0
    with (
        feature_path.open("w", encoding="utf-8", newline="") as feature_handle,
        label_path.open("w", encoding="utf-8", newline="") as label_handle,
    ):
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
                for metric_name in METRICS:
                    value = metrics.get((service, bucket, metric_name))
                    row[f"metric_{metric_name}"] = value[0] if value else 0.0
                    row[f"observed_{metric_name}"] = int(value is not None)
                row["telemetry_valid"] = int(
                    any(row[f"observed_{name}"] for name in METRICS)
                )
                row.update({name: 0 for name in LOG_FEATURES})
                row.update((logs or {}).get((service, bucket), {}))
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


def aggregate_business(
    archive: Path, bucket_seconds: int, source_timezone: str = "UTC",
    quality: Counter | None = None,
):
    """Read actual business logs, excluding injected run-log labels from inputs."""
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    if seven_zip is None:
        raise FileNotFoundError("7-Zip is required")
    listing = subprocess.run(
        [seven_zip, "l", "-slt", str(archive)], check=True, capture_output=True, text=True
    )
    members = [
        line.removeprefix("Path = ")
        for line in listing.stdout.splitlines()
        if line.startswith("Path = business/") and line.endswith("2021-07.csv")
    ]
    if not members:
        raise ValueError("no July business tables found")
    totals = defaultdict(
        lambda: {"log_count": 0, "log_error_count": 0, "log_warning_count": 0, "log_observed": 1}
    )
    counts = Counter()
    quality = quality if quality is not None else Counter()
    for member in members:
        process, stream = open_trace_member(seven_zip, archive, member)
        try:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"service", "message"}.issubset(reader.fieldnames):
                raise ValueError(f"business columns missing: {member}")
            for row in reader:
                quality["total_rows"] += 1
                service = (row.get("service") or "").strip()
                message = row.get("message") or ""
                if service not in SERVICES:
                    quality["unknown_service_rows"] += 1
                    continue
                if not message.strip():
                    quality["empty_message_rows"] += 1
                    continue
                # Timestamp and severity are inside message; datetime is date only.
                match = re.match(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[,.]\d+)?)\s*\|\s*(\w+)\s*\|", message
                )
                if not match:
                    raise ValueError(
                        f"cannot parse business timestamp/severity in {member}, "
                        f"CSV line {reader.line_num}: {message[:160]!r}"
                    )
                timestamp = parse_trace_timestamp(match.group(1).replace(",", "."), source_timezone)
                if timestamp is None:
                    raise ValueError("invalid business timestamp")
                aggregate = totals[(service, minute_epoch(timestamp, bucket_seconds))]
                aggregate["log_count"] += 1
                aggregate["log_error_count"] += int(
                    match.group(2).upper() in {"ERROR", "CRITICAL", "FATAL"}
                )
                aggregate["log_warning_count"] += int(match.group(2).upper() in {"WARN", "WARNING"})
                counts[service] += 1
                quality["parsed_rows"] += 1
        finally:
            failed = sys.exc_info()[0] is not None
            if failed and process.poll() is None:
                process.terminate()
            stream.close()
            code = process.wait()
            if code and not failed:
                raise RuntimeError(f"7-Zip failed reading business logs: {member}")
        print(f"Read business table: {member}", file=sys.stderr)
    return dict(totals), counts


def main() -> int:
    args = parse_args()
    try:
        root = locate_micross(args.gaia_root)
        business_archive = resolve_input(
            args.business_archive, root / "business" / "business_split.zip"
        )
        metric_archive = resolve_input(args.metric_archive, root / "metric" / "metric_full.zip")
        trace_archive = resolve_input(args.trace_archive, root / "trace" / "trace_split.zip")
        run_dir = (args.run_dir or root / "run").expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)

        input_report = check_micross_inputs(metric_archive, trace_archive, business_archive, run_dir)
        injections = load_injections(run_dir, args.source_timezone)
        if not injections:
            raise ValueError("no injection labels parsed from July run logs")
        if args.check_inputs_only:
            print(json.dumps(input_report, indent=2))
            return 0

        print(f"MicroSS root: {root}", file=sys.stderr)
        # Validate business parsing before spending time scanning the full trace archive.
        log_quality = Counter()
        logs, log_counts = aggregate_business(
            business_archive, args.window_seconds, args.source_timezone, log_quality
        )
        print(f"Business log quality: {dict(log_quality)}", file=sys.stderr)
        print(f"Reading metrics from {metric_archive}", file=sys.stderr)
        metric_quality = []
        metrics, metric_files = aggregate_metrics(
            metric_archive, args.window_seconds, metric_quality
        )
        print(f"Read {metric_files} metric series; aligned keys: {len(metrics)}", file=sys.stderr)
        print(f"Reading traces from original multipart archive {trace_archive}", file=sys.stderr)
        traces, trace_counts = aggregate_traces(
            trace_archive, args.window_seconds, args.source_timezone
        )
        print(f"Trace spans by service: {dict(trace_counts)}", file=sys.stderr)
        print(f"Unique injection events: {len(injections)}", file=sys.stderr)
        feature_rows, label_rows = write_outputs(
            args.output.expanduser().resolve(),
            args.window_seconds,
            metrics,
            traces,
            injections,
            logs,
        )
        output = args.output.expanduser().resolve()
        (output / "feature_schema.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "source_timezone": args.source_timezone,
                    "feature_columns": list(FEATURES),
                    "services": list(SERVICES),
                    "counter_rates": sorted(COUNTERS),
                    "window_validity": "each service has >=1 selected resource metric in every input and target minute",
                },
                indent=2,
            )
        )
        report = {
            "input_preflight": input_report,
            "window_validity": "each service has >=1 selected resource metric in every input and target minute",
            "invalid_service_minutes": feature_rows - len({
                (service, bucket) for service, bucket, _ in metrics
            }),
            "metric_series": metric_quality,
            "trace_spans_by_service": dict(trace_counts),
            "business_logs_by_service": dict(log_counts),
            "business_log_quality": dict(log_quality),
            "injections": len(injections),
            "service_minute_rows": feature_rows,
            "missing_policy": "zero fill plus metric observed flags and log_observed",
            "trace_bucket": "completion time",
            "source_timezone_assumption": args.source_timezone,
            "label_definition": "next-minute injection start",
        }
        (output / "quality_report.json").write_text(json.dumps(report, indent=2))
    except (
        OSError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {feature_rows:,} service-minute rows to {args.output / 'service_features_1m.csv'}"
    )
    print(f"Wrote {label_rows:,} next-minute labels to {args.output / 'next_minute_labels.csv'}")
    print("Label source: GAIA run-log injection starts (proxy labels, not paper SLA labels).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
