"""Build next-minute MicroSS application-failure and root-cause labels."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class InjectionLike(Protocol):
    service: str
    kind: str
    started_at: datetime
    duration_seconds: float


def _parse_epoch(value: str) -> int:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("feature timestamps must include an explicit UTC offset")
    return int(timestamp.timestamp())


def write_sla_labels(
    feature_csv: str | Path,
    label_csv: str | Path,
    schema_json: str | Path,
    services: Sequence[str],
    injections: Iterable[InjectionLike],
    entry_services: Sequence[str] = ("webservice1", "webservice2"),
    error_rate_threshold: float = 0.999,
    min_entry_spans: int = 1,
    bucket_seconds: int = 60,
    consecutive_minutes: int = 1,
    max_injection_seconds: float | None = None,
) -> dict[str, object]:
    """Label failure at t+1 from entry-service 5xx evidence in that target minute.

    Resource metrics and current-minute trace statistics remain model inputs. The
    target uses only the following minute, so the label is not copied from an
    input feature. A high-confidence root label is assigned only when exactly
    one reliable injection is active at the onset of a failure event. That root
    is then retained for the whole event; failures without a unique onset
    injection remain valid prediction targets but are masked from RCA metrics.
    """
    if not 0 < error_rate_threshold <= 1:
        raise ValueError("error_rate_threshold must be in (0, 1]")
    if min_entry_spans < 1:
        raise ValueError("min_entry_spans must be positive")
    if bucket_seconds < 1:
        raise ValueError("bucket_seconds must be positive")
    if consecutive_minutes < 1:
        raise ValueError("consecutive_minutes must be positive")
    if max_injection_seconds is not None and max_injection_seconds <= 0:
        raise ValueError("max_injection_seconds must be positive")
    services = tuple(services)
    entry_services = tuple(entry_services)
    unknown = sorted(set(entry_services) - set(services))
    if not entry_services or unknown:
        raise ValueError(f"entry services must be nonempty known services; unknown={unknown}")

    timestamps: list[tuple[str, int]] = []
    seen_timestamps: set[str] = set()
    entry_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    with Path(feature_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp_utc", "service", "trace_span_count", "trace_5xx_count"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"feature CSV is missing label evidence columns: {missing}")
        for row in reader:
            stamp = row["timestamp_utc"]
            if stamp not in seen_timestamps:
                seen_timestamps.add(stamp)
                timestamps.append((stamp, _parse_epoch(stamp)))
            if row["service"] not in entry_services:
                continue
            try:
                spans = int(float(row["trace_span_count"]))
                errors = int(float(row["trace_5xx_count"]))
            except (TypeError, ValueError) as exc:
                raise ValueError("trace label evidence must be numeric") from exc
            if spans < 0 or errors < 0 or errors > spans:
                raise ValueError("invalid trace span/error counts for label evidence")
            aggregate = entry_counts[_parse_epoch(stamp)]
            aggregate[0] += spans
            aggregate[1] += errors

    timestamps.sort(key=lambda item: item[1])
    injections = list(injections)
    excluded = [e for e in injections if max_injection_seconds is not None
                and e.duration_seconds > max_injection_seconds]
    reliable = [e for e in injections if e not in excluded]
    observed = {epoch for epoch, (count, _) in entry_counts.items() if count >= min_entry_spans}
    high = {epoch for epoch in observed
            if entry_counts[epoch][1] / entry_counts[epoch][0] >= error_rate_threshold}
    runs = []
    for epoch in sorted(high):
        if not runs or epoch != runs[-1][-1] + bucket_seconds:
            runs.append([])
        runs[-1].append(epoch)
    events = []
    event_by_epoch = {}
    event_root_by_id: dict[str, str] = {}
    event_kinds_by_id: dict[str, list[str]] = {}
    for run in runs:
        if len(run) < consecutive_minutes:
            continue
        event_id = f"failure-{len(events)+1:04d}"
        start, end = run[0], run[-1] + bucket_seconds
        onset_candidates: dict[str, set[str]] = defaultdict(set)
        for injection in reliable:
            injection_start = injection.started_at
            if injection_start.tzinfo is None:
                injection_start = injection_start.replace(tzinfo=timezone.utc)
            injection_start_epoch = injection_start.timestamp()
            injection_end_epoch = injection_start_epoch + injection.duration_seconds
            if injection_start_epoch <= start < injection_end_epoch:
                onset_candidates[injection.service].add(injection.kind)
        candidate_services = sorted(onset_candidates)
        root_service = candidate_services[0] if len(candidate_services) == 1 else ""
        root_kinds = sorted(onset_candidates[root_service]) if root_service else []
        if root_service:
            event_root_by_id[event_id] = root_service
            event_kinds_by_id[event_id] = root_kinds
        events.append({
            "event_id": event_id, "start_epoch": start, "end_epoch_exclusive": end,
            "start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_utc_exclusive": datetime.fromtimestamp(end, timezone.utc).isoformat(),
            "duration_minutes": len(run),
            "onset_observed": start - bucket_seconds in observed,
            "recovery_observed": end in observed,
            "request_count": sum(entry_counts[t][0] for t in run),
            "error_count": sum(entry_counts[t][1] for t in run),
            "root_service": root_service,
            "root_fault_types": ";".join(root_kinds),
            "root_label_valid": int(bool(root_service)),
            "root_candidate_count": len(candidate_services),
            "root_candidates": ";".join(candidate_services),
        })
        event_by_epoch.update({epoch: event_id for epoch in run})
    label_path = Path(label_csv)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "input_minute_utc",
        "target_minute_utc",
        "service",
        "failure_next_minute",
        "root_cause_next_minute",
        "fault_types",
        "entry_span_count",
        "entry_5xx_count",
        "entry_5xx_rate",
        "entry_observed",
        "event_id",
        "root_label_valid",
    ]
    positive_minutes = attributed_minutes = unobserved_minutes = 0
    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stamp, epoch in timestamps:
            target = epoch + bucket_seconds
            spans, errors = entry_counts.get(target, (0, 0))
            rate = errors / spans if spans else 0.0
            failure = int(target in event_by_epoch)
            # Conservative complete evidence through the confirmation lookahead.
            label_valid = all(target + step * bucket_seconds in observed
                              for step in range(consecutive_minutes))
            unobserved_minutes += int(not label_valid)
            event_id = event_by_epoch.get(target, "")
            event_root = event_root_by_id.get(event_id, "")
            roots = {event_root} if failure and event_root else set()
            positive_minutes += failure
            attributed_minutes += int(bool(roots))
            target_stamp = datetime.fromtimestamp(target, timezone.utc).isoformat()
            for service in services:
                kinds = event_kinds_by_id.get(event_id, []) if service in roots else []
                writer.writerow(
                    {
                        "input_minute_utc": stamp,
                        "target_minute_utc": target_stamp,
                        "service": service,
                        "failure_next_minute": failure,
                        "root_cause_next_minute": int(service in roots),
                        "fault_types": ";".join(kinds),
                        "entry_span_count": spans,
                        "entry_5xx_count": errors,
                        "entry_5xx_rate": rate,
                        "entry_observed": int(label_valid),
                        "event_id": event_id,
                        "root_label_valid": int(not failure or bool(roots)),
                    }
                )

    schema: dict[str, object] = {
        "version": 2,
        "prediction_horizon_seconds": bucket_seconds,
        "failure_target": "next-minute aggregate entry-service trace 5xx rate meets threshold",
        "prediction_task": "failure state, including ongoing failures; not onset-only forecasting",
        "entry_services": list(entry_services),
        "error_rate_threshold": error_rate_threshold,
        "min_entry_spans": min_entry_spans,
        "consecutive_minutes": consecutive_minutes,
        "event_count": len(events),
        "events": events,
        "split_policy": "purged_events" if consecutive_minutes > 1 else "chronological",
        "label_lookahead_minutes": consecutive_minutes,
        "max_injection_seconds": max_injection_seconds,
        "excluded_injections": [{"service": e.service, "start": e.started_at.isoformat(),
                                 "duration_seconds": e.duration_seconds} for e in excluded],
        "root_cause_target": (
            "high-confidence proxy: the unique reliable injected service active at failure-event onset; "
            "the event root is repeated across its failed minutes and temporal overlap does not prove causation"
        ),
        "root_event_protocol": {
            "valid_events": sum(int(event["root_label_valid"]) for event in events),
            "unattributed_events": sum(not int(event["root_candidate_count"]) for event in events),
            "ambiguous_events": sum(int(event["root_candidate_count"]) > 1 for event in events),
            "evaluation_unit": "failure event; aggregate window scores within an event before ranking",
        },
        "entry_service_limitation": (
            "webservice1/2 are an entry-service approximation; top-level request identity is not verified"
        ),
        "target_minutes": len(timestamps),
        "unobserved_target_minutes": unobserved_minutes,
        "missing_policy": "entry_observed=0 excludes the target from training; unknown roots are masked",
        "positive_failure_minutes": positive_minutes,
        "attributed_failure_minutes": attributed_minutes,
        "unattributed_failure_minutes": positive_minutes - attributed_minutes,
    }
    Path(schema_json).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    event_path = label_path.with_suffix(".events.csv")
    with event_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]) if events else ["event_id"])
        writer.writeheader()
        writer.writerows(events)
    return schema
