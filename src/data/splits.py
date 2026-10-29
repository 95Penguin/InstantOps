"""Shared chronological dataset split utilities."""

from __future__ import annotations


def chronological_split(
    sample_indices: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Split ordered sample endpoints into 70/15/15 train/validation/test ranges."""
    count = len(sample_indices)
    if count < 3:
        raise ValueError(f"need at least 3 usable windows for train/val/test; found {count}")
    train_end = min(max(1, int(count * 0.70)), count - 2)
    validation_end = min(max(train_end + 1, int(count * 0.85)), count - 1)
    splits = (
        sample_indices[:train_end],
        sample_indices[train_end:validation_end],
        sample_indices[validation_end:],
    )
    if any(not split for split in splits):
        raise ValueError("chronological split produced an empty partition")
    return splits


def purged_event_split(dataset, label_schema):
    """Chronological split with disjoint input/evidence intervals and incidents.

    Events crossing a fixed boundary are excluded in all partitions, including
    windows whose history touches them. Confirmation lookahead is purged too.
    """
    raw = chronological_split(dataset.sample_indices)
    seconds = dataset.bucket_seconds
    times = dataset.timestamp_epochs
    cuts = [times[part[0]] + seconds for part in raw[1:]]
    crossing = [event for event in label_schema.get("events", [])
                if any(event["start_epoch"] < cut < event["end_epoch_exclusive"] for cut in cuts)]
    lookahead = label_schema.get("label_lookahead_minutes", 1)
    result = []
    for part, lower, upper in zip(raw, [float('-inf'), *cuts], [*cuts, float('inf')]):
        keep = []
        for end in part:
            left = times[end - dataset.history + 1]
            right = times[end] + (lookahead + 1) * seconds
            if left < lower or right > upper:
                continue
            if any(left < e["end_epoch_exclusive"] and right > e["start_epoch"] for e in crossing):
                continue
            keep.append(end)
        if not keep:
            raise ValueError("event purge leaves an empty partition")
        result.append(keep)
    return tuple(result), {
        "policy": "purged_events", "boundary_epochs": cuts,
        "crossing_events_excluded": [e["event_id"] for e in crossing],
        "dropped_windows": [len(a)-len(b) for a,b in zip(raw,result)],
    }
