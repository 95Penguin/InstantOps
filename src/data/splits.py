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
