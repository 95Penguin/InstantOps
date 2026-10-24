"""Classification and service ranking metrics; ranks start at one."""

from __future__ import annotations

import numpy as np


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    cumulative = np.cumsum(ranked)[ends]
    precision = cumulative / (ends + 1)
    recalled = np.diff(np.r_[0, cumulative]) / positives
    return float(np.sum(precision * recalled))


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    predictions = probabilities >= 0.5
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "accuracy": float((predictions == labels).mean()),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": f1,
        "average_precision": average_precision(labels, probabilities),
    }


def rca_metrics(node_targets: np.ndarray, node_scores: np.ndarray) -> dict[str, float | int | None]:
    positive_rows = np.flatnonzero(node_targets.sum(axis=1) > 0)
    if not len(positive_rows):
        return {
            "positive_windows": 0,
            "hit_at_1": None,
            "hit_at_2": None,
            "hit_at_3": None,
            "mfr": None,
            "mar": None,
            "mrr": None,
        }
    first_ranks: list[float] = []
    average_ranks: list[float] = []
    hits_2: list[float] = []
    hits_1: list[float] = []
    hits_3: list[float] = []
    reciprocal_ranks: list[float] = []
    for row in positive_rows:
        order = np.argsort(-node_scores[row], kind="stable")
        truth = node_targets[row].astype(bool)
        ranks = np.flatnonzero(truth[order])
        first_rank = int(ranks[0])
        first_ranks.append(float(first_rank + 1))
        average_ranks.append(float(np.mean(ranks + 1)))
        hits_2.append(float(first_rank < 2))
        hits_1.append(float(first_rank < 1))
        hits_3.append(float(first_rank < 3))
        reciprocal_ranks.append(1.0 / (first_rank + 1))
    return {
        "positive_windows": int(len(positive_rows)),
        "hit_at_1": float(np.mean(hits_1)),
        "hit_at_2": float(np.mean(hits_2)),
        "mfr": float(np.mean(first_ranks)),
        "mar": float(np.mean(average_ranks)),
        "hit_at_3": float(np.mean(hits_3)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }
