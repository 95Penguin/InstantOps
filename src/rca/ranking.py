"""Root-cause ranking utilities."""

from __future__ import annotations

import torch


def rank_services(
    node_scores: torch.Tensor, service_names: list[str]
) -> list[list[tuple[str, float]]]:
    """Return descending per-sample service rankings from model node scores."""
    if node_scores.ndim == 1:
        node_scores = node_scores.unsqueeze(0)
    if node_scores.ndim != 2:
        raise ValueError("node_scores must have shape [services] or [batch, services]")
    if node_scores.size(1) != len(service_names):
        raise ValueError("service_names length must match the number of services")

    values, indices = node_scores.detach().cpu().sort(dim=1, descending=True)
    return [
        [
            (service_names[idx], float(score))
            for idx, score in zip(row_idx.tolist(), row_val.tolist())
        ]
        for row_idx, row_val in zip(indices, values)
    ]


def masking_importance(
    model,
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    target_class: int = 1,
) -> torch.Tensor:
    """Estimate service importance by masking one service at a time.

    Returns a [batch, services] nonnegative score measuring the drop in the
    target-class logit when that service's temporal features are zeroed.
    This is an interpretable starter implementation; compare it with the
    paper's exact permutation procedure once inputs and labels are prepared.
    """
    if node_features.ndim != 4:
        raise ValueError("node_features must have shape [B, T, N, F]")
    was_training = model.training
    model.eval()
    try:
        return _masking_scores(model, node_features, adjacency, target_class)
    finally:
        model.train(was_training)


@torch.no_grad()
def _masking_scores(model, node_features, adjacency, target_class):
    with torch.no_grad():
        baseline = model(node_features, adjacency)["failure_logits"][:, target_class]
        scores = []
        for service_idx in range(node_features.size(2)):
            masked = node_features.clone()
            masked[:, :, service_idx, :] = 0
            perturbed = model(masked, adjacency)["failure_logits"][:, target_class]
            scores.append((baseline - perturbed).clamp_min(0))
        return torch.stack(scores, dim=1)
