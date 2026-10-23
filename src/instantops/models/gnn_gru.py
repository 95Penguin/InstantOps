"""Starter GNN-GRU model for InstantOps-style graph time series."""

from __future__ import annotations

import torch
from torch import nn


class GraphConvolution(nn.Module):
    """One normalized graph-convolution layer for batched service graphs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Args: node_features [..., nodes, features], adjacency [..., nodes, nodes]."""
        nodes = adjacency.size(-1)
        eye = torch.eye(nodes, device=adjacency.device, dtype=adjacency.dtype)
        adjacency = adjacency + eye
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)
        return self.linear(torch.matmul(normalized, node_features))


class InstantOpsGNNGRU(nn.Module):
    """Spatial graph encoder + per-service temporal GRU.

    Expected inputs:
      node_features: [batch, time, services, feature_dim]
      adjacency:     [batch, time, services, services]

    Outputs include application-level failure logits and per-service scores.
    The initial implementation deliberately keeps dataset parsing and label
    conventions outside the model.
    """

    def __init__(
        self,
        feature_dim: int,
        graph_hidden: int = 64,
        temporal_hidden: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_conv = GraphConvolution(feature_dim, graph_hidden)
        self.graph_norm = nn.LayerNorm(graph_hidden)
        self.dropout = nn.Dropout(dropout)
        self.temporal = nn.GRU(graph_hidden, temporal_hidden, batch_first=True)
        self.failure_head = nn.Linear(temporal_hidden, num_classes)
        self.node_score_head = nn.Linear(temporal_hidden, 1)

    def forward(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if node_features.ndim != 4:
            raise ValueError("node_features must have shape [B, T, N, F]")
        if adjacency.ndim != 4:
            raise ValueError("adjacency must have shape [B, T, N, N]")
        if node_features.shape[:3] != adjacency.shape[:3] or adjacency.shape[-1] != node_features.shape[2]:
            raise ValueError("node_features and adjacency batch/time/node dimensions must match")

        batch, steps, services, _ = node_features.shape
        spatial = self.graph_conv(
            node_features.reshape(batch * steps, services, -1),
            adjacency.reshape(batch * steps, services, services),
        )
        spatial = torch.relu(self.graph_norm(spatial))
        spatial = self.dropout(spatial).reshape(batch, steps, services, -1)

        # Encode each service's sequence independently after spatial message passing.
        per_service = spatial.permute(0, 2, 1, 3).reshape(batch * services, steps, -1)
        temporal, _ = self.temporal(per_service)
        final_nodes = temporal[:, -1].reshape(batch, services, -1)

        # Pool node states for application-level failure prediction.
        app_state = final_nodes.mean(dim=1)
        return {
            "failure_logits": self.failure_head(app_state),
            "node_scores": self.node_score_head(final_nodes).squeeze(-1),
            "node_embeddings": final_nodes,
        }
