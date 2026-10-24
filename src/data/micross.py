"""MicroSS schema and adapter for the shared window dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from constants import SERVICES
from data.windows import GraphWindowDataset

FEATURE_COLUMNS = (
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
)


class MicrossWindowDataset(GraphWindowDataset):
    """Use GAIA service names and injection-start proxy labels."""

    def __init__(
        self,
        feature_csv: str | Path,
        label_csv: str | Path,
        adjacency: np.ndarray | torch.Tensor,
        history: int = 5,
        services: Sequence[str] = SERVICES,
        bucket_seconds: int = 60,
        graph_path: str | Path | None = None,
    ) -> None:
        schema_path = Path(feature_csv).parent / "feature_schema.json"
        schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}
        if schema and tuple(schema["services"]) != tuple(services):
            raise ValueError("feature schema service order mismatch")
        feature_columns = tuple(schema.get("feature_columns", FEATURE_COLUMNS))
        super().__init__(
            feature_csv,
            label_csv,
            adjacency,
            services=services,
            feature_columns=feature_columns,
            history=history,
            bucket_seconds=bucket_seconds,
        )

        if graph_path is not None:
            with np.load(graph_path, allow_pickle=False) as graph:
                if tuple(graph["services"].tolist()) != self.services:
                    raise ValueError("dynamic graph service order mismatch")
                if not np.array_equal(graph["timestamps"], self.timestamp_epochs):
                    raise ValueError("dynamic graph timestamps do not match feature panel")
                adjacency = graph["adjacency"]
                if adjacency.shape != (
                    len(self.timestamps),
                    len(self.services),
                    len(self.services),
                ):
                    raise ValueError("invalid dynamic adjacency shape")
                if not np.isfinite(adjacency).all() or (adjacency < 0).any():
                    raise ValueError("invalid dynamic adjacency values")
                self.adjacency = torch.from_numpy(adjacency.astype(np.float32))
