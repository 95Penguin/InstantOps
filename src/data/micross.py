"""PyTorch dataset for aligned GAIA MicroSS service telemetry."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import SERVICES

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


class MicrossWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Create five-minute graph sequences from the preprocessed CSV outputs.

    Each item contains:
      * ``node_features``: [history, services, features], float32
      * ``adjacency``: [history, services, services], float32
      * ``failure_target``: scalar; any service has an injection start next minute
      * ``node_targets``: [services]; per-service injection-start labels

    The adjacency matrix is supplied by the caller because the current
    preprocessing stage has not yet inferred a validated call graph.
    """

    def __init__(
        self,
        feature_csv: str | Path,
        label_csv: str | Path,
        adjacency: np.ndarray | torch.Tensor,
        history: int = 5,
        services: Sequence[str] = SERVICES,
        bucket_seconds: int = 60,
    ) -> None:
        if history < 1:
            raise ValueError("history must be at least 1")
        if bucket_seconds < 1:
            raise ValueError("bucket_seconds must be positive")

        self.services = tuple(services)
        self.history = history
        self.bucket_seconds = bucket_seconds
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        expected_shape = (len(self.services), len(self.services))
        if tuple(self.adjacency.shape) != expected_shape:
            raise ValueError(f"adjacency must have shape {expected_shape}")

        features_by_time = self._read_features(Path(feature_csv))
        labels_by_time, target_times = self._read_labels(Path(label_csv))
        self.timestamps = sorted(features_by_time)
        self.timestamp_epochs = [self._parse_timestamp(value) for value in self.timestamps]
        available_timestamps = set(self.timestamps)
        if not self.timestamps:
            raise ValueError("feature CSV contains no service-minute rows")

        self.features = np.stack([features_by_time[t] for t in self.timestamps])
        self.node_targets = np.stack(
            [
                labels_by_time.get(t, np.zeros(len(self.services), dtype=np.float32))
                for t in self.timestamps
            ]
        )
        self.sample_indices = [
            end
            for end in range(history - 1, len(self.timestamps))
            if self._is_contiguous(end)
            and self.timestamps[end] in labels_by_time
            and target_times[self.timestamps[end]] in available_timestamps
        ]
        if not self.sample_indices:
            raise ValueError(
                f"no contiguous {history}-minute sequences found in the feature CSV"
            )

    def _read_features(self, path: Path) -> dict[str, np.ndarray]:
        service_indices = {name: index for index, name in enumerate(self.services)}
        values: dict[str, np.ndarray] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"timestamp_utc", "service", *FEATURE_COLUMNS}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                missing = sorted(required - set(reader.fieldnames or ()))
                raise ValueError(f"feature CSV is missing columns: {missing}")
            for row in reader:
                service_index = service_indices.get(row["service"])
                if service_index is None:
                    continue
                timestamp = row["timestamp_utc"]
                matrix = values.setdefault(
                    timestamp,
                    np.zeros((len(self.services), len(FEATURE_COLUMNS)), dtype=np.float32),
                )
                matrix[service_index] = [
                    self._to_float(row[column], column) for column in FEATURE_COLUMNS
                ]
        return values

    def _read_labels(
        self, path: Path
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        service_indices = {name: index for index, name in enumerate(self.services)}
        labels: dict[str, np.ndarray] = {}
        target_times: dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "input_minute_utc",
                "target_minute_utc",
                "service",
                "fault_next_minute",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                missing = sorted(required - set(reader.fieldnames or ()))
                raise ValueError(f"label CSV is missing columns: {missing}")
            for row in reader:
                service_index = service_indices.get(row["service"])
                if service_index is None:
                    continue
                timestamp = row["input_minute_utc"]
                target_times[timestamp] = row["target_minute_utc"]
                target = labels.setdefault(
                    timestamp, np.zeros(len(self.services), dtype=np.float32)
                )
                target[service_index] = float(row["fault_next_minute"])
        return labels, target_times

    @staticmethod
    def _to_float(value: str, column: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric value in {column}: {value!r}") from exc
        if not np.isfinite(number):
            raise ValueError(f"non-finite numeric value in {column}: {value!r}")
        return number

    def _is_contiguous(self, end: int) -> bool:
        start = end - self.history + 1
        for index in range(start + 1, end + 1):
            gap = self.timestamp_epochs[index] - self.timestamp_epochs[index - 1]
            if gap != self.bucket_seconds:
                return False
        return True

    @staticmethod
    def _parse_timestamp(value: str) -> int:
        from datetime import datetime

        return int(datetime.fromisoformat(value).timestamp())

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        end = self.sample_indices[item]
        start = end - self.history + 1
        node_targets = torch.from_numpy(self.node_targets[end].copy())
        return {
            "node_features": torch.from_numpy(self.features[start : end + 1].copy()),
            "adjacency": self.adjacency.unsqueeze(0).expand(self.history, -1, -1),
            "failure_target": node_targets.max().to(torch.long),
            "node_targets": node_targets,
        }
