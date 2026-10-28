"""Shared aligned service-panel reader, independent of dataset identities."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GraphWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Create contiguous graph sequences from adapter-produced CSV panels.

    Each item contains:
      * ``node_features``: [history, services, features], float32
      * ``adjacency``: [history, services, services], float32
      * ``failure_target``: scalar; union of the adapter's binary node targets
      * ``node_targets``: [services]; adapter-defined binary node labels

    The adjacency matrix is supplied by the caller. The repository includes a
    trace-derived graph builder, but its full-dataset coverage still needs
    validation before the graph should be treated as authoritative.
    """

    def __init__(
        self,
        feature_csv: str | Path,
        label_csv: str | Path,
        adjacency: np.ndarray | torch.Tensor,
        services: Sequence[str],
        feature_columns: Sequence[str],
        history: int = 5,
        bucket_seconds: int = 60,
    ) -> None:
        if history < 1:
            raise ValueError("history must be at least 1")
        if bucket_seconds < 1:
            raise ValueError("bucket_seconds must be positive")

        self.feature_columns = tuple(feature_columns)
        if not services or len(set(services)) != len(services):
            raise ValueError("services must be nonempty and unique")
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature columns must be nonempty and unique")
        self.services = tuple(services)
        self.history = history
        self.bucket_seconds = bucket_seconds
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        expected_shape = (len(self.services), len(self.services))
        if tuple(self.adjacency.shape) != expected_shape:
            raise ValueError(f"adjacency must have shape {expected_shape}")
        if not torch.isfinite(self.adjacency).all():
            raise ValueError("adjacency must contain only finite values")
        if torch.any(self.adjacency < 0):
            raise ValueError("adjacency weights must be nonnegative")

        features_by_time = self._read_features(Path(feature_csv))
        labels_by_time, target_times = self._read_labels(Path(label_csv))
        self.timestamps = sorted(features_by_time)
        self.timestamp_epochs = [self._parse_timestamp(value) for value in self.timestamps]
        available_timestamps = set(self.timestamps)
        if not self.timestamps:
            raise ValueError("feature CSV contains no service-minute rows")

        self.features = np.stack([features_by_time[t] for t in self.timestamps])
        self.valid_minutes = np.array([self._valid_by_time[t] for t in self.timestamps])
        time_indices = {timestamp: index for index, timestamp in enumerate(self.timestamps)}
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
            and self.valid_minutes[end - history + 1 : end + 1].all()
            and self.valid_minutes[time_indices[target_times[self.timestamps[end]]]]
        ]
        self.invalid_minutes = int((~self.valid_minutes).sum())
        if not self.sample_indices:
            raise ValueError(
                f"no valid contiguous {history}-minute sequences with observed input/target coverage"
            )

    def _read_features(self, path: Path) -> dict[str, np.ndarray]:
        service_indices = {name: index for index, name in enumerate(self.services)}
        values: dict[str, np.ndarray] = {}
        self._valid_by_time: dict[str, bool] = {}
        seen: dict[str, set[int]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"timestamp_utc", "service", *self.feature_columns}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                missing = sorted(required - set(reader.fieldnames or ()))
                raise ValueError(f"feature CSV is missing columns: {missing}")
            observed_columns = [
                column for column in self.feature_columns if column.startswith("observed_")
            ]
            for row in reader:
                service_index = service_indices.get(row["service"])
                if service_index is None:
                    continue
                timestamp = row["timestamp_utc"]
                if "telemetry_valid" in row:
                    valid = self._to_float(row["telemetry_valid"], "telemetry_valid")
                    if valid not in (0.0, 1.0):
                        raise ValueError("telemetry_valid must be 0 or 1")
                    valid = bool(valid)
                elif observed_columns:
                    # Older MicroSS v2 panels already contain resource observed flags.
                    valid = any(
                        self._to_float(row[column], column) > 0 for column in observed_columns
                    )
                else:
                    valid = True
                self._valid_by_time[timestamp] = self._valid_by_time.get(timestamp, True) and valid
                indices = seen.setdefault(timestamp, set())
                if service_index in indices:
                    raise ValueError(f"duplicate feature row for {timestamp}/{row['service']}")
                indices.add(service_index)
                matrix = values.setdefault(
                    timestamp,
                    np.zeros((len(self.services), len(self.feature_columns)), dtype=np.float32),
                )
                matrix[service_index] = [
                    self._to_float(row[column], column) for column in self.feature_columns
                ]
        if any(len(indices) != len(self.services) for indices in seen.values()):
            raise ValueError("feature panel must contain every service at each timestamp")
        return values

    def _read_labels(self, path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        service_indices = {name: index for index, name in enumerate(self.services)}
        labels: dict[str, np.ndarray] = {}
        target_times: dict[str, str] = {}
        seen: dict[str, set[int]] = {}
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
                indices = seen.setdefault(timestamp, set())
                if service_index in indices:
                    raise ValueError(f"duplicate label row for {timestamp}/{row['service']}")
                indices.add(service_index)
                target_time = row["target_minute_utc"]
                if (
                    self._parse_timestamp(target_time) - self._parse_timestamp(timestamp)
                    != self.bucket_seconds
                ):
                    raise ValueError("target timestamp must be one bucket after input timestamp")
                if timestamp in target_times and target_times[timestamp] != target_time:
                    raise ValueError("inconsistent target timestamp across services")
                target_times[timestamp] = target_time
                try:
                    label = float(row["fault_next_minute"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid fault label: {row['fault_next_minute']!r}") from exc
                if label not in (0.0, 1.0):
                    raise ValueError(f"fault labels must be 0 or 1, got {label}")
                target = labels.setdefault(
                    timestamp, np.zeros(len(self.services), dtype=np.float32)
                )
                target[service_index] = label
        if any(len(indices) != len(self.services) for indices in seen.values()):
            raise ValueError("label panel must contain every service at each labeled timestamp")
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

        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("panel timestamps must include an explicit UTC offset")
        return int(timestamp.timestamp())

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        end = self.sample_indices[item]
        start = end - self.history + 1
        node_targets = torch.from_numpy(self.node_targets[end].copy())
        return {
            "node_features": torch.from_numpy(self.features[start : end + 1].copy()),
            "adjacency": (
                self.adjacency[start : end + 1]
                if self.adjacency.ndim == 3
                else self.adjacency.unsqueeze(0).expand(self.history, -1, -1)
            ),
            "failure_target": node_targets.max().to(torch.long),
            "node_targets": node_targets,
        }
