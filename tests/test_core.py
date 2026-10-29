"""Regression tests for shared metrics and deterministic explanations."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.dataset import chronological_split
from models.gnn_gru import InstantOpsGNNGRU
from rca.ranking import masking_importance
from training.metrics import average_precision, rca_metrics


class CoreTests(unittest.TestCase):
    def test_tied_probabilities_are_threshold_grouped(self):
        self.assertAlmostEqual(average_precision(np.array([1, 0, 0, 1]), np.ones(4)), 0.5)
        self.assertIsNone(average_precision(np.zeros(3), np.ones(3)))

    def test_multiple_root_causes(self):
        result = rca_metrics(np.array([[0, 1, 1, 0]]), np.array([[4.0, 3.0, 1.0, 2.0]]))
        self.assertEqual(result["mfr"], 2)
        self.assertEqual(result["mar"], 3)
        self.assertEqual(result["hit_at_1"], 0)
        self.assertEqual(result["hit_at_2"], 1)

    def test_masking_is_deterministic_and_restores_mode(self):
        model = InstantOpsGNNGRU(3, graph_hidden=4, temporal_hidden=4, dropout=0.8)
        x = torch.randn(2, 5, 3, 3)
        a = torch.zeros(2, 5, 3, 3)
        model.train()
        first = masking_importance(model, x, a)
        second = masking_importance(model, x, a)
        torch.testing.assert_close(first, second)
        self.assertTrue(model.training)
        self.assertEqual(first.shape, (2, 3))
        self.assertEqual(len(model.graph_convs), 2)

    def test_split_has_disjoint_ordered_targets(self):
        train, val, test = chronological_split(list(range(20)))
        self.assertLess(max(train), min(val))
        self.assertLess(max(val), min(test))


class WindowAdapterTests(unittest.TestCase):
    def test_custom_services_and_missing_labels(self):
        import csv
        import tempfile
        from datetime import datetime, timedelta, timezone

        from data.dataset import GraphWindowDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stamps = [
                (datetime(2021, 7, 1, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()
                for i in range(7)
            ]
            with (root / "features.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp_utc", "service", "cpu"])
                for stamp in stamps[:-1]:
                    for service in ("frontend", "database"):
                        writer.writerow([stamp, service, 0.2])
            rows = [
                [stamp, stamps[i + 1], service, int(i == 2 and service == "database")]
                for i, stamp in enumerate(stamps[:-1])
                for service in ("frontend", "database")
            ]

            def labels(records):
                with (root / "labels.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ["input_minute_utc", "target_minute_utc", "service", "fault_next_minute"]
                    )
                    writer.writerows(records)

            labels(rows)
            dataset = GraphWindowDataset(
                root / "features.csv",
                root / "labels.csv",
                np.zeros((2, 2)),
                services=("frontend", "database"),
                feature_columns=("cpu",),
                history=2,
            )
            self.assertEqual(dataset[0]["node_features"].shape, (2, 2, 1))
            self.assertEqual(len(dataset), 4)
            labels(rows[:-1])
            with self.assertRaisesRegex(ValueError, "label panel"):
                GraphWindowDataset(
                    root / "features.csv",
                    root / "labels.csv",
                    np.zeros((2, 2)),
                    services=("frontend", "database"),
                    feature_columns=("cpu",),
                    history=2,
                )


if __name__ == "__main__":
    unittest.main()
