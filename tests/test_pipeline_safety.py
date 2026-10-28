"""Regression checks for coverage, archive checks and evaluation correctness."""

import csv
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.preflight import check_split_archive
from data.windows import GraphWindowDataset
from rca.ranking import masking_importance
from training.engine import run_epoch


class SafetyTests(unittest.TestCase):
    def test_missing_minutes_exclude_history_and_target_but_observed_zero_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            times = [(datetime(2021, 7, 1, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()
                     for i in range(8)]
            for explicit in (False, True):
                fields = ["timestamp_utc", "service", "cpu", "observed_cpu"]
                if explicit:
                    fields.append("telemetry_valid")
                with (root / "features.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    for i, stamp in enumerate(times):
                        for service in ("a", "b"):
                            valid = int(not (i == 3 and service == "b"))
                            writer.writerow([stamp, service, 0, valid] + ([valid] if explicit else []))
                with (root / "labels.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["input_minute_utc", "target_minute_utc", "service", "fault_next_minute"])
                    writer.writerows([times[i], times[i + 1], service, 0]
                                     for i in range(7) for service in ("a", "b"))
                dataset = GraphWindowDataset(root / "features.csv", root / "labels.csv",
                                             np.eye(2), ("a", "b"), ("cpu", "observed_cpu"), history=2)
                # End 2 targets missing minute 3; ends 3/4 include it in history.
                self.assertEqual(dataset.sample_indices, [1, 5, 6])
                self.assertEqual(dataset.invalid_minutes, 1)

    def test_common_logit_shift_does_not_create_rca_importance(self):
        class CommonShift(nn.Module):
            def forward(self, features, adjacency):
                shift = features.sum(dim=(1, 2, 3))
                return {"failure_logits": torch.stack((shift, shift + 1), dim=-1)}
        scores = masking_importance(CommonShift(), torch.ones(2, 3, 2, 1), torch.zeros(2, 3, 2, 2))
        torch.testing.assert_close(scores, torch.zeros_like(scores))

    def test_probability_drop_ranks_influential_service_first(self):
        class Signal(nn.Module):
            def forward(self, features, adjacency):
                signal = features[:, :, 1].sum(dim=(1, 2))
                return {"failure_logits": torch.stack((torch.zeros_like(signal), signal), dim=-1)}
        scores = masking_importance(Signal(), torch.ones(1, 2, 2, 1), torch.zeros(1, 2, 2, 2))
        self.assertEqual(scores[0, 0].item(), 0)
        self.assertGreater(scores[0, 1].item(), 0)

    def test_weighted_evaluation_loss_is_batch_invariant(self):
        class Fixed(nn.Module):
            def forward(self, features, adjacency):
                value = features[:, 0, 0, 0]
                node = value[:, None]
                return {"failure_logits": torch.stack((-value, value), dim=-1),
                        "node_logits": node, "node_scores": node.sigmoid()}
        values = [-2., -1., 2., 1., .5]
        labels = [0, 0, 1, 1, 1]
        rows = [{"node_features": torch.tensor([[[value]]]), "adjacency": torch.zeros(1, 1, 1),
                 "failure_target": torch.tensor(label), "node_targets": torch.tensor([float(label)])}
                for value, label in zip(values, labels)]
        ce = nn.CrossEntropyLoss(weight=torch.tensor([1., 4.]))
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))
        losses = [run_epoch(Fixed(), DataLoader(rows, batch_size=size), torch.device("cpu"),
                            ce, bce, .7)[0] for size in (1, 2, 5)]
        for loss in losses:
            self.assertAlmostEqual(loss, losses[-1], places=6)

    def test_missing_or_empty_split_volume_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "trace_split.zip"
            archive.write_bytes(b"placeholder")
            listing = "Volumes = 3\nPath = trace/trace_table_a_2021-07.csv\n"
            result = subprocess.CompletedProcess([], 0, listing, "")
            with patch("data.preflight.subprocess.run", return_value=result):
                with self.assertRaises(FileNotFoundError):
                    check_split_archive(archive, "7zz", "trace")
                archive.with_suffix(".z01").write_bytes(b"volume")
                archive.with_suffix(".z02").write_bytes(b"")
                with self.assertRaises(FileNotFoundError):
                    check_split_archive(archive, "7zz", "trace")
                archive.with_suffix(".z02").write_bytes(b"volume")
                self.assertEqual(check_split_archive(archive, "7zz", "trace")["volumes"], 3)


if __name__ == "__main__":
    unittest.main()
