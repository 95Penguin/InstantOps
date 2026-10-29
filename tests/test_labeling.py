"""Check forecasting alignment and independent application/root labels."""
import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.labeling.micross import write_sla_labels
from data.preprocessing.prepare_micross import Injection
from data.windows import GraphWindowDataset


class LabelTests(unittest.TestCase):
    def test_failure_state_roots_unknown_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            start = datetime(2021, 7, 1, tzinfo=timezone.utc)
            stamps = [(start + timedelta(minutes=i)).isoformat() for i in range(7)]
            features, labels = root / "features.csv", root / "labels.csv"
            with features.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp_utc", "service", "cpu", "trace_span_count", "trace_5xx_count"])
                for i, stamp in enumerate(stamps):
                    for service in ("web", "db"):
                        # Two consecutive failures, one unrelated failure and one unknown minute.
                        writer.writerow([stamp, service, .1, 0 if i == 5 else 10,
                                         10 if service == "web" and i in (1, 2, 4) else 0])
            events = [Injection("db", "cpu", start + timedelta(minutes=1), 120)]
            report = write_sla_labels(features, labels, root / "schema.json", ("web", "db"),
                                      events, ("web",), .999)
            dataset = GraphWindowDataset(features, labels, np.eye(2), ("web", "db"), ("cpu",), history=1)
            self.assertEqual(dataset.sample_indices, [0, 1, 2, 3, 5])
            self.assertEqual(dataset[0]["failure_target"].item(), 1)
            self.assertEqual(dataset[1]["failure_target"].item(), 1)
            self.assertEqual(dataset[0]["node_targets"].tolist(), [0, 1])
            self.assertEqual(dataset[2]["failure_target"].item(), 0)
            self.assertEqual(dataset[3]["failure_target"].item(), 1)
            self.assertFalse(dataset[3]["node_label_valid"].item())
            self.assertEqual(report["unattributed_failure_minutes"], 1)
            self.assertEqual(report["root_event_protocol"]["valid_events"], 1)
            self.assertEqual(report["events"][0]["root_service"], "db")
            persistent = root / "persistent.csv"
            report = write_sla_labels(features, persistent, root / "persistent.schema.json",
                                      ("web", "db"), events, ("web",), .5,
                                      consecutive_minutes=2, max_injection_seconds=60)
            dataset = GraphWindowDataset(features, persistent, np.eye(2), ("web", "db"), ("cpu",), history=1)
            self.assertEqual(report["event_count"], 1)
            self.assertEqual(report["events"][0]["duration_minutes"], 2)
            self.assertEqual(len(report["excluded_injections"]), 1)
            self.assertEqual(dataset.sample_indices, [0, 1, 2])
            self.assertEqual(dataset[0]["failure_target"].item(), 1)
            self.assertFalse(dataset[0]["node_label_valid"].item())
            self.assertEqual(dataset[2]["failure_target"].item(), 0)

    def test_purge_removes_crossing_event_and_confirmation_leakage(self):
        from types import SimpleNamespace
        from data.splits import purged_event_split
        data = SimpleNamespace(sample_indices=list(range(4, 104)), bucket_seconds=60,
                               timestamp_epochs=list(range(0, 106*60, 60)), history=5)
        schema = {"label_lookahead_minutes": 2,
                  "events": [{"event_id": "crossing", "start_epoch": 70*60,
                              "end_epoch_exclusive": 78*60}]}
        parts, report = purged_event_split(data, schema)
        self.assertEqual(report["crossing_events_excluded"], ["crossing"])
        for part in parts:
            for end in part:
                self.assertFalse((end-4)*60 < 78*60 and (end+3)*60 > 70*60)
        for left, right in zip(parts, parts[1:]):
            self.assertLessEqual((left[-1]+3)*60, (right[0]-4)*60)


if __name__ == "__main__":
    unittest.main()
