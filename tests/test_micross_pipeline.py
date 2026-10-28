"""Exercise raw ZIP -> preprocessing -> temporal graph -> training on tiny data."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from constants import SERVICES
from data.micross_schema import FEATURES, METRICS


def csv_text(fields, rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(fields)
    writer.writerows(rows)
    return stream.getvalue()


class TimezoneTests(unittest.TestCase):
    def test_naive_source_zone_does_not_depend_on_host(self):
        from data.preprocessing.prepare_micross import parse_trace_timestamp

        utc = parse_trace_timestamp("2021-07-01 12:00:00", "UTC")
        beijing = parse_trace_timestamp("2021-07-01 12:00:00", "Asia/Shanghai")
        self.assertEqual(utc - beijing, 8 * 3600)


class TraceTimezoneTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("7zz") or shutil.which("7z"), "7-Zip required")
    def test_trace_aggregation_uses_configured_timezone(self):
        from data.preprocessing.prepare_micross import aggregate_traces

        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "trace.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(
                    "trace/trace_table_dbservice1_2021-07.csv",
                    csv_text(
                        ["service_name", "start_time", "end_time", "status_code"],
                        [["dbservice1", "2021-07-01 12:00:01", "2021-07-01 12:00:03", "200"]],
                    ),
                )
            observations, _ = aggregate_traces(archive, 60, "Asia/Shanghai")
            epoch = int(datetime(2021, 7, 1, 4, tzinfo=timezone.utc).timestamp())
            self.assertIn(("dbservice1", epoch), observations)
            self.assertEqual(observations[("dbservice1", epoch)]["trace_duration_mean_ms"], 2000)


@unittest.skipUnless(shutil.which("7zz") or shutil.which("7z"), "7-Zip required")
class MicrossPipelineTests(unittest.TestCase):
    def test_raw_to_model_and_temporal_locality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "MicroSS"
            for directory in ("metric", "trace", "business", "run"):
                (raw / directory).mkdir(parents=True)
            start = datetime(2021, 7, 1, tzinfo=timezone.utc)
            times = [start + timedelta(minutes=i) for i in range(40)]
            with zipfile.ZipFile(
                raw / "metric/metric_full.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                for service in SERVICES:
                    for metric in METRICS:
                        rows = [
                            [
                                int(t.timestamp() * 1000),
                                i * 6000 if "network" in metric else (i + 1) * 0.01,
                            ]
                            for i, t in enumerate(times)
                        ]
                        member = f"metric/{service}_0.0.0.1_{metric}_2021-07-01_2021-07-31.csv"
                        archive.writestr(member, csv_text(["timestamp", "value"], rows))
            with zipfile.ZipFile(
                raw / "trace/trace_split.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                fields = [
                    "trace_id",
                    "span_id",
                    "parent_id",
                    "service_name",
                    "start_time",
                    "end_time",
                    "status_code",
                ]
                for service in SERVICES:
                    rows = []
                    for i, t in enumerate(times):
                        # Extra calls in a future minute must not enter previous graphs.
                        repeats = 5 if i == 38 and service == "dbservice1" else 1
                        for repeat in range(repeats):
                            parent = service == "webservice1"
                            rows.append(
                                [
                                    f"t{i}",
                                    f"p{i}" if parent else f"{service}{i}-{repeat}",
                                    "0" if parent else f"p{i}",
                                    service,
                                    (t + timedelta(seconds=1 if parent else 5)).strftime(
                                        "%Y-%m-%d %H:%M:%S.%f"
                                    ),
                                    (t + timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S.%f"),
                                    "200",
                                ]
                            )
                    archive.writestr(
                        f"trace/trace_table_{service}_2021-07.csv", csv_text(fields, rows)
                    )
            rows = [
                [
                    t.strftime("%Y-%m-%d"),
                    "webservice1",
                    t.strftime("%Y-%m-%d %H:%M:%S") + ",123 | ERROR | business message",
                ]
                for t in times
            ]
            with zipfile.ZipFile(
                raw / "business/business_split.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    "business/business_table_webservice1_2021-07.csv",
                    csv_text(["datetime", "service", "message"], rows),
                )
            rows = [
                [
                    times[i].strftime("%Y-%m-%d"),
                    "dbservice1",
                    f"[memory_anomalies] trigger start at {times[i].strftime('%Y-%m-%d %H:%M:%S')} and lasts 600 seconds",
                ]
                for i in (5, 14, 24, 31, 38)
            ]
            (raw / "run/run_table_2021-07.csv").write_text(
                csv_text(["datetime", "service", "message"], rows)
            )
            data, output = root / "processed", root / "results"
            command = [
                sys.executable,
                str(ROOT / "scripts/run_micross.py"),
                str(raw),
                "--data-dir",
                str(data),
                "--output-dir",
                str(output),
                "--work-dir",
                str(root / "index"),
                "--epochs",
                "1",
                "--device",
                "cpu",
                "--batch-size",
                "16",
            ]
            preflight = subprocess.run(
                [*command, "--check-inputs-only"], capture_output=True, text=True, cwd=ROOT
            )
            self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
            self.assertFalse(data.exists(), "preflight must not run preprocessing")
            self.assertFalse(output.exists(), "preflight must not start training")
            result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            schema = json.loads((data / "feature_schema.json").read_text())
            self.assertEqual(tuple(schema["feature_columns"]), FEATURES)
            with np.load(data / "service_adjacency_1m.npz") as graph:
                db, web = SERVICES.index("dbservice1"), SERVICES.index("webservice1")
                self.assertEqual(graph["adjacency"][5, db, web], 1)
                self.assertEqual(graph["adjacency"][38, db, web], 5)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["rca_method"], "masking")
            self.assertGreater(metrics["root_cause_proxy"]["positive_windows"], 0)
            self.assertIn("mfr", metrics["root_cause_proxy"])
            self.assertTrue((output / "best_model.pt").is_file())
            predictions = [
                json.loads(row) for row in (output / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(predictions), metrics["split_sizes"]["test"])
            self.assertEqual(len(predictions[0]["ranking"]), len(SERVICES))
            quality = json.loads((data / "quality_report.json").read_text())
            self.assertEqual(quality["business_logs_by_service"]["webservice1"], 40)
            self.assertEqual(quality["injections"], 5)
            # The first network sample has no past observation: no invented rate.
            with (data / "service_features_1m.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            first = rows[0]
            self.assertEqual(float(first["observed_docker_network_in_bytes"]), 0)
            self.assertEqual(float(rows[10]["metric_docker_network_in_bytes"]), 100)


if __name__ == "__main__":
    unittest.main()
