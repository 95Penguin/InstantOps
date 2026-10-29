"""Business-log parsing and extractor-cleanup regressions."""

import csv
import io
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import prepare as module


def csv_text(messages):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["datetime", "service", "message"])
    for message in messages:
        writer.writerow(["2021-07-01", "webservice1", message])
    return stream.getvalue()


class TimestampTests(unittest.TestCase):
    def test_fast_parser_preserves_legacy_results(self):
        for value in ("2021-07-01 12:00:00", "2021-07-01 12:00:00.123456",
                      "2021-7-1 1:2:3", "2021-02-30 00:00:00", "", "bad"):
            expected = None
            for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    expected = datetime.strptime(value, pattern)
                    break
                except ValueError:
                    pass
            self.assertEqual(module.parse_datetime(value), expected)


@unittest.skipUnless(shutil.which("7zz") or shutil.which("7z"), "7-Zip required")
class BusinessLogTests(unittest.TestCase):
    def test_blank_messages_are_counted_and_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "business.zip"
            with ZipFile(archive, "w") as output:
                output.writestr("business/business_table_webservice1_2021-07.csv",
                                csv_text(["", "   ", "2021-07-01 12:00:00,123 | INFO | hello"]))
            quality = Counter()
            totals, counts = module.aggregate_business(archive, 60, "UTC", quality)
            self.assertEqual(quality["total_rows"], 3)
            self.assertEqual(quality["empty_message_rows"], 2)
            self.assertEqual(quality["parsed_rows"], 1)
            self.assertEqual(counts["webservice1"], 1)
            self.assertEqual(sum(row["log_count"] for row in totals.values()), 1)

    def test_primary_parse_error_is_not_hidden_by_extractor_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "business.zip"
            member = "business/business_table_webservice1_2021-07.csv"
            with ZipFile(archive, "w") as output:
                output.writestr(member, csv_text(["malformed nonempty message"]))
            process = Mock()
            process.poll.return_value = None
            process.wait.return_value = -15
            stream = io.StringIO(csv_text(["malformed nonempty message"]))
            with patch.object(module, "open_trace_member", return_value=(process, stream)):
                with self.assertRaisesRegex(ValueError, "CSV line 2"):
                    module.aggregate_business(archive, 60)
            process.terminate.assert_called_once()
            self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
