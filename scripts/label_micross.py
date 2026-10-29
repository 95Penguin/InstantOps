"""Relabel cached telemetry without rebuilding features or graphs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from constants import SERVICES
from data.labeling.micross import write_sla_labels
from data.preprocessing.prepare_micross import load_injections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/MicroSS/run")
    parser.add_argument("--error-rate-threshold", type=float, required=True)
    parser.add_argument("--min-entry-spans", type=int, default=1)
    parser.add_argument("--consecutive-minutes", type=int, default=1)
    parser.add_argument("--max-injection-seconds", type=float)
    parser.add_argument("--entry-services", nargs="+", default=["webservice1", "webservice2"])
    args = parser.parse_args()
    schema = json.loads((args.data_dir / "feature_schema.json").read_text())
    if any(p.exists() for p in (args.output, args.output.with_suffix(".schema.json"), args.output.with_suffix(".events.csv"))):
        parser.error("output exists; choose a new path to preserve prior experiments")
    injections = load_injections(args.run_dir, schema.get("source_timezone", "UTC"))
    report = write_sla_labels(
        args.data_dir / "service_features_1m.csv", args.output,
        args.output.with_suffix(".schema.json"), SERVICES, injections,
        args.entry_services, args.error_rate_threshold, args.min_entry_spans,
        consecutive_minutes=args.consecutive_minutes,
        max_injection_seconds=args.max_injection_seconds,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
