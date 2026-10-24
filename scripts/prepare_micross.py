#!/usr/bin/env python3
"""CLI entry point; implementation lives in data.preprocessing.prepare_micross."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.preprocessing.prepare_micross import main

if __name__ == "__main__":
    raise SystemExit(main())
