#!/usr/bin/env python3
"""CLI entry point; implementation lives in data.preprocessing.build_micross_graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.preprocessing.build_micross_graph import main

if __name__ == "__main__":
    raise SystemExit(main())
