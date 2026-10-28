"""Fast input checks; archive directories are checked without a full CRC scan."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path


def check_split_archive(archive: Path, executable: str, prefix: str) -> dict:
    if not archive.is_file() or archive.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty archive: {archive}")
    listing = subprocess.run(
        [executable, "l", "-slt", str(archive)], capture_output=True, text=True, check=False
    )
    if listing.returncode:
        detail = listing.stderr.strip() or listing.stdout[-1000:]
        raise ValueError(f"Cannot list {archive}: {detail}")
    match = re.search(r"^Volumes = (\d+)$", listing.stdout, re.M)
    count = int(match.group(1)) if match else 1
    volumes = [archive.with_suffix(f".z{index:02d}") for index in range(1, count)]
    missing = [str(path) for path in volumes if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty archive volumes: {missing}")
    members = [
        line[7:] for line in listing.stdout.splitlines()
        if line.startswith(f"Path = {prefix}/") and line.endswith("2021-07.csv")
    ]
    if not members:
        raise ValueError(f"No July {prefix} CSV members in {archive}")
    return {"archive": str(archive), "volumes": count, "july_members": len(members)}


def check_micross_inputs(metric: Path, trace: Path, business: Path, run_dir: Path) -> dict:
    executable = shutil.which("7zz") or shutil.which("7z")
    if executable is None:
        raise FileNotFoundError("7-Zip required: install 7zz or 7z before preprocessing")
    for path in (metric, trace, business):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty input: {path}")
    # Python's metric reader needs a regular, combined ZIP, not a split final volume.
    with zipfile.ZipFile(metric) as archive:
        members = [
            info for info in archive.infolist()
            if info.filename.endswith(".csv") and "2021-07-" in info.filename
        ]
        if not members:
            raise ValueError(f"No July metric CSV members in {metric}")
        if any(info.volume != 0 for info in archive.infolist()):
            raise ValueError("Metrics require a combined metric_full.zip, not metric_split.zip")
    runs = sorted(run_dir.glob("run_table_2021-07*.csv"))
    if not runs or any(path.stat().st_size == 0 for path in runs):
        raise FileNotFoundError(f"Missing or empty extracted July run CSV in {run_dir}")
    return {
        "integrity": "directory and volume presence only; no full CRC test",
        "metric": {"archive": str(metric), "july_members": len(members)},
        "trace": check_split_archive(trace, executable, "trace"),
        "business": check_split_archive(business, executable, "business"),
        "run_files": [str(path) for path in runs],
    }
