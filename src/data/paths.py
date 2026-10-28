"""Project-relative data paths, independent of the process working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MICROSS_RAW_DIR = PROJECT_ROOT / "data" / "MicroSS"
MICROSS_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "micross_v2"
MICROSS_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "micross_v2"
MICROSS_WORK_DIR = PROJECT_ROOT / "data" / "tmp" / "micross"


def resolve_micross_root(path: Path) -> Path:
    """Accept a MicroSS folder, its parent, a project root, or a GAIA checkout."""
    root = path.expanduser().resolve()
    candidates = (
        root,
        root / "MicroSS",
        root / "data" / "MicroSS",
        root / "GAIA-DataSet" / "MicroSS",
    )
    for candidate in candidates:
        if (candidate / "run").is_dir() and (candidate / "metric").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find MicroSS run/ and metric/ directories under {root}")
