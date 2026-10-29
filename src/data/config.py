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


"""Versioned MicroSS features; preserve metric units until train-only scaling."""

METRICS = (
    "docker_cpu_kernel_pct",
    "docker_cpu_user_pct",
    "docker_cpu_system_pct",
    "docker_cpu_total_pct",
    "docker_memory_usage_pct",
    "docker_memory_usage_total",
    "docker_diskio_summary_rate",
    "docker_network_in_bytes",
    "docker_network_out_bytes",
)
COUNTERS = {"docker_network_in_bytes", "docker_network_out_bytes"}
TRACE_FEATURES = ("trace_span_count", "trace_5xx_count", "trace_5xx_rate", "trace_duration_mean_ms")
LOG_FEATURES = ("log_count", "log_error_count", "log_warning_count", "log_observed")
FEATURES = (
    tuple(f"metric_{name}" for name in METRICS)
    + tuple(f"observed_{name}" for name in METRICS)
    + TRACE_FEATURES
    + LOG_FEATURES
)
