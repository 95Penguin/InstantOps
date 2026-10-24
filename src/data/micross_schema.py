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
