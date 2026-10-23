# GAIA MicroSS preprocessing decisions

## Inputs and streaming

The adapter expects GAIA's `MicroSS/metric/metric_full.zip`, the original multipart `MicroSS/trace/trace_split.zip` with its `.z01` through `.z31` volumes, and July run tables under `MicroSS/run/`. It uses Python's ZIP reader for the metric archive and `7zz x -so` to stream each July trace table. It deliberately avoids the merged `trace_full.zip`, which failed during extraction in the observed local workflow.

## Time alignment

- All output timestamps are UTC and bucketed by Unix epoch; the default bucket is 60 seconds.
- Metric samples use epoch milliseconds and are bucketed after conversion to seconds. For each individual metric series, values in a bucket are averaged first. The script then averages those series-level means within a service and metric family, so a series with more samples does not get extra weight.
- Trace spans are assigned using `start_time`, falling back to `timestamp`. Per service and bucket, the script records span count, 5xx span count/rate, and mean span duration in milliseconds.
- Labels for an input minute at time `t` refer to fault-injection starts in `[t + 60s, t + 120s)`. This provides the next-minute target convention; it does not label every minute while a long injection remains active.

## Labels and limitations

Run-log messages provide the injected fault type, service, start time, and duration. Duplicate log entries are deduplicated by service, type, start, and duration. The generated positive label means an injection begins on that service in the target minute. This is an available root-cause proxy, not proof that the application crossed an SLA threshold. Do not present its scores as directly comparable to the paper's failure prediction results.

The paper describes aligned windows, trace-derived service dependencies, a GNN, and a GRU, but does not specify enough detail to infer every parser, metric normalization, or missing-value rule. This adapter therefore records explicit choices: coarse metric families (`cpu`, `memory`, `network`, `disk`, `other`), zero-filled absent service-minute values, UTC 1-minute buckets, and run-log injection-start targets. Zero-filled feature values currently mean “no metric observation in this bucket” and are not statistically imputed.

The adapter currently covers July 2021, when metric, trace, and run-log data overlap. A five-minute PyTorch Dataset is implemented in `src/data/micross.py`; it builds contiguous service sequences and accepts a caller-supplied adjacency matrix. The Trace parent/child graph extractor is implemented in `scripts/build_micross_graph.py`. It derives the same chronological training interval as the trainer and counts edges only from that interval; full-data graph coverage and edge direction still need validation. Paper-style SLA labels remain future work. The training script provides a chronological train/validation/test split and evaluation for the available injection-start proxy labels; baseline comparisons are deferred. See the repository README for limitations and usage.
