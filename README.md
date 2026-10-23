# InstantOps reproduction

This repository is a workflow-level reproduction of **InstantOps: A Joint Approach to System Failure Prediction and Root Cause Identification in Microservices Cloud-Native Applications** (ICPE 2024). It is an independent implementation; it does not claim to reproduce the paper's reported scores.

## Environment

This project uses `uv` to manage Python and dependencies. From the repository root:

```bash
uv sync
```

Run project commands through `uv run` so they use the locked environment.

## Current milestone: GAIA / MicroSS preprocessing

The MicroSS adapter aligns service metrics, trace spans, and run-log injection events into UTC time buckets. The script reads the metric ZIP and streams the original multipart trace archive, so it does not expand the multi-gigabyte trace tables to disk.

Expected dataset files:

```text
GAIA-DataSet/MicroSS/
├── metric/metric_full.zip
├── trace/trace_split.zip
├── trace/trace_split.z01 ... trace_split.z31
└── run/run_table_2021-07.csv
```

Use the original `trace_split.zip` together with all its `.z01`, `.z02`, ... volumes. Do not pass the concatenated `trace_full.zip` created with `zip -s 0`; that archive was observed to fail during trace decompression. The script requires 7-Zip (`brew install sevenzip` on macOS).

From the InstantOps repository root:

```bash
uv run python scripts/prepare_micross.py ~/project/GAIA-DataSet/MicroSS
```

If the dataset is at the GAIA checkout root, pass `~/project/GAIA-DataSet` instead. Optional paths and bucket size can be set explicitly:

```bash
uv run python scripts/prepare_micross.py ~/project/GAIA-DataSet/MicroSS \
  --metric-archive ~/project/GAIA-DataSet/MicroSS/metric/metric_full.zip \
  --trace-archive ~/project/GAIA-DataSet/MicroSS/trace/trace_split.zip \
  --run-dir ~/project/GAIA-DataSet/MicroSS/run \
  --window-seconds 60 \
  --output data/processed/micross
```

The generated files are:

- `data/processed/micross/service_features_1m.csv`: one row per service and UTC minute, with coarse metric-family means, metric-series coverage counts, and trace span/error/duration aggregates.
- `data/processed/micross/next_minute_labels.csv`: one row per service and input minute; the label is positive when a GAIA fault injection starts in the following minute.

`src/data/micross.py` loads these files into five-minute samples compatible with the model input shapes. The trace parent/child graph extractor is provided separately in `scripts/build_micross_graph.py`; its graph coverage and directions must be checked on the full dataset before treating it as validated.

The label is a **GAIA injection-start proxy**, not the application-level SLA failure label used by InstantOps. Trace 5xx counts here count spans and must not be interpreted as failed top-level requests. Feature values are grouped into coarse families because GAIA exposes many individually named metric series; this is an initial reproducible adapter, not the paper's undocumented per-metric normalization. See [the preprocessing decisions](docs/preprocessing.md).

## Train the first baseline

After preprocessing, extract the service graph from span `parent_id` relationships. The extractor streams the original multipart trace archive and uses a temporary SQLite index to resolve parent spans to services:

```bash
uv run python scripts/build_micross_graph.py \
  ~/project/GAIA-DataSet/MicroSS/trace/trace_split.zip \
  --training-data-dir data/processed/micross \
  --history 5 \
  --output-dir data/processed/micross \
  --work-dir /path/to/a/directory/with/free/space
```

`--training-data-dir` derives the same first 70% chronological training range used by the trainer. The graph counts only parent/child edges whose child span starts in that interval; it still indexes span-to-service mappings from the full archive so early training spans can resolve parents that started before the interval. The temporary index can use substantial disk space. The script writes directed span counts to `service_call_edges.csv` and a symmetric binary `service_adjacency.npy` for the GNN. The edge CSV preserves call direction; the binary adjacency is symmetrized because the current graph convolution uses symmetric degree normalization.

Train the GNN-GRU and evaluate on a chronological held-out test segment:

```bash
uv run python scripts/train_micross.py \
  --data-dir data/processed/micross \
  --adjacency-npy data/processed/micross/service_adjacency.npy \
  --output-dir artifacts/micross \
  --epochs 20 \
  --device auto
```

`auto` selects CUDA on Linux servers when available, then Apple MPS, then CPU. The script fits feature standardization on training windows only, selects the checkpoint by validation loss, and reports binary classification metrics plus Hit@1, Hit@3, and MRR against injection-start service labels. Without `--adjacency-npy`, it uses an identity adjacency matrix; that run is a temporal-only baseline, not a graph-based reproduction. A supplied NumPy adjacency matrix must follow the service ordering in `src/constants.py`. The RCA numbers are proxy localization metrics because GAIA injection labels do not establish SLA root cause.

## Project layout

```text
docs/           Paper notes and preprocessing decisions
scripts/        Runnable data-preparation entry points
src/data/      Dataset adapters and window builders
src/models/    GNN-GRU and other model implementations
src/rca/       Root-cause ranking utilities
src/constants.py Shared service and dataset constants
```

The input dataset stays outside this repository. Generated outputs go under `data/processed/` (ignored by Git) and the directory is created when preprocessing runs.

## Planned milestones

1. Inspect the selected dataset and identify its modalities, timestamps, service identifiers, and fault labels. **Done for GAIA MicroSS.**
2. Convert available data into aligned fixed-size time windows. **1-minute service panel, next-minute injection labels, and a five-minute PyTorch Dataset implemented.**
3. **Trace parent/child graph extractor implemented.** Validate graph coverage and edge directions on the full MicroSS traces before reporting graph-based results.
4. **Baseline training/evaluation entry point implemented.** The default identity graph is temporal-only; supply the extracted adjacency for graph-based experiments.
5. **Proxy RCA training and evaluation implemented** with per-service injection labels, Hit@1, Hit@3, and MRR. Paper-style SLA labels and the paper's exact ranking procedure remain unavailable.
6. Validate the trace-derived call graph on the full dataset, then evaluate one focused improvement. Baseline comparisons are deferred.

TrainTicket and the paper's private QoTD collection need separate adapters. QoTD reproduction requires deployment, fault injection, and telemetry collection.
