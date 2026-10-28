# InstantOps reproduction

This repository is a workflow-level reproduction of **InstantOps: A Joint Approach to System Failure Prediction and Root Cause Identification in Microservices Cloud-Native Applications** (ICPE 2024). It is an independent implementation; it does not claim to reproduce the paper's reported scores.

## Environment

This project uses `uv` to manage Python and dependencies. From the repository root:

```bash
uv sync
```

Run project commands through `uv run` so they use the locked environment.

## MicroSS v2 pipeline

The recommended entry point is:

```bash
uv run python scripts/run_micross.py --epochs 20 --device auto
```

See [server instructions](docs/server.md) for the recorded server and input layout.
Raw data defaults to `<project>/data/MicroSS/`, processed data to
`data/processed/micross_v2/`, and temporary indexes to `data/tmp/micross/`.
These paths are ignored by Git. Defaults are resolved from the code location,
so the same layout works after copying the project to the server.
An optional positional path still supports data stored elsewhere, for example
`uv run python scripts/run_micross.py /other/disk/MicroSS --work-dir /large/disk/tmp`.
The pipeline streams metrics, original multipart trace and business ZIPs, builds
minute graphs, trains the two-layer GCN-GRU, and exports prediction/attribution results.
It requires the July run CSVs, metric/metric_full.zip, trace/trace_split.zip and
business/business_split.zip with all their respective volumes.

- Data preprocessing: `scripts/prepare_micross.py`; implementation in `src/data/preprocessing/`.
- Graph extraction: `scripts/build_micross_graph.py`; temporal graph emitted when using `--training-data-dir`.
- Training: `scripts/train_micross.py --data-dir ... --graph .../service_adjacency_1m.npz`.
- `--rca-method masking` (default) ranks score drops after masking each service. `node-head` ranks the auxiliary supervised head.
- `--skip-prepare --skip-graph` on the pipeline reuses completed v2 stages for another training run.

Standalone commands use the same v2 output defaults:

```bash
uv run python scripts/prepare_micross.py
uv run python scripts/build_micross_graph.py --training-data-dir data/processed/micross_v2
uv run python scripts/train_micross.py --graph data/processed/micross_v2/service_adjacency_1m.npz
```

Each resource metric retains its own unit/channel. Network byte counters become
rates using earlier observations. Metric observation flags and business log
counts accompany trace counts, errors, and duration features. Mean/std fitting
uses only training inputs. Fault labels remain next-minute injection starts.
Business logs are inputs; run injection records are labels only.

The shared model consumes [batch, history, services, features] plus time-aligned
weighted adjacency. Current graph weights sum directed completed child-span
calls in each minute into a symmetric graph. Graphs for held-out minutes are
built from telemetry of those same minutes, never future minutes.

Artifacts: feature_schema.json, quality_report.json, service_adjacency_1m.npz,
graph_quality.json, best_model.pt, metrics.json and predictions.jsonl.
Prediction metrics include precision/recall/F1/AP. Root service rankings include
Hit@1/2/3, MRR, MFR and MAR on proxy-positive windows; these are window-level
metrics, not incident-level scores or exact paper-result reproductions.

## Layout

```text
scripts/                 CLI entry points and pipeline orchestration
src/data/preprocessing/  MicroSS telemetry parsing and graphs
src/data/micross.py       Versioned MicroSS adapter
src/data/windows.py       Shared windows and binary node-target panel reader
src/models/              GNN-GRU
src/training/            Training, scaling and evaluation
src/rca/                 Deterministic masking and ranking
tests/                   Unit and synthetic end-to-end regression checks
```

The current repository focuses on the MicroSS workflow. See
[remaining work](docs/architecture.md) and [preprocessing decisions](docs/preprocessing.md).
For the next application-failure experiment, see the proposed
[labeling and evaluation protocol](docs/experiment-protocol.md); it is not yet implemented.

Old family-average data remain loadable by the standalone trainer, but should not
be used for v2 experiments. One-layer model checkpoints require retraining.

Before a server run, check uploaded inputs without scanning the telemetry:

```bash
uv run python scripts/run_micross.py --check-inputs-only
```

See [preprocessing decisions](docs/preprocessing.md) for resource coverage
filtering, archive-check limits and the probability-based RCA score. Regenerate
preprocessing outputs to persist explicit validity metadata and quality counts;
old v2 panels are also filtered using their existing resource observed flags.
