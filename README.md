# InstantOps reproduction

This repository is a workflow-level reproduction of **InstantOps: A Joint Approach to System Failure Prediction and Root Cause Identification in Microservices Cloud-Native Applications** (ICPE 2024). It is an independent implementation; it does not claim to reproduce the paper's reported scores.

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
python3 scripts/prepare_micross.py ~/project/GAIA-DataSet/MicroSS
```

If the dataset is at the GAIA checkout root, pass `~/project/GAIA-DataSet` instead. Optional paths and bucket size can be set explicitly:

```bash
python3 scripts/prepare_micross.py ~/project/GAIA-DataSet/MicroSS \
  --metric-archive ~/project/GAIA-DataSet/MicroSS/metric/metric_full.zip \
  --trace-archive ~/project/GAIA-DataSet/MicroSS/trace/trace_split.zip \
  --run-dir ~/project/GAIA-DataSet/MicroSS/run \
  --window-seconds 60 \
  --output data/processed/micross
```

The generated files are:

- `data/processed/micross/service_features_1m.csv`: one row per service and UTC minute, with coarse metric-family means, metric-series coverage counts, and trace span/error/duration aggregates.
- `data/processed/micross/next_minute_labels.csv`: one row per service and input minute; the label is positive when a GAIA fault injection starts in the following minute.

The label is a **GAIA injection-start proxy**, not the application-level SLA failure label used by InstantOps. Trace 5xx counts here count spans and must not be interpreted as failed top-level requests. Feature values are grouped into coarse families because GAIA exposes many individually named metric series; this is an initial reproducible adapter, not the paper's undocumented per-metric normalization. See [the preprocessing decisions](docs/preprocessing.md).

## Project layout

```text
docs/           Paper notes and preprocessing decisions
scripts/        Runnable data-preparation entry points
src/instantops/ Reusable model and RCA implementation
```

The input dataset stays outside this repository. Generated outputs go under `data/processed/` (ignored by Git) and the directory is created when preprocessing runs.

## Planned milestones

1. Inspect the selected dataset and identify its modalities, timestamps, service identifiers, and fault labels. **Done for GAIA MicroSS.**
2. Convert available data into aligned fixed-size time windows. **Initial 1-minute service panel and next-minute injection labels implemented.**
3. Build a service interaction graph from traces and attach per-service features.
4. Implement and train a GNN-GRU failure prediction baseline.
5. Rank candidate root-cause services and evaluate prediction/localization separately.
6. Add and evaluate one focused improvement.

TrainTicket and the paper's private QoTD collection need separate adapters. QoTD reproduction requires deployment, fault injection, and telemetry collection.
