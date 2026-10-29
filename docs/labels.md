# Local business-failure labels

## Event-root labels v2

`entry_failure_v2.csv` preserves the v1 failure definition and changes only
root attribution: exactly one service with an injection active at event onset
provides a candidate event root, repeated throughout the event. Injections
longer than 3600 seconds are excluded as unverified records. Their exclusion
does not prove that another simultaneous injection caused the failure.
Zero or multiple candidate services leave the event root unknown. These are
temporal-association proxy labels, not independently verified causal truth.

The July audit found 15 attributable events out of 160 (9 train, 0 validation,
6 test by onset boundaries, before window eligibility/purging). Some events
share an injection, so they are not necessarily independent interventions.
These counts support exploratory RCA only. Audit excluded records and expand
verified interventions before making strong RCA performance claims.

The trainer exports event IDs and `root_cause_event_proxy`, using mean scores
over each event's available windows. This is retrospective event evaluation;
it must not be described as localization at the first alarm. Historical v1
artifacts used minute-overlap attribution; the current label generator uses v2.

```bash
.venv/bin/python scripts/train_micross.py --data-dir data/processed/micross_mps_smoke_01 --graph data/processed/micross_mps_smoke_01/service_adjacency_1m.npz --labels data/processed/micross_mps_smoke_01/entry_failure_v2.csv --output-dir artifacts/entry_failure_mps_v2 --epochs 5 --device mps
```

## Entry failure v1 (local experiment)

Current candidate: pooled entry 5xx rate >= 50%, at least 20 completed requests
per minute, sustained for >=2 consecutive minutes. This is an experimental
definition of severe login-service degradation, not a verified production SLO.
The event starts at the first qualifying minute; online confirmation requires
the second minute to complete. Training still forecasts **next-minute state**,
including ongoing events; it is not an onset-only early-warning experiment.

```bash
.venv/bin/python scripts/label_micross.py --data-dir data/processed/micross_mps_smoke_01 --output data/processed/micross_mps_smoke_01/entry_failure_v1.csv --error-rate-threshold 0.5 --min-entry-spans 20 --consecutive-minutes 2 --max-injection-seconds 3600
.venv/bin/python scripts/train_micross.py --data-dir data/processed/micross_mps_smoke_01 --graph data/processed/micross_mps_smoke_01/service_adjacency_1m.npz --labels data/processed/micross_mps_smoke_01/entry_failure_v1.csv --output-dir artifacts/entry_failure_mps_v1 --epochs 5 --device mps
```

Outputs include `.events.csv` and `.schema.json`, recording event boundaries,
coverage of onset/recovery, thresholds and excluded injection records. A
60-minute injection-duration audit cap excludes the three implausibly long
records; it does not truncate or repair them. Failure windows overlapped by
such records have unknown roots and no node supervision. Other active
injections remain only candidate causes, not proven ground truth.

Two-minute labels automatically use a chronological event-purged split.
Entire events crossing either boundary are excluded, as are windows whose
history or label-confirmation lookahead crosses partitions. Temporal graphs
are required. `metrics.json` records purge counts and root-label coverage.
The current-minute error-threshold baseline uses only currently available
evidence; a high next-state score alone does not establish advance warning.

For a compact methods sentence: “We define a severe business-degradation event
as at least two consecutive one-minute windows with >=20 observed entry
requests and an entry-request error rate >=50%.” Keep the measurement and
event-split details in the experiment settings or appendix.

## Earlier single-minute exploratory labels

The model is unchanged. `scripts/label_micross.py` reads cached features and
creates an independent label CSV plus `.schema.json`. Pass the CSV with
`--labels` to the trainer or pipeline; omitting it preserves the old baseline.

The target is the **next minute's failure state**, including ongoing failures,
not only transitions from healthy to failed. It uses the pooled completed-span
5xx rate of configurable entry services (webservice1/2 by default). These spans
are an entry-service proxy; top-level request semantics remain unverified.
The threshold is required explicitly. 0.999 is an exploratory severe-failure
setting, not an established SLO. Do not select it using test performance or to
match the number of injections. CPU/memory remain predictor features.

No/insufficient entry spans means unknown and excludes that target. An active
injection overlapping a failed minute provides a candidate root label; overlap
does not establish causality. Failure without a known root still trains the
failure head but is masked from root supervision. RCA metrics cover only
attributed failure windows. Existing chronological splits remain window-level.

Local commands (reuse the completed preprocessing and temporal graph):

```bash
.venv/bin/python scripts/label_micross.py --data-dir data/processed/micross_mps_smoke_01 --output data/processed/micross_mps_smoke_01/sla_labels_0999.csv --error-rate-threshold 0.999
.venv/bin/python scripts/train_micross.py --data-dir data/processed/micross_mps_smoke_01 --graph data/processed/micross_mps_smoke_01/service_adjacency_1m.npz --labels data/processed/micross_mps_smoke_01/sla_labels_0999.csv --output-dir artifacts/micross_sla_mps_01 --epochs 1 --device mps
```

The label command refuses to overwrite an existing output. Historical model
artifacts remain valid for their original labels and should not be compared
directly against new-label metrics.
