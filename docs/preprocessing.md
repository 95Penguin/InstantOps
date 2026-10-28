# MicroSS v2 preprocessing decisions

1. Coverage is July 2021; timestamps in output have explicit UTC offsets. Naive
   source timestamps use --source-timezone (default UTC, an unverified assumption).
   Metric epoch timestamps are never shifted. Verify source timezone on the real data.
2. Retain nine selected container resource metrics separately (micross_schema.py).
   CPU cores and ticks are excluded. In/out byte counters become delta/elapsed-second
   rates; reset and invalid values are excluded and counted. Training fits each
   feature's scaler only on training windows. Missing metrics are zero-filled with
   explicit observed flags. Overlapping archive files contribute samples to a minute mean.
3. Completed trace spans provide count, 5xx count/rate and average duration.
   The end_time determines the input bucket so future duration information is not
   pulled into an earlier start-time bucket. Counts refer to spans, not top-level requests.
4. Business CSVs have date-only datetime fields; parse the timestamp and severity
   embedded in message. Features are total/error/warning counts and a log_observed
   indicator. log_observed means at least one parsed log, not verified collector uptime.
   Missing log services/buckets are zero-filled and service coverage appears in quality_report.
5. Run logs provide service/type/start/duration. Deduplicate exact events. Labels at
   input minute t indicate injection starts in [t+60,t+120). They do not label every
   active minute and do not establish an application SLA failure. Injection records
   never extend the feature timeline or enter node features.
6. The trace index resolves (trace_id,parent_id) to service. Each input minute's
   graph counts child spans completed in that minute. Opposite directed counts
   are summed symmetrically for the current GCN. Missing graph minutes have no
   service edges (GCN adds self loops). Legacy static graph uses training-period
   evidence only; temporal graph uses each input minute's own observed evidence.
7. A fixed 70/15/15 chronological split defines distinct label targets. Validation
   history may include earlier training observations, as in rolling forecasting;
   standardization never fits validation/test data. This is not an incident-held-out
   split. Reported RCA scores are per positive window.

Schema v2, resource quality, log/trace coverage and graph coverage are serialized.
Real-data schemas and graph resolution must be validated during the server run.

## Coverage filtering and startup checks

A service-minute is `telemetry_valid=1` only if at least one selected resource
metric is observed. A usable window requires every service to meet this rule in
all history minutes and in the next target minute. Zero values with an observed
flag remain valid; missing trace spans or logs alone do not establish collector
failure or service health. This conservative rule can exclude real incidents
with lost resource monitoring; invalid minutes are reported rather than labeled
normal. Existing v2 panels without this column use their metric observed flags.
Legacy panels without coverage indicators cannot receive this coverage check.
`telemetry_valid` is metadata, never a model feature.

Before telemetry scans, preprocessing checks 7-Zip availability, the combined
metric ZIP directory, original trace/business volume presence, July members,
and parsed July injection records. This is not a full CRC/integrity scan:
truncated nonempty files may still fail during streaming.

```bash
uv run python scripts/run_micross.py --check-inputs-only
```

Masking RCA uses the nonnegative decrease in failure probability. Equal scores
are ordered by service order, with all-service ties counted in metrics.json.
Validation/test loss accumulates weighted cross-entropy numerators and weight
sums across the entire epoch; node BCE uses its element count. Thus evaluation
loss does not depend on how positive samples happen to be grouped into batches.

Empty/whitespace-only business messages are counted and skipped, not treated as
errors or normal requests. Business parsing runs before the expensive trace scan.
Other malformed messages still fail with the member/line and a short preview.
When parsing raises, the extractor is terminated without hiding that original
exception behind a secondary nonzero 7-Zip exit. Canonical local timestamps use
an equivalent ISO parser; legacy accepted formats retain their strptime fallback.
