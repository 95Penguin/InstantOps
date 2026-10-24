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
