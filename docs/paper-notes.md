# InstantOps paper notes

Paper: Raphael Rouf et al., “InstantOps: A Joint Approach to System Failure Prediction and Root Cause Identification in Microservices Cloud-Native Applications,” ICPE 2024. DOI: <https://doi.org/10.1145/3629526.3645047>.

## Workflow described in the paper

1. Split telemetry into aligned time windows.
2. Use traces to construct a time-varying service-call graph.
3. Attach log, metric, and (where available) Kubernetes event features to service nodes.
4. Apply a GNN to model spatial/service dependencies and a GRU to model temporal evolution.
5. Predict imminent system failure and use node importance/scores to rank likely root-cause services.

## Datasets described

- **TrainTicket**: 41 microservices; the paper uses public data with injected application-bug, CPU-exhaustion, and network-congestion faults, focusing on faults injected into microservices.
- **MicroSS / GAIA**: 10 microservices, MySQL and Redis, five hosts; the dataset includes fault-injection records. The available modalities and field schema must be verified from the downloaded release.
- **QoTD**: eight services deployed by the authors on IBM OpenShift. They injected CPU, memory, and DNS faults with Chaos Mesh and collected telemetry using Instana, LogDNA, and a custom Kubernetes-event collector. Treat this as a separate, optional data-collection effort.

## Metrics to implement

- Failure prediction: precision, recall, F1, and accuracy (report class balance too).
- Root-cause localization: A@1/A@2/A@3 and rank-based metrics (MFR/MAR, after confirming the paper's exact definitions for the chosen dataset).

## Reproduction caveat

The paper uses different data sources and modalities across datasets. Do not assume GAIA, TrainTicket, and QoTD have identical fields or can be processed by one hard-coded parser. Keep dataset-specific parsing separate from the shared windowing/model code.
