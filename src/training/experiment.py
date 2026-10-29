"""Dataset-independent experiment orchestration for aligned graph windows."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.splits import chronological_split, purged_event_split
from data.windows import GraphWindowDataset
from models.gnn_gru import InstantOpsGNNGRU
from rca.ranking import masking_importance
from training.engine import (
    choose_device,
    class_pos_weight,
    fit_feature_scaler,
    run_epoch,
    seed_everything,
)
from training.metrics import classification_metrics, rca_metrics


def run_training(
    dataset: GraphWindowDataset, args: argparse.Namespace, graph_source: str, label_definition: str | dict
) -> dict:
    """Train one dataset with train-only normalization and held-out evaluation.

    Adapters must provide binary failure/node labels with documented semantics.
    Different datasets train separate weights and never share fitted scalers.
    """
    if getattr(dataset, "failure_labels_available", True) is False:
        raise ValueError("application failure labels are unavailable for this dataset")

    # Scaling replaces the feature array on a shallow copy, so repeated runs
    # on the same adapter still fit against the original raw feature values.
    dataset = copy.copy(dataset)
    seed_everything(args.seed)
    device = choose_device(args.device)
    split_report = {"policy": "chronological"}
    if isinstance(label_definition, dict) and label_definition.get("split_policy") == "purged_events":
        if dataset.adjacency.ndim != 3:
            raise ValueError("event protocol requires the temporal graph; cached static graph may use later evidence")
        (train_ends, val_ends, test_ends), split_report = purged_event_split(dataset, label_definition)
    else:
        train_ends, val_ends, test_ends = chronological_split(dataset.sample_indices)
    baseline = None
    if isinstance(label_definition, dict):
        entry = [dataset.services.index(s) for s in label_definition["entry_services"]]
        count_col = dataset.feature_columns.index("trace_span_count")
        error_col = dataset.feature_columns.index("trace_5xx_count")
        current = dataset.features[test_ends][:, entry, :]
        counts = current[:, :, count_col].sum(axis=1)
        errors = current[:, :, error_col].sum(axis=1)
        observed = counts >= label_definition["min_entry_spans"]
        rates = errors / np.maximum(counts, 1)
        flags = rates >= label_definition["error_rate_threshold"]
        baseline = {
            "definition": "current-minute entry error threshold predicts next-minute failure; no future confirmation",
            "evaluated_windows": int(observed.sum()),
            "metrics": classification_metrics(dataset.failure_targets[test_ends][observed], flags[observed].astype(float))
                if observed.any() else None,
        }
    mean, scale = fit_feature_scaler(dataset, train_ends)
    train_index = {end: index for index, end in enumerate(dataset.sample_indices)}
    subsets = [
        Subset(dataset, [train_index[end] for end in ends])
        for ends in (train_ends, val_ends, test_ends)
    ]
    loaders = [
        DataLoader(subset, batch_size=args.batch_size, shuffle=(index == 0), num_workers=0)
        for index, subset in enumerate(subsets)
    ]
    train_labels = torch.tensor(
        [int(dataset.failure_targets[end]) for end in train_ends], dtype=torch.float32
    )
    if train_labels.unique().numel() < 2:
        raise ValueError("training split needs both normal and fault windows")
    train_node_labels = torch.from_numpy(dataset.node_targets[train_ends])
    known_roots = (dataset.failure_targets[train_ends] == 0) | (dataset.node_targets[train_ends].max(axis=1) > 0)
    train_node_labels = train_node_labels[known_roots]
    failure_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, class_pos_weight(train_labels, torch.device("cpu")).item()],
            dtype=torch.float32,
            device=device,
        )
    )
    node_loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(train_node_labels, device))
    model = InstantOpsGNNGRU(feature_dim=len(dataset.feature_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(
            model,
            loaders[0],
            device,
            failure_loss_fn,
            node_loss_fn,
            args.node_loss_weight,
            optimizer,
        )
        val_loss, _ = run_epoch(
            model,
            loaders[1],
            device,
            failure_loss_fn,
            node_loss_fn,
            args.node_loss_weight,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": mean,
                    "feature_scale": scale,
                    "services": list(dataset.services),
                    "feature_columns": list(dataset.feature_columns),
                    "history": args.history,
                    "graph_source": graph_source,
                    "adjacency": dataset.adjacency.cpu(),
                    "model_config": {"feature_dim": len(dataset.feature_columns), "gcn_layers": 2},
                    "seed": args.seed,
                    "label_definition": label_definition,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_output = run_epoch(
        model, loaders[2], device, failure_loss_fn, node_loss_fn, args.node_loss_weight
    )
    method = getattr(args, "rca_method", "masking")
    if method == "masking":
        scores = []
        model.eval()
        for batch in loaders[2]:
            scores.append(
                masking_importance(
                    model, batch["node_features"].to(device), batch["adjacency"].to(device)
                )
                .cpu()
                .numpy()
            )
        test_output["node_scores"] = np.concatenate(scores)
    predictions = []
    test_times = [dataset.timestamps[end] for end in test_ends]
    test_event_ids = [dataset.event_ids[end] for end in test_ends]
    for index, (timestamp, event_id) in enumerate(zip(test_times, test_event_ids)):
        scores = test_output["node_scores"][index]
        order = sorted(range(len(dataset.services)), key=lambda node: (-float(scores[node]), node))
        predictions.append(
            {
                "input_minute_utc": timestamp,
                "failure_probability": float(test_output["probabilities"][index]),
                "failure_target": int(test_output["labels"][index]),
                "event_id": event_id,
                "rca_method": method,
                "true_services": [
                    service
                    for service, label in zip(dataset.services, test_output["node_targets"][index])
                    if label
                ],
                "ranking": [
                    {"service": dataset.services[node], "score": float(scores[node])}
                    for node in order
                ],
            }
        )
    event_rows: dict[str, list[int]] = {}
    for row, event_id in enumerate(test_event_ids):
        if event_id and test_output["node_targets"][row].sum() > 0:
            event_rows.setdefault(event_id, []).append(row)
    event_targets = []
    event_scores = []
    for event_id, rows in event_rows.items():
        targets = test_output["node_targets"][rows]
        if not np.all(targets == targets[0]):
            raise ValueError(f"root labels change within failure event {event_id}")
        event_targets.append(targets[0])
        event_scores.append(test_output["node_scores"][rows].mean(axis=0))
    event_rca = rca_metrics(
        np.stack(event_targets) if event_targets else np.empty((0, len(dataset.services))),
        np.stack(event_scores) if event_scores else np.empty((0, len(dataset.services))),
    )
    event_rca["positive_events"] = event_rca.pop("positive_windows")
    event_rca["aggregation"] = "mean scores over available attributed windows in each event; retrospective"
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    metrics = {
        "rca_method": method,
        "split_protocol": split_report,
        "current_error_baseline": baseline,
        "root_label_coverage": {
            "test_failure_windows": int(test_output["labels"].sum()),
            "test_attributed_failure_windows": int((test_output["node_targets"].max(axis=1) > 0).sum()),
        },
        "rca_score_definition": (
            "nonnegative target-probability drop" if method == "masking" else "node sigmoid"
        ),
        "coverage": {
            "invalid_panel_minutes": dataset.invalid_minutes,
            "usable_windows": len(dataset),
        },
        "rca_ties": {
            "all_services_equal_windows": int((
                test_output["node_scores"].max(axis=1)
                == test_output["node_scores"].min(axis=1)
            ).sum()),
            "tie_break": "service order; Hit/MRR may depend on this order",
        },
        "run_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "split_positive_samples": {
            name: int(dataset.failure_targets[ends].sum())
            for name, ends in zip(
                ("train", "validation", "test"), (train_ends, val_ends, test_ends)
            )
        },
        "test_loss": test_loss,
        "failure_prediction": classification_metrics(
            test_output["labels"], test_output["probabilities"]
        ),
        "root_cause_proxy": rca_metrics(test_output["node_targets"], test_output["node_scores"]),
        "root_cause_event_proxy": event_rca,
        "graph_source": graph_source,
        "label_definition": label_definition,
        "split_sizes": {
            "train": len(train_ends),
            "validation": len(val_ends),
            "test": len(test_ends),
        },
        "training_history": history,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    if graph_source.startswith("identity matrix"):
        print("WARNING: no call graph supplied; this run is a temporal-only baseline.")
    return metrics
