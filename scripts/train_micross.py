#!/usr/bin/env python3
"""Train/evaluate the GNN-GRU on the available GAIA MicroSS proxy labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import SERVICES  # noqa: E402
from data.micross import FEATURE_COLUMNS, MicrossWindowDataset  # noqa: E402
from data.splits import chronological_split  # noqa: E402
from models.gnn_gru import InstantOpsGNNGRU  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/processed/micross"
    )
    parser.add_argument("--adjacency-npy", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/micross")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    args = parser.parse_args()
    if args.history < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("history, epochs, and batch-size must be positive")
    if args.learning_rate <= 0 or args.node_loss_weight < 0:
        parser.error("learning-rate must be positive and node-loss-weight nonnegative")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch build")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fit_feature_scaler(
    dataset: MicrossWindowDataset, train_ends: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardization using unique timestamps covered by train windows only."""
    train_times: set[int] = set()
    for end in train_ends:
        train_times.update(range(end - dataset.history + 1, end + 1))
    matrix = dataset.features[sorted(train_times)].reshape(-1, len(FEATURE_COLUMNS))
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    dataset.features = ((dataset.features - mean) / scale).astype(np.float32)
    return mean, scale


def class_pos_weight(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    positives = labels.sum().item()
    negatives = labels.numel() - positives
    weight = negatives / positives if positives else 1.0
    return torch.tensor(weight, dtype=torch.float32, device=device)


def run_epoch(
    model: InstantOpsGNNGRU,
    loader: DataLoader,
    device: torch.device,
    failure_loss_fn: nn.Module,
    node_loss_fn: nn.Module,
    node_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    collected: dict[str, list[np.ndarray]] = {
        "labels": [], "probabilities": [], "node_targets": [], "node_scores": []
    }
    for batch in loader:
        features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        labels = batch["failure_target"].to(device)
        node_targets = batch["node_targets"].to(device)
        with torch.set_grad_enabled(training):
            output = model(features, adjacency)
            failure_loss = failure_loss_fn(output["failure_logits"], labels)
            node_loss = node_loss_fn(output["node_logits"], node_targets)
            loss = failure_loss + node_loss_weight * node_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        size = labels.size(0)
        total_loss += loss.item() * size
        total_items += size
        probabilities = output["failure_logits"].softmax(dim=-1)[:, 1]
        collected["labels"].append(labels.detach().cpu().numpy())
        collected["probabilities"].append(probabilities.detach().cpu().numpy())
        collected["node_targets"].append(node_targets.detach().cpu().numpy())
        collected["node_scores"].append(output["node_scores"].detach().cpu().numpy())
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    return total_loss / max(total_items, 1), arrays


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    precision_at_k = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision_at_k[ranked].mean())


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | None]:
    predictions = probabilities >= 0.5
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "accuracy": float((predictions == labels).mean()),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": f1,
        "average_precision": average_precision(labels, probabilities),
    }


def rca_metrics(node_targets: np.ndarray, node_scores: np.ndarray) -> dict[str, float | int | None]:
    positive_rows = np.flatnonzero(node_targets.sum(axis=1) > 0)
    if not len(positive_rows):
        return {"positive_windows": 0, "hit_at_1": None, "hit_at_3": None, "mrr": None}
    hits_1: list[float] = []
    hits_3: list[float] = []
    reciprocal_ranks: list[float] = []
    for row in positive_rows:
        order = np.argsort(-node_scores[row], kind="stable")
        truth = node_targets[row].astype(bool)
        ranks = np.flatnonzero(truth[order])
        first_rank = int(ranks[0])
        hits_1.append(float(first_rank < 1))
        hits_3.append(float(first_rank < 3))
        reciprocal_ranks.append(1.0 / (first_rank + 1))
    return {
        "positive_windows": int(len(positive_rows)),
        "hit_at_1": float(np.mean(hits_1)),
        "hit_at_3": float(np.mean(hits_3)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    feature_csv = args.data_dir / "service_features_1m.csv"
    label_csv = args.data_dir / "next_minute_labels.csv"
    if args.adjacency_npy:
        adjacency = np.load(args.adjacency_npy)
        graph_source = str(args.adjacency_npy)
    else:
        adjacency = np.eye(len(SERVICES), dtype=np.float32)
        graph_source = "identity matrix (no service edges); temporal-only baseline"

    dataset = MicrossWindowDataset(
        feature_csv,
        label_csv,
        adjacency=adjacency,
        history=args.history,
    )
    train_ends, val_ends, test_ends = chronological_split(dataset.sample_indices)
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
        [int(dataset.node_targets[end].max()) for end in train_ends], dtype=torch.float32
    )
    train_node_labels = torch.from_numpy(dataset.node_targets[train_ends])
    failure_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, class_pos_weight(train_labels, torch.device("cpu")).item()],
            dtype=torch.float32,
            device=device,
        )
    )
    node_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=class_pos_weight(train_node_labels, device)
    )
    model = InstantOpsGNNGRU(feature_dim=len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(
            model, loaders[0], device, failure_loss_fn, node_loss_fn,
            args.node_loss_weight, optimizer,
        )
        val_loss, _ = run_epoch(
            model, loaders[1], device, failure_loss_fn, node_loss_fn,
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
                    "services": list(SERVICES),
                    "feature_columns": list(FEATURE_COLUMNS),
                    "history": args.history,
                    "graph_source": graph_source,
                    "label_definition": "any GAIA injection start in the next minute",
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_output = run_epoch(
        model, loaders[2], device, failure_loss_fn, node_loss_fn, args.node_loss_weight
    )
    metrics = {
        "test_loss": test_loss,
        "failure_prediction": classification_metrics(
            test_output["labels"], test_output["probabilities"]
        ),
        "root_cause_proxy": rca_metrics(
            test_output["node_targets"], test_output["node_scores"]
        ),
        "graph_source": graph_source,
        "label_definition": (
            "next-minute GAIA fault-injection-start proxy; not SLA failure ground truth"
        ),
        "split_sizes": {
            "train": len(train_ends), "validation": len(val_ends), "test": len(test_ends)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
