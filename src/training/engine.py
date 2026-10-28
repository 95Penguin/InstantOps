from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.windows import GraphWindowDataset
from models.gnn_gru import InstantOpsGNNGRU


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
    dataset: GraphWindowDataset, train_ends: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardization using unique timestamps covered by train windows only."""
    train_times: set[int] = set()
    for end in train_ends:
        train_times.update(range(end - dataset.history + 1, end + 1))
    matrix = dataset.features[sorted(train_times)].reshape(-1, len(dataset.feature_columns))
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
    if failure_loss_fn.reduction != "mean" or node_loss_fn.reduction != "mean":
        raise ValueError("epoch aggregation requires mean-reduced losses")
    training = optimizer is not None
    model.train(training)
    failure_numerator = failure_denominator = 0.0
    node_numerator = node_denominator = 0.0
    collected: dict[str, list[np.ndarray]] = {
        "labels": [],
        "probabilities": [],
        "node_targets": [],
        "node_scores": [],
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
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss; check features and class weights")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        size = labels.size(0)
        # CrossEntropyLoss(mean) divides by target-class weight, not batch size.
        weight = getattr(failure_loss_fn, "weight", None)
        denominator = float(weight[labels].sum().item()) if weight is not None else size
        failure_numerator += failure_loss.item() * denominator
        failure_denominator += denominator
        node_numerator += node_loss.item() * node_targets.numel()
        node_denominator += node_targets.numel()
        probabilities = output["failure_logits"].softmax(dim=-1)[:, 1]
        collected["labels"].append(labels.detach().cpu().numpy())
        collected["probabilities"].append(probabilities.detach().cpu().numpy())
        collected["node_targets"].append(node_targets.detach().cpu().numpy())
        collected["node_scores"].append(output["node_scores"].detach().cpu().numpy())
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    epoch_loss = (
        failure_numerator / max(failure_denominator, 1e-12)
        + node_loss_weight * node_numerator / max(node_denominator, 1)
    )
    return epoch_loss, arrays
