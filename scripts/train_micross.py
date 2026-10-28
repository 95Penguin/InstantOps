#!/usr/bin/env python3
"""Train the shared GNN-GRU pipeline on the MicroSS adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from constants import SERVICES
from data.micross import MicrossWindowDataset
from data.paths import MICROSS_OUTPUT_DIR, MICROSS_PROCESSED_DIR
from training.experiment import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=MICROSS_PROCESSED_DIR)
    parser.add_argument("--adjacency-npy", "--graph", dest="adjacency_npy", type=Path)
    parser.add_argument("--rca-method", choices=("masking", "node-head"), default="masking")
    parser.add_argument("--output-dir", type=Path, default=MICROSS_OUTPUT_DIR)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.adjacency_npy and args.adjacency_npy.suffix not in {".npy", ".npz"}:
        parser.error("graph must be a .npy static matrix or .npz temporal graph")
    if args.history < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("history, epochs, and batch-size must be positive")
    if args.learning_rate <= 0 or args.node_loss_weight < 0:
        parser.error("learning-rate must be positive and node-loss-weight nonnegative")
    return args


def main() -> int:
    args = parse_args()
    adjacency = (
        np.load(args.adjacency_npy)
        if args.adjacency_npy and args.adjacency_npy.suffix == ".npy"
        else np.eye(len(SERVICES), dtype=np.float32)
    )
    source = (
        str(args.adjacency_npy)
        if args.adjacency_npy
        else "identity matrix (no service edges); temporal-only baseline"
    )
    dataset = MicrossWindowDataset(
        args.data_dir / "service_features_1m.csv",
        args.data_dir / "next_minute_labels.csv",
        adjacency=adjacency,
        history=args.history,
        graph_path=args.adjacency_npy
        if args.adjacency_npy and args.adjacency_npy.suffix == ".npz"
        else None,
    )
    run_training(dataset, args, source, "next-minute GAIA injection-start proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
