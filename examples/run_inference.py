#!/usr/bin/env python3
"""Load a checkpoint and run batched inference on synthetic features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ai_engine.config import load_config
from ai_engine.inference import InferenceEngine
from ai_engine.utils import setup_logging


def main() -> None:
    """CLI-style inference demo against a trained checkpoint."""
    parser = argparse.ArgumentParser(description="Run AI Engine inference demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "mlp_example.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/mlp_demo/best.pt"),
        help="Path to checkpoint file produced by training",
    )
    parser.add_argument("--n-samples", type=int, default=64)
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}. Train a model first "
            "(see examples/train_classifier.py)."
        )

    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=config)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(args.n_samples, config.model.input_dim)).astype(np.float32)
    predictions = engine.predict(features)
    logger.info("Predictions: %s", predictions.tolist())


if __name__ == "__main__":
    main()
