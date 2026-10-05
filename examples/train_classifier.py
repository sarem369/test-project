#!/usr/bin/env python3
"""End-to-end example: preprocess synthetic data, train an MLP, run inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_engine.config import EngineConfig, load_config
from ai_engine.data import (
    DataPreprocessor,
    create_dataloader,
    dataset_from_bundle,
    train_val_split,
)
from ai_engine.inference import InferenceEngine
from ai_engine.models import build_model
from ai_engine.training import Trainer
from ai_engine.utils import setup_logging


def make_synthetic_classification(
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linearly separable-ish classification dataset.

    Args:
        n_samples: Number of samples.
        n_features: Feature dimensionality.
        n_classes: Number of classes.
        seed: RNG seed.

    Returns:
        ``(features, labels)`` arrays.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    weights = rng.normal(size=(n_features, n_classes)).astype(np.float32)
    logits = features @ weights + rng.normal(scale=0.5, size=(n_samples, n_classes))
    labels = logits.argmax(axis=1).astype(np.int64)
    return features, labels


def main() -> None:
    """Train and evaluate a demo MLP classifier."""
    logger = setup_logging()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mlp_example.yaml"
    config: EngineConfig = load_config(config_path)

    features, labels = make_synthetic_classification(
        n_features=config.model.input_dim,
        n_classes=config.model.num_classes,
        seed=config.training.seed,
    )

    preprocessor = DataPreprocessor(config.data)
    bundle = preprocessor.fit_transform(features, labels)
    x_train, x_val, y_train, y_val = train_val_split(
        bundle.features,
        bundle.labels,
        val_ratio=0.2,
        seed=config.training.seed,
    )

    from ai_engine.data import PreprocessedBundle

    train_loader = create_dataloader(
        dataset_from_bundle(PreprocessedBundle(x_train, y_train)),
        config.data,
        shuffle=True,
    )
    val_loader = create_dataloader(
        dataset_from_bundle(PreprocessedBundle(x_val, y_val)),
        config.data,
        shuffle=False,
    )

    model = build_model(config.model)
    logger.info("Model parameters: %s", model.count_parameters())

    trainer = Trainer(model, config, train_loader, val_loader=val_loader)
    history = trainer.fit()
    logger.info("History: %s", history)

    engine = InferenceEngine(model, config)
    proba = engine.predict(x_val[:16], return_proba=True)
    preds = proba.argmax(axis=1)
    logger.info("Sample predictions: %s", preds.tolist())
    logger.info("Sample probabilities shape: %s", proba.shape)


if __name__ == "__main__":
    main()
