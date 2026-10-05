"""Command-line entry points for training and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ai_engine.config import load_config
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


def _build_parser_train() -> argparse.ArgumentParser:
    """Create the training CLI parser."""
    parser = argparse.ArgumentParser(description="Train a model with AI Engine")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML/JSON config")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    return parser


def _build_parser_infer() -> argparse.ArgumentParser:
    """Create the inference CLI parser."""
    parser = argparse.ArgumentParser(description="Run inference with AI Engine")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML/JSON config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--input", type=Path, required=True, help="Input .npy feature file")
    parser.add_argument("--output", type=Path, default=Path("predictions.npy"), help="Output path")
    parser.add_argument("--proba", action="store_true", help="Write probabilities instead of labels")
    return parser


def train_main(argv: list[str] | None = None) -> int:
    """CLI entry point for training.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = _build_parser_train().parse_args(argv)
    logger = setup_logging()
    config = load_config(args.config)
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir

    preprocessor = DataPreprocessor(config.data)
    if config.data.train_path is None:
        raise SystemExit("data.train_path must be set in the config")

    train_bundle = preprocessor.load_from_config("train")
    if config.data.val_path is not None:
        val_bundle = preprocessor.load_from_config("val")
    else:
        x_train, x_val, y_train, y_val = train_val_split(
            train_bundle.features,
            train_bundle.labels,
            seed=config.training.seed,
        )
        train_bundle.features, train_bundle.labels = x_train, y_train
        from ai_engine.data.preprocessing import PreprocessedBundle

        val_bundle = PreprocessedBundle(features=x_val, labels=y_val)

    # Align model input dim for tabular MLP when unspecified / mismatched.
    if config.model.name == "mlp":
        config.model.input_dim = int(train_bundle.features.shape[-1])
        if train_bundle.labels is not None:
            config.model.num_classes = int(len(np.unique(train_bundle.labels)))

    train_loader = create_dataloader(dataset_from_bundle(train_bundle), config.data, shuffle=True)
    val_loader = create_dataloader(dataset_from_bundle(val_bundle), config.data, shuffle=False)

    model = build_model(config.model)
    trainer = Trainer(model, config, train_loader, val_loader=val_loader)
    history = trainer.fit()
    logger.info("Training complete. Final metrics: %s", history[-1] if history else {})
    return 0


def infer_main(argv: list[str] | None = None) -> int:
    """CLI entry point for offline inference.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    args = _build_parser_infer().parse_args(argv)
    logger = setup_logging()
    config = load_config(args.config)
    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=config)
    features = np.load(args.input)
    predictions = engine.predict(features, return_proba=args.proba)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, predictions)
    logger.info("Wrote predictions to %s (shape=%s)", args.output, predictions.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
