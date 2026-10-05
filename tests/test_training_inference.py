"""Integration tests for training and inference on tiny synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ai_engine.config import EngineConfig
from ai_engine.data import (
    DataPreprocessor,
    PreprocessedBundle,
    create_dataloader,
    dataset_from_bundle,
    train_val_split,
)
from ai_engine.inference import InferenceEngine, export_torchscript
from ai_engine.models import build_model
from ai_engine.training import Trainer


def _tiny_config(tmp_path: Path) -> EngineConfig:
    """Build a minimal config suitable for CPU smoke tests."""
    return EngineConfig.model_validate(
        {
            "experiment_name": "smoke",
            "model": {
                "name": "mlp",
                "input_dim": 8,
                "num_classes": 2,
                "hidden_dims": [16],
                "dropout": 0.0,
            },
            "data": {"batch_size": 16, "normalize": True, "num_workers": 0},
            "optimizer": {"name": "adamw", "lr": 1e-2},
            "scheduler": {"name": "none"},
            "training": {
                "output_dir": str(tmp_path / "run"),
                "epochs": 2,
                "log_every": 1,
                "save_every": 1,
                "device": "cpu",
                "seed": 0,
            },
            "inference": {"batch_size": 16, "device": "cpu"},
        }
    )


def test_train_and_infer_mlp(tmp_path: Path) -> None:
    """Trainer should reduce loss and InferenceEngine should return predictions."""
    config = _tiny_config(tmp_path)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 8)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int64)

    preprocessor = DataPreprocessor(config.data)
    bundle = preprocessor.fit_transform(features, labels)
    x_train, x_val, y_train, y_val = train_val_split(
        bundle.features, bundle.labels, val_ratio=0.25, seed=0
    )

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
    trainer = Trainer(model, config, train_loader, val_loader=val_loader)
    history = trainer.fit()

    assert len(history) == 2
    assert "train_loss" in history[-1]
    assert "val_accuracy" in history[-1]
    assert (tmp_path / "run" / "best.pt").exists() or (tmp_path / "run" / "epoch_2.pt").exists()

    engine = InferenceEngine(model, config)
    preds = engine.predict(x_val)
    assert preds.shape == (len(x_val),)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = engine.predict(x_val[:8], return_proba=True)
    assert proba.shape == (8, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_export_torchscript(tmp_path: Path) -> None:
    """TorchScript export should produce a loadable artifact."""
    model = build_model(
        EngineConfig().model.model_copy(
            update={"name": "mlp", "input_dim": 4, "num_classes": 2, "hidden_dims": [8]}
        )
    )
    example = torch.randn(2, 4)
    path = tmp_path / "model.ts.pt"
    export_torchscript(model, example, path)
    loaded = torch.jit.load(str(path))
    out = loaded(example)
    assert out.shape == (2, 2)
