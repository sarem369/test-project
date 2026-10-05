"""Tests for model construction and forward shapes."""

from __future__ import annotations

import torch

from ai_engine.config import ModelConfig
from ai_engine.models import MLPClassifier, TransformerClassifier, build_model, list_models


def test_registry_contains_builtins() -> None:
    """Built-in models should be registered by name."""
    names = list_models()
    assert "mlp" in names
    assert "transformer_classifier" in names


def test_mlp_forward_shape() -> None:
    """MLP logits should match (batch, num_classes)."""
    model = MLPClassifier(input_dim=16, num_classes=5, hidden_dims=[32, 16])
    x = torch.randn(8, 16)
    logits = model(x)
    assert logits.shape == (8, 5)
    assert model.predict(x).shape == (8,)
    assert model.predict_proba(x).shape == (8, 5)


def test_transformer_forward_shape() -> None:
    """Transformer classifier should pool sequences to class logits."""
    model = TransformerClassifier(
        vocab_size=100,
        num_classes=4,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dim_feedforward=64,
        max_seq_len=32,
    )
    tokens = torch.randint(1, 100, (4, 16))
    tokens[:, -3:] = 0  # padding
    logits = model(tokens)
    assert logits.shape == (4, 4)


def test_build_model_from_config() -> None:
    """Registry factory should construct models from ModelConfig."""
    model = build_model(ModelConfig(name="mlp", input_dim=10, num_classes=2, hidden_dims=[16]))
    assert isinstance(model, MLPClassifier)
    assert model.count_parameters() > 0
