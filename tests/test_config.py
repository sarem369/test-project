"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_engine.config import EngineConfig, deep_merge, load_config, save_config
from ai_engine.config.schema import ModelConfig


def test_engine_config_defaults() -> None:
    """Default config should validate and expose nested sections."""
    config = EngineConfig()
    assert config.model.name == "mlp"
    assert config.training.epochs == 10
    assert config.data.batch_size == 32


def test_model_config_rejects_bad_heads() -> None:
    """d_model must be divisible by n_heads."""
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=100, n_heads=8)


def test_load_and_save_yaml(tmp_path: Path) -> None:
    """Round-trip a config through YAML."""
    source = Path(__file__).resolve().parents[1] / "configs" / "mlp_example.yaml"
    config = load_config(source)
    assert config.experiment_name == "mlp_synthetic_demo"

    out = tmp_path / "saved.yaml"
    save_config(config, out)
    reloaded = load_config(out)
    assert reloaded.model.hidden_dims == config.model.hidden_dims


def test_deep_merge_overrides_nested_keys() -> None:
    """Nested dictionaries should merge rather than replace wholesale."""
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3, "d": 4}})
    assert merged == {"a": {"b": 1, "c": 3, "d": 4}}
