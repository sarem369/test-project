"""Model registry for name-based architecture construction."""

from __future__ import annotations

from typing import Callable, TypeVar

from torch import nn

from ai_engine.config.schema import ModelConfig
from ai_engine.models.base import BaseModel
from ai_engine.models.mlp import MLPClassifier
from ai_engine.models.transformer import TransformerClassifier

ModelFactory = Callable[[ModelConfig], BaseModel]
T = TypeVar("T", bound=BaseModel)

_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator to register a model factory under ``name``.

    Args:
        name: Unique registry key (case-insensitive).

    Returns:
        Decorator that registers and returns the factory unchanged.
    """

    def decorator(factory: ModelFactory) -> ModelFactory:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Model {name!r} is already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def build_model(config: ModelConfig) -> BaseModel:
    """Instantiate a registered model from configuration.

    Args:
        config: Model configuration including the registered ``name``.

    Returns:
        Constructed :class:`BaseModel` instance.

    Raises:
        KeyError: If ``config.name`` is not registered.
    """
    key = config.name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown model {config.name!r}. Registered: {known}")
    model = _REGISTRY[key](config)
    if config.pretrained_path is not None:
        from ai_engine.utils.checkpoint import load_checkpoint

        load_checkpoint(config.pretrained_path, model=model)
    return model


def list_models() -> list[str]:
    """Return sorted registered model names."""
    return sorted(_REGISTRY)


@register_model("mlp")
def _build_mlp(config: ModelConfig) -> MLPClassifier:
    """Factory for :class:`MLPClassifier`."""
    return MLPClassifier.from_config(config)


@register_model("transformer_classifier")
def _build_transformer(config: ModelConfig) -> TransformerClassifier:
    """Factory for :class:`TransformerClassifier`."""
    return TransformerClassifier.from_config(config)


def as_module(model: nn.Module) -> nn.Module:
    """Identity helper for type-narrowing in generic pipelines."""
    return model
