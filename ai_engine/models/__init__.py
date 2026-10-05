"""Model architectures and registry."""

from ai_engine.models.base import BaseModel
from ai_engine.models.layers import (
    FeedForward,
    MultiHeadSelfAttention,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
    get_activation,
)
from ai_engine.models.mlp import MLPClassifier
from ai_engine.models.registry import build_model, list_models, register_model
from ai_engine.models.transformer import TransformerClassifier

__all__ = [
    "BaseModel",
    "FeedForward",
    "MLPClassifier",
    "MultiHeadSelfAttention",
    "SinusoidalPositionalEncoding",
    "TransformerClassifier",
    "TransformerEncoderBlock",
    "build_model",
    "get_activation",
    "list_models",
    "register_model",
]
