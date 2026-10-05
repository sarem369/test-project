"""Multi-layer perceptron classifier / regressor."""

from __future__ import annotations

from torch import nn
import torch

from ai_engine.config.schema import ModelConfig
from ai_engine.models.base import BaseModel
from ai_engine.models.layers import get_activation


class MLPClassifier(BaseModel):
    """Fully-connected classifier for tabular / flattened inputs.

    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
        hidden_dims: Sequence of hidden layer widths.
        dropout: Dropout probability between layers.
        activation: Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Compute classification logits.

        Args:
            features: Float tensor of shape ``(batch, input_dim)``.
            **kwargs: Ignored; accepted for API compatibility.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        del kwargs
        if features.ndim > 2:
            features = features.reshape(features.size(0), -1)
        return self.network(features)

    @classmethod
    def from_config(cls, config: ModelConfig) -> MLPClassifier:
        """Construct an :class:`MLPClassifier` from a :class:`ModelConfig`."""
        return cls(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dims=list(config.hidden_dims),
            dropout=config.dropout,
            activation=config.activation,
        )
