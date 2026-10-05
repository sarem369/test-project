"""Base model interfaces for the AI Engine framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class shared by all framework models.

    Subclasses must implement :meth:`forward`. Optional helpers provide a
    consistent prediction API used by the trainer and inference engine.
    """

    @abstractmethod
    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute model logits / outputs.

        Args:
            features: Input tensor (shape depends on the concrete model).
            **kwargs: Optional model-specific keyword arguments (e.g. masks).

        Returns:
            Output tensor (typically logits of shape ``(batch, num_classes)``).
        """

    @torch.inference_mode()
    def predict(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return class indices for classification models.

        Args:
            features: Input tensor.
            **kwargs: Forward keyword arguments.

        Returns:
            Predicted class indices of shape ``(batch,)``.
        """
        logits = self.forward(features, **kwargs)
        if logits.ndim == 1:
            return (logits > 0).long()
        return logits.argmax(dim=-1)

    @torch.inference_mode()
    def predict_proba(self, features: torch.Tensor, temperature: float = 1.0, **kwargs: Any) -> torch.Tensor:
        """Return softmax probabilities for classification models.

        Args:
            features: Input tensor.
            temperature: Softmax temperature (higher → flatter distribution).
            **kwargs: Forward keyword arguments.

        Returns:
            Probability tensor of shape ``(batch, num_classes)``.
        """
        logits = self.forward(features, **kwargs)
        return torch.softmax(logits / temperature, dim=-1)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: If ``True``, count only ``requires_grad`` parameters.

        Returns:
            Total parameter count.
        """
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)
