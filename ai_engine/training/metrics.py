"""Training / evaluation metric utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


class MetricTracker:
    """Accumulate scalar metrics across steps and compute running averages.

    Example:
        >>> tracker = MetricTracker()
        >>> tracker.update({"loss": 1.0}, batch_size=32)
        >>> tracker.update({"loss": 0.5}, batch_size=32)
        >>> tracker.compute()
        {'loss': 0.75}
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._weights: dict[str, float] = defaultdict(float)

    def update(self, metrics: dict[str, float | int], batch_size: int = 1) -> None:
        """Add a batch of metrics.

        Args:
            metrics: Metric name → scalar value mapping.
            batch_size: Weight for averaging (typically the batch size).
        """
        weight = float(batch_size)
        for key, value in metrics.items():
            self._sums[key] += float(value) * weight
            self._weights[key] += weight

    def compute(self) -> dict[str, float]:
        """Return weighted averages for all tracked metrics."""
        return {
            key: self._sums[key] / max(self._weights[key], 1e-12)
            for key in self._sums
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._sums.clear()
        self._weights.clear()


@torch.inference_mode()
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from logits and integer labels.

    Args:
        logits: Model logits of shape ``(batch, num_classes)``.
        labels: Ground-truth class indices of shape ``(batch,)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float().mean().item()
    return float(correct)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute a small suite of classification metrics.

    Args:
        logits: Model logits ``(batch, num_classes)``.
        labels: Integer labels ``(batch,)``.

    Returns:
        Dictionary containing at least ``accuracy`` and ``loss``-ready fields.
    """
    return {"accuracy": accuracy_from_logits(logits, labels)}


def merge_metric_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge metric dictionaries (later dicts win on conflicts)."""
    merged: dict[str, Any] = {}
    for item in dicts:
        merged.update(item)
    return merged
