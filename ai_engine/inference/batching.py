"""Dynamic batching helpers for efficient inference."""

from __future__ import annotations

from typing import Iterator, Sequence, TypeVar

import torch

T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield contiguous slices of ``items`` with size ``batch_size``.

    Args:
        items: Sequence to split.
        batch_size: Maximum items per batch.

    Yields:
        Slices of ``items``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def pad_sequences(
    sequences: Sequence[torch.Tensor],
    pad_value: int | float = 0,
    max_length: int | None = None,
) -> torch.Tensor:
    """Right-pad a list of 1D tensors to a common length.

    Args:
        sequences: List of 1D tensors.
        pad_value: Padding fill value.
        max_length: Optional hard cap on padded length.

    Returns:
        Padded tensor of shape ``(batch, length)``.
    """
    if not sequences:
        return torch.empty(0, 0)

    lengths = [int(seq.numel()) for seq in sequences]
    target = max(lengths)
    if max_length is not None:
        target = min(target, max_length)

    batch = torch.full((len(sequences), target), pad_value, dtype=sequences[0].dtype)
    for row, seq in enumerate(sequences):
        truncated = seq[:target]
        batch[row, : truncated.numel()] = truncated
    return batch


def collate_features(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a list of feature dictionaries into a batched dictionary.

    Args:
        batch: Sample dicts each containing a ``features`` tensor.

    Returns:
        Batched dictionary; sequence features are padded when ranks differ
        only in length (1D inputs).
    """
    features = [item["features"] for item in batch]
    if features[0].ndim == 1 and features[0].dtype in {
        torch.long,
        torch.int64,
        torch.int32,
    }:
        batched = pad_sequences(features, pad_value=0)
    else:
        batched = torch.stack(features, dim=0)

    output: dict[str, torch.Tensor] = {"features": batched}
    if "labels" in batch[0]:
        output["labels"] = torch.stack([item["labels"] for item in batch], dim=0)
    return output
