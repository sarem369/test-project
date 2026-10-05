"""PyTorch dataset adapters for the AI Engine framework."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from ai_engine.config.schema import DataConfig
from ai_engine.data.preprocessing import PreprocessedBundle


class ArrayDataset(Dataset):
    """Simple in-memory dataset wrapping NumPy / tensor features and labels.

    Args:
        features: Feature array of shape ``(n_samples, ...)``.
        labels: Optional label array of shape ``(n_samples,)``.
        transform: Optional per-sample callable applied to features.
    """

    def __init__(
        self,
        features: NDArray[Any] | torch.Tensor,
        labels: NDArray[Any] | torch.Tensor | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if isinstance(features, np.ndarray):
            features_tensor = torch.as_tensor(features)
        else:
            features_tensor = features
        if features_tensor.dtype in {torch.float64}:
            features_tensor = features_tensor.float()
        elif features_tensor.dtype in {torch.int64, torch.int32} and features_tensor.ndim == 2:
            # Token ids stay as long; continuous features become float.
            pass

        self.features = features_tensor
        self.labels: torch.Tensor | None
        if labels is None:
            self.labels = None
        else:
            label_tensor = torch.as_tensor(labels)
            if label_tensor.dtype == torch.float64:
                label_tensor = label_tensor.long()
            self.labels = label_tensor
        self.transform = transform

        if self.labels is not None and len(self.labels) != len(self.features):
            raise ValueError("features and labels must have the same length")

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a single sample dictionary.

        Args:
            index: Sample index.

        Returns:
            Dictionary with ``features`` and optionally ``labels``.
        """
        item_features = self.features[index]
        if self.transform is not None:
            item_features = self.transform(item_features)
        sample: dict[str, torch.Tensor] = {"features": item_features}
        if self.labels is not None:
            sample["labels"] = self.labels[index]
        return sample


def dataset_from_bundle(bundle: PreprocessedBundle) -> ArrayDataset:
    """Build an :class:`ArrayDataset` from a :class:`PreprocessedBundle`.

    Args:
        bundle: Preprocessed feature / label container.

    Returns:
        Ready-to-use :class:`ArrayDataset`.
    """
    features = bundle.features
    if features.dtype.kind == "f":
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    else:
        feature_tensor = torch.as_tensor(features, dtype=torch.long)
    labels = None if bundle.labels is None else torch.as_tensor(bundle.labels)
    return ArrayDataset(feature_tensor, labels)


def create_dataloader(
    dataset: Dataset,
    config: DataConfig,
    shuffle: bool | None = None,
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` from config defaults.

    Args:
        dataset: Dataset to wrap.
        config: Data configuration providing batch size / workers / pin memory.
        shuffle: Override for ``config.shuffle`` (useful for val/test loaders).

    Returns:
        Configured DataLoader.
    """
    pin_memory = bool(config.pin_memory and torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle if shuffle is None else shuffle,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
