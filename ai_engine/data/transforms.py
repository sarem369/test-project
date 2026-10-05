"""Composable feature transforms for tabular and sequence data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray


ArrayLike = NDArray[Any] | torch.Tensor | list[float] | list[int]


class Transform(ABC):
    """Abstract base class for stateful or stateless feature transforms."""

    @abstractmethod
    def fit(self, data: ArrayLike) -> Transform:
        """Estimate transform parameters from data.

        Args:
            data: Training features used to fit statistics.

        Returns:
            ``self`` for fluent chaining.
        """

    @abstractmethod
    def transform(self, data: ArrayLike) -> NDArray[np.floating[Any]]:
        """Apply the transform to ``data``.

        Args:
            data: Features to transform.

        Returns:
            Transformed NumPy array.
        """

    def fit_transform(self, data: ArrayLike) -> NDArray[np.floating[Any]]:
        """Fit on ``data`` then transform it.

        Args:
            data: Features to fit and transform.

        Returns:
            Transformed NumPy array.
        """
        return self.fit(data).transform(data)

    def __call__(self, data: ArrayLike) -> NDArray[np.floating[Any]]:
        """Alias for :meth:`transform`."""
        return self.transform(data)


def _to_numpy(data: ArrayLike) -> NDArray[Any]:
    """Convert common array-likes to a NumPy ndarray."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


class StandardScaler(Transform):
    """Feature-wise standardization to zero mean and unit variance.

    Attributes:
        eps: Numerical stability constant added to the standard deviation.
        mean_: Fitted per-feature means (set after :meth:`fit`).
        scale_: Fitted per-feature standard deviations (set after :meth:`fit`).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: NDArray[np.floating[Any]] | None = None
        self.scale_: NDArray[np.floating[Any]] | None = None

    def fit(self, data: ArrayLike) -> StandardScaler:
        """Compute mean and std from ``data``.

        Args:
            data: 2D feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            ``self``.
        """
        array = _to_numpy(data).astype(np.float64, copy=False)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.mean_ = array.mean(axis=0)
        self.scale_ = array.std(axis=0)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        return self

    def transform(self, data: ArrayLike) -> NDArray[np.floating[Any]]:
        """Standardize ``data`` using fitted statistics.

        Args:
            data: Feature matrix or vector.

        Returns:
            Standardized float64 array.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fit before transform()")
        array = _to_numpy(data).astype(np.float64, copy=False)
        squeeze = False
        if array.ndim == 1:
            array = array.reshape(-1, 1)
            squeeze = True
        result = (array - self.mean_) / self.scale_
        return result.ravel() if squeeze and result.shape[1] == 1 else result


class MinMaxScaler(Transform):
    """Scale features to a target range (default ``[0, 1]``)."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8) -> None:
        self.feature_range = feature_range
        self.eps = eps
        self.data_min_: NDArray[np.floating[Any]] | None = None
        self.data_max_: NDArray[np.floating[Any]] | None = None

    def fit(self, data: ArrayLike) -> MinMaxScaler:
        """Compute per-feature min / max from ``data``."""
        array = _to_numpy(data).astype(np.float64, copy=False)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.data_min_ = array.min(axis=0)
        self.data_max_ = array.max(axis=0)
        return self

    def transform(self, data: ArrayLike) -> NDArray[np.floating[Any]]:
        """Scale ``data`` into ``feature_range``."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("MinMaxScaler must be fit before transform()")
        array = _to_numpy(data).astype(np.float64, copy=False)
        squeeze = False
        if array.ndim == 1:
            array = array.reshape(-1, 1)
            squeeze = True
        low, high = self.feature_range
        scale = np.maximum(self.data_max_ - self.data_min_, self.eps)
        result = (array - self.data_min_) / scale
        result = result * (high - low) + low
        return result.ravel() if squeeze and result.shape[1] == 1 else result


class SequencePadTruncate(Transform):
    """Pad or truncate 1D integer token sequences to a fixed length.

    Attributes:
        max_length: Target sequence length.
        pad_value: Value used for right-padding.
    """

    def __init__(self, max_length: int = 512, pad_value: int = 0) -> None:
        self.max_length = max_length
        self.pad_value = pad_value

    def fit(self, data: ArrayLike) -> SequencePadTruncate:
        """No-op fit for API compatibility."""
        return self

    def transform(self, data: ArrayLike) -> NDArray[np.floating[Any]]:
        """Pad / truncate a single sequence or a batch of sequences."""
        array = _to_numpy(data)
        if array.ndim == 1:
            return self._pad_one(array).astype(np.float64)
        return np.stack([self._pad_one(row) for row in array]).astype(np.float64)

    def _pad_one(self, sequence: NDArray[Any]) -> NDArray[Any]:
        """Pad or truncate one sequence to ``max_length``."""
        sequence = np.asarray(sequence).ravel()
        if len(sequence) >= self.max_length:
            return sequence[: self.max_length]
        pad_width = self.max_length - len(sequence)
        return np.pad(sequence, (0, pad_width), constant_values=self.pad_value)


class Compose(Transform):
    """Compose a sequence of transforms applied left-to-right."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def fit(self, data: ArrayLike) -> Compose:
        """Fit each transform sequentially on the cascading output."""
        current = data
        for transform in self.transforms:
            current = transform.fit_transform(current)
        return self

    def transform(self, data: ArrayLike) -> NDArray[np.floating[Any]]:
        """Apply each transform in order."""
        current: ArrayLike = data
        for transform in self.transforms:
            current = transform.transform(current)
        return _to_numpy(current)
