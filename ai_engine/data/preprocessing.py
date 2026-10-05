"""High-level data preprocessing pipelines."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from ai_engine.config.schema import DataConfig
from ai_engine.data.transforms import Compose, SequencePadTruncate, StandardScaler, Transform


@dataclass
class PreprocessedBundle:
    """Container for preprocessed feature / label arrays and fitted transforms.

    Attributes:
        features: Feature matrix of shape ``(n_samples, n_features)`` or
            token matrix of shape ``(n_samples, seq_len)``.
        labels: Optional label vector of shape ``(n_samples,)``.
        transforms: Fitted transform pipeline applied to produce ``features``.
        metadata: Free-form metadata (column names, class maps, etc.).
    """

    features: NDArray[Any]
    labels: NDArray[Any] | None = None
    transforms: Transform | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataPreprocessor:
    """Orchestrates loading, cleaning, and transforming datasets.

    The preprocessor is intentionally format-agnostic at the edges: callers can
    feed NumPy arrays directly or point at CSV / NPY files via :class:`DataConfig`.

    Args:
        config: Data configuration controlling normalization, max length, etc.
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or DataConfig()
        self._pipeline: Transform | None = None
        self._label_map: dict[Any, int] = {}

    @property
    def pipeline(self) -> Transform | None:
        """Fitted transform pipeline, or ``None`` before :meth:`fit`."""
        return self._pipeline

    @property
    def label_map(self) -> dict[Any, int]:
        """Mapping from raw label values to integer class indices."""
        return dict(self._label_map)

    def fit(self, features: NDArray[Any], labels: NDArray[Any] | None = None) -> DataPreprocessor:
        """Fit preprocessing statistics on training features / labels.

        Args:
            features: Training feature matrix.
            labels: Optional training labels (used to build a label map).

        Returns:
            ``self``.
        """
        transforms: list[Transform] = []
        if self.config.normalize and features.dtype.kind == "f":
            transforms.append(StandardScaler())
        if features.ndim == 2 and features.dtype.kind in {"i", "u"}:
            transforms.append(SequencePadTruncate(max_length=self.config.max_length))

        if transforms:
            self._pipeline = Compose(transforms)
            self._pipeline.fit(features)
        else:
            self._pipeline = None

        if labels is not None:
            self._fit_label_map(labels)
        return self

    def transform(
        self,
        features: NDArray[Any],
        labels: NDArray[Any] | None = None,
    ) -> PreprocessedBundle:
        """Transform features (and optionally encode labels).

        Args:
            features: Raw feature matrix.
            labels: Optional raw labels.

        Returns:
            A :class:`PreprocessedBundle` with processed arrays.
        """
        processed = self._pipeline(features) if self._pipeline is not None else np.asarray(features)
        encoded_labels = self.encode_labels(labels) if labels is not None else None
        return PreprocessedBundle(
            features=np.asarray(processed),
            labels=encoded_labels,
            transforms=self._pipeline,
            metadata={"label_map": self.label_map},
        )

    def fit_transform(
        self,
        features: NDArray[Any],
        labels: NDArray[Any] | None = None,
    ) -> PreprocessedBundle:
        """Fit on training data then transform it."""
        return self.fit(features, labels).transform(features, labels)

    def encode_labels(self, labels: NDArray[Any] | Sequence[Any]) -> NDArray[np.int64]:
        """Encode raw labels using the fitted label map.

        Args:
            labels: Raw label values.

        Returns:
            Integer-encoded label array.

        Raises:
            RuntimeError: If the label map has not been fitted.
            KeyError: If an unseen label is encountered.
        """
        if not self._label_map:
            raise RuntimeError("Label map is empty; call fit() with labels first")
        array = np.asarray(labels, dtype=object)
        encoded = np.empty(array.shape[0], dtype=np.int64)
        for idx, value in enumerate(array.tolist()):
            if value not in self._label_map:
                raise KeyError(f"Unseen label: {value!r}")
            encoded[idx] = self._label_map[value]
        return encoded

    def load_from_config(self, split: str = "train") -> PreprocessedBundle:
        """Load a dataset split described by :attr:`config` and preprocess it.

        Args:
            split: One of ``train``, ``val``, or ``test``.

        Returns:
            Preprocessed data bundle for the requested split.

        Raises:
            ValueError: If the split path is unset or the format is unsupported.
        """
        path_map = {
            "train": self.config.train_path,
            "val": self.config.val_path,
            "test": self.config.test_path,
        }
        if split not in path_map:
            raise ValueError(f"Unknown split {split!r}; expected train|val|test")
        path = path_map[split]
        if path is None:
            raise ValueError(f"No path configured for split={split!r}")

        features, labels = self._load_file(Path(path))
        if split == "train":
            return self.fit_transform(features, labels)
        if self._pipeline is None and labels is not None and not self._label_map:
            # Allow standalone val/test usage when train was not fit in-process.
            self.fit(features, labels)
        return self.transform(features, labels)

    def _fit_label_map(self, labels: NDArray[Any]) -> None:
        """Build a stable sorted label → index mapping."""
        unique = sorted({value for value in np.asarray(labels, dtype=object).tolist()}, key=str)
        self._label_map = {value: idx for idx, value in enumerate(unique)}

    def _load_file(self, path: Path) -> tuple[NDArray[Any], NDArray[Any] | None]:
        """Load features / labels from CSV or NPY files."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(path)
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            return array, None
        if suffix == ".npz":
            payload = np.load(path, allow_pickle=False)
            features = payload["features"]
            labels = payload["labels"] if "labels" in payload.files else None
            return features, labels
        if suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            features = np.asarray(payload["features"])
            labels = np.asarray(payload["labels"]) if "labels" in payload else None
            return features, labels
        raise ValueError(f"Unsupported data format: {suffix}")

    def _load_csv(self, path: Path) -> tuple[NDArray[Any], NDArray[Any] | None]:
        """Load a tabular CSV into feature / label arrays."""
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV is empty: {path}")

        label_column = self.config.label_column
        feature_columns = self.config.feature_columns
        if feature_columns is None:
            feature_columns = [name for name in reader.fieldnames if name != label_column]

        features = np.asarray(
            [[float(row[col]) for col in feature_columns] for row in rows],
            dtype=np.float64,
        )
        labels: NDArray[Any] | None = None
        if label_column in reader.fieldnames:
            raw_labels = [row[label_column] for row in rows]
            # Prefer numeric labels when possible.
            try:
                labels = np.asarray([float(value) for value in raw_labels], dtype=np.float64)
                if np.all(labels == labels.astype(np.int64)):
                    labels = labels.astype(np.int64)
            except ValueError:
                labels = np.asarray(raw_labels, dtype=object)
        return features, labels


def train_val_split(
    features: NDArray[Any],
    labels: NDArray[Any] | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any] | None, NDArray[Any] | None]:
    """Randomly split arrays into train / validation subsets.

    Args:
        features: Feature matrix.
        labels: Optional label vector aligned with ``features``.
        val_ratio: Fraction of samples reserved for validation.
        seed: RNG seed for reproducible shuffling.

    Returns:
        ``(x_train, x_val, y_train, y_val)`` where label outputs may be ``None``.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive)")

    n_samples = features.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    val_count = max(1, int(round(n_samples * val_ratio)))
    val_idx, train_idx = indices[:val_count], indices[val_count:]

    x_train, x_val = features[train_idx], features[val_idx]
    if labels is None:
        return x_train, x_val, None, None
    return x_train, x_val, labels[train_idx], labels[val_idx]
