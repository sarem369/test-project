"""Tests for transforms and the data preprocessor."""

from __future__ import annotations

import numpy as np

from ai_engine.config import DataConfig
from ai_engine.data import DataPreprocessor, StandardScaler, train_val_split


def test_standard_scaler_zero_mean_unit_var() -> None:
    """Fitted StandardScaler should center and scale training data."""
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(500, 4))
    scaler = StandardScaler().fit(data)
    transformed = scaler.transform(data)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-6)


def test_preprocessor_fit_transform_and_split() -> None:
    """Preprocessor should encode labels and support train/val splitting."""
    features = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    labels = np.asarray(["cat", "dog", "cat", "bird"], dtype=object)
    preprocessor = DataPreprocessor(DataConfig(normalize=True))
    bundle = preprocessor.fit_transform(features, labels)

    assert bundle.features.shape == features.shape
    assert bundle.labels is not None
    assert set(bundle.labels.tolist()) == {0, 1, 2}
    assert preprocessor.label_map["bird"] == 0

    x_train, x_val, y_train, y_val = train_val_split(
        bundle.features, bundle.labels, val_ratio=0.25, seed=1
    )
    assert len(x_train) + len(x_val) == len(features)
    assert y_train is not None and y_val is not None
