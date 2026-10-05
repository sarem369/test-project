"""
AI Engine
=========

A modular, scalable deep-learning framework built on PyTorch.

Modules
-------
config
    Typed configuration schemas and YAML/JSON loading.
data
    Datasets, transforms, and preprocessing pipelines.
models
    Reusable layers, transformer backbones, and a model registry.
training
    Trainer, callbacks, metrics, optimizers, and schedulers.
inference
    Batched prediction, streaming decode helpers, and export utilities.
utils
    Logging, seeding, checkpoints, and device management.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
