"""Efficient inference engine with batching and AMP support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from ai_engine.config.schema import EngineConfig, InferenceConfig
from ai_engine.inference.batching import iter_batches
from ai_engine.models.base import BaseModel
from ai_engine.models.registry import build_model
from ai_engine.utils.checkpoint import load_checkpoint
from ai_engine.utils.device import maybe_autocast, resolve_device
from ai_engine.utils.logging import setup_logging


class InferenceEngine:
    """High-throughput inference wrapper around a trained model.

    Args:
        model: Trained model module.
        config: Inference configuration (or full :class:`EngineConfig`).
        logger: Optional logger instance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig | EngineConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if isinstance(config, EngineConfig):
            self.config = config.inference
            self.engine_config = config
        else:
            self.config = config or InferenceConfig()
            self.engine_config = None

        self.logger = logger or setup_logging()
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: EngineConfig | None = None,
        map_location: str | None = None,
    ) -> InferenceEngine:
        """Load a model + checkpoint and return an inference engine.

        Args:
            checkpoint_path: Path to ``.pt`` checkpoint or directory.
            config: Optional engine config; if omitted, config embedded in the
                checkpoint is used when available.
            map_location: Optional device mapping for ``torch.load``.

        Returns:
            Ready-to-serve :class:`InferenceEngine`.
        """
        device = resolve_device(
            config.inference.device if config is not None else "auto"
        )
        checkpoint = load_checkpoint(
            checkpoint_path,
            map_location=map_location or str(device),
        )

        if config is None:
            if "config" not in checkpoint:
                raise ValueError(
                    "No EngineConfig provided and checkpoint has no embedded config"
                )
            config = EngineConfig.model_validate(checkpoint["config"])

        model = build_model(config.model)
        model.load_state_dict(checkpoint["model_state_dict"])
        config.inference.checkpoint_path = Path(checkpoint_path)
        return cls(model=model, config=config)

    @torch.inference_mode()
    def predict_batch(self, features: torch.Tensor) -> torch.Tensor:
        """Run a single batched forward pass and return class indices.

        Args:
            features: Input tensor already shaped for the model.

        Returns:
            Predicted labels of shape ``(batch,)``.
        """
        features = features.to(self.device, non_blocking=True)
        with maybe_autocast(
            self.device,
            self.config.precision,
            enabled=self.config.use_amp,
        ):
            if isinstance(self.model, BaseModel):
                return self.model.predict(features)
            logits = self.model(features)
            return logits.argmax(dim=-1)

    @torch.inference_mode()
    def predict_proba_batch(self, features: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for a feature batch.

        Args:
            features: Input tensor.

        Returns:
            Probability tensor ``(batch, num_classes)``.
        """
        features = features.to(self.device, non_blocking=True)
        with maybe_autocast(
            self.device,
            self.config.precision,
            enabled=self.config.use_amp,
        ):
            if isinstance(self.model, BaseModel):
                return self.model.predict_proba(features, temperature=self.config.temperature)
            logits = self.model(features)
            return torch.softmax(logits / self.config.temperature, dim=-1)

    def predict(
        self,
        features: NDArray[Any] | torch.Tensor | Sequence[NDArray[Any]],
        return_proba: bool = False,
    ) -> NDArray[Any]:
        """Run inference over array-like inputs with automatic micro-batching.

        Args:
            features: NumPy array, tensor, or sequence of sample arrays.
            return_proba: If ``True``, return probabilities instead of labels.

        Returns:
            NumPy array of predictions or probabilities.
        """
        tensor = self._to_tensor(features)
        outputs: list[torch.Tensor] = []
        for start in range(0, tensor.size(0), self.config.batch_size):
            batch = tensor[start : start + self.config.batch_size]
            if return_proba:
                outputs.append(self.predict_proba_batch(batch).cpu())
            else:
                outputs.append(self.predict_batch(batch).cpu())
        return torch.cat(outputs, dim=0).numpy()

    @torch.inference_mode()
    def predict_loader(self, loader: DataLoader, return_proba: bool = False) -> NDArray[Any]:
        """Run inference over an entire :class:`DataLoader`.

        Args:
            loader: DataLoader yielding ``{\"features\": ...}`` batches.
            return_proba: Whether to return probabilities.

        Returns:
            Concatenated NumPy predictions.
        """
        chunks: list[torch.Tensor] = []
        for batch in loader:
            features = batch["features"]
            if return_proba:
                chunks.append(self.predict_proba_batch(features).cpu())
            else:
                chunks.append(self.predict_batch(features).cpu())
        if not chunks:
            return np.asarray([])
        return torch.cat(chunks, dim=0).numpy()

    def stream_predict(
        self,
        samples: Sequence[NDArray[Any] | torch.Tensor],
        return_proba: bool = False,
    ) -> IteratorResult:
        """Yield predictions for an iterable of samples in micro-batches.

        Args:
            samples: Sequence of per-sample feature arrays / tensors.
            return_proba: Whether to yield probabilities.

        Returns:
            An :class:`IteratorResult` wrapping a generator of NumPy arrays.
        """

        def generator():
            for batch_samples in iter_batches(list(samples), self.config.batch_size):
                batch = self._to_tensor(batch_samples)
                if return_proba:
                    yield self.predict_proba_batch(batch).cpu().numpy()
                else:
                    yield self.predict_batch(batch).cpu().numpy()

        return IteratorResult(generator())

    def _to_tensor(
        self,
        features: NDArray[Any] | torch.Tensor | Sequence[NDArray[Any] | torch.Tensor],
    ) -> torch.Tensor:
        """Normalize heterogeneous inputs into a batched tensor."""
        if isinstance(features, torch.Tensor):
            tensor = features
        elif isinstance(features, np.ndarray):
            tensor = torch.as_tensor(features)
        else:
            arrays = [
                torch.as_tensor(sample) if not isinstance(sample, torch.Tensor) else sample
                for sample in features
            ]
            if arrays and arrays[0].ndim == 0:
                tensor = torch.stack(arrays)
            elif arrays and arrays[0].ndim == 1 and arrays[0].dtype in {
                torch.long,
                torch.int32,
                torch.int64,
            }:
                from ai_engine.inference.batching import pad_sequences

                tensor = pad_sequences(arrays, pad_value=0, max_length=self.config.max_length)
            else:
                tensor = torch.stack(arrays)

        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        if tensor.ndim == 1:
            # Single tabular sample → batch dimension.
            tensor = tensor.unsqueeze(0)
        return tensor


class IteratorResult:
    """Thin wrapper so streaming helpers remain easy to iterate / materialize."""

    def __init__(self, iterator: Any) -> None:
        self._iterator = iterator

    def __iter__(self):
        return self._iterator

    def concat(self) -> NDArray[Any]:
        """Materialize and concatenate all streamed batches."""
        parts = list(self._iterator)
        if not parts:
            return np.asarray([])
        return np.concatenate(parts, axis=0)
