"""High-level training loop with AMP, grad accumulation, and callbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ai_engine.config.schema import EngineConfig, Precision
from ai_engine.training.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from ai_engine.training.metrics import MetricTracker, classification_metrics
from ai_engine.training.optim import build_optimizer
from ai_engine.training.scheduler import build_scheduler
from ai_engine.utils.device import maybe_autocast, resolve_device
from ai_engine.utils.logging import setup_logging
from ai_engine.utils.seeding import seed_everything


class Trainer:
    """Scalable training orchestrator for :class:`~ai_engine.models.base.BaseModel`.

    Features
    --------
    - Automatic device placement and optional ``torch.compile``
    - Mixed precision (fp16 / bf16) via ``torch.autocast`` + GradScaler
    - Gradient accumulation and global-norm clipping
    - Callback hooks (logging, checkpointing, early stopping)
    - Epoch- and step-based evaluation

    Args:
        model: PyTorch module to train.
        config: Aggregate engine configuration.
        train_loader: Training DataLoader yielding dict batches with
            ``features`` and ``labels`` keys.
        val_loader: Optional validation DataLoader.
        loss_fn: Optional loss module; defaults to ``CrossEntropyLoss``.
        callbacks: Optional extra callbacks.
        logger: Optional logger; a package logger is created when omitted.
    """

    def __init__(
        self,
        model: nn.Module,
        config: EngineConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
        callbacks: list[Callback] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.logger = logger or setup_logging(
            log_file=Path(config.training.output_dir) / "train.log"
        )

        seed_everything(config.training.seed)
        self.device = resolve_device(config.training.device)
        self.model = model.to(self.device)

        if config.training.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        steps_per_epoch = max(len(train_loader), 1)
        total_steps = config.training.max_steps or (
            config.training.epochs * steps_per_epoch // max(config.training.accumulate_grad_batches, 1)
        )
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, config.scheduler, total_steps=total_steps)

        self.use_scaler = (
            config.training.precision == Precision.FP16 and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.global_step = 0
        self.should_stop = False
        self.history: list[dict[str, float]] = []

        output_dir = Path(config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        default_callbacks: list[Callback] = [
            LoggingCallback(self.logger),
            CheckpointCallback(
                output_dir=output_dir,
                monitor=config.training.early_stopping_metric,
                mode=config.training.early_stopping_mode,
                save_every=config.training.save_every,
            ),
        ]
        if config.training.early_stopping_patience is not None:
            default_callbacks.append(
                EarlyStoppingCallback(
                    monitor=config.training.early_stopping_metric,
                    mode=config.training.early_stopping_mode,
                    patience=config.training.early_stopping_patience,
                )
            )
        self.callbacks = default_callbacks + (callbacks or [])

    def fit(self) -> list[dict[str, float]]:
        """Run the full training loop.

        Returns:
            Per-epoch metric history.
        """
        self._emit("on_train_begin")
        for epoch in range(self.config.training.epochs):
            if self.should_stop:
                break
            self._emit("on_epoch_begin", epoch)
            train_metrics = self._train_epoch(epoch)
            metrics = {f"train_{k}": v for k, v in train_metrics.items()}

            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                self._emit("on_eval_end", metrics)

            self.history.append(metrics)
            self._emit("on_epoch_end", epoch, metrics)

            if self.config.training.max_steps and self.global_step >= self.config.training.max_steps:
                self.should_stop = True

        self._emit("on_train_end")
        return self.history

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one training epoch.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            Aggregated training metrics.
        """
        self.model.train()
        tracker = MetricTracker()
        accumulation = self.config.training.accumulate_grad_batches
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(self.train_loader, desc=f"epoch {epoch + 1}", leave=False)
        for batch_idx, batch in enumerate(progress):
            if self.config.training.max_steps and self.global_step >= self.config.training.max_steps:
                self.should_stop = True
                break

            features = batch["features"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with maybe_autocast(self.device, self.config.training.precision):
                logits = self.model(features)
                loss = self.loss_fn(logits, labels) / accumulation

            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (batch_idx + 1) % accumulation == 0
            if should_step:
                if self.use_scaler:
                    if self.config.training.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.grad_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.config.training.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.grad_clip_norm,
                        )
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.scheduler, "step") and self.config.scheduler.name.value != "step":
                    # Step-based schedules advance per optimizer step.
                    self.scheduler.step()
                self.global_step += 1

                batch_metrics = {
                    "loss": float(loss.detach().item() * accumulation),
                    **classification_metrics(logits.detach(), labels),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }
                tracker.update(batch_metrics, batch_size=features.size(0))
                self._emit("on_batch_end", self.global_step, batch_metrics)
                progress.set_postfix(loss=batch_metrics["loss"])

                if (
                    self.config.training.eval_every
                    and self.val_loader is not None
                    and self.global_step % self.config.training.eval_every == 0
                ):
                    eval_metrics = self.evaluate(self.val_loader)
                    self._emit("on_eval_end", {f"val_{k}": v for k, v in eval_metrics.items()})
                    self.model.train()

            if self.should_stop:
                break

        # Epoch-level StepLR advances once per epoch.
        if self.config.scheduler.name.value == "step":
            self.scheduler.step()

        return tracker.compute()

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader | None = None) -> dict[str, float]:
        """Evaluate the model on a loader.

        Args:
            loader: DataLoader to evaluate; defaults to ``val_loader``.

        Returns:
            Aggregated metrics including ``loss`` and ``accuracy``.
        """
        loader = loader or self.val_loader
        if loader is None:
            raise ValueError("No evaluation loader provided")

        self.model.eval()
        tracker = MetricTracker()
        for batch in loader:
            features = batch["features"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with maybe_autocast(self.device, self.config.training.precision):
                logits = self.model(features)
                loss = self.loss_fn(logits, labels)
            metrics = {"loss": float(loss.item()), **classification_metrics(logits, labels)}
            tracker.update(metrics, batch_size=features.size(0))
        return tracker.compute()

    def _emit(self, hook: str, *args: Any) -> None:
        """Invoke ``hook`` on all registered callbacks."""
        for callback in self.callbacks:
            method: Callable[..., None] = getattr(callback, hook)
            method(self, *args)
