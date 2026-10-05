"""Training callbacks for checkpointing, early stopping, and logging."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_engine.utils.checkpoint import save_checkpoint
from ai_engine.utils.logging import log_metrics

if TYPE_CHECKING:
    from ai_engine.training.trainer import Trainer


class Callback(ABC):
    """Base class for trainer lifecycle hooks."""

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after training finishes."""

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch with aggregated metrics."""

    def on_batch_end(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        """Called after each optimizer step."""

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """Called after a validation pass."""


class LoggingCallback(Callback):
    """Emit structured logs for steps and epochs."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ai_engine.training")

    def on_batch_end(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        """Log metrics every ``trainer.config.training.log_every`` steps."""
        if step % trainer.config.training.log_every == 0:
            log_metrics(self.logger, metrics, step=step)

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch-level aggregates."""
        payload = {"epoch": epoch, **metrics}
        log_metrics(self.logger, payload)


class CheckpointCallback(Callback):
    """Persist checkpoints on a fixed epoch interval and keep the best model."""

    def __init__(
        self,
        output_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_every: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every
        self.best_score: float | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_improved(self, score: float) -> bool:
        """Return whether ``score`` improves on ``best_score``."""
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        """Save periodic and best checkpoints."""
        if (epoch + 1) % self.save_every == 0:
            save_checkpoint(
                self.output_dir / f"epoch_{epoch + 1}.pt",
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                step=trainer.global_step,
                metrics=metrics,
                config=trainer.config,
            )

        if self.monitor in metrics and self._is_improved(metrics[self.monitor]):
            self.best_score = metrics[self.monitor]
            save_checkpoint(
                self.output_dir / "best.pt",
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                step=trainer.global_step,
                metrics=metrics,
                config=trainer.config,
            )


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.bad_epochs = 0

    def _is_improved(self, score: float) -> bool:
        """Check whether ``score`` improved beyond ``min_delta``."""
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """Update patience counters and request stop if exhausted."""
        if self.monitor not in metrics:
            return
        score = metrics[self.monitor]
        if self._is_improved(score):
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                trainer.should_stop = True
