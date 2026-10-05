"""Learning-rate scheduler factories."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer

from ai_engine.config.schema import SchedulerConfig, SchedulerName


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay to ``min_lr``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps for the cosine phase horizon.
        min_lr: Final learning rate after cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        """Advance the schedule by one optimizer step."""
        self._step += 1
        lr = self._lr_at(self._step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            # Scale each group relative to its base lr.
            scale = lr / max(base_lr, 1e-12) if self._step <= self.warmup_steps else None
            if self._step <= self.warmup_steps:
                group["lr"] = base_lr * (self._step / max(self.warmup_steps, 1))
            else:
                progress = (self._step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                group["lr"] = self.min_lr + (base_lr - self.min_lr) * cosine
            del scale

    def _lr_at(self, step: int) -> float:
        """Compute the primary group's LR at ``step`` (for introspection)."""
        base = self.base_lrs[0]
        if step <= self.warmup_steps:
            return base * (step / max(self.warmup_steps, 1))
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (base - self.min_lr) * cosine

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state."""
        return {
            "step": self._step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state."""
        self._step = int(state["step"])
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.min_lr = float(state["min_lr"])
        self.base_lrs = list(state["base_lrs"])


class NoOpScheduler:
    """Scheduler stub that leaves learning rates unchanged."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        """No-op."""

    def state_dict(self) -> dict[str, Any]:
        """Empty state."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Ignore state."""
        del state


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    total_steps: int,
) -> Any:
    """Build a learning-rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        config: Scheduler configuration.
        total_steps: Estimated total optimizer steps for cosine schedules.

    Returns:
        Scheduler instance with ``step`` / ``state_dict`` / ``load_state_dict``.
    """
    if config.name == SchedulerName.NONE:
        return NoOpScheduler(optimizer)
    if config.name == SchedulerName.COSINE:
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr=config.min_lr,
        )
    if config.name == SchedulerName.STEP:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    raise ValueError(f"Unsupported scheduler: {config.name}")
