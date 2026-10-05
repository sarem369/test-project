"""Checkpoint save / load utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ai_engine.config.schema import EngineConfig


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    config: EngineConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a training checkpoint to disk.

    Args:
        path: Destination ``.pt`` file or directory (writes ``checkpoint.pt``).
        model: Model whose ``state_dict`` should be saved.
        optimizer: Optional optimizer to include.
        scheduler: Optional LR scheduler with a ``state_dict`` method.
        epoch: Epoch index associated with this snapshot.
        step: Global optimizer step associated with this snapshot.
        metrics: Optional metric dictionary to embed.
        config: Optional engine config to serialize alongside weights.
        extra: Arbitrary extra metadata.

    Returns:
        Path to the written checkpoint file.
    """
    destination = Path(path)
    if destination.is_dir() or destination.suffix == "":
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / "checkpoint.pt"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        payload["config"] = config.model_dump(mode="json")
        config_path = destination.with_name("config.yaml")
        from ai_engine.config.loader import save_config

        save_config(config, config_path)

    torch.save(payload, destination)
    meta_path = destination.with_name("checkpoint_meta.json")
    meta_path.write_text(
        json.dumps({"epoch": epoch, "step": step, "metrics": metrics or {}}, indent=2),
        encoding="utf-8",
    )
    return destination.resolve()


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore module state.

    Args:
        path: Checkpoint ``.pt`` file or directory containing ``checkpoint.pt``.
        model: Optional model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        map_location: Device mapping passed to ``torch.load``.
        strict: Whether to enforce exact ``state_dict`` key matching.

    Returns:
        The deserialized checkpoint dictionary.
    """
    source = Path(path)
    if source.is_dir():
        source = source / "checkpoint.pt"
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
