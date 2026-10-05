"""Optimizer factory helpers."""

from __future__ import annotations

import torch
from torch import nn

from ai_engine.config.schema import OptimizerConfig, OptimizerName


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Construct an optimizer for ``model`` from configuration.

    Applies a common best practice: no weight decay on bias and LayerNorm /
    normalization parameters.

    Args:
        model: Model whose parameters will be optimized.
        config: Optimizer hyperparameters.

    Returns:
        Instantiated :class:`torch.optim.Optimizer`.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if config.name == OptimizerName.ADAMW:
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
        )
    if config.name == OptimizerName.ADAM:
        return torch.optim.Adam(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
        )
    if config.name == OptimizerName.SGD:
        return torch.optim.SGD(
            param_groups,
            lr=config.lr,
            momentum=config.momentum,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")
