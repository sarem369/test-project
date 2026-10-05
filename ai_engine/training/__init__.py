"""Training pipelines, metrics, and callbacks."""

from ai_engine.training.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from ai_engine.training.metrics import MetricTracker, accuracy_from_logits, classification_metrics
from ai_engine.training.optim import build_optimizer
from ai_engine.training.scheduler import WarmupCosineScheduler, build_scheduler
from ai_engine.training.trainer import Trainer

__all__ = [
    "Callback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "MetricTracker",
    "Trainer",
    "WarmupCosineScheduler",
    "accuracy_from_logits",
    "build_optimizer",
    "build_scheduler",
    "classification_metrics",
]
