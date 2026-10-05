"""Structured logging helpers for training and inference."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


_CONFIGURED = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    name: str = "ai_engine",
) -> logging.Logger:
    """Configure root application logging once and return the package logger.

    Args:
        level: Logging level name or numeric level.
        log_file: Optional file path for a rotating-style plain file handler.
        name: Logger name to return (default: ``ai_engine``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if not _CONFIGURED:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(path, encoding="utf-8"))

        root = logging.getLogger("ai_engine")
        root.setLevel(level)
        root.handlers.clear()
        for handler in handlers:
            handler.setFormatter(formatter)
            root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    else:
        logger.setLevel(level)

    return logger


def log_metrics(logger: logging.Logger, metrics: dict[str, Any], step: int | None = None) -> None:
    """Emit a compact metrics line.

    Args:
        logger: Logger to write to.
        metrics: Metric name → value mapping.
        step: Optional global step to prefix.
    """
    parts = [f"{key}={_format_value(value)}" for key, value in sorted(metrics.items())]
    prefix = f"step={step} " if step is not None else ""
    logger.info("%s%s", prefix, " ".join(parts))


def _format_value(value: Any) -> str:
    """Format a metric value for compact log output."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
