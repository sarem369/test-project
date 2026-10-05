"""Device selection and mixed-precision helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from ai_engine.config.schema import Precision


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a user device string to a :class:`torch.device`.

    Args:
        device: One of ``auto``, ``cpu``, ``cuda``, ``cuda:N``, or ``mps``.

    Returns:
        A concrete :class:`torch.device`.

    Raises:
        ValueError: If a requested accelerator is unavailable.
    """
    requested = device.lower().strip()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine")
    if resolved.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available on this machine")
    return resolved


def autocast_dtype(precision: Precision) -> torch.dtype | None:
    """Map a :class:`Precision` enum to an autocast dtype.

    Args:
        precision: Desired precision mode.

    Returns:
        ``torch.float16``, ``torch.bfloat16``, or ``None`` for FP32 (no autocast).
    """
    if precision == Precision.FP16:
        return torch.float16
    if precision == Precision.BF16:
        return torch.bfloat16
    return None


@contextmanager
def maybe_autocast(
    device: torch.device,
    precision: Precision,
    enabled: bool = True,
) -> Iterator[None]:
    """Context manager that enables autocast when beneficial.

    Args:
        device: Target device.
        precision: Desired precision mode.
        enabled: Master switch; when ``False``, yields without autocast.

    Yields:
        Control to the caller under an optional autocast region.
    """
    dtype = autocast_dtype(precision)
    use_autocast = enabled and dtype is not None and device.type in {"cuda", "cpu"}
    if use_autocast:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield
    else:
        yield
