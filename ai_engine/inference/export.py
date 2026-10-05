"""Model export utilities (TorchScript / optional ONNX)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger("ai_engine.inference.export")


def export_torchscript(
    model: nn.Module,
    example_inputs: torch.Tensor,
    path: str | Path,
    method: str = "trace",
) -> Path:
    """Export a model to TorchScript.

    Args:
        model: Module to export (should already be on the desired device).
        example_inputs: Example feature batch for tracing / scripting checks.
        path: Destination ``.pt`` path.
        method: ``trace`` (default) or ``script``.

    Returns:
        Resolved path to the exported artifact.
    """
    model.eval()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if method == "trace":
        scripted = torch.jit.trace(model, example_inputs)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError("method must be 'trace' or 'script'")

    scripted.save(str(destination))
    logger.info("Wrote TorchScript model to %s", destination)
    return destination.resolve()


def export_onnx(
    model: nn.Module,
    example_inputs: torch.Tensor,
    path: str | Path,
    opset_version: int = 17,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export a model to ONNX (requires ``torch.onnx``).

    Args:
        model: Module to export.
        example_inputs: Example feature batch.
        path: Destination ``.onnx`` path.
        opset_version: ONNX opset version.
        input_names: Optional ONNX input names.
        output_names: Optional ONNX output names.
        dynamic_axes: Optional dynamic axis specification.

    Returns:
        Resolved path to the ONNX file.
    """
    model.eval()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        example_inputs,
        str(destination),
        input_names=input_names or ["features"],
        output_names=output_names or ["logits"],
        dynamic_axes=dynamic_axes
        or {
            "features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset_version,
    )
    logger.info("Wrote ONNX model to %s", destination)
    return destination.resolve()
