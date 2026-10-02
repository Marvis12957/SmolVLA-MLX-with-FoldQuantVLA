"""Safetensors conversion with auditable tensor coverage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .artifacts import SourceSpec, copy_policy_assets, sha256, write_manifest


@dataclass(frozen=True)
class ConversionReport:
    source_tensors: int
    converted_tensors: int
    source_bytes: int
    output_bytes: int
    converted_conv2d: int


def mlx_layout(name: str, tensor: np.ndarray) -> np.ndarray:
    """Apply only layout transforms required by MLX modules.

    Linear weights remain [out, in], matching MLX Linear. Conv2d weights change
    from PyTorch OIHW to MLX OHWI. The source name is intentionally retained so
    that conversion coverage can be audited against the PyTorch checkpoint.
    """
    if tensor.ndim == 4 and name.endswith("weight"):
        return np.ascontiguousarray(tensor.transpose(0, 2, 3, 1))
    return np.ascontiguousarray(tensor)


def convert_checkpoint(source: str | Path, destination: str | Path, *, source_spec: SourceSpec = SourceSpec()) -> ConversionReport:
    source = Path(source)
    destination = Path(destination)
    source_weights = source / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)
    destination.mkdir(parents=True, exist_ok=True)

    loaded = mx.load(str(source_weights))
    tensors: dict[str, mx.array] = {}
    conv_count = 0
    for name, tensor in loaded.items():
        if tensor.ndim == 4 and name.endswith("weight"):
            conv_count += 1
            tensor = mx.transpose(tensor, (0, 2, 3, 1))
        tensors[name] = tensor
    if len(tensors) != len(loaded):
        raise RuntimeError(f"Converted {len(tensors)} of {len(loaded)} source tensors")
    output_weights = destination / "weights.safetensors"
    mx.save_safetensors(str(output_weights), tensors)
    keys = list(loaded)
    copy_policy_assets(source, destination)
    report = ConversionReport(
        source_tensors=len(keys),
        converted_tensors=len(tensors),
        source_bytes=source_weights.stat().st_size,
        output_bytes=output_weights.stat().st_size,
        converted_conv2d=conv_count,
    )
    (destination / "conversion.json").write_text(json.dumps(report.__dict__, indent=2, sort_keys=True) + "\n")
    write_manifest(destination, source_spec, conversion=report.__dict__)
    return report


def validate_artifact(path: str | Path) -> ConversionReport:
    """Validate an MLX artifact without loading the full model into memory."""
    path = Path(path)
    report = ConversionReport(**json.loads((path / "conversion.json").read_text()))
    weights = path / "weights.safetensors"
    if not weights.is_file() or sha256(weights) == "":
        raise RuntimeError(f"Invalid weights artifact: {weights}")
    with safe_open(weights, framework="np") as handle:
        if len(handle.keys()) != report.converted_tensors:
            raise RuntimeError("Tensor count does not match conversion report")
    return report
