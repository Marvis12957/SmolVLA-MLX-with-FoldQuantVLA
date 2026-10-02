"""Build a FoldQuant artifact from a SmolVLA safetensors checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from .foldquant import fold_weight, pack_int4, quantize_weight_per_row, smoothquant_scale

_LINEAR_ENDS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)
_HIGH_PRECISION = ("o_proj", "out_proj", "down_proj", "fc2")


def quantize_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    scheme: str = "w8a8",
    block_size: int = 64,
    alpha: float = 0.4,
    group_size: int = 64,
    activation_amax: dict[str, np.ndarray] | None = None,
) -> dict:
    """Quantize transformer projections. Action and state heads stay in floating point.

    ``mlx-w8`` and ``mlx-w4`` use MLX affine group quantization and ``mx.quantized_matmul``.
    Layers whose input width is not divisible by 32, 64, or 128 stay in floating point,
    because that is the only group size the MLX kernel accepts. Expert projections with
    width 720 are in that set.

    FoldQuant schemes apply SmoothQuant, then a block FWHT when the width allows it.
    ``activation_amax`` maps a layer name to the per-input-channel maximum. Missing
    entries keep a scale of 1. Widths divisible by the group size are stored with
    MLX affine ``mx.quantize``. The rest keep one symmetric scale per row.
    """
    source = Path(source)
    destination = Path(destination)
    weight_file = source / "weights.safetensors" if (source / "weights.safetensors").is_file() else source / "model.safetensors"
    if not weight_file.is_file():
        raise FileNotFoundError(weight_file)
    kind, bits, selective = _scheme(scheme)
    weights = mx.load(str(weight_file))
    destination.mkdir(parents=True, exist_ok=True)
    layers: dict[str, dict] = {}
    skipped: list[str] = []
    exported: dict[str, mx.array] = {}
    calibrated = 0
    for name, value in weights.items():
        if kind == "mlx":
            if not _mlx_quantizable(name, value, group_size):
                if _would_quantize(name, value) and int(value.shape[1]) % group_size:
                    skipped.append(name[: -len(".weight")])
                exported[name] = value
                continue
            matrix = value.astype(mx.float32)
            packed, scales, biases = mx.quantize(matrix, group_size=group_size, bits=bits, mode="affine")
            stem = name[: -len(".weight")]
            exported[stem + ".qweight"] = packed
            exported[stem + ".qscales"] = scales
            exported[stem + ".qbiases"] = biases
            bias_name = stem + ".bias"
            if bias_name in weights:
                exported[bias_name] = weights[bias_name]
            layers[stem] = {"bits": bits, "group_size": group_size, "mode": "affine"}
            continue
        if not _quantizable(name, value, bits, selective, block_size):
            exported[name] = value
            continue
        matrix = np.array(value.astype(mx.float32))
        stem = name[: -len(".weight")]
        observed = None if activation_amax is None else activation_amax.get(stem)
        if observed is not None and np.asarray(observed).shape == (matrix.shape[1],):
            scale = smoothquant_scale(np.asarray(observed, dtype=np.float32), matrix, alpha)
            calibrated += 1
        else:
            scale = np.ones(matrix.shape[1], dtype=np.float32)
        rotate = matrix.shape[1] % block_size == 0
        folded = fold_weight(matrix, scale, rotate=rotate, block_size=block_size if rotate else 2)
        bias_name = stem + ".bias"
        if bias_name in weights:
            exported[bias_name] = weights[bias_name]
        exported[stem + ".smooth_scale"] = mx.array(scale)
        record = {"bits": bits, "rotate": rotate, "block_size": block_size if rotate else 1, "alpha": alpha}
        if matrix.shape[1] % group_size == 0 and group_size in (32, 64, 128):
            packed, qscales, qbiases = mx.quantize(mx.array(folded), group_size=group_size, bits=bits, mode="affine")
            exported[stem + ".qweight"] = packed
            exported[stem + ".qscales"] = qscales
            exported[stem + ".qbiases"] = qbiases
            record.update({"group_size": group_size, "mode": "affine", "weight_quant": "group"})
        else:
            quantized = quantize_weight_per_row(folded, bits)
            exported[stem + ".codes"] = mx.array(quantized.codes)
            exported[stem + ".row_scales"] = mx.array(quantized.scales)
            if bits == 4:
                exported[stem + ".packed"] = mx.array(pack_int4(quantized.codes))
            record["weight_quant"] = "row"
        layers[stem] = record
    mx.save_safetensors(str(destination / "weights.safetensors"), exported)
    manifest = {"scheme": scheme, "layers": layers}
    if kind == "mlx":
        manifest["group_size"] = group_size
        manifest["skipped"] = skipped
    (destination / "quantization.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Processor mean/std live in small safetensors next to the JSON. The weight
    # file itself is replaced by weights.safetensors and must not be copied.
    skipped_weights = {"model.safetensors", "weights.safetensors", weight_file.name}
    for asset in source.iterdir():
        if not asset.is_file() or asset.name == "quantization.json" or asset.name in skipped_weights:
            continue
        if asset.suffix in {".json", ".md", ".safetensors"}:
            shutil.copy2(asset, destination / asset.name)
    return {
        "scheme": scheme,
        "quantized_layers": len(layers),
        "skipped_layers": len(skipped),
        "calibrated_layers": calibrated,
        "tensors": len(exported),
    }


def _scheme(scheme: str) -> tuple[str, int, bool]:
    table = {
        "mlx-w8": ("mlx", 8, False),
        "mlx-w4": ("mlx", 4, False),
        "w8a8": ("fold", 8, False),
        "w4a4": ("fold", 4, False),
        "w4a4-selective": ("fold", 4, True),
    }
    if scheme not in table:
        raise ValueError(f"Unknown scheme {scheme}. Choose from {sorted(table)}")
    return table[scheme]


def _would_quantize(name: str, value: mx.array) -> bool:
    if value.ndim != 2 or not name.endswith(_LINEAR_ENDS):
        return False
    if "action_" in name or name.endswith("state_proj.weight"):
        return False
    return True


def _mlx_quantizable(name: str, value: mx.array, group_size: int) -> bool:
    if group_size not in (32, 64, 128):
        raise ValueError("MLX affine quantization only accepts group size 32, 64, or 128.")
    if not _would_quantize(name, value):
        return False
    return int(value.shape[1]) % group_size == 0


def _quantizable(name: str, value: mx.array, bits: int, selective: bool, block_size: int) -> bool:
    if value.ndim != 2 or not name.endswith(_LINEAR_ENDS):
        return False
    if "action_" in name or name.endswith("state_proj.weight"):
        return False
    if selective and any(part in name for part in _HIGH_PRECISION):
        return False
    width = int(value.shape[1])
    if bits == 4 and width % 8:
        return False
    if width % block_size and width < 2:
        return False
    return True
