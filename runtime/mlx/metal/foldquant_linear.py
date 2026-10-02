"""FoldQuant product on top of MLX's GEMM.

MLX quantizes weights by group. FoldQuant instead uses one scale per row and
one scale per token, so this file only does that product. The matmul itself is
``mx.matmul``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

_MODEL_ROOT = Path(__file__).resolve().parents[3] / "models" / "smolvla"
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from quantization.foldquant import QuantizedWeight, unpack_int4


def _as_2d(x: mx.array) -> tuple[mx.array, tuple[int, ...]]:
    if x.ndim < 2:
        raise ValueError("Quantized linear requires at least two dimensions.")
    return x.reshape((-1, x.shape[-1])), tuple(x.shape[:-1])


def dynamic_quantize_mx(x: mx.array, bits: int) -> tuple[mx.array, mx.array]:
    maximum = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), mx.array(1e-8, dtype=x.dtype))
    scale = maximum / ((1 << (bits - 1)) - 1)
    codes = mx.clip(mx.round(x / scale), -((1 << (bits - 1)) - 1), (1 << (bits - 1)) - 1).astype(mx.int8)
    return codes, scale.squeeze(-1)


def code_matrix(weight: QuantizedWeight, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Integer codes as a dense matrix. INT8 values are exact in BF16 and FP16."""
    if weight.bits == 4:
        codes = unpack_int4(weight.packed, weight.codes.shape[-1])
    else:
        codes = weight.codes
    return mx.array(np.ascontiguousarray(codes)).astype(dtype)


def gemm_integer_linear(
    x_codes: mx.array,
    x_scales: mx.array,
    codes: mx.array,
    row_scales: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """(xq @ wq.T) * token_scale * row_scale, computed by ``mx.matmul``."""
    x_codes, prefix = _as_2d(x_codes)
    accumulator = mx.matmul(x_codes.astype(codes.dtype), mx.swapaxes(codes, -1, -2)).astype(mx.float32)
    output = accumulator * mx.array(x_scales).astype(mx.float32).reshape((-1, 1))
    output = output * mx.array(row_scales).astype(mx.float32).reshape((1, -1))
    if bias is not None:
        output = output + mx.array(bias).astype(mx.float32)
    return output.astype(codes.dtype).reshape((*prefix, codes.shape[0]))
