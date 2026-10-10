from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models" / "smolvla" / "src"))

from smolvla_mlx.runtime import QuantizedProjection, _mlx_linear, _proj_in


def test_mlx_quantized_linear_matches_qqmm_and_adds_bias():
    """Test that MLX QuantizedLinear with bias produces correct outputs."""
    rng = np.random.default_rng(0)
    weight = mx.array(rng.normal(size=(16, 64)).astype(np.float32))
    bias = mx.array(rng.normal(size=(16,)).astype(np.float32))
    packed, scales = mx.quantize(weight, group_size=16, bits=4, mode="nvfp4")
    spec = QuantizedProjection(
        kind="mlx",
        bits=4,
        bias=bias,
        qweight=packed,
        qscales=scales,
        group_size=16,
        mode="nvfp4",
        activation_bits=4,
    )
    layer = _mlx_linear(spec)
    assert isinstance(layer, nn.QuantizedLinear)
    x = mx.array(rng.normal(size=(2, 5, 64)).astype(np.float32))
    got = layer(x)
    # Reference: dequantize + matmul + bias
    dequant = mx.dequantize(packed, scales, bits=4, group_size=16, mode="nvfp4")
    ref = mx.matmul(x, dequant.T) + bias
    mx.eval(got, ref)
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-5)
    assert _proj_in(layer) == 64


def test_weight_only_nvfp4_stays_quantized_linear():
    """Test that weight-only quantization uses QuantizedLinear without bias."""
    weight = mx.ones((8, 32))
    packed, scales = mx.quantize(weight, group_size=16, bits=4, mode="nvfp4")
    spec = QuantizedProjection(
        kind="mlx",
        bits=4,
        bias=None,
        qweight=packed,
        qscales=scales,
        group_size=16,
        mode="nvfp4",
        activation_bits=None,
    )
    layer = _mlx_linear(spec)
    assert isinstance(layer, nn.QuantizedLinear)
