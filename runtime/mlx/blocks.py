"""Pieces MLX's stock blocks do not cover.

``nn.RoPE`` only accepts one offset per example, while SmolVLA positions come
from the padding mask. ``nn.MultiHeadAttention`` has no grouped-query layout.
The OpenPI mask, SmolVLM pixel shuffle, and flow-matching time embedding are
not modules either. Norms, linears, convolution, and GELU are ``mlx.nn``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def apply_rope(x: mx.array, positions: mx.array, max_wavelength: float = 10_000.0) -> mx.array:
    """Apply RoPE to x [B, L, H, D] with positions [B, L]."""
    half = x.shape[-1] // 2
    x32 = x.astype(mx.float32)
    freq = (2.0 / x.shape[-1]) * mx.arange(half, dtype=mx.float32)
    timescale = max_wavelength**freq
    radians = positions.astype(mx.float32)[..., None] / timescale
    radians = radians[..., None, :]
    sin = mx.sin(radians)
    cos = mx.cos(radians)
    first, second = x32[..., :half], x32[..., half:]
    rotated = mx.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)
    return rotated.astype(x.dtype)


def sinusoidal_time(time: mx.array, dimension: int, min_period: float, max_period: float) -> mx.array:
    """Sine-cosine embedding. Frequencies are computed in float64, matching OpenPI."""
    if dimension % 2:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    half = dimension // 2
    fraction = np.linspace(0.0, 1.0, half, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period * 2.0 * math.pi).astype(np.float32)
    angle = time.astype(mx.float32)[..., None] * mx.array(scaling)
    return mx.concatenate([mx.sin(angle), mx.cos(angle)], axis=-1)


def attention_mask_2d(pad_masks: mx.array, att_masks: mx.array) -> mx.array:
    """OpenPI block-causal mask. True means the query may attend to the key."""
    cumulative = mx.cumsum(att_masks.astype(mx.int32), axis=1)
    allowed = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = pad_masks[:, None, :] & pad_masks[:, :, None]
    return allowed & valid


def pixel_shuffle(hidden: mx.array, scale_factor: int) -> mx.array:
    """SmolVLM connector pixel shuffle. hidden is [B, S, C] with S a square."""
    batch, sequence, channels = hidden.shape
    height = width = int(sequence**0.5)
    if height * width != sequence:
        raise ValueError(f"Vision sequence {sequence} is not square")
    value = hidden.reshape(batch, height, width, channels)
    value = value.reshape(batch, height, width // scale_factor, channels * scale_factor)
    value = mx.transpose(value, (0, 2, 1, 3))
    value = value.reshape(batch, width // scale_factor, height // scale_factor, channels * scale_factor * scale_factor)
    value = mx.transpose(value, (0, 2, 1, 3))
    return value.reshape(batch, sequence // (scale_factor**2), channels * scale_factor * scale_factor)


def hadamard(x: mx.array, block_size: int = 64) -> mx.array:
    """Normalized block FWHT along the last axis."""
    width = x.shape[-1]
    if width % block_size:
        raise ValueError(f"Width {width} must be divisible by FWHT block size {block_size}")
    leading = x.shape[:-1]
    value = x.reshape(*leading, width // block_size, block_size)
    step = 1
    while step < block_size:
        view = value.reshape(*leading, width // block_size, block_size // (2 * step), 2, step)
        left, right = view[..., 0, :], view[..., 1, :]
        value = mx.stack([left + right, left - right], axis=-2).reshape(*leading, width // block_size, block_size)
        step *= 2
    return (value / math.sqrt(block_size)).reshape(x.shape)


def attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array | None,
    num_heads: int,
    num_kv_heads: int,
) -> mx.array:
    """Fused attention. query/key/value are [B, S, heads, dim].

    ``mask`` is [B, Q, K] with True meaning the query may attend, or None when
    every position is valid. Key and value stay in grouped-query layout; MLX
    expands the groups inside the kernel.
    """
    if query.shape[2] != num_heads or key.shape[2] != num_kv_heads:
        raise ValueError(
            f"Head counts {(query.shape[2], key.shape[2])} do not match {(num_heads, num_kv_heads)}"
        )
    batch, length, _, head_dim = query.shape
    query = mx.transpose(query, (0, 2, 1, 3))
    key = mx.transpose(key, (0, 2, 1, 3))
    value = mx.transpose(value, (0, 2, 1, 3))
    output = mx.fast.scaled_dot_product_attention(query, key, value, scale=head_dim**-0.5, mask=mask)
    output = mx.transpose(output, (0, 2, 1, 3))
    return output.reshape(batch, length, num_heads * head_dim)
