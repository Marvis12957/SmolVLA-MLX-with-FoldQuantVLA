"""MLX inference runtime for the pinned SmolVLA checkpoint.

The forward path follows LeRobot ``VLAFlowMatching`` and ``SmolVLMWithExpertModel``:
SigLIP vision, SmolVLM pixel-shuffle connector, 16-layer SmolLM prefix with KV cache,
and a width-0.75 action expert. Even expert layers self-attend with the cached prefix.
Odd layers cross-attend to it. Ten Euler steps integrate the velocity field.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .processor import ProcessedObservation

_ROOT = Path(__file__).resolve().parents[4]
for _extra in (_ROOT / "runtime" / "mlx", _ROOT / "runtime" / "mlx" / "metal", _ROOT / "models" / "smolvla"):
    _entry = str(_extra)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from blocks import (  # noqa: E402
    apply_rope,
    attention,
    attention_mask_2d,
    hadamard,
    pixel_shuffle,
    sinusoidal_time,
)
from foldquant_linear import code_matrix, dynamic_quantize_mx  # noqa: E402
from quantization.foldquant import QuantizedWeight  # noqa: E402

_COMPUTE = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}

VISION_HEADS = 12
HEAD_DIM = 64
RMS_EPS = 1e-5
LAYER_NORM_EPS = 1e-6
PREFIX = "model."


@dataclass
class QuantizedProjection:
    kind: str
    bits: int
    bias: mx.array | None
    codes: np.ndarray | None = None
    scales: np.ndarray | None = None
    packed: np.ndarray | None = None
    smooth_scale: np.ndarray | None = None
    rotate: bool = False
    block_size: int = 64
    qweight: mx.array | None = None
    qscales: mx.array | None = None
    qbiases: mx.array | None = None
    group_size: int = 64
    mode: str = "affine"


@dataclass
class PreparedActivation:
    """Pre-computed activation transforms for QKV sharing.

    Fields:
        hidden_dtype: Original activation dtype before quantization.
        width: Original activation width before padding.
        pad: Optional padding array applied before scale/rotate.
        scaled: SmoothQuant-scaled activation (or original if no smooth).
        rotated: Hadamard-rotated activation (or scaled if no rotate).
        quantized: (codes, token_scales) tuple ready for projection.
        uses_qqmm: Whether projection should use mx.qqmm path.
        uses_int4: Whether weight is INT4 (for code unpacking).
    """
    hidden_dtype: mx.Dtype
    width: int
    pad: mx.array | None
    scaled: mx.array
    rotated: mx.array
    quantized: tuple[mx.array, mx.array]
    uses_qqmm: bool
    uses_int4: bool


# Feature flag for QKV activation sharing (default True for quantized)
QKV_SHARING_ENABLED = True


class FoldLinear(nn.Module):
    """SmoothQuant, FWHT, and per-token activation quant around an MLX linear.

    Weights that fit group 32/64/128 use ``nn.QuantizedLinear``. The others keep
    one scale per row and a dense ``nn.Linear`` of the dequantized matrix.

    Supports QKV activation sharing via ``prepare_activation`` and ``forward_prepared``.
    """

    def __init__(self, spec: QuantizedProjection, proj: nn.Module) -> None:
        super().__init__()
        self.proj = proj
        self.smooth = mx.array(np.ascontiguousarray(spec.smooth_scale)) if spec.smooth_scale is not None else None
        self.bits = spec.bits
        self.rotate = spec.rotate
        self.block_size = spec.block_size
        self.has_smooth = spec.smooth_scale is not None

    def prepare_activation(self, hidden: mx.array) -> PreparedActivation:
        """Prepare activation for QKV sharing: performs pad/scale/rotate/quant but NOT projection.

        Returns a PreparedActivation that can be reused across Q, K, V projections.

        Args:
            hidden: Input activation tensor [batch, seq, hidden_dim]

        Returns:
            PreparedActivation with pre-computed transforms
        """
        hidden_dtype = hidden.dtype
        original_width = hidden.shape[-1]

        # Determine target width based on projection input size
        proj_in = _proj_in(self.proj)
        target_width = proj_in

        # Handle padding if needed
        pad = None
        working = hidden
        if original_width != target_width:
            if original_width > target_width:
                working = hidden[..., :target_width]
            else:
                pad_width = target_width - original_width
                pad = mx.zeros((*hidden.shape[:-1], pad_width), dtype=hidden.dtype)
                working = mx.concatenate([hidden, pad], axis=-1)

        # Apply SmoothQuant scale if present
        if self.has_smooth:
            scaled = working / self.smooth.astype(working.dtype)
        else:
            scaled = working

        # Apply Hadamard rotation if enabled
        if self.rotate:
            rotated = hadamard(scaled, self.block_size)
        else:
            rotated = scaled

        # Quantize: (codes, token_scales)
        codes, token_scales = dynamic_quantize_mx(rotated, self.bits)

        # Determine if using qqmm path and int4
        uses_qqmm = hasattr(self.proj, "group_size") and self.proj.group_size > 0
        uses_int4 = self.bits == 4

        return PreparedActivation(
            hidden_dtype=hidden_dtype,
            width=original_width,
            pad=pad,
            scaled=scaled,
            rotated=rotated,
            quantized=(codes, token_scales),
            uses_qqmm=uses_qqmm,
            uses_int4=uses_int4,
        )

    def forward_prepared(self, prepared: PreparedActivation) -> mx.array:
        """Run projection using pre-computed activation.

        Args:
            prepared: Pre-computed activation from prepare_activation

        Returns:
            Projection output with dtype matching original hidden
        """
        codes, token_scales = prepared.quantized

        # Dequantize back to activation dtype
        restored = codes.astype(prepared.hidden_dtype) * token_scales.astype(prepared.hidden_dtype)[..., None]

        # Run through projection
        output = self.proj(restored)
        return output if output.dtype == prepared.hidden_dtype else output.astype(prepared.hidden_dtype)

    def __call__(self, hidden: mx.array) -> mx.array:
        if QKV_SHARING_ENABLED and hasattr(self, "_shared_prepared"):
            return self.forward_prepared(self._shared_prepared)
        value = hidden
        if self.has_smooth:
            value = value / self.smooth.astype(value.dtype)
        if self.rotate:
            value = hadamard(value, self.block_size)
        codes, token_scales = dynamic_quantize_mx(value, self.bits)
        restored = codes.astype(hidden.dtype) * token_scales.astype(hidden.dtype)[..., None]
        output = self.proj(restored)
        return output if output.dtype == hidden.dtype else output.astype(hidden.dtype)


def _float_linear(weight: mx.array, bias: mx.array | None) -> nn.Linear:
    layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
    layer.weight = weight
    if bias is not None:
        layer.bias = bias
    return layer


def _row_linear(spec: QuantizedProjection, compute: mx.Dtype) -> nn.Linear:
    packed = QuantizedWeight(codes=spec.codes, scales=spec.scales, bits=spec.bits, packed=spec.packed)
    weight = code_matrix(packed, mx.float32) * mx.array(np.ascontiguousarray(spec.scales)).reshape((-1, 1))
    return _float_linear(weight.astype(compute), spec.bias)


def _mlx_linear(spec: QuantizedProjection) -> nn.QuantizedLinear:
    out, inn = _projection_width(spec)
    layer = nn.QuantizedLinear(
        inn,
        out,
        bias=spec.bias is not None,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
    )
    layer.weight = spec.qweight
    layer.scales = spec.qscales
    layer.biases = spec.qbiases
    if spec.bias is not None:
        layer.bias = spec.bias
    return layer


def _rms(weight: mx.array) -> nn.RMSNorm:
    norm = nn.RMSNorm(int(weight.shape[0]), eps=RMS_EPS)
    norm.weight = weight
    return norm


def _layer_norm(weight: mx.array, bias: mx.array) -> nn.LayerNorm:
    norm = nn.LayerNorm(int(weight.shape[0]), eps=LAYER_NORM_EPS, bias=True)
    norm.weight = weight
    norm.bias = bias
    return norm


@dataclass
class TextLayer:
    input_norm: nn.RMSNorm
    post_norm: nn.RMSNorm
    q: nn.Module
    k: nn.Module
    v: nn.Module
    o: nn.Module
    gate: nn.Module
    up: nn.Module
    down: nn.Module
    heads: int
    kv_heads: int
    cross_kv: bool


@dataclass
class VisionLayer:
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    q: nn.Module
    k: nn.Module
    v: nn.Module
    out: nn.Module
    fc1: nn.Module
    fc2: nn.Module


def _as_image(image: np.ndarray) -> mx.array:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected CHW image, got {array.shape}")
    return mx.array(np.transpose(array, (1, 2, 0)))


class MLXSmolVLARuntime:
    def __init__(
        self,
        weights: dict[str, mx.array],
        config: dict[str, Any],
        quant: dict[str, QuantizedProjection],
        compute: mx.Dtype = mx.bfloat16,
    ) -> None:
        self.config = config
        self.weights = weights
        self.num_layers = int(config["num_vlm_layers"])
        self.self_attn_every = int(config["self_attn_every_n_layers"])
        self.chunk_size = int(config["chunk_size"])
        self.num_steps = int(config["num_steps"])
        self.max_action_dim = int(config["max_action_dim"])
        self.min_period = float(config["min_period"])
        self.max_period = float(config["max_period"])
        text_q = f"{PREFIX}vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj"
        text_k = f"{PREFIX}vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj"
        q_out, q_in = _matrix_shape(weights, quant, text_q)
        k_out, _ = _matrix_shape(weights, quant, text_k)
        self.hidden = q_in
        self.heads = q_out // HEAD_DIM
        self.kv_heads = k_out // HEAD_DIM
        expert_q_out, expert_q_in = _matrix_shape(weights, quant, f"{PREFIX}vlm_with_expert.lm_expert.layers.0.self_attn.q_proj")
        self.expert_hidden = expert_q_in
        self.expert_heads = expert_q_out // HEAD_DIM
        connector = weights[f"{PREFIX}vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"]
        vision_hidden = int(weights[f"{PREFIX}vlm_with_expert.vlm.model.vision_model.post_layernorm.weight"].shape[0])
        self.scale_factor = int(math.isqrt(int(connector.shape[1] // vision_hidden)))
        self.connector = _float_linear(connector, None)
        patch = weights[f"{PREFIX}vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"]
        self.patch_size = int(patch.shape[1])
        self.patch_embed = nn.Conv2d(
            int(patch.shape[-1]),
            int(patch.shape[0]),
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.patch_embed.weight = patch
        self.patch_embed.bias = weights[f"{PREFIX}vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias"]
        positions = weights[f"{PREFIX}vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight"]
        self.position_embed = nn.Embedding(int(positions.shape[0]), int(positions.shape[1]))
        self.position_embed.weight = positions
        self.vision_norm = _layer_norm(
            weights[f"{PREFIX}vlm_with_expert.vlm.model.vision_model.post_layernorm.weight"],
            weights[f"{PREFIX}vlm_with_expert.vlm.model.vision_model.post_layernorm.bias"],
        )
        self.compute = compute
        self.projections: dict[str, nn.Module] = {}
        self.vision = self._vision_layers(quant)
        self.text = self._text_layers("vlm.model.text_model", quant, cross=False)
        self.expert = self._text_layers("lm_expert", quant, cross=True)
        self.state_proj = self._projection(f"{PREFIX}state_proj", quant)
        self.action_in = self._projection(f"{PREFIX}action_in_proj", quant)
        self.action_out = self._projection(f"{PREFIX}action_out_proj", quant)
        self.time_in = self._projection(f"{PREFIX}action_time_mlp_in", quant)
        self.time_out = self._projection(f"{PREFIX}action_time_mlp_out", quant)
        embed = weights[f"{PREFIX}vlm_with_expert.vlm.model.text_model.embed_tokens.weight"]
        self.embed = nn.Embedding(int(embed.shape[0]), int(embed.shape[1]))
        self.embed.weight = embed
        self.text_norm = _rms(weights[f"{PREFIX}vlm_with_expert.vlm.model.text_model.norm.weight"])
        self.expert_norm = _rms(weights[f"{PREFIX}vlm_with_expert.lm_expert.norm.weight"])

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: str = "bfloat16") -> "MLXSmolVLARuntime":
        root = Path(path)
        weight_file = root / "weights.safetensors"
        if not weight_file.is_file():
            weight_file = root / "model.safetensors"
        if not weight_file.is_file():
            raise FileNotFoundError(f"No safetensors weights in {root}")
        config = json.loads((root / "config.json").read_text())
        loaded = mx.load(str(weight_file))
        weights = {name: _prepare_weight(name, value, dtype) for name, value in loaded.items()}
        quant = _load_quantized(root, weights)
        return cls(weights, config, quant, compute=_COMPUTE[dtype])

    def predict_action_chunk(
        self, processed: ProcessedObservation, noise: np.ndarray | None = None, seed: int | None = None
    ) -> np.ndarray:
        images = mx.stack([_as_image(image) for image in processed.images])[None, ...]
        # Processor stores one observation. Vision still sees every camera, including the empty one.
        image_tokens = self.embed_images(images[0])
        image_masks = mx.array(np.asarray(processed.image_masks, dtype=np.bool_))[None, :]
        tokens = mx.array(np.asarray(processed.language_tokens, dtype=np.int32))[None, :]
        token_mask = mx.array(np.asarray(processed.language_attention_mask, dtype=np.bool_))[None, :]
        state = mx.array(np.asarray(processed.state, dtype=np.float32))[None, :]
        prefix, prefix_pad, prefix_att = self.embed_prefix(image_tokens, image_masks, tokens, token_mask, state)
        prefix_mask = attention_mask_2d(prefix_pad, prefix_att)
        positions = mx.cumsum(prefix_pad.astype(mx.int32), axis=1) - 1
        cache: list[tuple[mx.array, mx.array] | None] = [None] * self.num_layers
        _, cache = self._transformer([prefix, None], positions, prefix_mask, cache, fill_cache=True)
        mx.eval(cache)

        batch = state.shape[0]
        if noise is None:
            if seed is not None:
                mx.random.seed(seed)
            sample = mx.random.normal((batch, self.chunk_size, self.max_action_dim))
        else:
            sample = mx.array(np.asarray(noise, dtype=np.float32))
            if sample.ndim == 2:
                sample = sample[None, ...]
        step = -1.0 / self.num_steps
        current = sample
        prefix_len = prefix.shape[1]
        for index in range(self.num_steps):
            timestep = mx.full((batch,), 1.0 + index * step, dtype=mx.float32)
            velocity = self.denoise_step(prefix_pad, cache, current, timestep)
            current = current + step * velocity
            self._crop_cache(cache, prefix_len)
            mx.eval(current, cache)
        return np.asarray(current.astype(mx.float32))

    def embed_images(self, images: mx.array) -> mx.array:
        """images: [N, H, W, 3] in [-1, 1]. Returns [N, tokens, hidden]."""
        hidden = self._vision(images)
        hidden = pixel_shuffle(hidden, self.scale_factor)
        hidden = self.connector(hidden)
        return hidden * math.sqrt(hidden.shape[-1])

    def embed_prefix(self, image_tokens, image_masks, lang_tokens, lang_masks, state):
        batch = image_tokens.shape[0] if image_tokens.ndim == 4 else 1
        if image_tokens.ndim == 3:
            image_tokens = image_tokens[None, ...]
        pieces = []
        pads = []
        marks: list[int] = []
        cameras = image_tokens.shape[1]
        for camera in range(cameras):
            tokens = image_tokens[:, camera]
            count = tokens.shape[1]
            valid = image_masks[:, camera : camera + 1]
            pieces.append(tokens)
            pads.append(mx.broadcast_to(valid, (batch, count)))
            marks.extend([0] * count)
        language = self.embed(lang_tokens) * math.sqrt(self.hidden)
        pieces.append(language)
        pads.append(lang_masks)
        marks.extend([0] * language.shape[1])
        state_tokens = self.state_proj(state.astype(mx.float32))[:, None, :]
        pieces.append(state_tokens)
        pads.append(mx.ones((batch, 1), dtype=mx.bool_))
        marks.extend([1])
        embeddings = mx.concatenate(pieces, axis=1)
        pad = mx.concatenate(pads, axis=1)
        att = mx.broadcast_to(mx.array(np.asarray(marks, dtype=np.int32))[None, :], pad.shape)
        return embeddings, pad, att

    def denoise_step(self, prefix_pad, cache, actions, timestep) -> mx.array:
        suffix, suffix_pad, suffix_att = self.embed_suffix(actions, timestep)
        suffix_len = suffix.shape[1]
        prefix_len = prefix_pad.shape[1]
        prefix_keep = mx.broadcast_to(prefix_pad[:, None, :], (prefix_pad.shape[0], suffix_len, prefix_len))
        suffix_mask = attention_mask_2d(suffix_pad, suffix_att)
        full_mask = mx.concatenate([prefix_keep, suffix_mask], axis=2)
        offset = mx.sum(prefix_pad.astype(mx.int32), axis=-1)[:, None]
        positions = offset + mx.cumsum(suffix_pad.astype(mx.int32), axis=1) - 1
        outputs, _ = self._transformer([None, suffix], positions, full_mask, cache, fill_cache=False)
        expert_out = outputs[1][:, -self.chunk_size :].astype(mx.float32)
        return self.action_out(expert_out)

    def embed_suffix(self, actions, timestep):
        action_embed = self.action_in(actions.astype(mx.float32))
        time_embed = sinusoidal_time(timestep, self.expert_hidden, self.min_period, self.max_period)
        time_embed = mx.broadcast_to(time_embed[:, None, :], action_embed.shape)
        fused = self.time_out(nn.silu(self.time_in(mx.concatenate([action_embed, time_embed], axis=-1))))
        batch, length, _ = fused.shape
        pad = mx.ones((batch, length), dtype=mx.bool_)
        att = mx.ones((batch, length), dtype=mx.int32)
        return fused, pad, att

    def _transformer(self, streams, positions, mask, cache, fill_cache: bool):
        values = list(streams)
        for layer_idx in range(self.num_layers):
            self_attn = fill_cache or (self.self_attn_every > 0 and layer_idx % self.self_attn_every == 0)
            if self_attn:
                attended = self._self_attention(values, layer_idx, positions, mask, cache, fill_cache or True)
            else:
                attended = self._cross_attention(values, layer_idx, positions, mask, cache)
            values = self._mlp(values, attended, layer_idx)
        outputs = []
        for index, hidden in enumerate(values):
            if hidden is None:
                outputs.append(None)
            else:
                norm = self.text_norm if index == 0 else self.expert_norm
                outputs.append(norm(hidden))
        return outputs, cache

    def _self_attention(self, streams, layer_idx, positions, mask, cache, use_cache: bool):
        queries, keys, values = [], [], []
        layers = (self.text[layer_idx], self.expert[layer_idx])

        # Determine if this layer uses QKV sharing
        # Language layers 0-15: share Q, K, V
        # Expert odd layers (1, 3, 5, ...): share K, V only
        # QKV sharing requires ALL projections to be FoldLinear
        use_qkv_sharing = QKV_SHARING_ENABLED

        for stream_idx, (hidden, layer) in enumerate(zip(streams, layers, strict=True)):
            if hidden is None:
                continue
            normed = layer.input_norm(hidden)

            # Check if all projections support QKV sharing
            supports_sharing = (
                isinstance(layer.q, FoldLinear)
                and isinstance(layer.k, FoldLinear)
                and isinstance(layer.v, FoldLinear)
            )
            use_sharing = use_qkv_sharing and supports_sharing

            if use_sharing:
                # Prepare activation once, reuse for Q, K, V (or K, V only)
                is_expert_odd = stream_idx == 1 and layer_idx % 2 == 1
                if is_expert_odd:
                    # Expert odd: share K, V only (Q is unique)
                    prepared_kv = layer.k.prepare_activation(normed)
                    q_proj = layer.q(normed)
                    queries.append(_heads(q_proj, layer.heads, HEAD_DIM))
                    keys.append(_heads(layer.k.forward_prepared(prepared_kv), layer.kv_heads, HEAD_DIM))
                    values.append(_heads(layer.v.forward_prepared(prepared_kv), layer.kv_heads, HEAD_DIM))
                else:
                    # Language layers or expert even: share Q, K, V
                    prepared = layer.q.prepare_activation(normed)
                    queries.append(_heads(layer.q.forward_prepared(prepared), layer.heads, HEAD_DIM))
                    keys.append(_heads(layer.k.forward_prepared(prepared), layer.kv_heads, HEAD_DIM))
                    values.append(_heads(layer.v.forward_prepared(prepared), layer.kv_heads, HEAD_DIM))
            else:
                # Original path: separate projections
                queries.append(_heads(layer.q(normed), layer.heads, HEAD_DIM))
                keys.append(_heads(layer.k(normed), layer.kv_heads, HEAD_DIM))
                values.append(_heads(layer.v(normed), layer.kv_heads, HEAD_DIM))

        query = mx.concatenate(queries, axis=1)
        key = mx.concatenate(keys, axis=1)
        value = mx.concatenate(values, axis=1)
        length = query.shape[1]
        pos = positions[:, :length] if positions.shape[1] != length else positions
        attn_mask = mask
        if length < mask.shape[1] and mask.shape[1] == mask.shape[2]:
            attn_mask = mask[:, :length, :length]
            pos = positions[:, :length]
        query = apply_rope(query, pos)
        key = apply_rope(key, pos)
        if use_cache:
            key, value = _cache_update(cache, layer_idx, key, value)
        heads = queries[0].shape[2]
        kv_heads = keys[0].shape[2]
        return [attention(query, key, value, attn_mask, heads, kv_heads)]

    def _cross_attention(self, streams, layer_idx, positions, mask, cache):
        expert = self.expert[layer_idx]
        hidden = streams[1]
        normed = expert.input_norm(hidden)
        query = _heads(expert.q(normed), expert.heads, HEAD_DIM)
        cached_k, cached_v = cache[layer_idx]
        flat_k = cached_k.reshape(*cached_k.shape[:2], -1)
        flat_v = cached_v.reshape(*cached_v.shape[:2], -1)
        key = _heads(expert.k(flat_k), expert.kv_heads, HEAD_DIM)
        value = _heads(expert.v(flat_v), expert.kv_heads, HEAD_DIM)
        relative = positions - mx.min(positions, axis=1, keepdims=True)
        query = apply_rope(query, relative)
        attn_mask = mask[:, -hidden.shape[1] :, : key.shape[1]]
        return [None, attention(query, key, value, attn_mask, expert.heads, expert.kv_heads)]

    def _mlp(self, streams, attended, layer_idx):
        layers = (self.text[layer_idx], self.expert[layer_idx])
        outputs = []
        start = 0
        single = len(attended) == 1
        for index, (hidden, layer) in enumerate(zip(streams, layers, strict=True)):
            if hidden is None or layer is None:
                outputs.append(hidden)
                continue
            end = start + hidden.shape[1]
            block = attended[0] if single else attended[index]
            piece = block[:, start:end] if single else block
            if piece.shape[-1] != _proj_in(layer.o):
                raise RuntimeError(f"Attention width {piece.shape[-1]} does not match o_proj")
            residual = hidden + layer.o(piece)
            normed = layer.post_norm(residual)
            mlp = layer.down(nn.silu(layer.gate(normed)) * layer.up(normed))
            outputs.append(residual + mlp)
            start = end if single else 0
        return outputs

    def _vision(self, images: mx.array) -> mx.array:
        patches = self.patch_embed(images.astype(self.patch_embed.weight.dtype))
        batch, height, width, channels = patches.shape
        hidden = patches.reshape(batch, height * width, channels)
        hidden = hidden + self.position_embed(mx.arange(hidden.shape[1]))
        for layer in self.vision:
            residual = hidden
            normed = layer.norm1(hidden)
            query = _heads(layer.q(normed), VISION_HEADS, channels // VISION_HEADS)
            key = _heads(layer.k(normed), VISION_HEADS, channels // VISION_HEADS)
            value = _heads(layer.v(normed), VISION_HEADS, channels // VISION_HEADS)
            attended = attention(query, key, value, None, VISION_HEADS, VISION_HEADS)
            hidden = residual + layer.out(attended)
            residual = hidden
            hidden = residual + layer.fc2(nn.gelu_approx(layer.fc1(layer.norm2(hidden))))
        return self.vision_norm(hidden)

    def _projection(self, name: str, quant: dict[str, QuantizedProjection]) -> nn.Module:
        spec = quant.get(name)
        if spec is None:
            layer = _float_linear(self.weights[name + ".weight"], self.weights.get(name + ".bias"))
        elif spec.kind == "mlx":
            layer = _mlx_linear(spec)
        else:
            inner = _mlx_linear(spec) if spec.qweight is not None else _row_linear(spec, self.compute)
            layer = FoldLinear(spec, inner)
        self.projections[name] = layer
        return layer

    def _text_layers(self, group: str, quant: dict[str, QuantizedProjection], cross: bool) -> list[TextLayer]:
        layers = []
        root = f"{PREFIX}vlm_with_expert.{group}"
        for index in range(self.num_layers):
            base = f"{root}.layers.{index}"
            q_name = base + ".self_attn.q_proj"
            k_name = base + ".self_attn.k_proj"
            q_out, q_in = _matrix_shape(self.weights, quant, q_name)
            k_out, k_in = _matrix_shape(self.weights, quant, k_name)
            layers.append(
                TextLayer(
                    input_norm=_rms(self.weights[base + ".input_layernorm.weight"]),
                    post_norm=_rms(self.weights[base + ".post_attention_layernorm.weight"]),
                    q=self._projection(q_name, quant),
                    k=self._projection(k_name, quant),
                    v=self._projection(base + ".self_attn.v_proj", quant),
                    o=self._projection(base + ".self_attn.o_proj", quant),
                    gate=self._projection(base + ".mlp.gate_proj", quant),
                    up=self._projection(base + ".mlp.up_proj", quant),
                    down=self._projection(base + ".mlp.down_proj", quant),
                    heads=q_out // HEAD_DIM,
                    kv_heads=k_out // HEAD_DIM,
                    cross_kv=cross and k_in != q_in,
                )
            )
        return layers

    def _vision_layers(self, quant: dict[str, QuantizedProjection]) -> list[VisionLayer]:
        layers = []
        root = f"{PREFIX}vlm_with_expert.vlm.model.vision_model.encoder.layers"
        count = len({name.split(".")[7] for name in self.weights if name.startswith(root)})
        for index in range(count):
            base = f"{root}.{index}"
            layers.append(
                VisionLayer(
                    norm1=_layer_norm(self.weights[base + ".layer_norm1.weight"], self.weights[base + ".layer_norm1.bias"]),
                    norm2=_layer_norm(self.weights[base + ".layer_norm2.weight"], self.weights[base + ".layer_norm2.bias"]),
                    q=self._projection(base + ".self_attn.q_proj", quant),
                    k=self._projection(base + ".self_attn.k_proj", quant),
                    v=self._projection(base + ".self_attn.v_proj", quant),
                    out=self._projection(base + ".self_attn.out_proj", quant),
                    fc1=self._projection(base + ".mlp.fc1", quant),
                    fc2=self._projection(base + ".mlp.fc2", quant),
                )
            )
        return layers

    @staticmethod
    def _crop_cache(cache, length: int) -> None:
        for index, item in enumerate(cache):
            if item is not None and item[0].shape[1] > length:
                cache[index] = (item[0][:, :length], item[1][:, :length])


def _heads(value: mx.array, heads: int, dim: int) -> mx.array:
    batch, length, _ = value.shape
    return value.reshape(batch, length, heads, dim)


def _projection_width(spec: QuantizedProjection) -> tuple[int, int]:
    if spec.qscales is not None:
        return int(spec.qscales.shape[0]), int(spec.qscales.shape[1]) * spec.group_size
    return int(spec.codes.shape[0]), int(spec.codes.shape[1])


def _matrix_shape(weights: dict[str, mx.array], quant: dict[str, QuantizedProjection], name: str) -> tuple[int, int]:
    weight = weights.get(name + ".weight")
    if weight is not None:
        return int(weight.shape[0]), int(weight.shape[1])
    return _projection_width(quant[name])


def _proj_in(proj: nn.Module) -> int:
    if isinstance(proj, FoldLinear):
        return _proj_in(proj.proj)
    if isinstance(proj, nn.QuantizedLinear):
        return int(proj.scales.shape[1]) * int(proj.group_size)
    return int(proj.weight.shape[1])


def _cache_update(cache, layer_idx, key, value):
    current = cache[layer_idx]
    if current is None:
        cache[layer_idx] = (key, value)
    else:
        cache[layer_idx] = (
            mx.concatenate([current[0], key], axis=1),
            mx.concatenate([current[1], value], axis=1),
        )
    return cache[layer_idx]


def _prepare_weight(name: str, value: mx.array, dtype: str) -> mx.array:
    if value.ndim == 4 and name.endswith("weight") and value.shape[1] == 3:
        value = mx.transpose(value, (0, 2, 3, 1))
    if dtype == "float16" and value.dtype == mx.bfloat16:
        return value.astype(mx.float16)
    if dtype == "float32" and value.dtype == mx.bfloat16:
        return value.astype(mx.float32)
    return value


def _load_quantized(root: Path, weights: dict[str, mx.array]) -> dict[str, QuantizedProjection]:
    manifest_path = root / "quantization.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text())
    loaded: dict[str, QuantizedProjection] = {}
    scheme = str(manifest.get("scheme", ""))
    for name, spec in manifest["layers"].items():
        bias = weights.pop(name + ".bias", None)
        smooth = weights.pop(name + ".smooth_scale", None)
        if scheme.startswith("mlx") or name + ".qweight" in weights:
            loaded[name] = QuantizedProjection(
                kind="mlx" if scheme.startswith("mlx") else "fold",
                bits=int(spec["bits"]),
                bias=bias,
                qweight=weights.pop(name + ".qweight"),
                qscales=weights.pop(name + ".qscales"),
                qbiases=weights.pop(name + ".qbiases"),
                group_size=int(spec.get("group_size", 64)),
                mode=str(spec.get("mode", "affine")),
                smooth_scale=None if smooth is None else np.asarray(smooth),
                rotate=bool(spec.get("rotate", False)),
                block_size=int(spec.get("block_size", 1)),
            )
            continue
        loaded[name] = QuantizedProjection(
            kind="fold",
            bits=int(spec["bits"]),
            bias=bias,
            codes=np.asarray(weights.pop(name + ".codes")),
            scales=np.asarray(weights.pop(name + ".row_scales")),
            packed=np.asarray(weights.pop(name + ".packed")) if spec["bits"] == 4 else None,
            smooth_scale=np.asarray(smooth),
            rotate=bool(spec["rotate"]),
            block_size=int(spec["block_size"]),
        )
        weights.pop(name + ".weight", None)
    return loaded
