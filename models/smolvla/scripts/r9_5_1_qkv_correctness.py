"""R9.5.1 focused QKV-sharing parity and dispatch investigation.

This is deliberately a read-only diagnostic: it loads the locked artifact,
uses saved reference observations/noise, and never changes artifact contents.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "models" / "smolvla" / "src"), str(ROOT / "models" / "smolvla"), str(ROOT / "runtime" / "mlx")]

import smolvla_mlx.runtime as runtime_module
from smolvla_mlx.processor import PolicyProcessor
from smolvla_mlx.runtime import FoldLinear, MLXSmolVLARuntime, _heads
from blocks import attention, attention_mask_2d, apply_rope


ARTIFACT = ROOT / "results/quantization/clean_track/artifacts/fq-pad768-language-downproj-w8"


def metrics(left: mx.array, right: mx.array) -> dict[str, float]:
    a, b = np.asarray(left, dtype=np.float32).ravel(), np.asarray(right, dtype=np.float32).ravel()
    diff = a - b
    return {
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(b)),
    }


def observation(seed: int) -> tuple[dict[str, Any], str, np.ndarray]:
    path = ROOT / "reference_inputs" / f"seed_{seed}"
    meta = json.loads((path / "metadata.json").read_text())
    state = np.load(path / "state.npy")
    if state.shape[-1] < 8:
        state = np.pad(state, (0, 8 - state.shape[-1]))
    # The locked LIBERO artifact expects its padded processor representation.
    return ({
        "observation.images.camera1": np.load(path / "camera1.npy"),
        "observation.images.camera2": np.load(path / "camera2.npy"),
        "observation.images.camera3": np.load(path / "camera3.npy"),
        "observation.state": state,
    }, meta.get("task", "test task"), np.random.default_rng(10_000 + seed).standard_normal((1, 50, 32)).astype(np.float32))


def prefix_inputs(runtime: MLXSmolVLARuntime, processed):
    images = mx.stack([mx.array(np.transpose(np.asarray(image, dtype=np.float32), (1, 2, 0))) for image in processed.images])[None]
    image_tokens = runtime.embed_images(images[0])
    image_masks = mx.array(np.asarray(processed.image_masks, dtype=np.bool_))[None]
    tokens = mx.array(np.asarray(processed.language_tokens, dtype=np.int32))[None]
    token_mask = mx.array(np.asarray(processed.language_attention_mask, dtype=np.bool_))[None]
    state = mx.array(np.asarray(processed.state, dtype=np.float32))[None]
    prefix, pad, att = runtime.embed_prefix(image_tokens, image_masks, tokens, token_mask, state)
    positions = mx.cumsum(pad.astype(mx.int32), axis=1) - 1
    cache: list = [None] * runtime.num_layers
    _, cache = runtime._transformer([prefix, None], positions, attention_mask_2d(pad, att), cache, fill_cache=True)
    mx.eval(cache)
    return prefix, pad, positions, cache


def projection_parity(runtime, processed) -> dict[str, Any]:
    _, pad, _, cache = prefix_inputs(runtime, processed)
    actions = mx.array(np.random.default_rng(123).standard_normal((1, 50, 32)).astype(np.float32))
    suffix, suffix_pad, suffix_att = runtime.embed_suffix(actions, mx.array([0.5], dtype=mx.float32))
    prefix_keep = mx.broadcast_to(pad[:, None, :], (1, suffix.shape[1], pad.shape[1]))
    mask = mx.concatenate([prefix_keep, attention_mask_2d(suffix_pad, suffix_att)], axis=2)
    positions = mx.sum(pad.astype(mx.int32), axis=-1)[:, None] + mx.cumsum(suffix_pad.astype(mx.int32), axis=1) - 1
    result: dict[str, Any] = {"even_self_attention": {}, "odd_cross_attention": {}}
    for index in range(runtime.num_layers):
        layer = runtime.expert[index]
        if index % 2 == 0:
            normed = layer.input_norm(suffix)
            direct = {name: getattr(layer, name)(normed) for name in ("q", "k", "v")}
            prepared = layer.q.prepare_activation(normed)
            shared = {"q": layer.q.forward_prepared(prepared), "k": layer.k.forward_prepared(prepared), "v": layer.v.forward_prepared(prepared)}
            # attention parity, retaining the prefix cache exactly as runtime does.
            dq, dk, dv = (_heads(direct[n], layer.heads if n == "q" else layer.kv_heads, 64) for n in ("q", "k", "v"))
            sq, sk, sv = (_heads(shared[n], layer.heads if n == "q" else layer.kv_heads, 64) for n in ("q", "k", "v"))
            dq, dk, sq, sk = apply_rope(dq, positions), apply_rope(dk, positions), apply_rope(sq, positions), apply_rope(sk, positions)
            direct_attn = attention(dq, mx.concatenate([cache[index][0], dk], axis=1), mx.concatenate([cache[index][1], dv], axis=1), mask, layer.heads, layer.kv_heads)
            shared_attn = attention(sq, mx.concatenate([cache[index][0], sk], axis=1), mx.concatenate([cache[index][1], sv], axis=1), mask, layer.heads, layer.kv_heads)
            result["even_self_attention"][str(index)] = {name: metrics(direct[name], shared[name]) for name in ("q", "k", "v")}
            result["even_self_attention"][str(index)]["attention"] = metrics(direct_attn, shared_attn)
        else:
            ck, cv = cache[index]
            flat_k, flat_v = ck.reshape(*ck.shape[:2], -1), cv.reshape(*cv.shape[:2], -1)
            # Actual cross path is independent; contrasting a deliberately wrong K-input reuse proves the distinction.
            k = layer.k(flat_k)
            v = layer.v(flat_v)
            wrong_v = layer.v.forward_prepared(layer.k.prepare_activation(flat_k))
            result["odd_cross_attention"][str(index)] = {"k_shape": list(k.shape), "v_shape": list(v.shape), "wrong_k_input_reused_for_v": metrics(v, wrong_v)}
    mx.eval(mx.array(0))
    return result


def action_parity(runtime, processor, seeds: list[int]) -> dict[str, Any]:
    original = runtime_module.QKV_SHARING_ENABLED
    output: dict[str, Any] = {}
    try:
        for seed in seeds:
            obs, task, noise = observation(seed)
            processed = processor.prepare(obs, task)
            runtime_module.QKV_SHARING_ENABLED = True
            shared = runtime.predict_action_chunk(processed, noise=noise)
            runtime_module.QKV_SHARING_ENABLED = False
            independent = runtime.predict_action_chunk(processed, noise=noise)
            output[str(seed)] = metrics(mx.array(shared), mx.array(independent))
    finally:
        runtime_module.QKV_SHARING_ENABLED = original
    return output


def call_counts(runtime, processor) -> dict[str, int]:
    counts: Counter[str] = Counter()
    ids = {id(proj): name for name, proj in runtime.projections.items() if isinstance(proj, FoldLinear)}
    original_call = FoldLinear.__call__
    original_prepare = FoldLinear.prepare_activation
    original_forward = FoldLinear.forward_prepared
    def wrapped(self, hidden):
        counts[f"call:{ids[id(self)]}"] += 1
        return original_call(self, hidden)
    def prepared(self, hidden):
        counts[f"prepare:{ids[id(self)]}"] += 1
        return original_prepare(self, hidden)
    def forwarded(self, activation):
        counts[f"forward_prepared:{ids[id(self)]}"] += 1
        return original_forward(self, activation)
    FoldLinear.__call__ = wrapped
    FoldLinear.prepare_activation = prepared
    FoldLinear.forward_prepared = forwarded
    try:
        obs, task, noise = observation(0)
        runtime.predict_action_chunk(processor.prepare(obs, task), noise=noise)
    finally:
        FoldLinear.__call__ = original_call
        FoldLinear.prepare_activation = original_prepare
        FoldLinear.forward_prepared = original_forward
    odd = [f"model.vlm_with_expert.lm_expert.layers.{index}.self_attn" for index in range(1, runtime.num_layers, 2)]
    return {
        "all": dict(counts),
        "odd_cross_k_calls": sum(counts[f"call:{name}.k_proj"] for name in odd),
        "odd_cross_v_calls": sum(counts[f"call:{name}.v_proj"] for name in odd),
        "odd_cross_k_prepare": sum(counts[f"prepare:{name}.k_proj"] for name in odd),
        "odd_cross_v_prepare": sum(counts[f"prepare:{name}.v_proj"] for name in odd),
    }


def benchmark(runtime, processor, seed: int, iterations: int) -> dict[str, float]:
    obs, task, noise = observation(seed)
    processed = processor.prepare(obs, task)
    for _ in range(5):
        mx.eval(mx.array(runtime.predict_action_chunk(processed, noise=noise)))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        value = runtime.predict_action_chunk(processed, noise=noise)
        mx.eval(mx.array(value))
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"p50_ms": float(np.percentile(samples, 50)), "p90_ms": float(np.percentile(samples, 90)), "mean_ms": float(np.mean(samples)), "iterations": iterations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    processor, runtime = PolicyProcessor.from_pretrained(ARTIFACT), MLXSmolVLARuntime.from_pretrained(ARTIFACT)
    obs, task, _ = observation(0)
    processed = processor.prepare(obs, task)
    result: dict[str, Any] = {
        "artifact": str(ARTIFACT), "flags": {"QKV_SHARING_ENABLED": runtime_module.QKV_SHARING_ENABLED, "compile": False},
        "projection_parity": projection_parity(runtime, processed),
        "action_parity_shared_vs_independent": action_parity(runtime, processor, [0, 1, 2]),
        "call_counts": call_counts(runtime, processor),
        "dispatch": {"foldlinear_count": sum(isinstance(x, FoldLinear) for x in runtime.projections.values()), "mlx_quantized_count": sum(type(getattr(x, "proj", None)).__name__ == "QuantizedLinear" for x in runtime.projections.values())},
    }
    if args.benchmark:
        original = runtime_module.QKV_SHARING_ENABLED
        try:
            runtime_module.QKV_SHARING_ENABLED = True; result["benchmark_shared"] = benchmark(runtime, processor, 0, 50)
            runtime_module.QKV_SHARING_ENABLED = False; result["benchmark_independent"] = benchmark(runtime, processor, 0, 50)
        finally:
            runtime_module.QKV_SHARING_ENABLED = original
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
