"""Phase 0 hardware audit and SmolVLA module-tree inspection."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
from collections import defaultdict
from pathlib import Path

from smolvla_mlx.constants import LOCAL_CHECKPOINT


def command_audit(args: argparse.Namespace) -> None:
    tree = _module_tree(Path(args.checkpoint))
    payload = {
        **_hardware(),
        "mlx_quantization": _mlx_quantization_api(),
        "checkpoint": str(args.checkpoint),
        "parameters": tree["parameters"],
        "tensors": tree["tensors"],
        "subsystems": tree["subsystems"],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    tree_path = Path(args.tree_output)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(json.dumps(payload["module_tree"], indent=2) + "\n")
    print(destination)
    print(tree_path)
    print(json.dumps({key: payload[key] for key in ("chip", "memory_gb", "macos", "mlx_version", "python", "metal_available")}, indent=2))
    print(json.dumps(payload["mlx_quantization"], indent=2))
    print(json.dumps(payload["subsystems"], indent=2))


def _hardware() -> dict:
    from metrics import environment_metadata, git_commit

    meta = environment_metadata()
    hardware = str(meta.get("hardware", ""))
    chip = hardware.split(",")[0].strip() if hardware else "unknown"
    memory_gb = 0
    if "GB" in hardware:
        token = hardware.split(",")[-1].replace("GB", "").strip().split()[0]
        try:
            memory_gb = int(float(token))
        except ValueError:
            memory_gb = 0
    metal_available = False
    try:
        import mlx.core as mx

        metal = getattr(mx, "metal", None)
        metal_available = bool(metal is not None and getattr(metal, "is_available", lambda: False)())
    except Exception:
        metal_available = False
    return {
        "chip": chip,
        "memory_gb": memory_gb,
        "macos": platform.mac_ver()[0] or meta.get("platform"),
        "mlx_version": meta.get("mlx"),
        "python": meta.get("python"),
        "metal_available": metal_available,
        "platform": meta.get("platform"),
        "commit": git_commit(),
    }


def _mlx_quantization_api() -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    def present(owner, name: str) -> dict:
        value = getattr(owner, name, None)
        payload = {"available": value is not None}
        if value is not None and callable(value):
            try:
                payload["signature"] = str(inspect.signature(value))
            except (TypeError, ValueError):
                payload["signature"] = None
        return payload

    bits: list[int] = []
    group_sizes: list[int] = []
    if hasattr(nn, "QuantizedLinear"):
        linear = nn.Linear(256, 128)
        for bit in (2, 3, 4, 5, 6, 8):
            for group in (32, 64, 128):
                try:
                    nn.QuantizedLinear.from_linear(linear, bits=bit, group_size=group)
                    if bit not in bits:
                        bits.append(bit)
                    if group not in group_sizes:
                        group_sizes.append(group)
                except Exception:
                    continue
    return {
        "QuantizedLinear": present(nn, "QuantizedLinear"),
        "quantize_module": present(nn, "quantize"),
        "quantize": present(mx, "quantize"),
        "quantized_matmul": present(mx, "quantized_matmul"),
        "hadamard_transform": present(mx, "hadamard_transform"),
        "QQLinear": present(nn, "QQLinear"),
        "qqmm": present(mx, "qqmm"),
        "quantized_linear_bits": bits,
        "quantized_linear_group_sizes": group_sizes,
        "reset_peak_memory": present(mx, "reset_peak_memory"),
        "get_peak_memory": present(mx, "get_peak_memory"),
        "get_active_memory": present(mx, "get_active_memory"),
    }


def _module_tree(checkpoint: Path) -> dict:
    from safetensors import safe_open

    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        weights = checkpoint / "weights.safetensors"
    layers: list[dict] = []
    with safe_open(weights, framework="pt") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            layers.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "parameters": int(tensor.numel()),
                }
            )
    subsystems: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for layer in layers:
        group = _subsystem(layer["name"])
        grouped[group].append(layer)
    for group, items in grouped.items():
        subsystems[group] = {
            "tensors": len(items),
            "parameters": sum(item["parameters"] for item in items),
            "bytes": sum(_nbytes(item) for item in items),
        }
    return {
        "weight_file": str(weights),
        "tensors": len(layers),
        "parameters": sum(item["parameters"] for item in layers),
        "subsystems": subsystems,
        "layers": layers,
    }


def _subsystem(name: str) -> str:
    if "vision_model" in name or "vision_tower" in name:
        return "vision"
    if ".connector." in name or name.endswith("connector") or ".connector." in f".{name}.":
        return "connector"
    if "embed" in name or "lm_head" in name:
        return "embeddings"
    if "norm" in name:
        return "normalization"
    if "lm_expert" in name:
        if "mlp" in name:
            return "expert_mlp"
        if any(token in name for token in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attn")):
            return "expert_attention"
        return "expert_other"
    if "text_model" in name:
        if "mlp" in name:
            return "vlm_mlp"
        if any(token in name for token in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attn")):
            return "vlm_attention"
        return "vlm_other"
    if "state_proj" in name:
        return "state_proj"
    if "action_in_proj" in name:
        return "action_in_proj"
    if "action_out_proj" in name:
        return "action_out_proj"
    if "action_time_mlp" in name:
        return "action_time_mlp"
    return "other"


def _nbytes(layer: dict) -> int:
    itemsize = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1, "uint32": 4}.get(layer["dtype"], 2)
    return int(layer["parameters"]) * itemsize


def register(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("audit", help="Record hardware, MLX quant APIs, and SmolVLA module tree")
    command.add_argument("--checkpoint", default=LOCAL_CHECKPOINT)
    command.add_argument("--output", default="results/system.json")
    command.add_argument("--tree-output", default="results/baseline/module_tree.json")
    command.set_defaults(func=command_audit)
