"""Build a FoldQuant quantized artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantization.artifact import quantize_checkpoint


def _activation_amax(args: argparse.Namespace) -> dict[str, np.ndarray] | None:
    if args.amax:
        loaded = np.load(args.amax)
        return {key: loaded[key] for key in loaded.files}
    if args.scheme.startswith("mlx"):
        return None
    from quantization.collect import collect_activation_amax

    observations = None
    if args.dataset:
        import json

        from quantization.dataset import load_calibration_observations

        count = args.samples if args.samples > 0 else 128
        split, observations = load_calibration_observations(args.dataset, seed=args.seed, num_calibration=count)
        split_path = Path(args.amax_out).with_name("calibration_split.json") if args.amax_out else Path("results/smolvla_mlx/calibration_split.json")
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split.to_manifest(), indent=2, sort_keys=True) + "\n")
    elif args.samples < 1:
        return None
    stats = collect_activation_amax(args.source, args.samples, args.task, observations=observations)
    if args.amax_out:
        destination = Path(args.amax_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **stats)
    return stats


def command_quantize(args: argparse.Namespace) -> None:
    report = quantize_checkpoint(
        args.source,
        args.destination,
        scheme=args.scheme,
        block_size=args.block_size,
        group_size=args.group_size,
        activation_amax=_activation_amax(args),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def register(subparsers: argparse._SubParsersAction) -> None:
    quantize = subparsers.add_parser("quantize", help="Build an MLX or FoldQuant quantized artifact")
    quantize.add_argument("--source", default="checkpoints/raw/smolvla/ai20k")
    quantize.add_argument("--destination", default="checkpoints/mlx/smolvla/w8a8")
    quantize.add_argument("--scheme", choices=("mlx-w8", "mlx-w4", "w8a8", "w4a4", "w4a4-selective"), default="w8a8")
    quantize.add_argument("--block-size", type=int, default=64)
    quantize.add_argument("--group-size", type=int, default=64)
    quantize.add_argument("--samples", type=int, default=0, help="Synthetic BF16 forwards used to fit SmoothQuant scales")
    quantize.add_argument("--task", default="pick up the block")
    quantize.add_argument("--dataset", default="", help="Local LeRobot root used for episode-disjoint SmoothQuant calibration")
    quantize.add_argument("--seed", type=int, default=1000)
    quantize.add_argument("--amax", default="", help="Reuse a saved activation-amax npz")
    quantize.add_argument("--amax-out", default="", help="Where to write activation amax when --samples is set")
    quantize.set_defaults(func=command_quantize)
