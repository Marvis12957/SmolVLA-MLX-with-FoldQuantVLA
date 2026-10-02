"""Convert a PyTorch SmolVLA checkpoint into an MLX artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smolvla_mlx.converter import convert_checkpoint


def command_convert(args: argparse.Namespace) -> None:
    report = convert_checkpoint(Path(args.source).expanduser(), Path(args.destination).expanduser())
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))


def register(subparsers: argparse._SubParsersAction) -> None:
    convert = subparsers.add_parser("convert", help="Convert checkpoint tensors and write an MLX artifact")
    convert.add_argument("--source", default="checkpoints/raw/smolvla/base")
    convert.add_argument("--destination", default="checkpoints/mlx/smolvla/bf16")
    convert.set_defaults(func=command_convert)
