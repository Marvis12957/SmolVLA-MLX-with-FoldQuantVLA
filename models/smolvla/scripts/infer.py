"""Run MLX action-chunk inference on a local artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deployment.session import InferenceSession, synthetic_observation


def command_infer(args: argparse.Namespace) -> None:
    session = InferenceSession(args.artifact, dtype=args.dtype)
    observation = synthetic_observation(args.seed)
    if args.camera1 is not None:
        from PIL import Image

        observation["observation.images.camera1"] = np.asarray(Image.open(args.camera1).convert("RGB"))
        observation["observation.images.camera2"] = np.asarray(Image.open(args.camera2).convert("RGB"))
    if args.state is not None:
        observation["observation.state"] = np.asarray(args.state, dtype=np.float32)
    session.runtime.num_steps = args.steps
    normalized = session.predict_normalized(observation, args.task, seed=args.seed)
    actions = session.processor.unnormalize_action(normalized)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, actions)
    stem = destination.stem[: -len("_actions")] if destination.stem.endswith("_actions") else destination.stem
    normalized_path = destination.with_name(f"{stem}_normalized.npy")
    np.save(normalized_path, normalized)
    print(destination, actions.shape)
    print(normalized_path, normalized.shape)


def register(subparsers: argparse._SubParsersAction) -> None:
    infer = subparsers.add_parser("infer", help="Run MLX action-chunk inference")
    infer.add_argument("--artifact", default="checkpoints/raw/smolvla/base")
    infer.add_argument("--task", default="pick up the block")
    infer.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    infer.add_argument("--steps", type=int, default=10)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--camera1", type=Path)
    infer.add_argument("--camera2", type=Path)
    infer.add_argument("--state", type=float, nargs=6)
    infer.add_argument("--output", default="results/smolvla_mlx/actions.npy")
    infer.set_defaults(func=command_infer)
