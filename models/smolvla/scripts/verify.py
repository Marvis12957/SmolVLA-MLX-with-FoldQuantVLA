"""Compare an MLX action chunk with a saved reference or a PyTorch forward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_ROOT / "benchmarks"))
from metrics import action_metrics


def command_verify(args: argparse.Namespace) -> None:
    if args.pytorch:
        report = _compare_pytorch(args)
    else:
        candidate = np.load(args.candidate) if args.candidate else _mlx_robot_actions(args, _shared_noise(args))
        reference = np.load(args.reference)
        report = {"unnormalized": action_metrics(reference, candidate)}
    print(json.dumps(report, indent=2, sort_keys=True))
    normalized = report.get("normalized", report.get("unnormalized"))
    if normalized["max_abs"] > args.max_abs or normalized["mae"] > args.mean_abs:
        raise SystemExit(
            f"drift max_abs {normalized['max_abs']} mae {normalized['mae']} exceeds {args.max_abs} / {args.mean_abs}"
        )


def _chw_observation(observation: dict) -> dict:
    """LeRobot's image resize expects channel-first float images in [0, 1]."""
    prepared = {}
    for key, value in observation.items():
        if not str(key).startswith("observation.images"):
            prepared[key] = value
            continue
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))
        array = array.astype(np.float32, copy=False)
        if float(array.max(initial=0.0)) > 1.0:
            array = array / 255.0
        prepared[key] = array[None, ...]
    if "observation.state" in prepared:
        state = np.asarray(prepared["observation.state"], dtype=np.float32)
        prepared["observation.state"] = state if state.ndim == 2 else state[None, ...]
    return prepared


def _shared_noise(args: argparse.Namespace) -> np.ndarray:
    return np.random.default_rng(args.seed).standard_normal((1, 50, 32)).astype(np.float32)


def _mlx_robot_actions(args: argparse.Namespace, noise: np.ndarray) -> np.ndarray:
    from deployment.session import InferenceSession, synthetic_observation

    session = InferenceSession(args.artifact, dtype=args.dtype)
    session.runtime.num_steps = args.steps
    return session.predict(synthetic_observation(args.seed), args.task, noise=noise)


def _compare_pytorch(args: argparse.Namespace) -> dict:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    from deployment.session import InferenceSession, synthetic_observation
    from smolvla_mlx.processor import PolicyProcessor

    raw = synthetic_observation(args.seed)
    noise = _shared_noise(args)
    session = InferenceSession(args.artifact, dtype=args.dtype)
    session.runtime.num_steps = args.steps
    mlx_normalized = session.predict_normalized(raw, args.task, noise=noise)
    observation = _chw_observation(raw)
    observation = {
        key: torch.from_numpy(np.asarray(value)) if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }
    observation["task"] = args.task

    policy = SmolVLAPolicy.from_pretrained(str(args.artifact)).float().to("cpu")
    policy.eval()
    policy.config.num_steps = args.steps
    preprocessor = DataProcessorPipeline.from_pretrained(
        str(args.artifact),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    batch = preprocessor(observation)
    batch = {key: value.to("cpu") if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        pytorch = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise))
    pytorch_normalized = pytorch.float().cpu().numpy()
    processor = PolicyProcessor.from_pretrained(args.artifact)
    return {
        "normalized": action_metrics(pytorch_normalized, mlx_normalized),
        "unnormalized": action_metrics(
            processor.unnormalize_action(pytorch_normalized),
            processor.unnormalize_action(mlx_normalized),
        ),
    }


def register(subparsers: argparse._SubParsersAction) -> None:
    verify = subparsers.add_parser("verify", help="Compare PyTorch and MLX action chunks")
    verify.add_argument("--artifact", default="checkpoints/raw/smolvla/base")
    verify.add_argument("--reference", type=Path)
    verify.add_argument("--candidate", type=Path)
    verify.add_argument("--pytorch", action="store_true")
    verify.add_argument("--dtype", default="bfloat16")
    verify.add_argument("--steps", type=int, default=10)
    verify.add_argument("--seed", type=int, default=0)
    verify.add_argument("--task", default="pick up the block")
    verify.add_argument("--max-abs", type=float, default=0.02)
    verify.add_argument("--mean-abs", type=float, default=0.005)
    verify.set_defaults(func=command_verify)
