"""Compare an MLX candidate against the frozen PyTorch reference chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smolvla_mlx.constants import LOCAL_CHECKPOINT


def command_evaluate_fidelity(args: argparse.Namespace) -> None:
    from deployment.session import InferenceSession
    from metrics import action_metrics, write_report
    from smolvla_mlx.processor import PolicyProcessor

    session = InferenceSession(args.artifact, dtype=args.dtype)
    session.runtime.num_steps = args.steps
    processor = PolicyProcessor.from_pretrained(args.artifact)
    rows = []
    for seed_dir in sorted(Path(args.inputs).glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        metadata = json.loads((seed_dir / "metadata.json").read_text())
        observation = {
            "observation.images.camera1": np.load(seed_dir / "camera1.npy"),
            "observation.images.camera2": np.load(seed_dir / "camera2.npy"),
            "observation.state": np.load(seed_dir / "state.npy"),
        }
        noise = np.load(seed_dir / "noise.npy")
        predicted = session.predict_normalized(observation, metadata["task"], noise=noise)
        reference = np.load(Path(args.outputs) / seed_dir.name / "pytorch_fp32_normalized.npy")
        width = min(predicted.shape[-1], reference.shape[-1])
        predicted = predicted[..., :width]
        reference = reference[..., :width]
        normalized = action_metrics(reference, predicted)
        unnormalized = action_metrics(
            processor.unnormalize_action(reference),
            processor.unnormalize_action(predicted),
        )
        rows.append({"seed": seed, "normalized": normalized, "unnormalized": unnormalized})
        print(f"seed {seed} cosine {normalized['cosine']:.6f} mae {normalized['mae']:.6f} max_abs {normalized['max_abs']:.6f}")
    if not rows:
        raise SystemExit(f"No frozen references in {args.inputs}")
    payload = {
        "artifact": str(args.artifact),
        "dtype": args.dtype,
        "steps": args.steps,
        "median_cosine": float(np.median([row["normalized"]["cosine"] for row in rows])),
        "min_cosine": float(np.min([row["normalized"]["cosine"] for row in rows])),
        "mae": float(np.mean([row["normalized"]["mae"] for row in rows])),
        "max_abs": float(np.max([row["normalized"]["max_abs"] for row in rows])),
        "seeds": rows,
    }
    path = write_report(Path(args.report).parent, Path(args.report).name, payload)
    print(path)
    if payload["median_cosine"] < args.min_cosine:
        raise SystemExit(f"median cosine {payload['median_cosine']} below {args.min_cosine}")


def register(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("evaluate-fidelity", help="Compare MLX actions with frozen PyTorch references")
    command.add_argument("--artifact", default=LOCAL_CHECKPOINT)
    command.add_argument("--dtype", default="bfloat16")
    command.add_argument("--steps", type=int, default=10)
    command.add_argument("--inputs", default="reference_inputs")
    command.add_argument("--outputs", default="reference_outputs")
    command.add_argument("--report", default="results/baseline/mlx_bf16_vs_pytorch.json")
    command.add_argument("--min-cosine", type=float, default=0.999)
    command.set_defaults(func=command_evaluate_fidelity)
