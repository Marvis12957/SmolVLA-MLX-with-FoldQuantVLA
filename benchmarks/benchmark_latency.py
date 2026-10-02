"""Measure one SmolVLA MLX action-chunk forward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "models" / "smolvla"))
from deployment.session import InferenceSession, synthetic_observation
from metrics import latency_report, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure SmolVLA MLX inference latency")
    parser.add_argument("--artifact", default="checkpoints/raw/smolvla/base")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="results/smolvla_mlx/latency.json")
    args = parser.parse_args(argv)
    session = InferenceSession(args.artifact, dtype=args.dtype)
    session.runtime.num_steps = args.steps
    observation = synthetic_observation(0)

    def run() -> None:
        session.predict(observation, "pick up the block", seed=0)
        mx.eval(mx.zeros((1,)))

    report = latency_report(run, lambda: None, warmups=args.warmups, iterations=args.iterations)
    path = write_report(
        Path(args.output).parent,
        Path(args.output).name,
        {"dtype": args.dtype, "steps": args.steps, "latency": report.__dict__},
    )
    print(path)


if __name__ == "__main__":
    main()
