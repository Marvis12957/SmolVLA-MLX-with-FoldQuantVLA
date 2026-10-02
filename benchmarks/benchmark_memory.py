"""Measure MLX peak memory for one SmolVLA forward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "models" / "smolvla"))
from deployment.session import InferenceSession, synthetic_observation
from metrics import assert_metal, mlx_memory_bytes, reset_mlx_peak_memory, write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure SmolVLA MLX peak memory")
    parser.add_argument("--artifact", default="checkpoints/raw/smolvla/base")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", default="results/baseline/memory_bf16.json")
    args = parser.parse_args(argv)
    assert_metal()
    reset_mlx_peak_memory()
    session = InferenceSession(args.artifact, dtype=args.dtype)
    session.runtime.num_steps = args.steps
    actions = session.predict(synthetic_observation(0), "pick up the block", seed=0)
    mx.eval(mx.array(actions))
    peak_bytes, active_bytes = mlx_memory_bytes()
    payload = {
        "dtype": args.dtype,
        "steps": args.steps,
        "peak_bytes": peak_bytes,
        "active_bytes": active_bytes,
        "action_shape": list(actions.shape),
    }
    path = write_report(Path(args.output).parent, Path(args.output).name, payload)
    print(path)


if __name__ == "__main__":
    main()
