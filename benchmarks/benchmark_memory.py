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
from metrics import write_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure SmolVLA MLX peak memory")
    parser.add_argument("--artifact", default="checkpoints/raw/smolvla/base")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", default="results/smolvla_mlx/memory.json")
    args = parser.parse_args(argv)
    mx.metal.reset_peak_memory()
    session = InferenceSession(args.artifact, dtype=args.dtype)
    session.runtime.num_steps = args.steps
    actions = session.predict(synthetic_observation(0), "pick up the block", seed=0)
    mx.eval(mx.array(actions))
    payload = {
        "dtype": args.dtype,
        "steps": args.steps,
        "peak_bytes": int(mx.metal.get_peak_memory()),
        "active_bytes": int(mx.metal.get_active_memory()),
        "action_shape": list(actions.shape),
    }
    path = write_report(Path(args.output).parent, Path(args.output).name, payload)
    print(path)


if __name__ == "__main__":
    main()
