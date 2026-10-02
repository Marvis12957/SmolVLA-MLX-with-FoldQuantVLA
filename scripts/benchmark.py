"""Plan-facing wrapper around `smolvla-mlx benchmark`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smolvla import main as smolvla_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark SmolVLA MLX latency, memory, or size")
    parser.add_argument("--model", default="smolvla")
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "bfloat16", "fp16", "float16"))
    parser.add_argument("--metric", default="latency", choices=("latency", "memory", "size"))
    args, rest = parser.parse_known_args(argv)
    dtype = "float16" if args.precision in {"fp16", "float16"} else "bfloat16"
    stem = "fp16" if dtype == "float16" else "bf16"
    if args.metric == "latency":
        smolvla_main(["benchmark", "latency", "--dtype", dtype, "--output", f"results/baseline/latency_{stem}.json", *rest])
        return
    if args.metric == "memory":
        smolvla_main(["benchmark", "memory", "--dtype", dtype, "--output", f"results/baseline/memory_{stem}.json", *rest])
        return
    smolvla_main(["benchmark", "size", "checkpoints/raw/smolvla/base", *rest])


if __name__ == "__main__":
    main()
