"""Plan-facing wrapper around `smolvla-mlx evaluate-fidelity`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smolvla import main as smolvla_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare an MLX candidate against the frozen PyTorch reference")
    parser.add_argument("--reference", default="pytorch")
    parser.add_argument("--candidate", default="mlx-bf16")
    parser.add_argument("--artifact", default="checkpoints/raw/smolvla/base")
    args, rest = parser.parse_known_args(argv)
    dtype = "float16" if "fp16" in args.candidate else "bfloat16"
    report = "results/baseline/mlx_fp16_vs_pytorch.json" if dtype == "float16" else "results/baseline/mlx_bf16_vs_pytorch.json"
    smolvla_main(["evaluate-fidelity", "--artifact", args.artifact, "--dtype", dtype, "--report", report, *rest])


if __name__ == "__main__":
    main()
