"""Compare two saved robot-unit action arrays in both action spaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "smolvla" / "src"))
from metrics import action_metrics
from smolvla_mlx.processor import PolicyProcessor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare reference and candidate action arrays")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--stats", default="checkpoints/raw/smolvla/base", help="Artifact with action mean/std")
    args = parser.parse_args(argv)
    processor = PolicyProcessor.from_pretrained(args.stats)
    reference = np.load(args.reference.expanduser())
    candidate = np.load(args.candidate.expanduser())
    payload = {
        "unnormalized": action_metrics(reference, candidate),
        "normalized": action_metrics(processor.normalize_action(reference), processor.normalize_action(candidate)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
