"""Plan-facing wrapper around `smolvla-mlx audit`."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from smolvla import main as smolvla_main


def main(argv: list[str] | None = None) -> None:
    smolvla_main(["audit", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
