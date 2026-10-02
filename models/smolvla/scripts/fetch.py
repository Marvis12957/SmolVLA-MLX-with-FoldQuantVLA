"""Download the pinned SmolVLA checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from smolvla_mlx.artifacts import SourceSpec, fetch_checkpoint
from smolvla_mlx.constants import LOCAL_CHECKPOINT, MODEL_REPO_ID, MODEL_REVISION


def command_fetch(args: argparse.Namespace) -> None:
    destination = Path(args.destination).expanduser()
    result = fetch_checkpoint(destination, SourceSpec(repo_id=args.repo_id, revision=args.revision))
    print(result)


def register(subparsers: argparse._SubParsersAction) -> None:
    fetch = subparsers.add_parser("fetch", help="Download the pinned source checkpoint")
    fetch.add_argument("--repo-id", default=MODEL_REPO_ID)
    fetch.add_argument("--revision", default=MODEL_REVISION)
    fetch.add_argument("--destination", default=LOCAL_CHECKPOINT)
    fetch.set_defaults(func=command_fetch)
