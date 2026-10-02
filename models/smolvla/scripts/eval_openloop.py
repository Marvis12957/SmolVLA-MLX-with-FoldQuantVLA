"""Open-loop action error of a SmolVLA checkpoint on local dataset frames.

Each test episode contributes one frame. The predicted 50-step chunk is compared
with the recorded joint targets, in degrees and in the checkpoint's normalized
action space. A hold-state baseline repeats the current joints for the chunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHUNK = 50
ACTION_DIM = 6
PADDED_ACTION_DIM = 32


def command_eval(args: argparse.Namespace) -> None:
    if args.reuse:
        samples = _reuse_samples(Path(args.dataset), Path(args.reuse), args.limit)
    else:
        samples = _test_samples(Path(args.dataset), Path(args.split), args.seed, args.limit)
    reports = [_evaluate(Path(path), samples, args.device, args.seed) for path in args.checkpoints]
    payload = {
        "dataset": str(args.dataset),
        "split": "test",
        "seed": args.seed,
        "frames": len(samples),
        "hold_state_mae_deg": _mean([sample["hold_mae_deg"] for sample in samples]),
        "checkpoints": reports,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def _evaluate(checkpoint: Path, samples: list[dict], device_name: str, seed: int) -> dict:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    from smolvla_mlx.processor import PolicyProcessor

    device = _device(device_name)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).float().to(device).eval()
    policy.config.num_steps = 10
    preprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device.type}},
    )
    processor = PolicyProcessor.from_pretrained(checkpoint)
    rng = np.random.default_rng(seed)
    rows = []
    for index, sample in enumerate(samples):
        noise = torch.from_numpy(rng.standard_normal((1, CHUNK, PADDED_ACTION_DIM)).astype(np.float32)).to(device)
        batch = preprocessor(_observation(sample))
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(batch, noise=noise).float().cpu().numpy()
        predicted = processor.unnormalize_action(normalized[..., :ACTION_DIM])[0]
        reference = sample["actions"]
        error = np.abs(predicted - reference)
        normalized_error = np.abs(normalized[0, :, :ACTION_DIM] - processor.normalize_action(reference))
        row = {
            "episode_index": sample["episode_index"],
            "frame_index": sample["frame_index"],
            "task": sample["task"],
            "mae_deg": float(error.mean()),
            "first_mae_deg": float(error[0].mean()),
            "normalized_mae": float(normalized_error.mean()),
        }
        rows.append(row)
        print(
            f"{checkpoint.parent.name} {index + 1}/{len(samples)} episode {sample['episode_index']} mae {row['mae_deg']:.3f}",
            flush=True,
        )
    return {
        "checkpoint": str(checkpoint),
        "step": _training_step(checkpoint),
        "mae_deg": _mean([row["mae_deg"] for row in rows]),
        "first_mae_deg": _mean([row["first_mae_deg"] for row in rows]),
        "normalized_mae": _mean([row["normalized_mae"] for row in rows]),
        "per_task": _per_task(rows),
        "episodes": rows,
    }


def _reuse_samples(root: Path, previous: Path, limit: int) -> list[dict]:
    """Score the same frames already recorded in an open-loop result."""
    payload = json.loads(previous.read_text())
    refs = payload["checkpoints"][0]["episodes"]
    if limit > 0:
        refs = refs[:limit]
    chosen = [(int(row["episode_index"]), int(row["frame_index"])) for row in refs]
    return _load_frames(root, chosen)


def _test_samples(root: Path, split_path: Path, seed: int, limit: int) -> list[dict]:
    split = json.loads(split_path.read_text())
    test_ids = [int(episode) for episode in split["test_episodes"]]
    if limit > 0:
        test_ids = test_ids[:limit]
    rng = np.random.default_rng(seed)
    chosen = []
    episodes = _episode_table(root)
    for episode_index in test_ids:
        episode = episodes[episode_index]
        start, end = episode["start"], episode["end"]
        if end - start <= CHUNK:
            continue
        chosen.append((episode_index, int(rng.integers(start, end - CHUNK))))
    return _load_frames(root, chosen)


def _load_frames(root: Path, chosen: list[tuple[int, int]]) -> list[dict]:
    import pyarrow.parquet as pq
    from lerobot.datasets.video_utils import decode_video_frames

    episodes = _episode_table(root)
    table = pq.read_table(
        root / "data" / "chunk-000" / "file-000.parquet",
        columns=["timestamp", "observation.state", "action"],
    )
    timestamps = table.column("timestamp").to_numpy()
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    samples = []
    for episode_index, frame in chosen:
        episode = episodes[episode_index]
        future = actions[frame : frame + CHUNK]
        state = states[frame]
        images = {}
        for camera, video in episode["videos"].items():
            decoded = decode_video_frames(
                video["path"],
                [video["from_timestamp"] + float(timestamps[frame])],
                1e-4,
                return_uint8=True,
            )
            images[camera] = _chw_float(decoded[0])
        samples.append(
            {
                "episode_index": episode_index,
                "frame_index": frame,
                "task": episode["task"],
                "state": state,
                "actions": future,
                "images": images,
                "hold_mae_deg": float(np.abs(np.broadcast_to(state, future.shape) - future).mean()),
            }
        )
    if not samples:
        raise RuntimeError(f"No test frames could be read from {root}")
    return samples


def _episode_table(root: Path) -> dict[int, dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        columns=[
            "episode_index",
            "dataset_from_index",
            "dataset_to_index",
            "tasks",
            "videos/observation.images.wrist/chunk_index",
            "videos/observation.images.wrist/file_index",
            "videos/observation.images.wrist/from_timestamp",
            "videos/observation.images.top/chunk_index",
            "videos/observation.images.top/file_index",
            "videos/observation.images.top/from_timestamp",
        ],
    ).to_pydict()
    episodes = {}
    for index, episode in enumerate(table["episode_index"]):
        videos = {}
        for camera in ("wrist", "top"):
            chunk = table[f"videos/observation.images.{camera}/chunk_index"][index]
            file_index = table[f"videos/observation.images.{camera}/file_index"][index]
            videos[camera] = {
                "path": root / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4",
                "from_timestamp": float(table[f"videos/observation.images.{camera}/from_timestamp"][index]),
            }
        episodes[int(episode)] = {
            "start": int(table["dataset_from_index"][index]),
            "end": int(table["dataset_to_index"][index]),
            "task": table["tasks"][index][0],
            "videos": videos,
        }
    return episodes


def _observation(sample: dict) -> dict:
    observation = {
        "observation.images.wrist": sample["images"]["wrist"],
        "observation.images.top": sample["images"]["top"],
        "observation.state": np.asarray(sample["state"], dtype=np.float32)[None, ...],
        "task": sample["task"],
    }
    return observation


def _chw_float(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 3:
        array = np.transpose(array, (2, 0, 1))
    array = array.astype(np.float32, copy=False)
    if float(array.max(initial=0.0)) > 1.0:
        array = array / 255.0
    return array[None, ...]


def _device(name: str):
    import torch

    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def _training_step(checkpoint: Path) -> int | None:
    path = checkpoint.parent / "training_state" / "training_step.json"
    if not path.is_file():
        return None
    return int(json.loads(path.read_text()).get("step", 0))


def _per_task(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    return {
        task: {
            "episodes": len(items),
            "mae_deg": _mean([item["mae_deg"] for item in items]),
            "first_mae_deg": _mean([item["first_mae_deg"] for item in items]),
        }
        for task, items in grouped.items()
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def register(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("eval-openloop", help="Score a checkpoint against recorded test-split actions")
    command.add_argument("--checkpoints", nargs="+", required=True)
    command.add_argument("--dataset", default="data/so101-ai20k-multitask")
    command.add_argument("--split", default="outputs/splits/split.json")
    command.add_argument("--reuse", default="", help="Existing open-loop JSON whose frames should be scored again")
    command.add_argument("--device", default="auto")
    command.add_argument("--seed", type=int, default=1000)
    command.add_argument("--limit", type=int, default=0, help="Evaluate only the first N test episodes")
    command.add_argument("--output", default="results/smolvla_mlx/openloop_test.json")
    command.set_defaults(func=command_eval)
