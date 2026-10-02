"""Open-loop action error of an MLX SmolVLA policy on a local LeRobot dataset.

Each selected episode contributes one frame whose future action chunk stays
inside the episode. The predicted chunk is compared with the recorded actions
in the dataset's own units and in the checkpoint's normalized space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smolvla_mlx.constants import (
    CHUNK_SIZE,
    LIBERO_DATASET,
    LIBERO_DATASET_REPO_ID,
    LIBERO_MLX_CHECKPOINT,
    PADDED_ACTION_DIM,
)


def command_eval_dataset(args: argparse.Namespace) -> None:
    samples = _samples(Path(args.dataset), args.repo_id, args.seed, args.limit)
    report = _evaluate(Path(args.artifact), samples, args.steps, args.seed, not args.skip_pytorch)
    payload = {
        "dataset": str(args.dataset),
        "repo_id": args.repo_id,
        "artifact": str(args.artifact),
        "seed": args.seed,
        "frames": len(samples),
        **report,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def _evaluate(artifact: Path, samples: list[dict], steps: int, seed: int, compare_pytorch: bool) -> dict:
    from smolvla_mlx.policy import SmolVLAPolicyMLX

    policy = SmolVLAPolicyMLX.from_pretrained(artifact)
    policy.model.num_steps = steps
    action_dim = policy.processor.action_dim
    pytorch = _pytorch_policy(artifact, steps) if compare_pytorch else None
    rng = np.random.default_rng(seed)
    rows = []
    for index, sample in enumerate(samples):
        noise = rng.standard_normal((1, CHUNK_SIZE, PADDED_ACTION_DIM)).astype(np.float32)
        predicted = policy.predict_action_chunk(sample["observation"], sample["task"], noise=noise)[0]
        reference = np.asarray(sample["actions"], dtype=np.float32)[:, :action_dim]
        error = np.abs(predicted - reference)
        normalized_error = np.abs(
            policy.processor.normalize_action(predicted) - policy.processor.normalize_action(reference)
        )
        row = {
            "episode_index": sample["episode_index"],
            "frame_index": sample["frame_index"],
            "task": sample["task"],
            "mae": float(error.mean()),
            "first_mae": float(error[0].mean()),
            "normalized_mae": float(normalized_error.mean()),
        }
        if pytorch is not None:
            row["pytorch_normalized_mae"] = _pytorch_normalized_mae(pytorch, policy, sample, noise)
        rows.append(row)
        print(
            f"{index + 1}/{len(samples)} episode {sample['episode_index']} mae {row['mae']:.4f}",
            flush=True,
        )
    summary = {
        "action_dim": action_dim,
        "mae": _mean([row["mae"] for row in rows]),
        "first_mae": _mean([row["first_mae"] for row in rows]),
        "normalized_mae": _mean([row["normalized_mae"] for row in rows]),
        "episodes": rows,
    }
    if pytorch is not None:
        summary["pytorch_normalized_mae"] = _mean([row["pytorch_normalized_mae"] for row in rows])
    return summary


def _pytorch_policy(artifact: Path, steps: int):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    policy = SmolVLAPolicy.from_pretrained(str(artifact)).float().to("cpu").eval()
    policy.config.num_steps = steps
    preprocessor = DataProcessorPipeline.from_pretrained(
        str(artifact),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, torch


def _pytorch_normalized_mae(bundle, policy, sample: dict, noise: np.ndarray) -> float:
    pytorch, preprocessor, torch = bundle
    observation = {key: _torch_value(value, torch) for key, value in sample["observation"].items()}
    observation["task"] = sample["task"]
    batch = preprocessor(observation)
    batch = {key: value.to("cpu") if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.inference_mode():
        reference = pytorch.predict_action_chunk(batch, noise=torch.from_numpy(noise)).float().cpu().numpy()
    predicted = policy.predict_normalized(sample["observation"], sample["task"], noise=noise)
    width = min(reference.shape[-1], predicted.shape[-1])
    return float(np.abs(reference[..., :width] - predicted[..., :width]).mean())


def _torch_value(value, torch):
    array = np.asarray(value)
    if str(getattr(value, "dtype", "")).startswith("float") or array.dtype.kind == "f":
        array = array.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        return tensor[None, ...]
    if tensor.ndim == 1:
        return tensor[None, ...]
    return tensor


def _samples(root: Path, repo_id: str, seed: int, limit: int) -> list[dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = LeRobotDataset(repo_id, root=root, download_videos=False)
    fps = float(meta.fps)
    chosen = _choose_frames(meta, seed, limit)
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        download_videos=False,
        delta_timestamps={"action": [index / fps for index in range(CHUNK_SIZE)]},
    )
    samples = []
    for episode_index, frame_index in chosen:
        item = dataset[frame_index]
        task = item.get("task")
        if isinstance(task, (list, tuple)):
            task = task[0]
        observation = {
            key: _array(item[key])
            for key in item
            if key.startswith("observation.images.") or key == "observation.state"
        }
        state = observation["observation.state"]
        if state.ndim > 1:
            observation["observation.state"] = state.reshape(-1)
        samples.append(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": str(task),
                "actions": _array(item["action"]),
                "observation": observation,
            }
        )
    if not samples:
        raise RuntimeError(f"No frames could be read from {root}")
    return samples


def _choose_frames(dataset, seed: int, limit: int) -> list[tuple[int, int]]:
    episodes = _episode_table(dataset)
    eligible = [row for row in episodes if row["end"] - row["start"] > CHUNK_SIZE]
    if not eligible:
        raise RuntimeError("Dataset has no episode long enough for a 50-step action chunk")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    chosen = []
    seen_tasks = set()
    for index in order:
        row = eligible[int(index)]
        if row["task"] in seen_tasks:
            continue
        seen_tasks.add(row["task"])
        start = row["start"]
        frame = int(rng.integers(start, row["end"] - CHUNK_SIZE))
        chosen.append((row["episode_index"], frame))
        if limit > 0 and len(chosen) >= limit:
            break
    return chosen


def _episode_table(dataset) -> list[dict]:
    episodes = dataset.meta.episodes
    if hasattr(episodes, "to_pandas"):
        table = episodes.to_pandas()
        rows = table.to_dict(orient="records")
    elif hasattr(episodes, "to_dict"):
        raw = episodes.to_dict()
        count = len(next(iter(raw.values())))
        rows = [{key: raw[key][index] for key in raw} for index in range(count)]
    else:
        rows = [dict(row) for row in episodes]
    parsed = []
    for row in rows:
        tasks = row.get("tasks") or row.get("task") or ""
        if isinstance(tasks, (list, tuple, np.ndarray)):
            task = str(tasks[0]) if len(tasks) else ""
        else:
            task = str(tasks)
        parsed.append(
            {
                "episode_index": int(row["episode_index"]),
                "start": int(row["dataset_from_index"]),
                "end": int(row["dataset_to_index"]),
                "task": task,
            }
        )
    return parsed


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def register(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("eval-dataset", help="Score MLX actions against a local LeRobot dataset")
    command.add_argument("--artifact", default=LIBERO_MLX_CHECKPOINT)
    command.add_argument("--dataset", default=LIBERO_DATASET)
    command.add_argument("--repo-id", default=LIBERO_DATASET_REPO_ID)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--limit", type=int, default=8, help="Number of episodes, one frame each")
    command.add_argument("--steps", type=int, default=10)
    command.add_argument("--skip-pytorch", action="store_true")
    command.add_argument("--output", default="results/smolvla_mlx/libero_openloop.json")
    command.set_defaults(func=command_eval_dataset)
