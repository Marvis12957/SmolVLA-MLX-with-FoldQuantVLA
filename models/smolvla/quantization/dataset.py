"""LeRobot v3 metadata helpers used for calibration and offline verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import CalibrationSplit, make_episode_disjoint_split


def calibration_split_from_lerobot_root(
    root: str | Path, *, seed: int = 1000, num_calibration: int = 128, num_heldout: int = 32
) -> CalibrationSplit:
    """Read v3 episode metadata without decoding video frames."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency/environment specific
        raise RuntimeError("Dataset metadata reading requires the reference dependency extra.") from exc
    root = Path(root)
    episode_file = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episode_file.is_file():
        raise FileNotFoundError(f"Expected LeRobot v3 episode metadata at {episode_file}")
    table = pq.read_table(episode_file)
    columns = table.to_pydict()
    required = ("episode_index", "dataset_from_index", "dataset_to_index")
    missing = [name for name in required if name not in columns]
    if missing:
        raise RuntimeError(f"Episode metadata missing columns {missing}")
    if "task_index" in columns:
        task_index = columns["task_index"]
    else:
        task_index = _task_index(root, columns.get("tasks"))
    rows = zip(
        columns["episode_index"],
        columns["dataset_from_index"],
        columns["dataset_to_index"],
        task_index,
        strict=True,
    )
    return make_episode_disjoint_split(rows, seed=seed, num_calibration=num_calibration, num_heldout=num_heldout)


def _task_index(root: Path, tasks: list | None) -> list[int]:
    """Map each episode's first task string onto ``meta/tasks.parquet``."""
    import pyarrow.parquet as pq

    task_file = root / "meta" / "tasks.parquet"
    if tasks is None or not task_file.is_file():
        raise RuntimeError("Episode metadata has no task_index and no tasks.parquet.")
    table = pq.read_table(task_file).to_pydict()
    lookup = {name: int(index) for name, index in zip(table["task"], table["task_index"], strict=True)}
    resolved = []
    for episode_tasks in tasks:
        if not episode_tasks:
            raise RuntimeError("An episode has an empty task list.")
        name = episode_tasks[0]
        if name not in lookup:
            raise RuntimeError(f"Task {name!r} is not in {task_file}.")
        resolved.append(lookup[name])
    return resolved


def local_episode_rows(root: str | Path) -> list[tuple[int, int, int, int]]:
    """Episodes whose wrist and top video files are already on disk."""
    return [
        (int(episode["episode_index"]), int(episode["start"]), int(episode["end"]), int(episode["task_index"]))
        for episode in _local_episodes(root).values()
    ]


def _local_episodes(root: Path) -> dict[int, dict]:
    import pyarrow.parquet as pq

    present = {path.relative_to(root / "videos").as_posix() for path in (root / "videos").rglob("*.mp4")}
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
    task_index = _task_index(root, table["tasks"])
    episodes = {}
    for index, episode in enumerate(table["episode_index"]):
        videos = {}
        missing = False
        for camera in ("wrist", "top"):
            chunk = table[f"videos/observation.images.{camera}/chunk_index"][index]
            file_index = table[f"videos/observation.images.{camera}/file_index"][index]
            relative = f"observation.images.{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            if relative not in present:
                missing = True
                break
            videos[camera] = {
                "path": root / "videos" / relative,
                "from_timestamp": float(table[f"videos/observation.images.{camera}/from_timestamp"][index]),
            }
        if missing:
            continue
        episodes[int(episode)] = {
            "episode_index": int(episode),
            "start": int(table["dataset_from_index"][index]),
            "end": int(table["dataset_to_index"][index]),
            "task_index": task_index[index],
            "videos": videos,
        }
    if not episodes:
        raise FileNotFoundError(f"No episode in {root} has both camera videos on disk.")
    return episodes


def load_calibration_observations(
    root: str | Path,
    *,
    seed: int = 1000,
    num_calibration: int = 128,
    num_heldout: int = 32,
) -> tuple[CalibrationSplit, list[dict]]:
    """Load one real frame from each local calibration episode.

    Images stay in the dataset's ``wrist`` / ``top`` names. The policy processor
    renames them to the checkpoint cameras. Episodes whose video file was not
    downloaded are skipped.
    """
    root = Path(root)
    episodes = _local_episodes(root)
    available = len(episodes)
    heldout = min(num_heldout, max(0, available // 5))
    calibration_count = min(num_calibration, available - heldout)
    rows = [(ep["episode_index"], ep["start"], ep["end"], ep["task_index"]) for ep in episodes.values()]
    split = make_episode_disjoint_split(rows, seed=seed, num_calibration=calibration_count, num_heldout=heldout)
    frames = _read_frame_rows(root, [ref.frame_index for ref in split.calibration])
    names = _task_names(root)
    observations = [_observation(episodes[ref.episode_index], frames[ref.frame_index], names) for ref in split.calibration]
    return split, observations


def _task_names(root: Path) -> dict[int, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    return {int(index): str(name) for name, index in zip(table["task"], table["task_index"], strict=True)}


def _read_frame_rows(root: Path, indices: list[int]) -> dict[int, dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        root / "data" / "chunk-000" / "file-000.parquet",
        columns=["index", "timestamp", "observation.state", "task_index"],
        filters=[("index", "in", indices)],
    ).to_pydict()
    rows = {
        int(index): {
            "timestamp": float(timestamp),
            "state": state,
            "task_index": int(task_index),
        }
        for index, timestamp, state, task_index in zip(
            table["index"], table["timestamp"], table["observation.state"], table["task_index"], strict=True
        )
    }
    missing = [index for index in indices if index not in rows]
    if missing:
        raise KeyError(f"Frame indices missing from local parquet: {missing[:5]}")
    return rows


def _observation(episode: dict, frame: dict, names: dict[int, str]) -> dict:
    from lerobot.datasets.video_utils import decode_video_frames

    images = {}
    for camera, video in episode["videos"].items():
        decoded = decode_video_frames(
            video["path"],
            [video["from_timestamp"] + frame["timestamp"]],
            1e-4,
            return_uint8=True,
        )
        images[f"observation.images.{camera}"] = _hwc_uint8(decoded[0])
    return {
        **images,
        "observation.state": np.asarray(frame["state"], dtype=np.float32).reshape(-1),
        "task": names[frame["task_index"]],
    }


def _hwc_uint8(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
