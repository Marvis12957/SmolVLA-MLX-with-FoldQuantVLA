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


def load_calibration_observations(
    root: str | Path,
    *,
    seed: int = 1000,
    num_calibration: int = 128,
    num_heldout: int = 32,
) -> tuple[CalibrationSplit, list[dict]]:
    """Load one real frame from each calibration episode.

    Images stay in the dataset's ``wrist`` / ``top`` names. The policy processor
    renames them to the checkpoint cameras.
    """
    root = Path(root)
    split = calibration_split_from_lerobot_root(
        root, seed=seed, num_calibration=num_calibration, num_heldout=num_heldout
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "hungho77/so101-ai20k-multitask",
        root=root,
        download_videos=False,
        return_uint8=True,
    )
    observations = []
    for ref in split.calibration:
        item = dataset[ref.frame_index]
        task = item.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0]
        observations.append(
            {
                "observation.images.wrist": _hwc_uint8(item["observation.images.wrist"]),
                "observation.images.top": _hwc_uint8(item["observation.images.top"]),
                "observation.state": np.asarray(item["observation.state"], dtype=np.float32).reshape(-1),
                "task": str(task),
            }
        )
    return split, observations


def _hwc_uint8(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
