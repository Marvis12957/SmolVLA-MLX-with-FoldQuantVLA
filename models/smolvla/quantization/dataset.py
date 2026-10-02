"""LeRobot v3 metadata helpers used for calibration and offline verification."""

from __future__ import annotations

from pathlib import Path

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
