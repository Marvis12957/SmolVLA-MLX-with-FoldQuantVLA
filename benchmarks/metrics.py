"""Repeatable latency, memory, model-size, and fidelity measurements."""

from __future__ import annotations

import json
import importlib.metadata
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class LatencyReport:
    warmups: int
    iterations: int
    samples_ms: list[float]
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    std_ms: float


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def assert_metal() -> None:
    import mlx.core as mx

    metal = getattr(mx, "metal", None)
    available = bool(metal is not None and getattr(metal, "is_available", lambda: False)())
    if not available:
        raise RuntimeError("Metal is required; refusing to fall back to CPU")


def mlx_memory_bytes() -> tuple[int, int]:
    import mlx.core as mx

    peak = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", mx), "get_peak_memory")
    active = getattr(mx, "get_active_memory", None) or getattr(getattr(mx, "metal", mx), "get_active_memory")
    return int(peak()), int(active())


def reset_mlx_peak_memory() -> None:
    import mlx.core as mx

    reset = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", mx), "reset_peak_memory")
    reset()


def latency_report(fn: Callable[[], Any], synchronize: Callable[[], None], *, warmups: int = 10, iterations: int = 50) -> LatencyReport:
    for _ in range(warmups):
        fn()
        synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn()
        synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return LatencyReport(
        warmups=warmups,
        iterations=iterations,
        samples_ms=samples,
        mean_ms=statistics.fmean(samples),
        p50_ms=float(np.percentile(samples, 50)),
        p90_ms=float(np.percentile(samples, 90)),
        p95_ms=float(np.percentile(samples, 95)),
        std_ms=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    )


def action_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError(f"Mismatched action shapes: {reference.shape} vs {candidate.shape}")
    difference = candidate - reference
    flat_ref = reference.reshape(-1)
    flat_cand = candidate.reshape(-1)
    denom = max(float(np.linalg.norm(flat_ref) * np.linalg.norm(flat_cand)), 1e-12)
    per_coordinate = np.mean(np.abs(difference), axis=tuple(range(difference.ndim - 1)))
    return {
        "cosine": float(np.dot(flat_ref, flat_cand) / denom),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
        "per_coordinate_mae": [float(value) for value in np.atleast_1d(per_coordinate)],
    }


def environment_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        text = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True, timeout=10)
        fields = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        chip = fields.get("Chip", "unknown chip")
        memory = fields.get("Memory", "unknown memory")
        result["hardware"] = f"{chip}, {memory}"
    except (OSError, subprocess.SubprocessError):
        result["hardware"] = "unavailable"
    try:
        result["mlx"] = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        result["mlx"] = "unavailable"
    return result


def write_report(destination: Path, name: str, payload: dict[str, Any]) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / name
    payload = {"commit": git_commit(), "environment": environment_metadata(), **payload}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def checkpoint_size(path: Path) -> dict[str, int]:
    return {file.name: file.stat().st_size for file in sorted(path.glob("*.safetensors"))}
