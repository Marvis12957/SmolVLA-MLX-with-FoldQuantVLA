"""Record per-input-channel activation maxima from a floating-point runtime."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from deployment.session import InferenceSession, synthetic_observation


def _attach_amax(module, name: str, captured: list[tuple[str, mx.array]]) -> None:
    """Record inputs by changing the instance class.

    ``module(x)`` looks up ``__call__`` on the class, so assigning
    ``module.__call__`` does not run.
    """
    base = type(module)

    def __call__(self, hidden, *args, **kwargs):
        reduced = mx.max(mx.abs(hidden), axis=tuple(range(hidden.ndim - 1)))
        captured.append((name, reduced))
        return base.__call__(self, hidden, *args, **kwargs)

    module.__class__ = type(f"{base.__name__}Amax", (base,), {"__call__": __call__})


def collect_activation_amax(
    artifact: str,
    samples: int = 0,
    task: str = "pick up the block",
    observations: list[dict] | None = None,
) -> dict[str, np.ndarray]:
    """Max absolute activation per input channel.

    ``observations`` are real frames. Without them, ``samples`` synthetic
    forwards use the benchmark camera and state distribution.
    """
    if observations is None:
        if samples < 1:
            raise ValueError("Calibration needs at least one forward.")
        observations = [{**synthetic_observation(seed), "task": task} for seed in range(samples)]
    session = InferenceSession(artifact, dtype="bfloat16")
    stats: dict[str, np.ndarray] = {}
    captured: list[tuple[str, mx.array]] = []
    for name, module in session.runtime.projections.items():
        _attach_amax(module, name, captured)
    for index, observation in enumerate(observations):
        captured.clear()
        sample_task = str(observation.get("task", task))
        session.predict(observation, sample_task, seed=index)
        mx.eval(*(value for _, value in captured))
        for name, reduced in captured:
            vector = np.asarray(reduced.astype(mx.float32))
            previous = stats.get(name)
            stats[name] = vector if previous is None else np.maximum(previous, vector)
    return stats
