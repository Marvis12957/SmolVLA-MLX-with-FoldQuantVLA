"""Record per-input-channel activation maxima from a floating-point runtime."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from deployment.session import InferenceSession, synthetic_observation


def collect_activation_amax(
    artifact: str,
    samples: int,
    task: str = "pick up the block",
) -> dict[str, np.ndarray]:
    """Max absolute activation per input channel, over ``samples`` synthetic forwards.

    The LeRobot episode split is not on disk, so this uses the same synthetic
    camera and state distribution as the benchmark.
    """
    if samples < 1:
        raise ValueError("Calibration needs at least one forward.")
    session = InferenceSession(artifact, dtype="bfloat16")
    stats: dict[str, np.ndarray] = {}
    for name, module in session.runtime.projections.items():
        call = module.__call__

        def wrapped(hidden, *, name=name, call=call):
            reduced = mx.max(mx.abs(hidden), axis=tuple(range(hidden.ndim - 1)))
            mx.eval(reduced)
            vector = np.asarray(reduced.astype(mx.float32))
            previous = stats.get(name)
            stats[name] = vector if previous is None else np.maximum(previous, vector)
            return call(hidden)

        module.__call__ = wrapped
    for seed in range(samples):
        session.predict(synthetic_observation(seed), task, seed=seed)
    return stats
