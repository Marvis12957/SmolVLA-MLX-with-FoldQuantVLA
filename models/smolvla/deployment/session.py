"""Load a local artifact and predict one SO101 action chunk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smolvla_mlx.policy import SmolVLAPolicyMLX
from smolvla_mlx.processor import PolicyProcessor
from smolvla_mlx.runtime import MLXSmolVLARuntime


class InferenceSession:
    def __init__(self, artifact: str | Path, dtype: str = "bfloat16") -> None:
        root = Path(artifact)
        self.processor = PolicyProcessor.from_pretrained(root)
        self.runtime = MLXSmolVLARuntime.from_pretrained(root, dtype=dtype)
        self.policy = SmolVLAPolicyMLX(self.processor, self.runtime)

    def predict_normalized(self, observation: dict, task: str, noise: np.ndarray | None = None, seed: int | None = 0) -> np.ndarray:
        return self.policy.predict_normalized(observation, task, noise=noise, seed=seed)

    def predict(self, observation: dict, task: str, noise: np.ndarray | None = None, seed: int | None = 0) -> np.ndarray:
        return self.policy.predict_action_chunk(observation, task, noise=noise, seed=seed)


def synthetic_observation(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "observation.images.camera1": rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.camera2": rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8),
        "observation.state": rng.normal(size=(6,)).astype(np.float32),
    }
