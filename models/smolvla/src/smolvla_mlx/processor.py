"""Exact SO101 observation preparation used by the pinned policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from safetensors.numpy import load_file

from .constants import (
    ACTION_DIM,
    CHUNK_SIZE,
    IMAGE_KEYS,
    PADDED_ACTION_DIM,
    PADDED_STATE_DIM,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    TOKENIZER_MAX_LENGTH,
)


def _as_chw_float(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW image, got {array.shape}")
    if array.shape[-1] in (1, 3, 4):
        array = array[..., :3].transpose(2, 0, 1)
    elif array.shape[0] in (1, 3, 4):
        array = array[:3]
    else:
        raise ValueError(f"Cannot infer image channels from {array.shape}")
    array = array.astype(np.float32, copy=False)
    if array.max(initial=0.0) > 1.0:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def _bilinear(image: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Channel-first bilinear resize matching ``F.interpolate(..., align_corners=False)``."""
    _, src_h, src_w = image.shape
    if src_h == dst_h and src_w == dst_w:
        return image
    ys = (np.arange(dst_h, dtype=np.float32) + 0.5) * (src_h / dst_h) - 0.5
    xs = (np.arange(dst_w, dtype=np.float32) + 0.5) * (src_w / dst_w) - 0.5
    y0_raw = np.floor(ys).astype(np.int32)
    x0_raw = np.floor(xs).astype(np.int32)
    wy = (ys - y0_raw).reshape(1, -1, 1)
    wx = (xs - x0_raw).reshape(1, 1, -1)
    y0 = np.clip(y0_raw, 0, src_h - 1)
    x0 = np.clip(x0_raw, 0, src_w - 1)
    y1 = np.clip(y0_raw + 1, 0, src_h - 1)
    x1 = np.clip(x0_raw + 1, 0, src_w - 1)
    top_left = image[:, y0][:, :, x0]
    top_right = image[:, y0][:, :, x1]
    bottom_left = image[:, y1][:, :, x0]
    bottom_right = image[:, y1][:, :, x1]
    return (top_left * (1.0 - wy) + bottom_left * wy) * (1.0 - wx) + (top_right * (1.0 - wy) + bottom_right * wy) * wx


def resize_with_pad(image: np.ndarray, height: int = RESIZE_HEIGHT, width: int = RESIZE_WIDTH) -> np.ndarray:
    """Aspect-preserving resize with SmolVLA's left and top zero pad."""
    image = _as_chw_float(image)
    _, src_h, src_w = image.shape
    if src_h == height and src_w == width:
        return image
    ratio = max(src_w / width, src_h / height)
    dst_h, dst_w = int(src_h / ratio), int(src_w / ratio)
    resized = np.clip(_bilinear(image, dst_h, dst_w), 0.0, 1.0)
    canvas = np.zeros((image.shape[0], height, width), dtype=np.float32)
    pad_h, pad_w = height - dst_h, width - dst_w
    canvas[:, pad_h : pad_h + dst_h, pad_w : pad_w + dst_w] = resized
    return canvas


def pad_vector(values: Any, target_dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad {array.shape[-1]} values into target dimension {target_dim}")
    result = np.zeros((*array.shape[:-1], target_dim), dtype=np.float32)
    result[..., : array.shape[-1]] = array
    return result


@dataclass(frozen=True)
class ProcessedObservation:
    images: tuple[np.ndarray, ...]
    image_masks: np.ndarray
    state: np.ndarray
    language_tokens: np.ndarray
    language_attention_mask: np.ndarray


class PolicyProcessor:
    """Loads LeRobot pre/post-processing state without requiring a robot runtime."""

    def __init__(
        self,
        preprocessor: Mapping[str, Any],
        postprocessor: Mapping[str, Any],
        normalizer_state: Mapping[str, np.ndarray] | None = None,
        unnormalizer_state: Mapping[str, np.ndarray] | None = None,
        image_features: tuple[str, ...] = IMAGE_KEYS,
        empty_cameras: int = 0,
    ) -> None:
        self.preprocessor = dict(preprocessor)
        self.postprocessor = dict(postprocessor)
        self.normalizer_state = dict(normalizer_state or {})
        self.unnormalizer_state = dict(unnormalizer_state or {})
        self.rename_map = self._find_rename_map()
        self.image_features = tuple(image_features)
        self.empty_cameras = int(empty_cameras)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "PolicyProcessor":
        root = Path(path)
        preprocessor = json.loads((root / "policy_preprocessor.json").read_text())
        postprocessor = json.loads((root / "policy_postprocessor.json").read_text())

        def state_for(processor: Mapping[str, Any], parent: Path) -> Mapping[str, np.ndarray]:
            for step in processor.get("steps", []):
                if step.get("registry_name") in {"normalizer_processor", "unnormalizer_processor"}:
                    state_file = step.get("state_file")
                    if state_file and (parent / state_file).is_file():
                        return load_file(parent / state_file)
            return {}

        image_features = IMAGE_KEYS
        empty_cameras = 0
        config_path = root / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            features = config.get("input_features") or {}
            if isinstance(features, dict):
                found = tuple(key for key in features if str(key).startswith("observation.images."))
                if found:
                    image_features = found
            empty_cameras = int(config.get("empty_cameras", 0))
        return cls(
            preprocessor,
            postprocessor,
            state_for(preprocessor, root),
            state_for(postprocessor, root),
            image_features=image_features,
            empty_cameras=empty_cameras,
        )

    def _find_rename_map(self) -> dict[str, str]:
        for step in self.preprocessor.get("steps", []):
            if step.get("registry_name") == "rename_observations_processor":
                return dict(step.get("config", {}).get("rename_map", {}))
        return {}

    def _normalization_pair(self, feature: str, state: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
        # lerobot/smolvla_base stores the SO-100 embodiment under this prefix.
        candidates = (
            (f"{feature}.mean", f"{feature}.std"),
            (f"{feature}.min", f"{feature}.max"),
            (f"so100.buffer.{feature}.mean", f"so100.buffer.{feature}.std"),
        )
        for first, second in candidates:
            if first in state and second in state:
                return np.asarray(state[first], dtype=np.float32), np.asarray(state[second], dtype=np.float32)
        return None

    def normalize_state(self, state: Any) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        pair = self._normalization_pair("observation.state", self.normalizer_state)
        if pair is None:
            return state
        mean, std = pair
        return (state - mean) / np.maximum(std, 1e-8)

    def unnormalize_action(self, action: Any) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        pair = self._normalization_pair("action", self.unnormalizer_state)
        if pair is None:
            return action
        mean, std = pair
        return action * np.maximum(std, 1e-8) + mean

    def normalize_action(self, action: Any) -> np.ndarray:
        """Map robot-unit actions back into the model's MEAN_STD space."""
        action = np.asarray(action, dtype=np.float32)
        pair = self._normalization_pair("action", self.unnormalizer_state)
        if pair is None:
            return action
        mean, std = pair
        return (action - mean) / np.maximum(std, 1e-8)

    @property
    def action_dim(self) -> int:
        """Robot action width stored in the checkpoint stats. Falls back to the SO-100 default."""
        for state in (self.unnormalizer_state, self.normalizer_state):
            pair = self._normalization_pair("action", state)
            if pair is not None:
                return int(np.asarray(pair[0]).shape[-1])
        return ACTION_DIM

    def tokenize(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """Use the checkpoint tokenizer when transformers is installed."""
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Tokenization requires the 'reference' dependency extra.") from exc
        tokenizer_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        encoded = tokenizer(task + "\n", max_length=TOKENIZER_MAX_LENGTH, padding="max_length", truncation=True)
        return np.asarray(encoded["input_ids"], dtype=np.int32), np.asarray(encoded["attention_mask"], dtype=np.bool_)

    def prepare(self, observation: Mapping[str, Any], task: str) -> ProcessedObservation:
        renamed = {self.rename_map.get(key, key): value for key, value in observation.items()}
        present = [key for key in self.image_features if key in renamed]
        if not present:
            raise KeyError(f"Observation is missing required camera(s): {list(self.image_features)}")
        images = [resize_with_pad(renamed[key]) * 2.0 - 1.0 for key in present]
        masks = [True] * len(images)
        missing = [key for key in self.image_features if key not in renamed]
        for _ in missing[: self.empty_cameras]:
            images.append(np.full_like(images[-1], -1.0))
            masks.append(False)
        tokens, attention_mask = self.tokenize(task)
        if "observation.state" not in renamed:
            raise KeyError("Observation is missing observation.state")
        state = pad_vector(self.normalize_state(renamed["observation.state"]), PADDED_STATE_DIM)
        return ProcessedObservation(
            images=tuple(images),
            image_masks=np.asarray(masks, dtype=np.bool_),
            state=state,
            language_tokens=tokens,
            language_attention_mask=attention_mask,
        )

    def pad_actions(self, actions: Any) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim}-DoF actions, got {actions.shape}")
        return pad_vector(actions, PADDED_ACTION_DIM)

    def crop_actions(self, actions: Any) -> np.ndarray:
        return np.asarray(actions)[..., : self.action_dim]

    @staticmethod
    def empty_action_chunk(batch_size: int = 1) -> np.ndarray:
        return np.zeros((batch_size, CHUNK_SIZE, PADDED_ACTION_DIM), dtype=np.float32)
