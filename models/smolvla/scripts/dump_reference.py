"""Freeze deterministic PyTorch SmolVLA reference observations and action chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smolvla_mlx.constants import LOCAL_CHECKPOINT


SEEDS = (0, 1, 2)
TASK = "pick up the block"
STEPS = 10


def command_dump_reference(args: argparse.Namespace) -> None:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    from deployment.session import synthetic_observation
    from smolvla_mlx.processor import PolicyProcessor

    checkpoint = Path(args.checkpoint)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).float().to("cpu").eval()
    policy.config.num_steps = args.steps
    preprocessor = DataProcessorPipeline.from_pretrained(
        str(checkpoint),
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
    )
    processor = PolicyProcessor.from_pretrained(checkpoint)
    inputs_root = Path(args.inputs)
    outputs_root = Path(args.outputs)
    inputs_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        raw = synthetic_observation(seed)
        noise = np.random.default_rng(seed).standard_normal((1, 50, 32)).astype(np.float32)
        processed = processor.prepare(raw, args.task)
        observation = _chw_observation(raw)
        observation = {
            key: torch.from_numpy(np.asarray(value)) if isinstance(value, np.ndarray) else value
            for key, value in observation.items()
        }
        observation["task"] = args.task
        batch = preprocessor(observation)
        batch = {key: value.to("cpu") if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise)).float().cpu().numpy()
        unnormalized = processor.unnormalize_action(normalized)
        seed_in = inputs_root / f"seed_{seed}"
        seed_out = outputs_root / f"seed_{seed}"
        seed_in.mkdir(parents=True, exist_ok=True)
        seed_out.mkdir(parents=True, exist_ok=True)
        np.save(seed_in / "camera1.npy", raw["observation.images.camera1"])
        np.save(seed_in / "camera2.npy", raw["observation.images.camera2"])
        np.save(seed_in / "state.npy", raw["observation.state"])
        np.save(seed_in / "noise.npy", noise)
        np.save(seed_in / "language_tokens.npy", processed.language_tokens)
        np.save(seed_in / "language_mask.npy", processed.language_attention_mask)
        np.save(seed_in / "processed_state.npy", processed.state)
        (seed_in / "metadata.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "task": args.task,
                    "steps": args.steps,
                    "checkpoint": str(checkpoint),
                    "image_masks": processed.image_masks.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        np.save(seed_out / "pytorch_fp32_normalized.npy", normalized)
        np.save(seed_out / "pytorch_fp32_unnormalized.npy", unnormalized)
        print(seed_out / "pytorch_fp32_normalized.npy", normalized.shape)


def _chw_observation(observation: dict) -> dict:
    prepared = {}
    for key, value in observation.items():
        if not str(key).startswith("observation.images"):
            prepared[key] = value
            continue
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))
        array = array.astype(np.float32, copy=False)
        if float(array.max(initial=0.0)) > 1.0:
            array = array / 255.0
        prepared[key] = array[None, ...]
    if "observation.state" in prepared:
        state = np.asarray(prepared["observation.state"], dtype=np.float32)
        prepared["observation.state"] = state if state.ndim == 2 else state[None, ...]
    return prepared


def register(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("dump-reference", help="Freeze PyTorch action chunks for later MLX comparison")
    command.add_argument("--checkpoint", default=LOCAL_CHECKPOINT)
    command.add_argument("--inputs", default="reference_inputs")
    command.add_argument("--outputs", default="reference_outputs")
    command.add_argument("--task", default=TASK)
    command.add_argument("--steps", type=int, default=STEPS)
    command.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    command.set_defaults(func=command_dump_reference)
