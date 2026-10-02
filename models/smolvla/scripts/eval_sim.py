"""Closed-loop success of an MLX SmolVLA policy in the LIBERO simulator.

The loop matches LeRobot's eval: flip the raw cameras, build the 8-D state,
then execute the unnormalized 7-D chunk one step at a time. A new chunk is
sampled only after the previous `n_action_steps` actions have been used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from smolvla_mlx.constants import LIBERO_MLX_CHECKPOINT


def command_eval_sim(args: argparse.Namespace) -> None:
    rows = []
    for task_id in range(args.task_id, args.task_id + args.tasks):
        for episode in range(args.episodes):
            row = _rollout(
                artifact=Path(args.artifact),
                suite=args.suite,
                task_id=task_id,
                episode_index=episode,
                image_size=args.image_size,
                steps=args.steps,
                seed=args.seed + episode,
            )
            rows.append(row)
            print(
                f"task {task_id} episode {episode} success {row['success']} steps {row['steps']}",
                flush=True,
            )
    payload = {
        "artifact": str(args.artifact),
        "suite": args.suite,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "rollouts": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def _rollout(
    artifact: Path,
    suite: str,
    task_id: int,
    episode_index: int,
    image_size: int,
    steps: int,
    seed: int,
) -> dict:
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from lerobot.processor.env_processor import LiberoProcessorStep

    from smolvla_mlx.policy import SmolVLAPolicyMLX

    policy = SmolVLAPolicyMLX.from_pretrained(artifact)
    policy.model.num_steps = steps
    config = json.loads((artifact / "config.json").read_text())
    horizon = int(config.get("n_action_steps", config.get("chunk_size", 50)))
    task_suite = _get_suite(suite)
    env = LiberoEnv(
        task_suite,
        task_id=task_id,
        task_suite_name=suite,
        observation_height=image_size,
        observation_width=image_size,
        episode_index=episode_index,
    )
    libero_step = LiberoProcessorStep()
    try:
        observation, _info = env.reset(seed=seed)
        task = env.task_description
        queue: list[np.ndarray] = []
        success = False
        taken = 0
        for _ in range(int(env._max_episode_steps)):
            if not queue:
                predicted = policy.predict_action_chunk(_policy_observation(observation, libero_step), task)
                queue = [np.asarray(step, dtype=np.float32) for step in predicted[0, :horizon]]
            action = queue.pop(0)
            observation, _reward, terminated, truncated, info = env.step(action)
            taken += 1
            success = bool(info.get("is_success", False))
            if success or terminated or truncated:
                break
        return {
            "suite": suite,
            "task_id": task_id,
            "task": task,
            "episode_index": episode_index,
            "success": success,
            "steps": taken,
        }
    finally:
        env.close()


def _policy_observation(observation: dict, libero_step) -> dict:
    from lerobot.envs.utils import preprocess_observation

    processed = libero_step.observation(preprocess_observation(observation))
    prepared = {}
    for key, value in processed.items():
        if not (key.startswith("observation.images.") or key == "observation.state"):
            continue
        array = value.detach().cpu().numpy()
        prepared[key] = array[0] if array.ndim >= 1 and array.shape[0] == 1 else array
    return prepared


def register(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("eval-sim", help="Roll out the MLX policy in the LIBERO simulator")
    command.add_argument("--artifact", default=LIBERO_MLX_CHECKPOINT)
    command.add_argument("--suite", default="libero_spatial")
    command.add_argument("--task-id", type=int, default=0)
    command.add_argument("--tasks", type=int, default=1)
    command.add_argument("--episodes", type=int, default=1)
    command.add_argument("--image-size", type=int, default=256)
    command.add_argument("--steps", type=int, default=10)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--output", default="results/smolvla_mlx/libero_sim.json")
    command.set_defaults(func=command_eval_sim)
