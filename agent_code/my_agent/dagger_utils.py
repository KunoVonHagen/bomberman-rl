from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tqdm

from agent_code.my_agent.gym_environment import BombermanGymEnv, ACTION_INDICES
from agent_code.my_agent.callbacks import act as expert_act, setup as expert_setup


@dataclass
class Transitions:
    obs: dict
    acts: np.ndarray
    next_obs: dict
    dones: np.ndarray
    infos: np.ndarray = field(default=None)


def _copy_obs(obs: dict) -> dict:
    return {k: v.copy() for k, v in obs.items()}


def dagger_collect(env: BombermanGymEnv, policy, n_episodes: int, cutoff_step: float = 400.0) -> Transitions:
    if expert_setup is not None:
        expert_setup(env.agent)

    all_obs, all_next_obs, all_actions, all_dones = [], [], [], []

    for _ in tqdm.tqdm(range(n_episodes), desc="Collecting DAgger data"):
        obs, _ = env.reset()
        obs = _copy_obs(obs)
        done = False

        while not done and env.step_count < cutoff_step:
            policy_action, _ = policy.predict(obs, deterministic=True)

            game_state = env.get_state_for_agent(env.agent)
            expert_action = expert_act(env.agent, game_state)
            expert_action = ACTION_INDICES[expert_action]

            next_obs, reward, terminated, truncated, _ = env.step(policy_action)
            next_obs = _copy_obs(next_obs)
            done = terminated or truncated

            all_obs.append(obs)
            all_next_obs.append(next_obs)
            all_actions.append(expert_action)
            all_dones.append(done)

            obs = next_obs

    obs = {
        "grid_tensor": np.stack([o["grid_tensor"] for o in all_obs]),
        "features": np.stack([o["features"] for o in all_obs]),
    }
    next_obs = {
        "grid_tensor": np.stack([o["grid_tensor"] for o in all_next_obs]),
        "features": np.stack([o["features"] for o in all_next_obs]),
    }

    return Transitions(
        obs=obs,
        acts=np.array(all_actions),
        next_obs=next_obs,
        dones=np.array(all_dones, dtype=bool),
        infos=np.array([{}] * len(all_actions), dtype=object),
    )


def merge_transitions(old: Transitions | None, new: Transitions) -> Transitions:
    if old is None:
        return new

    obs = {
        "grid_tensor": np.concatenate([old.obs["grid_tensor"], new.obs["grid_tensor"]]),
        "features": np.concatenate([old.obs["features"], new.obs["features"]]),
    }
    next_obs = {
        "grid_tensor": np.concatenate([old.next_obs["grid_tensor"], new.next_obs["grid_tensor"]]),
        "features": np.concatenate([old.next_obs["features"], new.next_obs["features"]]),
    }

    return Transitions(
        obs=obs,
        acts=np.concatenate([old.acts, new.acts]),
        next_obs=next_obs,
        dones=np.concatenate([old.dones, new.dones]),
        infos=np.concatenate([old.infos, new.infos]),
    )