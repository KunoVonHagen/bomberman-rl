"""
train.py
-----------------------------------------------------------------------------
Main training entrypoint. All tunable behaviour lives in config.py — edit
`DEFAULT_CONFIG` there rather than this file.

    python train.py
        Start a fresh run using DEFAULT_CONFIG from config.py.

    python train.py --resume run_20260101-101500
        Resume the given run from its latest checkpoint, using the config
        that run was originally created with (stored in its run_manifest.json).

    python train.py --resume runs/run_20260101-101500 --checkpoint checkpoint_0016777216
        Resume from a specific checkpoint instead of the latest one.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import copy
from multiprocessing import freeze_support

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from environment import WorldArgs

from agent_code.my_agent.gym_environment import BombermanGymEnv
from agent_code.my_agent.model import BombermanFeatureExtractor
from agent_code.my_agent.config import DEFAULT_CONFIG, TrainingConfig
from agent_code.my_agent.checkpoint_manager import CheckpointManager
from agent_code.my_agent.opponent_pool import OpponentPool


def mask_fn(env):
    return env.action_masks()


def build_world_args(
    cfg: TrainingConfig,
    log_dir: str,
    save_replay: bool,
    replay_path: str | None = None,
) -> WorldArgs:
    e = cfg.env
    return WorldArgs(
        scenario=e.scenario,
        seed=e.seed,
        silence_errors=e.silence_errors,
        no_gui=e.no_gui,
        make_video=e.make_video,
        save_replay=save_replay,
        save_stats=e.save_stats,
        turn_based=e.turn_based,
        update_interval=e.update_interval,
        log_dir=log_dir,
        match_name=e.match_name,
        fps=e.fps,
        replay=replay_path or e.replay,
        continue_without_training=e.continue_without_training,
    )


# --- ADD THIS NEW WRAPPER CLASS ---
class NativeBatchedVecEnv(VecEnv):
    """
    Adapter to make a natively batched Gymnasium Env (like BombermanGymEnv)
    compatible with Stable-Baselines3's VecEnv API.
    """

    def __init__(self, batched_env):
        self.env = batched_env
        super().__init__(
            num_envs=self.env.n_envs,
            observation_space=self.env.single_observation_space,  # NOT self.env.observation_space
            action_space=self.env.single_action_space  # NOT self.env.action_space
        )
        self._actions = None
        self._last_infos: list[dict] = []

    def reset(self, seed=None, options=None):
        # Unpack the tuple from the underlying env
        obs, infos = self.env.reset(seed=seed, options=options)
        self._last_infos = infos
        # Return ONLY obs, not the tuple - works around sb3_contrib bug
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.env.step(self._actions)
        dones = terminated | truncated
        self._last_infos = infos
        return obs, rewards, dones, infos

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        return self.env.close()

    def env_method(self, method_name: str, *args, **kwargs) -> list:
        # Extract 'indices' before calling the underlying method
        # (natively batched envs operate on all envs at once, so we filter after)
        indices = kwargs.pop('indices', None)

        result = getattr(self.env, method_name)(*args, **kwargs)

        # Convert batched results to per-env list as SB3 expects
        if isinstance(result, np.ndarray) and result.ndim > 0 and result.shape[0] == self.num_envs:
            # Batched array like action_masks: (n_envs, n_actions) -> list of (n_actions,) arrays
            per_env = [result[i] for i in range(self.num_envs)]
            if indices is not None:
                return [per_env[i] for i in indices]
            return per_env
        elif isinstance(result, list) and len(result) == self.num_envs:
            if indices is not None:
                return [result[i] for i in indices]
            return list(result)
        else:
            # Scalar or single value - replicate for requested envs
            n = len(indices) if indices is not None else self.num_envs
            return [result] * n

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def env_is_wrapped(self, wrapper_class: type, indices: list[int] | None = None) -> list[bool]:
        if indices is None:
            return [False] * self.num_envs
        return [False] * len(indices)

    def get_attr(self, attr_name: str, indices: list[int] | None = None) -> list:
        attr = getattr(self.env, attr_name)
        if indices is None:
            if isinstance(attr, (list, np.ndarray)) and len(attr) == self.num_envs:
                return list(attr)
            return [attr] * self.num_envs
        if isinstance(attr, (list, np.ndarray)) and len(attr) == self.num_envs:
            return [attr[i] for i in indices]
        return [attr] * len(indices)

    def set_attr(self, attr_name: str, values, indices: list[int] | None = None) -> None:
        if indices is None:
            if isinstance(values, (list, np.ndarray)) and len(values) == self.num_envs:
                setattr(self.env, attr_name, values)
            else:
                setattr(self.env, attr_name, values)
        else:
            current = getattr(self.env, attr_name)
            if isinstance(current, (list, np.ndarray)) and len(current) == self.num_envs:
                if isinstance(current, np.ndarray):
                    current[indices] = values
                else:
                    for idx, val in zip(indices, values):
                        current[idx] = val
                setattr(self.env, attr_name, current)
            else:
                setattr(self.env, attr_name, values[0] if isinstance(values, (list, np.ndarray)) else values)

    def get_images(self) -> list[np.ndarray]:
        return [None] * self.num_envs


# -----------------------------------

def make_train_env(cfg: TrainingConfig, opponents, log_dir: str) -> VecMonitor:
    world_args = build_world_args(cfg, log_dir, save_replay=False)
    env = BombermanGymEnv(
        world_args,
        opponents=opponents,
        layer_config=cfg.env.layer_config,
        n_envs=cfg.n_envs
    )
    vec_env = NativeBatchedVecEnv(env)
    # Wrap with VecMonitor to track ep_rew_mean and ep_len_mean
    return VecMonitor(vec_env, filename=None)


def make_test_env(cfg: TrainingConfig, opponents, log_dir: str, replay_path: str) -> NativeBatchedVecEnv:
    world_args = build_world_args(cfg, log_dir, save_replay=True, replay_path=replay_path)

    # For testing, we usually just want 1 game playing out
    env = BombermanGymEnv(
        world_args,
        opponents=opponents,
        layer_config=cfg.env.layer_config,
        n_envs=1
    )

    return NativeBatchedVecEnv(env)

def architecture_info(cfg: TrainingConfig) -> dict:
    """Static description of the model, written once into run_manifest.json
    so a run folder is self-describing without needing this script."""
    return {
        "policy": "MultiInputPolicy",
        "algorithm": "MaskablePPO",
        "features_extractor_class": BombermanFeatureExtractor.__name__,
        "layer_config": cfg.env.layer_config,
    }


def build_model(env: VecEnv, cfg: TrainingConfig, tensorboard_log: str) -> MaskablePPO:
    policy_kwargs = dict(features_extractor_class=BombermanFeatureExtractor)
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        clip_range_vf=cfg.ppo.clip_range_vf,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        target_kl=cfg.ppo.target_kl,
    )


def play_test_game(
    model,
    cfg: TrainingConfig,
    opponents,
    ckman: CheckpointManager,
    timesteps_done: int,
) -> None:
    match_name = cfg.env.match_name or "match"
    replay_path = ckman.replays_dir / f"{match_name}_{timesteps_done:010d}.pkl"

    test_env = make_test_env(cfg, opponents, str(ckman.logs_dir), str(replay_path))

    obs = test_env.reset()
    done = False
    total_reward = 0
    while not done:
        action_masks = get_action_masks(test_env)
        action, _ = model.predict(obs, deterministic=False, action_masks=action_masks)
        obs, reward, dones, info = test_env.step(action)
        total_reward += reward[0]
        done = dones[0]
    test_env.close()
    print(f"Eval game finished at {timesteps_done} timesteps, total_reward={total_reward}")
    print(f"Saved eval replay -> {replay_path}")


def run(cfg: TrainingConfig, resume_from: str | None = None, resume_checkpoint: str | None = None) -> None:
    freeze_support()

    if resume_from:
        ckman = CheckpointManager.resume(resume_from, runs_dir=cfg.runs_dir)
        cfg = ckman.config
        print(f"Resuming run '{cfg.run_name}' from {ckman.run_dir}")
    else:
        ckman = CheckpointManager.new(cfg, architecture_info(cfg))
        print(f"Starting new run '{cfg.run_name}' in {ckman.run_dir}")

    pool = OpponentPool(ckman, cfg.self_play)
    opponents = pool.current_opponents()

    env = make_train_env(cfg, opponents, str(ckman.logs_dir))
    model = build_model(env, cfg, str(ckman.tensorboard_dir))

    timesteps_done = 0
    if resume_from:
        checkpoint_dir = (
            ckman.get_checkpoint(resume_checkpoint) if resume_checkpoint
            else ckman.latest_checkpoint()
        )
        if checkpoint_dir is not None:
            loaded_model = ckman.load_model(MaskablePPO, checkpoint_dir, env=env)
            model = loaded_model
            timesteps_done = ckman.resolved_timesteps(checkpoint_dir)
            print(f"Loaded checkpoint {checkpoint_dir.name} ({timesteps_done} timesteps)")
        else:
            print("No checkpoint found in this run yet — starting from scratch.")

    while timesteps_done < cfg.total_timesteps:
        chunk = min(cfg.save_every_timesteps, cfg.total_timesteps - timesteps_done)

        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="PPO",
        )
        timesteps_done = model.num_timesteps

        # Get metrics from the monitor (now they'll be available!)
        ep_rew_mean = model.logger.name_to_value.get("rollout/ep_rew_mean")
        ep_len_mean = model.logger.name_to_value.get("rollout/ep_len_mean")

        # Print them explicitly
        if ep_rew_mean is not None:
            print(f"  ep_rew_mean: {ep_rew_mean:.4f}")
        if ep_len_mean is not None:
            print(f"  ep_len_mean: {ep_len_mean:.1f}")

        ckpt_dir = ckman.save_checkpoint(
            model,
            timesteps_done,
            extra_metadata={
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
                "opponents": pool.last_opponent_descriptions(),
            },
        )
        print(f"Saved checkpoint at {timesteps_done} timesteps -> {ckpt_dir}")

        if cfg.self_play.enabled:
            pool.maybe_add_checkpoint(ckpt_dir, timesteps_done)
            opponents = pool.current_opponents()
            env.env_method("set_opponents", opponents)  # Note: env is now VecMonitor

        if cfg.eval_every_save:
            play_test_game(model, cfg, opponents, ckman, timesteps_done)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Run name or path to resume from")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint name to resume from (default: latest)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(DEFAULT_CONFIG, resume_from=args.resume, resume_checkpoint=args.checkpoint)
