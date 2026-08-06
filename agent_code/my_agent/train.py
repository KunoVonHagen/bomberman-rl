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

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from agent_code.my_agent.gym_environment import BombermanGymEnv
from environment import WorldArgs
from model import BombermanFeatureExtractor

from config import DEFAULT_CONFIG, TrainingConfig
from checkpoint_manager import CheckpointManager
from opponent_pool import OpponentPool


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


def make_train_env(cfg: TrainingConfig, opponents, log_dir: str) -> VecNormalize:
    world_args = build_world_args(cfg, log_dir, save_replay=False)
    env = make_vec_env(
        lambda: ActionMasker(
            BombermanGymEnv(world_args, opponents=opponents, layer_config=cfg.env.layer_config),
            mask_fn,
        ),
        n_envs=cfg.n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def make_test_env(cfg: TrainingConfig, opponents, log_dir: str, replay_path: str) -> VecNormalize:
    world_args = build_world_args(cfg, log_dir, save_replay=True, replay_path=replay_path)
    env = make_vec_env(
        lambda: ActionMasker(
            BombermanGymEnv(world_args, opponents=opponents, layer_config=cfg.env.layer_config),
            mask_fn,
        ),
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
    )
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def architecture_info(cfg: TrainingConfig) -> dict:
    """Static description of the model, written once into run_manifest.json
    so a run folder is self-describing without needing this script."""
    return {
        "policy": "MultiInputPolicy",
        "algorithm": "MaskablePPO",
        "features_extractor_class": BombermanFeatureExtractor.__name__,
        "layer_config": cfg.env.layer_config,
    }


def build_model(env: VecNormalize, cfg: TrainingConfig, tensorboard_log: str) -> MaskablePPO:
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
    obs_rms,
    ret_rms,
    ckman: CheckpointManager,
    timesteps_done: int,
) -> None:
    match_name = cfg.env.match_name or "match"
    replay_path = ckman.replays_dir / f"{match_name}_{timesteps_done:010d}.pkl"

    test_env = make_test_env(cfg, opponents, str(ckman.logs_dir), str(replay_path))
    test_env.obs_rms = obs_rms
    test_env.ret_rms = ret_rms
    test_env.training = False

    obs = test_env.reset()
    done = False
    while not done:
        action_masks = get_action_masks(test_env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, dones, info = test_env.step(action)
        done = dones[0]
    test_env.close()
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
            loaded_model, vecnorm_path = ckman.load_model(MaskablePPO, checkpoint_dir, env=env)
            model = loaded_model
            if vecnorm_path is not None:
                env = VecNormalize.load(str(vecnorm_path), env.venv)
                model.set_env(env)
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

        vec_env = model.get_vec_normalize_env()
        ckpt_dir = ckman.save_checkpoint(
            model,
            vec_env,
            timesteps_done,
            extra_metadata={
                "ep_rew_mean": model.logger.name_to_value.get("rollout/ep_rew_mean"),
                "ep_len_mean": model.logger.name_to_value.get("rollout/ep_len_mean"),
                "opponents": pool.last_opponent_descriptions(),
            },
        )
        print(f"Saved checkpoint at {timesteps_done} timesteps -> {ckpt_dir}")

        if cfg.self_play.enabled:
            pool.maybe_add_checkpoint(ckpt_dir, timesteps_done)
            opponents = pool.current_opponents()
            env.env_method("set_opponents", opponents)

        if cfg.eval_every_save:
            obs_rms = copy.deepcopy(vec_env.obs_rms)
            ret_rms = copy.deepcopy(vec_env.ret_rms)
            play_test_game(model, cfg, opponents, obs_rms, ret_rms, ckman, timesteps_done)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Run name or path to resume from")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint name to resume from (default: latest)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(DEFAULT_CONFIG, resume_from=args.resume, resume_checkpoint=args.checkpoint)
