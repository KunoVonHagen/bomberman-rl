from __future__ import annotations

import argparse
import copy
from multiprocessing import freeze_support
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn, LinearSchedule
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv

from agent_code.ppo_agent.train import (
    resolve_device,
    make_train_env,
    OpponentResampleCallback,
)
from agent_code.ppo_agent.config import load_overrides_file
from agent_code.ppo_agent.training_schedule import DEFAULT_SCHEDULE, load_schedule

from .config import DEFAULT_CONFIG, TrainingConfig, DQNConfig
from .checkpoint_manager import CheckpointManager
from .opponent_pool import OpponentPool, OpponentSampler
from .model import BombermanFeatureExtractor, MaskableDQN


def architecture_info(cfg: TrainingConfig) -> dict:
    """Describe the model architecture and policy configuration for this run."""
    return {
        "policy": "MultiInputPolicy",
        "algorithm": "MaskableDQN",
        "features_extractor_class": BombermanFeatureExtractor.__name__,
        "layer_config": cfg.env.layer_config,
    }


def build_model(env: VecEnv, cfg: TrainingConfig, tensorboard_log: str, device: str) -> MaskableDQN:
    policy_kwargs = dict(features_extractor_class=BombermanFeatureExtractor)
    return MaskableDQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
        learning_rate=cfg.dqn.learning_rate,
        buffer_size=cfg.dqn.buffer_size,
        learning_starts=cfg.dqn.learning_starts,
        batch_size=cfg.dqn.batch_size,
        tau=cfg.dqn.tau,
        gamma=cfg.dqn.gamma,
        train_freq=cfg.dqn.train_freq,
        gradient_steps=cfg.dqn.gradient_steps,
        target_update_interval=cfg.dqn.target_update_interval,
        exploration_fraction=cfg.dqn.exploration_fraction,
        exploration_initial_eps=cfg.dqn.exploration_initial_eps,
        exploration_final_eps=cfg.dqn.exploration_final_eps,
        max_grad_norm=cfg.dqn.max_grad_norm,
    )


def apply_dqn_hyperparams(model: MaskableDQN, dqn_cfg: DQNConfig) -> None:
    """Update a live MaskableDQN model's hyperparameters to match a new DQNConfig."""
    model.learning_rate = dqn_cfg.learning_rate
    model.lr_schedule = get_schedule_fn(dqn_cfg.learning_rate)
    model.tau = dqn_cfg.tau
    model.gamma = dqn_cfg.gamma
    model.train_freq = TrainFreq(dqn_cfg.train_freq, TrainFrequencyUnit.STEP)
    model.gradient_steps = dqn_cfg.gradient_steps
    model.target_update_interval = dqn_cfg.target_update_interval
    model.max_grad_norm = dqn_cfg.max_grad_norm
    model.exploration_initial_eps = dqn_cfg.exploration_initial_eps
    model.exploration_final_eps = dqn_cfg.exploration_final_eps
    model.exploration_fraction = dqn_cfg.exploration_fraction
    model.exploration_schedule = LinearSchedule(
        dqn_cfg.exploration_initial_eps, dqn_cfg.exploration_final_eps, dqn_cfg.exploration_fraction,
    )


def apply_schedule_up_to(
    cfg: TrainingConfig,
    schedule: list[dict],
    timesteps_done: int,
    applied_idx: int,
    *,
    model: Optional[MaskableDQN] = None,
    env=None,
    ckman: Optional[CheckpointManager] = None,
    verbose: bool = True,
) -> int:
    """Apply all schedule stages up to timesteps_done, returning the index of the last one applied."""
    target_idx = applied_idx
    for i, stage in enumerate(schedule):
        if timesteps_done >= stage["at_timesteps"]:
            target_idx = i
    if target_idx == applied_idx:
        return applied_idx

    any_applied = False
    for i in range(applied_idx + 1, target_idx + 1):
        stage = schedule[i]
        applied = cfg.apply_overrides(stage["overrides"], restrict_to=TrainingConfig.RESUMABLE_FIELDS)
        if applied:
            any_applied = True
            if verbose:
                print(f"[schedule] stage {i} (at_timesteps>={stage['at_timesteps']}, "
                      f"timesteps={timesteps_done}/{cfg.total_timesteps}) activated:")
                for dotted_key, old, new in applied:
                    print(f"    {dotted_key}: {old!r} -> {new!r}")

    if any_applied:
        if model is not None:
            apply_dqn_hyperparams(model, cfg.dqn)
        if env is not None:
            env.env_method("set_reward_config", cfg.rewards)
        if ckman is not None:
            ckman.update_manifest_config()

    return target_idx


class ScheduleCallback(BaseCallback):
    """Applies training schedule overrides at the start of each rollout."""

    def __init__(self, cfg: TrainingConfig, ckman: CheckpointManager, schedule: list[dict],
                 applied_idx: int = -1, verbose: int = 1):
        super().__init__(verbose)
        self.cfg = cfg
        self.ckman = ckman
        self.schedule = schedule
        self.applied_idx = applied_idx

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self.applied_idx = apply_schedule_up_to(
            self.cfg, self.schedule, self.model.num_timesteps, self.applied_idx,
            model=self.model, env=self.training_env, ckman=self.ckman, verbose=bool(self.verbose),
        )


def run(
    cfg: TrainingConfig,
    resume_from: str | None = None,
    resume_checkpoint: str | None = None,
    overrides: dict | None = None,
    overrides_file: str | None = None,
    schedule: list[dict] | None = DEFAULT_SCHEDULE,
) -> None:
    freeze_support()

    if resume_from:
        ckman = CheckpointManager.resume(resume_from, runs_dir=cfg.runs_dir)
        cfg = ckman.config

        merged: dict = {}
        auto_path = ckman.run_dir / "config_overrides.json"
        if auto_path.exists():
            merged.update(load_overrides_file(auto_path))
            print(f"Found {auto_path}, applying its overrides")
        if overrides_file:
            merged.update(load_overrides_file(overrides_file))
        if overrides:
            merged.update(overrides)

        if merged:
            applied = cfg.apply_overrides(merged, restrict_to=TrainingConfig.RESUMABLE_FIELDS)
            ckman.update_manifest_config()
            print("Applied overrides on resume:")
            for path, old, new in applied:
                print(f"  {path}: {old!r} -> {new!r}")
        print(f"Resuming run '{cfg.run_name}' from {ckman.run_dir}")
    else:
        merged = {}
        if overrides_file:
            merged.update(load_overrides_file(overrides_file))
        if overrides:
            merged.update(overrides)
        if merged:
            cfg = copy.deepcopy(cfg)
            cfg.apply_overrides(merged)
        ckman = CheckpointManager.new(cfg, architecture_info(cfg))
        cfg = ckman.config
        print(f"Starting new run '{cfg.run_name}' in {ckman.run_dir}")

    device = resolve_device(cfg.device)

    pool = OpponentPool(ckman, cfg.self_play)
    opponents = pool.current_opponents()

    env = make_train_env(cfg, opponents, str(ckman.logs_dir))
    env.env_method("set_opponent_resampler", OpponentSampler(pool))

    model = build_model(env, cfg, str(ckman.tensorboard_dir), device)

    timesteps_done = 0
    if resume_from:
        checkpoint_dir = (
            ckman.get_checkpoint(resume_checkpoint) if resume_checkpoint
            else ckman.latest_checkpoint()
        )
        if checkpoint_dir is not None:
            model = ckman.load_model(MaskableDQN, checkpoint_dir, env=env, device=device)
            timesteps_done = ckman.resolved_timesteps(checkpoint_dir)
            apply_dqn_hyperparams(model, cfg.dqn)
            print(f"Loaded checkpoint {checkpoint_dir.name} ({timesteps_done} timesteps)")
        else:
            print("No checkpoint found in this run yet -- starting from scratch.")

    schedule_applied_idx = -1
    if schedule:
        schedule_applied_idx = apply_schedule_up_to(
            cfg, schedule, timesteps_done, schedule_applied_idx,
            model=model, env=env, ckman=ckman, verbose=True,
        )

    opponent_callback = OpponentResampleCallback(
        pool, every_n_rollouts=cfg.self_play.resample_every_n_rollouts, verbose=1,
    )
    schedule_callback = ScheduleCallback(
        cfg, ckman, schedule or [], applied_idx=schedule_applied_idx, verbose=1,
    )
    learn_callback = CallbackList([opponent_callback, schedule_callback])

    while timesteps_done < cfg.total_timesteps:
        chunk = min(cfg.save_every_timesteps, cfg.total_timesteps - timesteps_done)

        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="DQN",
            callback=learn_callback,
        )
        timesteps_done = model.num_timesteps

        ep_rew_mean = model.logger.name_to_value.get("rollout/ep_rew_mean")
        ep_len_mean = model.logger.name_to_value.get("rollout/ep_len_mean")
        if ep_rew_mean is not None:
            print(f"  ep_rew_mean: {ep_rew_mean:.4f}")
        if ep_len_mean is not None:
            print(f"  ep_len_mean: {ep_len_mean:.1f}")

        ckpt_dir = ckman.save_checkpoint(
            model,
            timesteps_done,
            extra_metadata={
                "ep_rew_mean": float(ep_rew_mean) if ep_rew_mean is not None else None,
                "ep_len_mean": float(ep_len_mean) if ep_len_mean is not None else None,
                "opponents": pool.last_opponent_descriptions(),
            },
        )
        print(f"Saved checkpoint at {timesteps_done} timesteps -> {ckpt_dir}")

        if cfg.self_play.enabled:
            pool.maybe_add_checkpoint(ckpt_dir, timesteps_done)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Run name or path to resume from")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to resume from (default: latest)")
    p.add_argument("--overrides-file", type=str, default=None, help="Path to a JSON file of dotted-key overrides")
    p.add_argument("--schedule-file", type=str, default=None, help="Path to a JSON curriculum schedule file")
    p.add_argument("--no-schedule", action="store_true", help="Disable the curriculum schedule for this run")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--n-shards", type=int, default=None)
    p.add_argument("--set", dest="overrides", type=str, nargs="*", default=[],
                   metavar="path.to.field=value",
                   help="Override any TrainingConfig field by dotted path, e.g. --set dqn.learning_rate=5e-5")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = DEFAULT_CONFIG

    overrides: dict[str, object] = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    if args.n_shards is not None:
        overrides["n_shards"] = args.n_shards
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects path.to.field=value, got: {item!r}")
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()

    if args.schedule_file is not None:
        schedule = load_schedule(args.schedule_file)
    elif args.no_schedule:
        schedule = None
    else:
        schedule = DEFAULT_SCHEDULE

    run(
        cfg,
        resume_from=args.resume,
        resume_checkpoint=args.checkpoint,
        overrides=overrides or None,
        overrides_file=args.overrides_file,
        schedule=schedule,
    )