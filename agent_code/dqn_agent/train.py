from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import pathlib
from collections import deque
from multiprocessing import freeze_support
from typing import Optional

import numpy as np
import torch

from environment import WorldArgs

from agent_code.dqn_agent.config import DEFAULT_CONFIG, TrainingConfig, DQNConfig, RewardConfig, load_overrides_file
from agent_code.dqn_agent.checkpoint_manager import CheckpointManager
from agent_code.dqn_agent.gym_environment import BombermanGymEnv
from agent_code.dqn_agent.model import BombermanFeatureExtractor, MaskableDQN
from agent_code.dqn_agent.opponent_pool import OpponentPool, OpponentSampler
from agent_code.dqn_agent.schedules import LinearSchedule
from agent_code.dqn_agent.training_schedule import DEFAULT_SCHEDULE, load_schedule


def resolve_device(requested: str = "auto") -> str:
    """Resolve the device string for training."""
    if requested != "auto":
        device = requested
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"[device] requested={requested!r} -> resolved={device!r}")
    print(f"[device] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"[device] GPU: {torch.cuda.get_device_name(idx)}")
        props = torch.cuda.get_device_properties(idx)
        print(f"[device] Total VRAM: {props.total_memory / 1024**3:.1f} GB")
    elif device == "cuda":
        raise RuntimeError(
            "device='cuda' was requested/forced but torch.cuda.is_available() "
            "is False. Check CUDA module / driver / torch build on this node."
        )
    else:
        print("[device] Running on CPU. This will be slow for the CNN "
              "feature extractor -- check that a GPU partition/module was "
              "requested if that wasn't intentional.")
    print("=" * 60)
    return device


def build_world_args(
    cfg: TrainingConfig,
    log_dir: str,
    save_replay: bool,
    replay_path: str | None = None,
) -> WorldArgs:
    """Build the environment arguments for a training run."""
    env_cfg = cfg.env
    return WorldArgs(
        scenario=env_cfg.scenario,
        seed=env_cfg.seed,
        silence_errors=env_cfg.silence_errors,
        no_gui=env_cfg.no_gui,
        make_video=env_cfg.make_video,
        save_replay=save_replay,
        save_stats=env_cfg.save_stats,
        turn_based=env_cfg.turn_based,
        update_interval=env_cfg.update_interval,
        log_dir=log_dir,
        match_name=env_cfg.match_name,
        fps=env_cfg.fps,
        replay=replay_path or env_cfg.replay,
        continue_without_training=env_cfg.continue_without_training,
    )


class NativeBatchedVecEnv:
    """Wrap a native batched Bomberman env with a vec-env-like interface."""

    def __init__(self, batched_env: BombermanGymEnv):
        self.env = batched_env
        self.num_envs = self.env.n_envs
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self._last_infos: list[dict] = []

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._last_infos = infos
        return obs

    def step(self, actions: np.ndarray):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = terminated | truncated
        self._last_infos = infos
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.env.close()

    def env_method(self, method_name: str, *args, **kwargs) -> list:
        indices = kwargs.pop("indices", None)
        result = getattr(self.env, method_name)(*args, **kwargs)

        if isinstance(result, np.ndarray) and result.ndim > 0 and result.shape[0] == self.num_envs:
            per_env = [result[i] for i in range(self.num_envs)]
            return [per_env[i] for i in indices] if indices is not None else per_env
        elif isinstance(result, list) and len(result) == self.num_envs:
            return [result[i] for i in indices] if indices is not None else list(result)
        n = len(indices) if indices is not None else self.num_envs
        return [result] * n

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

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


def _concat_obs(obs_list: list) -> dict:
    """Concatenate observations across shards along the batch axis."""
    keys = obs_list[0].keys()
    return {k: np.concatenate([o[k] for o in obs_list], axis=0) for k in keys}


def _shard_worker(remote, parent_remote, world_args_kwargs: dict, opponents, layer_config,
                   shard_n_envs: int, reward_config: Optional[RewardConfig] = None):
    """Run a single shard's BombermanGymEnv, serving commands from the parent process."""
    parent_remote.close()

    world_args = WorldArgs(**world_args_kwargs)
    env = BombermanGymEnv(
        world_args, opponents=opponents, layer_config=layer_config,
        n_envs=shard_n_envs, reward_config=reward_config,
    )

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            obs, rewards, terminated, truncated, infos = env.step(data)
            remote.send((obs, rewards, terminated, truncated, infos))
        elif cmd == "reset":
            seed, options = data
            obs, infos = env.reset(seed=seed, options=options)
            remote.send((obs, infos))
        elif cmd == "action_masks":
            remote.send(env.action_masks())
        elif cmd == "env_method":
            method_name, args, kwargs = data
            remote.send(getattr(env, method_name)(*args, **kwargs))
        elif cmd == "get_spaces":
            remote.send((env.single_observation_space, env.single_action_space))
        elif cmd == "get_attr":
            remote.send(getattr(env, data))
        elif cmd == "set_attr":
            attr_name, value = data
            setattr(env, attr_name, value)
            remote.send(None)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown shard command: {cmd!r}")


class ShardedNativeBatchedVecEnv:
    """Wraps N BombermanGymEnv shards, each natively batching M envs, as a
    single duck-typed vec-env, using multiprocessing pipes."""

    def __init__(self, cfg: TrainingConfig, opponents, log_dir: str, n_shards: int):
        if cfg.n_envs % n_shards != 0:
            raise ValueError(f"cfg.n_envs ({cfg.n_envs}) must be divisible by n_shards ({n_shards})")
        self.n_shards = n_shards
        self.shard_size = cfg.n_envs // n_shards
        self.num_envs = cfg.n_envs
        self._cfg = cfg

        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_shards)])
        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            e = cfg.env
            world_args_kwargs = dict(
                scenario=e.scenario,
                seed=e.seed,
                silence_errors=e.silence_errors,
                no_gui=e.no_gui,
                make_video=e.make_video,
                save_replay=False,
                save_stats=e.save_stats,
                turn_based=e.turn_based,
                update_interval=e.update_interval,
                log_dir=str(pathlib.Path(log_dir) / f"shard_{i}"),
                match_name=e.match_name,
                fps=e.fps,
                replay=e.replay,
                continue_without_training=e.continue_without_training,
            )
            p = ctx.Process(
                target=_shard_worker,
                args=(work_remote, remote, world_args_kwargs, opponents, e.layer_config,
                      self.shard_size, cfg.rewards),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        self.single_observation_space, self.single_action_space = self.remotes[0].recv()
        self._last_infos: list[dict] = []

    def reset(self, seed=None, options=None):
        for remote in self.remotes:
            remote.send(("reset", (seed, options)))
        results = [remote.recv() for remote in self.remotes]
        obs_list, info_lists = zip(*results)
        self._last_infos = [info for infos in info_lists for info in infos]
        return _concat_obs(list(obs_list))

    def step(self, actions: np.ndarray):
        shards = np.split(np.asarray(actions), self.n_shards)
        for remote, shard_actions in zip(self.remotes, shards):
            remote.send(("step", shard_actions))
        results = [remote.recv() for remote in self.remotes]
        obs_list, rew_list, term_list, trunc_list, info_lists = zip(*results)

        obs = _concat_obs(list(obs_list))
        rewards = np.concatenate(rew_list, axis=0)
        terminated = np.concatenate(term_list, axis=0)
        truncated = np.concatenate(trunc_list, axis=0)
        dones = terminated | truncated
        infos = [info for infos in info_lists for info in infos]
        self._last_infos = infos
        return obs, rewards, dones, infos

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)

    def action_masks(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("action_masks", None))
        return np.concatenate([remote.recv() for remote in self.remotes], axis=0)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        for remote in self.remotes:
            remote.send(("env_method", (method_name, args, kwargs)))
        results = [remote.recv() for remote in self.remotes]

        per_env = []
        for result in results:
            if isinstance(result, np.ndarray) and result.ndim > 0 and result.shape[0] == self.shard_size:
                per_env.extend(result[i] for i in range(self.shard_size))
            elif isinstance(result, list) and len(result) == self.shard_size:
                per_env.extend(result)
            else:
                per_env.extend([result] * self.shard_size)

        if indices is not None:
            return [per_env[i] for i in indices]
        return per_env

    def get_attr(self, attr_name: str, indices=None) -> list:
        for remote in self.remotes:
            remote.send(("get_attr", attr_name))

        per_env = []
        for remote in self.remotes:
            value = remote.recv()
            per_env.extend([value] * self.shard_size)
        if indices is not None:
            return [per_env[i] for i in indices]
        return per_env

    def set_attr(self, attr_name: str, values, indices=None) -> None:
        if indices is not None:
            raise NotImplementedError(
                "set_attr with per-index targeting isn't supported by "
                "ShardedNativeBatchedVecEnv -- shard boundaries don't map "
                "cleanly onto arbitrary env indices."
            )
        value = values[0] if isinstance(values, (list, np.ndarray)) else values
        for remote in self.remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in self.remotes:
            remote.recv()


class NativeMonitor:
    """Tracks per-episode reward/length over a trailing window, replacing
    SB3's VecMonitor. Transparently forwards everything else to the wrapped
    vec-env."""

    def __init__(self, venv, window: int = 100):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.single_observation_space = venv.single_observation_space
        self.single_action_space = venv.single_action_space

        self._ep_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._rew_window: deque = deque(maxlen=window)
        self._len_window: deque = deque(maxlen=window)

    def reset(self, seed=None, options=None):
        self._ep_rewards[:] = 0
        self._ep_lengths[:] = 0
        return self.venv.reset(seed=seed, options=options)

    def step(self, actions):
        obs, rewards, dones, infos = self.venv.step(actions)
        self._ep_rewards += rewards
        self._ep_lengths += 1
        for i, done in enumerate(dones):
            if done:
                self._rew_window.append(self._ep_rewards[i])
                self._len_window.append(self._ep_lengths[i])
                self._ep_rewards[i] = 0
                self._ep_lengths[i] = 0
        return obs, rewards, dones, infos

    def action_masks(self):
        return self.venv.action_masks()

    def env_method(self, *args, **kwargs):
        return self.venv.env_method(*args, **kwargs)

    def get_attr(self, *args, **kwargs):
        return self.venv.get_attr(*args, **kwargs)

    def set_attr(self, *args, **kwargs):
        return self.venv.set_attr(*args, **kwargs)

    def close(self):
        return self.venv.close()

    def ep_rew_mean(self) -> Optional[float]:
        return float(np.mean(self._rew_window)) if self._rew_window else None

    def ep_len_mean(self) -> Optional[float]:
        return float(np.mean(self._len_window)) if self._len_window else None


def make_train_env(cfg: TrainingConfig, opponents, log_dir: str) -> NativeMonitor:
    if cfg.n_shards > 1:
        vec_env = ShardedNativeBatchedVecEnv(cfg, opponents, log_dir, n_shards=cfg.n_shards)
    else:
        world_args = build_world_args(cfg, log_dir, save_replay=False)
        env = BombermanGymEnv(
            world_args,
            opponents=opponents,
            layer_config=cfg.env.layer_config,
            n_envs=cfg.n_envs,
            reward_config=cfg.rewards,
        )
        vec_env = NativeBatchedVecEnv(env)

    return NativeMonitor(vec_env)


class OpponentResampleCallback:
    """Resamples opponents every `every_n_timesteps` environment steps."""

    def __init__(self, pool: OpponentPool, every_n_timesteps: int, verbose: int = 0):
        self.pool = pool
        self.every_n_timesteps = max(1, every_n_timesteps)
        self.verbose = verbose
        self._next_resample_at = 0

    def on_training_start(self, model: MaskableDQN, env) -> None:
        self._next_resample_at = model.num_timesteps + self.every_n_timesteps

    def on_rollout_start(self, model: MaskableDQN, env) -> None:
        pass

    def on_step(self, model: MaskableDQN, env) -> None:
        if model.num_timesteps < self._next_resample_at:
            return
        self._next_resample_at = model.num_timesteps + self.every_n_timesteps

        env.env_method("set_opponent_resampler", OpponentSampler(self.pool))

        if self.verbose:
            arrangement_str = ", ".join(
                f"{label} {p:.0%}" for label, p in self.pool.arrangement_distribution()
            )
            scenario_str = ", ".join(
                f"{scenario} {p:.0%}" for scenario, p in self.pool.scenario_distribution()
            )
            print(f"[opponents] timesteps={model.num_timesteps}: resynced self-play pool "
                  f"({self.pool.num_checkpoints} checkpoint(s)) to training env "
                  f"-- distribution: {arrangement_str}")
            print(f"[scenario] timesteps={model.num_timesteps}: distribution: {scenario_str}")


class ProgressLoggingCallback:
    """Logs rolling training statistics periodically to stdout and optionally TensorBoard."""

    def __init__(self, tensorboard_dir: Optional[str] = None, every_n_timesteps: int = 5_000, verbose: int = 1):
        self.every_n_timesteps = max(1, every_n_timesteps)
        self.verbose = verbose
        self._next_log_at = 0
        self._writer = None
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(tensorboard_dir)

    def on_training_start(self, model: MaskableDQN, env) -> None:
        self._next_log_at = model.num_timesteps + self.every_n_timesteps

    def on_rollout_start(self, model: MaskableDQN, env) -> None:
        pass

    def on_step(self, model: MaskableDQN, env) -> None:
        if model.num_timesteps < self._next_log_at:
            return
        self._next_log_at = model.num_timesteps + self.every_n_timesteps

        ep_rew_mean = env.ep_rew_mean()
        ep_len_mean = env.ep_len_mean()

        if self.verbose:
            parts = [f"[progress] timesteps={model.num_timesteps}"]
            if ep_rew_mean is not None:
                parts.append(f"ep_rew_mean={ep_rew_mean:.4f}")
            if ep_len_mean is not None:
                parts.append(f"ep_len_mean={ep_len_mean:.1f}")
            if model.last_loss_mean is not None:
                parts.append(f"loss_mean={model.last_loss_mean:.5f}")
            parts.append(f"exploration_rate={model.exploration_rate:.4f}")
            print("  ".join(parts))

        if self._writer is not None:
            if ep_rew_mean is not None:
                self._writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, model.num_timesteps)
            if ep_len_mean is not None:
                self._writer.add_scalar("rollout/ep_len_mean", ep_len_mean, model.num_timesteps)
            if model.last_loss_mean is not None:
                self._writer.add_scalar("train/loss_mean", model.last_loss_mean, model.num_timesteps)
            self._writer.add_scalar("train/exploration_rate", model.exploration_rate, model.num_timesteps)
            self._writer.flush()


class ScheduleCallback:
    """Applies training schedule overrides at the start of each rollout."""

    def __init__(self, cfg: TrainingConfig, ckman: CheckpointManager, schedule: list[dict],
                 applied_idx: int = -1, verbose: int = 1):
        self.cfg = cfg
        self.ckman = ckman
        self.schedule = schedule
        self.applied_idx = applied_idx
        self.verbose = verbose

    def on_training_start(self, model: MaskableDQN, env) -> None:
        pass

    def on_rollout_start(self, model: MaskableDQN, env) -> None:
        self.applied_idx = apply_schedule_up_to(
            self.cfg, self.schedule, model.num_timesteps, self.applied_idx,
            model=model, env=env, ckman=self.ckman, verbose=bool(self.verbose),
        )

    def on_step(self, model: MaskableDQN, env) -> None:
        pass


def architecture_info(cfg: TrainingConfig) -> dict:
    """Describe the model architecture and policy configuration for this run."""
    return {
        "policy": "MultiInputPolicy",
        "algorithm": "MaskableDQN",
        "features_extractor_class": BombermanFeatureExtractor.__name__,
        "layer_config": cfg.env.layer_config,
    }


def build_model(env, cfg: TrainingConfig, device: str) -> MaskableDQN:
    exploration_duration = max(1, int(cfg.dqn.exploration_fraction * cfg.total_timesteps))
    return MaskableDQN(
        env.single_observation_space,
        env.single_action_space,
        n_envs=cfg.n_envs,
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
        exploration_duration=exploration_duration,
    )


def apply_dqn_hyperparams(model: MaskableDQN, dqn_cfg: DQNConfig, total_timesteps: int) -> None:
    """Update a live MaskableDQN model's hyperparameters to match a new DQNConfig."""
    model.learning_rate = dqn_cfg.learning_rate
    for group in model.optimizer.param_groups:
        group["lr"] = dqn_cfg.learning_rate
    model.tau = dqn_cfg.tau
    model.gamma = dqn_cfg.gamma
    model.train_freq = max(1, int(dqn_cfg.train_freq))
    model.gradient_steps = dqn_cfg.gradient_steps
    model.target_update_interval = max(1, int(dqn_cfg.target_update_interval))
    model.max_grad_norm = dqn_cfg.max_grad_norm
    model.exploration_initial_eps = dqn_cfg.exploration_initial_eps
    model.exploration_final_eps = dqn_cfg.exploration_final_eps
    model.exploration_fraction = dqn_cfg.exploration_fraction
    model.exploration_duration = max(1, int(dqn_cfg.exploration_fraction * total_timesteps))
    model.exploration_schedule = LinearSchedule(
        dqn_cfg.exploration_initial_eps, dqn_cfg.exploration_final_eps, model.exploration_duration,
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
            apply_dqn_hyperparams(model, cfg.dqn, cfg.total_timesteps)
        if env is not None:
            env.env_method("set_reward_config", cfg.rewards)
        if ckman is not None:
            ckman.update_manifest_config()

    return target_idx


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

    model = build_model(env, cfg, device)

    timesteps_done = 0
    if resume_from:
        checkpoint_dir = (
            ckman.get_checkpoint(resume_checkpoint) if resume_checkpoint
            else ckman.latest_checkpoint()
        )
        if checkpoint_dir is not None:
            model = ckman.load_model(MaskableDQN, checkpoint_dir, env=env, device=device)
            timesteps_done = ckman.resolved_timesteps(checkpoint_dir)
            apply_dqn_hyperparams(model, cfg.dqn, cfg.total_timesteps)
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
        pool, every_n_timesteps=cfg.self_play.resample_every_n_timesteps, verbose=1,
    )
    schedule_callback = ScheduleCallback(
        cfg, ckman, schedule or [], applied_idx=schedule_applied_idx, verbose=1,
    )
    progress_callback = ProgressLoggingCallback(
        tensorboard_dir=str(ckman.tensorboard_dir), every_n_timesteps=5_000, verbose=1,
    )
    learn_callbacks = [opponent_callback, schedule_callback, progress_callback]

    while timesteps_done < cfg.total_timesteps:
        chunk = min(cfg.save_every_timesteps, cfg.total_timesteps - timesteps_done)

        model.learn(env=env, total_timesteps=chunk, callbacks=learn_callbacks)
        timesteps_done = model.num_timesteps

        ep_rew_mean = env.ep_rew_mean()
        ep_len_mean = env.ep_len_mean()
        if ep_rew_mean is not None:
            print(f"  ep_rew_mean: {ep_rew_mean:.4f}")
        if ep_len_mean is not None:
            print(f"  ep_len_mean: {ep_len_mean:.1f}")
        if model.last_loss_mean is not None:
            print(f"  loss_mean: {model.last_loss_mean:.5f}")
        print(f"  exploration_rate: {model.exploration_rate:.4f}")

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