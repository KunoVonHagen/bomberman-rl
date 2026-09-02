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

    python train.py --resume run_20260101-101500 --set ppo.learning_rate=1e-4 ppo.ent_coef=0.0 total_timesteps=2_000_000_000
        Resume, but override arbitrary model-independent hyperparameters for
        the next phase of training. Only fields in TrainingConfig.RESUMABLE_FIELDS
        are allowed here -- anything that would change the model's architecture
        or the environment's observation format is rejected, since that would
        desync a resumed run from the checkpoint it's loading.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import pathlib
from datetime import datetime
from multiprocessing import freeze_support

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from environment import WorldArgs

from agent_code.my_agent.gym_environment import BombermanGymEnv
from agent_code.my_agent.model import BombermanFeatureExtractor
from agent_code.my_agent.config import DEFAULT_CONFIG, TrainingConfig, PPOConfig
from agent_code.my_agent.checkpoint_manager import CheckpointManager
from agent_code.my_agent.opponent_pool import OpponentPool


def mask_fn(env):
    return env.action_masks()


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' -> cuda if available else cpu, and print a clear,
    unmissable diagnostic so a silent CPU fallback never goes unnoticed
    in a scrollback log on the cluster."""
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


def _shard_worker(remote, parent_remote, world_args_kwargs: dict, opponents, layer_config, shard_n_envs: int):
    """
    Entry point for one shard subprocess. Owns a real BombermanGymEnv
    simulating `shard_n_envs` games natively-batched (i.e. via its own
    internal Python for-loop over that shard's envs). Talks to the parent
    over a Pipe using the same verb set SB3's own SubprocVecEnv workers use,
    so shard results can be concatenated like any other VecEnv batch.
    """
    parent_remote.close()
    from environment import WorldArgs
    from agent_code.my_agent.gym_environment import BombermanGymEnv

    world_args = WorldArgs(**world_args_kwargs)
    env = BombermanGymEnv(world_args, opponents=opponents, layer_config=layer_config, n_envs=shard_n_envs)

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


def _concat_obs(obs_list: list) -> dict:
    """Dict obs -> concatenate each key across shards along the batch axis."""
    keys = obs_list[0].keys()
    return {k: np.concatenate([o[k] for o in obs_list], axis=0) for k in keys}


class ShardedNativeBatchedVecEnv(VecEnv):
    """
    Runs several BombermanGymEnv batches in separate OS processes ("shards"),
    each natively-batched in-process, and concatenates their outputs into one
    big batch for SB3 -- functionally a drop-in replacement for
    NativeBatchedVecEnv when you want actual multi-core usage.

    Why this exists: BombermanGymEnv.step()/reset() loop over envs with a
    plain Python `for env in range(self.n_envs): self._advance(env, ...)`.
    That means simulating N envs in ONE process is single-threaded no matter
    how many CPU cores are requested -- more n_envs just makes that loop
    longer, and extra cores sit idle. Sharding cfg.n_envs across
    `n_shards` OS processes lets those per-shard loops run in parallel
    instead of sequentially, without touching the env's internals. The
    long-term fix is vectorizing _advance() itself with numpy across the
    batch dimension; this is the practical fix that doesn't require
    rewriting the game logic.
    """

    def __init__(self, cfg: TrainingConfig, opponents, log_dir: str, n_shards: int):
        if cfg.n_envs % n_shards != 0:
            raise ValueError(f"cfg.n_envs ({cfg.n_envs}) must be divisible by n_shards ({n_shards})")
        self.n_shards = n_shards
        self.shard_size = cfg.n_envs // n_shards
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
                args=(work_remote, remote, world_args_kwargs, opponents, e.layer_config, self.shard_size),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        obs_space, act_space = self.remotes[0].recv()

        super().__init__(num_envs=cfg.n_envs, observation_space=obs_space, action_space=act_space)
        self._actions = None

    def reset(self, seed=None, options=None):
        for remote in self.remotes:
            remote.send(("reset", (seed, options)))
        results = [remote.recv() for remote in self.remotes]
        obs_list, info_lists = zip(*results)
        self._last_infos = [info for infos in info_lists for info in infos]
        return _concat_obs(list(obs_list))

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        shards = np.split(np.asarray(self._actions), self.n_shards)
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

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

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

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        n = len(indices) if indices is not None else self.num_envs
        return [False] * n

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

    def get_images(self) -> list:
        return [None] * self.num_envs


def make_train_env(cfg: TrainingConfig, opponents, log_dir: str) -> VecMonitor:
    if cfg.n_shards > 1:
        vec_env = ShardedNativeBatchedVecEnv(cfg, opponents, log_dir, n_shards=cfg.n_shards)
    else:
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


def build_model(env: VecEnv, cfg: TrainingConfig, tensorboard_log: str, device: str) -> MaskablePPO:
    policy_kwargs = dict(features_extractor_class=BombermanFeatureExtractor)
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
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


def apply_ppo_hyperparams(model: MaskablePPO, ppo_cfg: PPOConfig) -> None:
    """Re-apply every PPOConfig field onto an already-constructed/loaded model.

    This exists because loading a checkpoint (MaskablePPO.load / ckman.load_model)
    restores the hyperparameters that were saved *into that checkpoint* -- so
    building a fresh model from an updated cfg and then loading weights on top
    of it silently reverts learning_rate, clip_range, ent_coef, etc. back to
    whatever they were when the checkpoint was written. Call this right after
    loading to make --set overrides on --resume actually take effect.

    Every field is safe to just assign except `n_steps`, which sizes the
    rollout buffer that was already allocated at model-construction time --
    that one requires rebuilding the buffer, not just flipping an int.
    """
    model.learning_rate = ppo_cfg.learning_rate
    model.lr_schedule = get_schedule_fn(ppo_cfg.learning_rate)
    model.clip_range = get_schedule_fn(ppo_cfg.clip_range)
    model.clip_range_vf = (
        get_schedule_fn(ppo_cfg.clip_range_vf) if ppo_cfg.clip_range_vf is not None else None
    )
    model.gamma = ppo_cfg.gamma
    model.gae_lambda = ppo_cfg.gae_lambda
    model.ent_coef = ppo_cfg.ent_coef
    model.vf_coef = ppo_cfg.vf_coef
    model.n_epochs = ppo_cfg.n_epochs
    model.target_kl = ppo_cfg.target_kl
    model.batch_size = ppo_cfg.batch_size  # only read at train()-time; safe to assign directly

    if model.n_steps != ppo_cfg.n_steps:
        old_n_steps = model.n_steps
        buffer_cls = type(model.rollout_buffer)
        model.n_steps = ppo_cfg.n_steps
        model.rollout_buffer = buffer_cls(
            ppo_cfg.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )
        print(f"  ppo.n_steps changed ({old_n_steps} -> {ppo_cfg.n_steps}): rebuilt rollout buffer")


class OpponentResampleCallback(BaseCallback):
    """Draws a fresh OpponentPool arrangement and pushes it into the training
    env at the start of every rollout (i.e. every `n_steps * n_envs`
    timesteps SB3 collects), instead of only after every checkpoint save.

    Without this, every one of the n_envs parallel games in a rollout -- and
    every rollout for a full `save_every_timesteps` chunk of them -- plays
    against the exact same fixed trio of opponents. That's exactly the kind
    of narrow, memorizable target that lets a policy settle into a
    single-opponent Nash equilibrium, or pick up exploits specific to one
    opponent (or one team composition) that don't generalize. Resampling
    frequently, combined with OpponentPool drawing a different arrangement
    (mix of static bots vs self-play snapshots) each time, keeps the
    training distribution of opponents wide throughout the run.
    """

    def __init__(self, pool: OpponentPool, every_n_rollouts: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.pool = pool
        self.every_n_rollouts = max(1, every_n_rollouts)
        self._rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self._rollout_count += 1
        if (self._rollout_count - 1) % self.every_n_rollouts != 0:
            return
        opponents = self.pool.current_opponents()
        self.training_env.env_method("set_opponents", opponents)
        if self.verbose:
            print(f"[opponents] rollout {self._rollout_count}: {self.pool.last_opponent_descriptions()}")


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
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, dones, info = test_env.step(action)
        total_reward += reward[0]
        done = dones[0]
    test_env.close()
    print(f"Eval game finished at {timesteps_done} timesteps, total_reward={total_reward}")
    print(f"Saved eval replay -> {replay_path}")


EVAL_SUITE_BOTS = [
    "agent_code.rule_based_agent.callbacks",
    "agent_code.coin_collector_agent.callbacks",
    "agent_code.peaceful_agent.callbacks",
]


def run_eval_suite(
    model,
    cfg: TrainingConfig,
    ckman: CheckpointManager,
    timesteps_done: int,
    n_episodes: int = 10,
    bot_paths: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate the current model (deterministic policy) against a battery
    of fixed scripted bots, independent of the self-play opponent mix used
    for training. Returns {bot_path: mean_reward}. Use this to tell whether
    self-play reward gains reflect real skill growth or just co-evolved
    exploitation of your own policy's blind spots."""
    bot_paths = bot_paths if bot_paths is not None else EVAL_SUITE_BOTS
    results: dict[str, float] = {}

    for bot_path in bot_paths:
        opponents = [OpponentPool._resolve_static(bot_path)] * 3
        bot_short_name = bot_path.split(".")[1]
        rewards = []

        for i in range(n_episodes):
            replay_path = (
                ckman.replays_dir
                / f"eval_{bot_short_name}_{timesteps_done:010d}_{i}.pkl"
            )
            test_env = make_test_env(cfg, opponents, str(ckman.logs_dir), str(replay_path))
            obs = test_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action_masks = get_action_masks(test_env)
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, dones, info = test_env.step(action)
                total_reward += float(reward[0])
                done = bool(dones[0])
            test_env.close()
            rewards.append(total_reward)

        mean_reward = float(sum(rewards) / len(rewards))
        results[bot_path] = mean_reward
        print(f"  eval vs {bot_short_name}: {mean_reward:.2f} (n={n_episodes})")

    return results


def run(
    cfg: TrainingConfig,
    resume_from: str | None = None,
    resume_checkpoint: str | None = None,
    overrides: dict | None = None,
) -> None:
    freeze_support()

    if resume_from:
        ckman = CheckpointManager.resume(resume_from, runs_dir=cfg.runs_dir)
        cfg = ckman.config
        if overrides:
            applied = cfg.apply_overrides(overrides, restrict_to=TrainingConfig.RESUMABLE_FIELDS)
            ckman.update_manifest_config()
            print("Applied overrides on resume:")
            for path, old, new in applied:
                print(f"  {path}: {old!r} -> {new!r}")
        print(f"Resuming run '{cfg.run_name}' from {ckman.run_dir}")
    else:
        if overrides:
            cfg = copy.deepcopy(cfg)
            cfg.apply_overrides(overrides)
        ckman = CheckpointManager.new(cfg, architecture_info(cfg))
        print(f"Starting new run '{cfg.run_name}' in {ckman.run_dir}")

    device = resolve_device(cfg.device)

    pool = OpponentPool(ckman, cfg.self_play)
    opponents = pool.current_opponents()

    env = make_train_env(cfg, opponents, str(ckman.logs_dir))
    model = build_model(env, cfg, str(ckman.tensorboard_dir), device)

    opponent_callback = OpponentResampleCallback(
        pool, every_n_rollouts=cfg.self_play.resample_every_n_rollouts, verbose=1,
    )

    timesteps_done = 0
    if resume_from:
        checkpoint_dir = (
            ckman.get_checkpoint(resume_checkpoint) if resume_checkpoint
            else ckman.latest_checkpoint()
        )
        if checkpoint_dir is not None:
            loaded_model = ckman.load_model(MaskablePPO, checkpoint_dir, env=env, device=device)
            model = loaded_model
            timesteps_done = ckman.resolved_timesteps(checkpoint_dir)
            apply_ppo_hyperparams(model, cfg.ppo)
            print(f"Loaded checkpoint {checkpoint_dir.name} ({timesteps_done} timesteps)")
        else:
            print("No checkpoint found in this run yet — starting from scratch.")

    while timesteps_done < cfg.total_timesteps:
        chunk = min(cfg.save_every_timesteps, cfg.total_timesteps - timesteps_done)

        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="PPO",
            callback=opponent_callback,
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

        eval_suite_results = None
        if cfg.eval_every_save:
            eval_suite_results = run_eval_suite(model, cfg, ckman, timesteps_done)

        ckpt_dir = ckman.save_checkpoint(
            model,
            timesteps_done,
            extra_metadata={
                "ep_rew_mean": float(ep_rew_mean) if ep_rew_mean is not None else None,
                "ep_len_mean": float(ep_len_mean) if ep_len_mean is not None else None,
                "opponents": pool.last_opponent_descriptions(),
                "eval_suite": eval_suite_results,
            },
        )
        print(f"Saved checkpoint at {timesteps_done} timesteps -> {ckpt_dir}")

        if cfg.self_play.enabled:
            pool.maybe_add_checkpoint(ckpt_dir, timesteps_done)

        if cfg.eval_every_save:
            test_opponents = pool.current_opponents()
            play_test_game(model, cfg, test_opponents, ckman, timesteps_done)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Run name or path to resume from")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint name to resume from (default: latest)")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"],
                   help="Override cfg.device. Use 'cuda' to force GPU and hard-fail if unavailable.")
    p.add_argument("--n-envs", type=int, default=None, help="Override cfg.n_envs for this run.")
    p.add_argument("--n-shards", type=int, default=None,
                   help="Override cfg.n_shards -- split n_envs across this many subprocesses for real "
                        "multi-core usage (n_envs must be divisible by n_shards).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a tiny, fast config (few envs, few timesteps, frequent saves, "
                        "eval disabled, self-play disabled, distinct run_name) to sanity-check "
                        "the whole pipeline end-to-end before a long cluster job.")
    p.add_argument("--static-opponents", type=str, nargs="*", default=None,
                   help="Shorthand for --set self_play.static_opponents=... (comma-joined "
                        "internally). Override self_play.static_opponents with these module "
                        "paths (e.g. agent_code.rule_based_agent.callbacks).")
    p.add_argument("--arrangements", type=str, default=None,
                   help="Shorthand for --set self_play.arrangements=.... Describes the possible "
                        "opponent-lineup shapes; every rollout draws one at random, weighted. "
                        "All entries must sum to the same total (the game's fixed opponent-seat "
                        "count). Two accepted formats -- shorthand (recommended on Windows/"
                        "PowerShell, no quote characters needed): semicolon-separated "
                        "'n_static,n_self_play,weight' triples, e.g. "
                        "'0,3,3;1,2,3;2,1,2;3,0,1'. Or JSON (needs careful quoting on "
                        "PowerShell): '[{\"n_static\":0,\"n_self_play\":3,\"weight\":3}]'")
    p.add_argument("--set", dest="overrides", type=str, nargs="*", default=[],
                   metavar="path.to.field=value",
                   help="Override any TrainingConfig field by dotted path, e.g. "
                        "--set ppo.learning_rate=1e-4 ppo.ent_coef=0.0 total_timesteps=2_000_000_000. "
                        "Repeatable / space-separated. On --resume, only fields in "
                        "TrainingConfig.RESUMABLE_FIELDS are accepted (anything that would "
                        "change the model's architecture or the environment's observation "
                        "format is rejected, to avoid desyncing a run from its checkpoint).")
    return p.parse_args()


def build_smoke_test_config(base_cfg: TrainingConfig) -> TrainingConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.run_name = f"smoketest_{datetime.now():%Y%m%d-%H%M%S}"
    cfg.total_timesteps = cfg.ppo.n_steps * cfg.n_envs * 3  # a handful of rollouts
    cfg.save_every_timesteps = cfg.ppo.n_steps * cfg.n_envs  # save after every rollout
    cfg.eval_every_save = True  # exercise play_test_game() too
    cfg.self_play.enabled = False  # keep it simple/fast; opponent pool has its own paths to check separately
    return cfg


if __name__ == "__main__":
    args = parse_args()

    cfg = DEFAULT_CONFIG
    if args.smoke_test:
        cfg = build_smoke_test_config(cfg)

    overrides: dict[str, object] = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.n_envs is not None:
        overrides["n_envs"] = args.n_envs
    if args.n_shards is not None:
        overrides["n_shards"] = args.n_shards
    if args.static_opponents is not None:
        overrides["self_play.static_opponents"] = ",".join(args.static_opponents)
    if args.arrangements is not None:
        overrides["self_play.arrangements"] = args.arrangements
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects path.to.field=value, got: {item!r}")
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()

    run(
        cfg,
        resume_from=args.resume,
        resume_checkpoint=args.checkpoint,
        overrides=overrides or None,
    )