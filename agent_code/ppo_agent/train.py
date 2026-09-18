from __future__ import annotations

import argparse
import copy
import ctypes
import multiprocessing as mp
import os
import pathlib
from collections import deque
from datetime import datetime
from multiprocessing import freeze_support
from typing import Optional

import numpy as np

import events as e
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from environment import WorldArgs

from .gym_environment import ACTIONS, MAX_OPPONENTS, BombermanGymEnv, observation_shapes
from .model import BombermanFeatureExtractor
from .config import (
    DEFAULT_CONFIG,
    TrainingConfig,
    PPOConfig,
    RewardConfig,
    load_overrides_file,
)
from .checkpoint_manager import CheckpointManager
from .rollout_buffer import PinnedMaskableDictRolloutBuffer
from .symmetry import augment_rollout_buffer
from .opponent_pool import (LearnerOpponentInference, OpponentPool, OpponentSampler, load_inference_model,
                            set_inference_device)
from .training_schedule import DEFAULT_SCHEDULE, load_schedule


def mask_fn(env):
    return env.action_masks()


def resolve_device(requested: str = "auto") -> str:
    """
    Resolve the device string to use for PyTorch (and SB3) training.
    """
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


class NativeBatchedVecEnv(VecEnv):
    """
    Wraps a BombermanGymEnv that already natively batches N envs in-process
    to make it compatible with Stable-Baselines3's VecEnv API.
    """

    def __init__(self, batched_env):
        self.env = batched_env
        super().__init__(
            num_envs=self.env.n_envs,
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space
        )
        self._actions = None
        self._last_infos: list[dict] = []

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._last_infos = infos
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
        indices = kwargs.pop('indices', None)

        result = getattr(self.env, method_name)(*args, **kwargs)

        if isinstance(result, np.ndarray) and result.ndim > 0 and result.shape[0] == self.num_envs:
            per_env = [result[i] for i in range(self.num_envs)]
            if indices is not None:
                return [per_env[i] for i in indices]
            return per_env
        elif isinstance(result, list) and len(result) == self.num_envs:
            if indices is not None:
                return [result[i] for i in indices]
            return list(result)
        else:
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


_WORKER_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def _start_single_threaded(process) -> None:
    saved = {name: os.environ.get(name) for name in _WORKER_THREAD_VARS}
    os.environ.update({name: "1" for name in _WORKER_THREAD_VARS})
    try:
        process.start()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class SharedStepBuffers:
    def __init__(self, ctx, n_envs: int, grid_shape: tuple, n_features: int, n_actions: int,
                 n_shards: int = 1, opponent_rows: int = 0):
        self.n_envs = int(n_envs)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.n_shards = int(n_shards)
        self.opponent_rows = int(opponent_rows)
        cells = 2 * self.n_envs
        rows = 2 * max(1, self.opponent_rows)
        self.raw = {
            "grid": ctx.RawArray(ctypes.c_float, cells * int(np.prod(self.grid_shape))),
            "features": ctx.RawArray(ctypes.c_float, cells * self.n_features),
            "rewards": ctx.RawArray(ctypes.c_float, cells),
            "terminated": ctx.RawArray(ctypes.c_bool, cells),
            "truncated": ctx.RawArray(ctypes.c_bool, cells),
            "masks": ctx.RawArray(ctypes.c_bool, cells * self.n_actions),
            "opp_grid": ctx.RawArray(ctypes.c_float, rows * int(np.prod(self.grid_shape)) if self.opponent_rows else 1),
            "opp_features": ctx.RawArray(ctypes.c_float, rows * self.n_features if self.opponent_rows else 1),
            "opp_masks": ctx.RawArray(ctypes.c_bool, rows * self.n_actions if self.opponent_rows else 1),
            "opp_model": ctx.RawArray(ctypes.c_int64, rows if self.opponent_rows else 1),
            "opp_count": ctx.RawArray(ctypes.c_int64, 2 * self.n_shards),
        }

    def views(self) -> dict:
        def view(name, dtype, shape):
            return np.frombuffer(self.raw[name], dtype=dtype).reshape((2, self.n_envs) + shape)

        out = {
            "grid": view("grid", np.float32, self.grid_shape),
            "features": view("features", np.float32, (self.n_features,)),
            "rewards": view("rewards", np.float32, ()),
            "terminated": view("terminated", np.bool_, ()),
            "truncated": view("truncated", np.bool_, ()),
            "masks": view("masks", np.bool_, (self.n_actions,)),
            "opp_count": np.frombuffer(self.raw["opp_count"], dtype=np.int64).reshape(2, self.n_shards),
        }
        if self.opponent_rows:
            rows = self.opponent_rows
            out["opp_grid"] = np.frombuffer(self.raw["opp_grid"], dtype=np.float32).reshape((2, rows) + self.grid_shape)
            out["opp_features"] = np.frombuffer(self.raw["opp_features"], dtype=np.float32).reshape(2, rows, self.n_features)
            out["opp_masks"] = np.frombuffer(self.raw["opp_masks"], dtype=np.bool_).reshape(2, rows, self.n_actions)
            out["opp_model"] = np.frombuffer(self.raw["opp_model"], dtype=np.int64).reshape(2, rows)
        return out

    def check(self, observation_space, action_space) -> None:
        grid_shape = tuple(observation_space["grid_tensor"].shape)
        n_features = int(observation_space["features"].shape[0])
        if grid_shape != self.grid_shape or n_features != self.n_features or int(action_space.n) != self.n_actions:
            raise RuntimeError(
                f"shared step buffers were sized for grid {self.grid_shape}, {self.n_features} features, "
                f"{self.n_actions} actions but the shard reports grid {grid_shape}, {n_features} features, "
                f"{int(action_space.n)} actions"
            )


def _shard_worker(remote, parent_remote, world_args_kwargs: dict, opponents, layer_config,
                   shard_n_envs: int, reward_config: Optional[RewardConfig] = None, env_version: int = 1,
                   shared: Optional[SharedStepBuffers] = None, shard_index: int = 0):
    """
    Worker function for a single shard process.
    It creates a BombermanGymEnv with the given world_args and handles commands from the parent process via the remote pipe.
    """
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parent_remote.close()
    torch.set_num_threads(1)

    world_args = WorldArgs(**world_args_kwargs)
    env = BombermanGymEnv(
        world_args, opponents=opponents, layer_config=layer_config,
        n_envs=shard_n_envs, reward_config=reward_config, env_version=env_version,
    )
    views = shared.views()
    lo, hi = shard_index * shard_n_envs, (shard_index + 1) * shard_n_envs
    opp_rows = shared.opponent_rows // max(1, shared.n_shards)
    olo, ohi = shard_index * opp_rows, (shard_index + 1) * opp_rows

    def publish(slot):
        if opp_rows:
            views["opp_count"][slot, shard_index] = env.publish_opponent_observations(
                views["opp_grid"][slot, olo:ohi], views["opp_features"][slot, olo:ohi],
                views["opp_masks"][slot, olo:ohi], views["opp_model"][slot, olo:ohi])

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            slot, actions, opponent_actions = data
            if opponent_actions is not None:
                env.set_opponent_actions(opponent_actions)
            obs, rewards, terminated, truncated, infos = env.step(actions)
            views["grid"][slot, lo:hi] = obs["grid_tensor"]
            views["features"][slot, lo:hi] = obs["features"]
            views["rewards"][slot, lo:hi] = rewards
            views["terminated"][slot, lo:hi] = terminated
            views["truncated"][slot, lo:hi] = truncated
            views["masks"][slot, lo:hi] = env.action_masks()
            publish(slot)
            remote.send(infos)
        elif cmd == "reset":
            seed, options, slot = data
            obs, infos = env.reset(seed=seed, options=options)
            publish(slot)
            remote.send((obs, infos, env.action_masks()))
        elif cmd == "action_masks":
            remote.send(env.action_masks())
        elif cmd == "has_attr":
            remote.send(hasattr(env, data))
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
    Wraps N BombermanGymEnv shards, each of which natively batches M envs in-process,
    to make it compatible with Stable-Baselines3's VecEnv API.
    """

    def __init__(self, cfg: TrainingConfig, opponents, log_dir: str, n_shards: int):
        if cfg.n_envs % n_shards != 0:
            raise ValueError(f"cfg.n_envs ({cfg.n_envs}) must be divisible by n_shards ({n_shards})")
        self.n_shards = n_shards
        self.shard_size = cfg.n_envs // n_shards
        self._cfg = cfg

        ctx = mp.get_context("spawn")
        grid_shape, n_features = observation_shapes(cfg.env.layer_config, cfg.env.env_version)
        opponent_rows = cfg.n_envs * MAX_OPPONENTS if cfg.self_play.opponent_inference == "learner" else 0
        self._shared = SharedStepBuffers(ctx, cfg.n_envs, grid_shape, n_features, len(ACTIONS),
                                         n_shards=n_shards, opponent_rows=opponent_rows)
        self._views = self._shared.views()
        self._slot = 1
        self._last_action_masks: Optional[np.ndarray] = None
        self._opponent_evaluator = None
        self._opponent_actions: Optional[list] = None
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
                      self.shard_size, cfg.rewards, e.env_version, self._shared, i),
                daemon=True,
            )
            _start_single_threaded(p)
            self.processes.append(p)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        self._shared.check(obs_space, act_space)

        super().__init__(num_envs=cfg.n_envs, observation_space=obs_space, action_space=act_space)
        self._actions = None

    def set_opponent_evaluator(self, evaluator) -> None:
        self._opponent_evaluator = evaluator

    def _evaluate_opponents(self, slot: int) -> Optional[list]:
        if self._opponent_evaluator is None or not self._shared.opponent_rows:
            return None
        views = self._views
        counts = views["opp_count"][slot].copy()
        if counts.sum() == 0:
            return [np.zeros(0, dtype=np.int64) for _ in range(self.n_shards)]
        per_shard = self._shared.opponent_rows // self.n_shards
        sel = np.concatenate([np.arange(s * per_shard, s * per_shard + int(counts[s])) for s in range(self.n_shards)])
        obs = {"grid_tensor": views["opp_grid"][slot][sel], "features": views["opp_features"][slot][sel]}
        greedy = self._opponent_evaluator(obs, views["opp_masks"][slot][sel], views["opp_model"][slot][sel])
        return list(np.split(np.asarray(greedy, dtype=np.int64), np.cumsum(counts)[:-1]))

    def reset(self, seed=None, options=None):
        for remote in self.remotes:
            remote.send(("reset", (seed, options, self._slot)))
        results = [remote.recv() for remote in self.remotes]
        obs_list, info_lists, mask_list = zip(*results)
        self._last_infos = [info for infos in info_lists for info in infos]
        self._last_action_masks = np.concatenate(mask_list, axis=0)
        self._opponent_actions = self._evaluate_opponents(self._slot)
        return _concat_obs(list(obs_list))

    def step_async(self, actions: np.ndarray) -> None:
        opponent_actions = self._opponent_actions
        self._opponent_actions = None
        self._slot ^= 1
        shards = np.split(np.asarray(actions), self.n_shards)
        for i, (remote, shard_actions) in enumerate(zip(self.remotes, shards)):
            remote.send(("step", (self._slot, shard_actions, None if opponent_actions is None else opponent_actions[i])))

    def step_wait(self):
        infos = [info for remote in self.remotes for info in remote.recv()]
        slot, views = self._slot, self._views
        obs = {"grid_tensor": views["grid"][slot], "features": views["features"][slot]}
        rewards = views["rewards"][slot].copy()
        dones = views["terminated"][slot] | views["truncated"][slot]
        self._last_infos = infos
        self._last_action_masks = views["masks"][slot].copy()
        self._opponent_actions = self._evaluate_opponents(slot)
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
        if self._last_action_masks is None:
            for remote in self.remotes:
                remote.send(("action_masks", None))
            self._last_action_masks = np.concatenate([remote.recv() for remote in self.remotes], axis=0)
        return self._last_action_masks

    def has_attr(self, attr_name: str) -> bool:
        for remote in self.remotes:
            remote.send(("has_attr", attr_name))
        return all(remote.recv() for remote in self.remotes)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        if method_name == "action_masks" and not args and not kwargs:
            per_env = list(self.action_masks())
            return [per_env[i] for i in indices] if indices is not None else per_env
        self._last_action_masks = None
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


def _sync_learner_inference(env, pool: OpponentPool) -> None:
    inference = getattr(getattr(env, "venv", env), "_opponent_evaluator", None)
    if inference is None:
        return
    env.env_method("set_opponent_model_index", inference.update(pool._checkpoints))


class GateStatsVecMonitor(VecMonitor):
    def __init__(self, venv, window: int = 200):
        super().__init__(venv, filename=None)
        self._score_window: deque = deque(maxlen=window)
        self._suicide_window: deque = deque(maxlen=window)

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        for i, done in enumerate(dones):
            if done:
                self._score_window.append(float(infos[i].get("score", 0.0)))
                self._suicide_window.append(float(e.KILLED_SELF in infos[i].get("events", ())))
        return obs, rewards, dones, infos

    def gate_stats(self) -> Optional[dict]:
        if not self._score_window:
            return None
        return {
            "episodes": len(self._score_window),
            "score_mean": float(np.mean(self._score_window)),
            "suicide_rate": float(np.mean(self._suicide_window)),
        }


def make_train_env(cfg: TrainingConfig, opponents, log_dir: str) -> VecMonitor:
    if cfg.n_shards > 1:
        vec_env = ShardedNativeBatchedVecEnv(cfg, opponents, log_dir, n_shards=cfg.n_shards)
    else:
        world_args = build_world_args(cfg, log_dir, save_replay=False)
        env = BombermanGymEnv(
            world_args,
            opponents=opponents,
            layer_config=cfg.env.layer_config,
            env_version=cfg.env.env_version,
            n_envs=cfg.n_envs,
            reward_config=cfg.rewards,
        )
        vec_env = NativeBatchedVecEnv(env)

    return GateStatsVecMonitor(vec_env)


def make_test_env(cfg: TrainingConfig, opponents, log_dir: str, replay_path: str | None) -> NativeBatchedVecEnv:
    world_args = build_world_args(cfg, log_dir, save_replay=replay_path is not None, replay_path=replay_path)

    env = BombermanGymEnv(
        world_args,
        opponents=opponents,
        layer_config=cfg.env.layer_config,
        env_version=cfg.env.env_version,
        n_envs=1,
        reward_config=cfg.rewards,
    )

    return NativeBatchedVecEnv(env)

def architecture_info(cfg: TrainingConfig) -> dict:
    """
    Return a dict describing the model architecture and policy configuration for this run.
    """
    return {
        "policy": "MultiInputPolicy",
        "algorithm": "MaskablePPO",
        "features_extractor_class": BombermanFeatureExtractor.__name__,
        "dropout": cfg.ppo.dropout,
        "layer_config": cfg.env.layer_config,
    }


def build_model(env: VecEnv, cfg: TrainingConfig, tensorboard_log: str, device: str) -> MaskablePPO:
    policy_kwargs = dict(
        features_extractor_class=BombermanFeatureExtractor,
        features_extractor_kwargs=dict(dropout=cfg.ppo.dropout),
        optimizer_kwargs=dict(weight_decay=cfg.ppo.weight_decay),
    )
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
        rollout_buffer_class=PinnedMaskableDictRolloutBuffer,
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


def _resize_rollout_buffer(model: MaskablePPO, new_n_steps: int) -> None:
    """
    Resize the rollout buffer of a MaskablePPO model to accommodate a new number of steps.
    """
    old_size = model.rollout_buffer.buffer_size
    buffer = model.rollout_buffer
    buffer_cls = type(buffer)
    new_buffer = buffer_cls(
        new_n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )
    buffer.__dict__.update(new_buffer.__dict__)
    print(f"  rollout buffer resized ({old_size} -> {new_n_steps} steps)")


def apply_ppo_hyperparams(
    model: MaskablePPO, ppo_cfg: PPOConfig, *, defer_n_steps_resize: bool = False,
) -> None:
    """
    Update an existing MaskablePPO model's hyperparameters to match a new PPOConfig.
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
    model.batch_size = ppo_cfg.batch_size
    for group in model.policy.optimizer.param_groups:
        group["weight_decay"] = float(ppo_cfg.weight_decay)

    if model.n_steps == ppo_cfg.n_steps:
        return

    if not defer_n_steps_resize:
        _resize_rollout_buffer(model, ppo_cfg.n_steps)
        return

    model._pending_n_steps = ppo_cfg.n_steps


class SymmetryAugmentationCallback(BaseCallback):
    def __init__(self, cfg: TrainingConfig, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self.cfg.ppo.symmetry_augmentation:
            return
        augment_rollout_buffer(self.model.rollout_buffer, self.model.policy, self.model.batch_size)


class OpponentResampleCallback(BaseCallback):
    """
    A callback that resamples opponents at the end of each training rollout.
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

        self.training_env.env_method("set_opponent_resampler", OpponentSampler(self.pool))
        _sync_learner_inference(self.training_env, self.pool)

        if self.verbose:
            arrangement_str = ", ".join(
                f"{label} {p:.0%}" for label, p in self.pool.arrangement_distribution()
            )
            scenario_str = ", ".join(
                f"{scenario} {p:.0%}" for scenario, p in self.pool.scenario_distribution()
            )
            print(f"[opponents] rollout {self._rollout_count}: resynced self-play pool "
                  f"({self.pool.num_checkpoints} checkpoint(s)) to training env "
                  f"-- distribution: {arrangement_str}")
            print(f"[scenario] rollout {self._rollout_count}: distribution: {scenario_str}")


def schedule_gate_open(stage: dict, timesteps_done: int, stats: Optional[dict]) -> bool:
    gate = stage.get("gate")
    if not gate:
        return True
    if timesteps_done >= stage["at_timesteps"] + int(gate.get("max_delay_timesteps", 0)):
        return True
    if not stats or stats["episodes"] < int(gate.get("min_episodes", 100)):
        return False
    if "min_score" in gate and stats["score_mean"] < float(gate["min_score"]):
        return False
    if "max_suicide_rate" in gate and stats["suicide_rate"] > float(gate["max_suicide_rate"]):
        return False
    return True


def apply_schedule_up_to(
    cfg: TrainingConfig,
    schedule: list[dict],
    timesteps_done: int,
    applied_idx: int,
    *,
    model=None,
    env=None,
    ckman: Optional[CheckpointManager] = None,
    verbose: bool = True,
) -> int:
    """
    Apply all stages in the training schedule up to the current count of timesteps done.
    Returns the index of the last stage that was applied.
    """
    target_idx = applied_idx
    stats = env.gate_stats() if env is not None and hasattr(env, "gate_stats") else None
    for i, stage in enumerate(schedule):
        if i <= applied_idx:
            continue
        if timesteps_done < stage["at_timesteps"] or not schedule_gate_open(stage, timesteps_done, stats):
            break
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
            apply_ppo_hyperparams(model, cfg.ppo, defer_n_steps_resize=True)
        if env is not None:
            env.env_method("set_reward_config", cfg.rewards)
        if ckman is not None:
            ckman.update_manifest_config()

    return target_idx


class ScheduleCallback(BaseCallback):
    """
    A callback that applies training schedule overrides at the start of each rollout.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        ckman: CheckpointManager,
        pool: OpponentPool,
        schedule: list[dict],
        applied_idx: int = -1,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.cfg = cfg
        self.ckman = ckman
        self.pool = pool
        self.schedule = schedule
        self.applied_idx = applied_idx

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        resize_to = getattr(self.model, "_pending_buffer_resize", None)
        if resize_to is not None:
            if resize_to != self.model.rollout_buffer.buffer_size:
                _resize_rollout_buffer(self.model, resize_to)
            self.model._pending_buffer_resize = None

        self.applied_idx = apply_schedule_up_to(
            self.cfg,
            self.schedule,
            self.model.num_timesteps,
            self.applied_idx,
            model=self.model,
            env=self.training_env,
            ckman=self.ckman,
            verbose=bool(self.verbose),
        )

    def _on_rollout_end(self) -> None:
        pending = getattr(self.model, "_pending_n_steps", None)
        if pending is not None:
            if pending != self.model.n_steps:
                self.model.n_steps = pending
                self.model._pending_buffer_resize = pending
            self.model._pending_n_steps = None


class MaxRolloutsCallback(BaseCallback):
    """
    A callback that stops training after a maximum number of rollouts.
    """

    def __init__(self, max_rollouts: int | None, verbose: int = 0):
        super().__init__(verbose)
        self.max_rollouts = max_rollouts
        self.rollouts_started = 0
        self.limit_reached = False

    def _on_rollout_start(self) -> None:
        self.rollouts_started += 1

    def _on_step(self) -> bool:
        if self.max_rollouts is not None and self.rollouts_started > self.max_rollouts:
            self.limit_reached = True
            return False
        return True


def play_test_game(
    model,
    cfg: TrainingConfig,
    opponents,
    replays_dir: pathlib.Path,
    logs_dir: pathlib.Path,
    timesteps_done: int,
) -> None:
    match_name = cfg.env.match_name or "match"
    replay_path = pathlib.Path(replays_dir) / f"{match_name}_{timesteps_done:010d}.pkl"

    test_env = make_test_env(cfg, opponents, str(logs_dir), str(replay_path))

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
    print(f"Eval game finished at {timesteps_done} timesteps, score={info[0]['score']}, "
          f"alive={info[0]['alive']}, total_reward={total_reward}", flush=True)
    print(f"Saved eval replay -> {replay_path}", flush=True)


EVAL_SUITE_BOTS = [
    "agent_code.rule_based_agent.callbacks",
    "agent_code.coin_collector_agent.callbacks",
    "agent_code.peaceful_agent.callbacks",
]

EVAL_SUITE_GENERALIZATION_CASES: list[tuple[str, int]] = [
    ("coin-heaven", 0),
    ("empty", 0),
    ("classic", 1),
    ("classic", 2),
]


def eval_suite_cases(
    cfg: TrainingConfig,
    bot_paths: list[str] | None = None,
    generalization_cases: list[tuple[str, int]] | None = None,
    generalization_opponent: str = "agent_code.rule_based_agent.callbacks",
) -> list[dict]:
    bot_paths = bot_paths if bot_paths is not None else EVAL_SUITE_BOTS
    generalization_cases = (
        generalization_cases if generalization_cases is not None else EVAL_SUITE_GENERALIZATION_CASES
    )
    cases = []
    for bot_path in bot_paths:
        short = bot_path.split(".")[1]
        cases.append(dict(key=bot_path, opponent=bot_path, n_opponents=3, scenario=cfg.env.scenario, tag=short,
                          label=f"vs {short} ({cfg.env.scenario}, 3 opp)"))
    for scenario, n_opponents in generalization_cases:
        tag = f"gen_{scenario}_{n_opponents}opp"
        cases.append(dict(key=tag, opponent=generalization_opponent, n_opponents=n_opponents, scenario=scenario,
                          tag=tag, label=f"generalization [{scenario}, {n_opponents} opp]"))
    return cases


def run_eval_case(
    model,
    cfg: TrainingConfig,
    case: dict,
    replays_dir: pathlib.Path,
    logs_dir: pathlib.Path,
    timesteps_done: int,
    n_episodes: int,
    save_replays: bool,
) -> dict[str, float | None]:
    case_cfg = cfg
    if case["scenario"] != cfg.env.scenario:
        case_cfg = copy.deepcopy(cfg)
        case_cfg.env.scenario = case["scenario"]
    opponents = [OpponentPool._resolve_static(case["opponent"])] * case["n_opponents"]

    stats: dict[str, list[float]] = {"score": [], "win": [], "survived": [], "length": [], "reward": []}
    for i in range(n_episodes):
        replay_path = None
        if save_replays:
            replay_path = str(pathlib.Path(replays_dir) / f"eval_{case['tag']}_{timesteps_done:010d}_{i}.pkl")
        test_env = make_test_env(case_cfg, opponents, str(logs_dir), replay_path)
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
        final = info[0]
        stats["score"].append(float(final["score"]))
        stats["survived"].append(float(final["alive"]))
        stats["length"].append(float(final["step"]))
        stats["reward"].append(total_reward)
        if final["opponent_scores"]:
            stats["win"].append(float(final["score"] > max(final["opponent_scores"])))
    return {key: (sum(values) / len(values) if values else None) for key, values in stats.items()}


def _format_eval_summary(summary: dict[str, float | None]) -> str:
    win = "-" if summary["win"] is None else f"{100 * summary['win']:.0f}%"
    return (f"score {summary['score']:.2f}, win {win}, survived {100 * summary['survived']:.0f}%, "
            f"length {summary['length']:.0f}, reward {summary['reward']:.2f}")


def _print_eval_case(case: dict, summary: dict, timesteps_done: int, n_episodes: int) -> None:
    print(f"  eval {case['label']}: {_format_eval_summary(summary)} (n={n_episodes})", flush=True)


_EVAL_POOL_MODEL = None


def _eval_pool_init(model_path: str) -> None:
    global _EVAL_POOL_MODEL
    torch.set_num_threads(1)
    _EVAL_POOL_MODEL = _load_eval_model(model_path)


def _eval_pool_case(args) -> tuple[str, dict]:
    cfg, case, replays_dir, logs_dir, timesteps_done, n_episodes, save_replays = args
    summary = run_eval_case(_EVAL_POOL_MODEL, cfg, case, replays_dir, logs_dir, timesteps_done, n_episodes, save_replays)
    _print_eval_case(case, summary, timesteps_done, n_episodes)
    return case["key"], summary


def _load_eval_model(model_path: str):
    return load_inference_model(model_path)


def run_eval_suite(
    model,
    cfg: TrainingConfig,
    replays_dir: pathlib.Path,
    logs_dir: pathlib.Path,
    timesteps_done: int,
    n_episodes: int | None = None,
    save_replays: bool | None = None,
    workers: int = 1,
    model_path: str | None = None,
) -> dict[str, dict[str, float | None]]:
    """
    Run a suite of evaluation games against a set of bots (all at the standard 1v3
    "classic" table) plus a set of generalization cases that vary the scenario and/or
    opponent count, and return the average rewards for each.
    """
    n_episodes = cfg.eval_suite_episodes if n_episodes is None else int(n_episodes)
    save_replays = cfg.eval_suite_replays if save_replays is None else bool(save_replays)
    cases = eval_suite_cases(cfg)
    results: dict[str, dict[str, float | None]] = {}

    if workers > 1 and model_path is not None:
        jobs = [(cfg, case, replays_dir, logs_dir, timesteps_done, n_episodes, save_replays) for case in cases]
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(jobs)), initializer=_eval_pool_init, initargs=(model_path,)) as pool:
            for key, summary in pool.map(_eval_pool_case, jobs, chunksize=1):
                results[key] = summary
        return {case["key"]: results[case["key"]] for case in cases}

    for case in cases:
        summary = run_eval_case(model, cfg, case, replays_dir, logs_dir, timesteps_done, n_episodes, save_replays)
        results[case["key"]] = summary
        _print_eval_case(case, summary, timesteps_done, n_episodes)
    return results


def evaluate_checkpoint(
    cfg: TrainingConfig,
    run_dir: str,
    ckpt_dir: str,
    timesteps_done: int,
    test_opponents,
    model=None,
) -> dict:
    run_path = pathlib.Path(run_dir)
    replays_dir, logs_dir = run_path / "replays", run_path / "logs"
    model_path = str(pathlib.Path(ckpt_dir) / "model.zip")
    if model is None:
        model = _load_eval_model(model_path)
    started = datetime.now()
    results = run_eval_suite(
        model, cfg, replays_dir, logs_dir, timesteps_done,
        workers=cfg.eval_suite_workers, model_path=model_path,
    )
    CheckpointManager.update_metadata(pathlib.Path(ckpt_dir), {"eval_suite": results})
    seconds = (datetime.now() - started).total_seconds()
    print(f"  eval suite for {timesteps_done} timesteps finished in {seconds:.0f}s "
          f"-> {pathlib.Path(ckpt_dir).name}/metadata.json", flush=True)
    play_test_game(model, cfg, test_opponents, replays_dir, logs_dir, timesteps_done)
    return results


def _background_eval(cfg: TrainingConfig, run_dir: str, ckpt_dir: str, timesteps_done: int, test_opponents) -> None:
    torch.set_num_threads(1)
    evaluate_checkpoint(cfg, run_dir, ckpt_dir, timesteps_done, test_opponents)


class BackgroundEvaluator:
    def __init__(self, cfg: TrainingConfig, ckman: CheckpointManager):
        self.cfg = cfg
        self.ckman = ckman
        self.process = None
        self.timesteps = None

    def wait(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            print(f"  waiting for the background eval of {self.timesteps} timesteps to finish...", flush=True)
        self.process.join()
        if self.process.exitcode != 0:
            print(f"  background eval of {self.timesteps} timesteps exited with code {self.process.exitcode}", flush=True)
        self.process = None

    def start(self, ckpt_dir: pathlib.Path, timesteps_done: int, test_opponents) -> None:
        self.wait()
        ctx = mp.get_context("spawn")
        self.process = ctx.Process(
            target=_background_eval,
            args=(self.cfg, str(self.ckman.run_dir), str(ckpt_dir), timesteps_done, test_opponents),
        )
        self.timesteps = timesteps_done
        _start_single_threaded(self.process)
        print(f"  eval suite for {timesteps_done} timesteps started in the background (pid {self.process.pid})", flush=True)


def run(
    cfg: TrainingConfig,
    resume_from: str | None = None,
    resume_checkpoint: str | None = None,
    overrides: dict | None = None,
    overrides_file: str | None = None,
    max_rollouts: int | None = None,
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
        print(
            f"Tip: drop a 'config_overrides.json' into {ckman.run_dir} and it will be "
            f"applied automatically next time you --resume this run."
        )

    device = resolve_device(cfg.device)
    if cfg.self_play.opponent_inference == "learner" and cfg.n_shards <= 1:
        set_inference_device(device)

    pool = OpponentPool(ckman, cfg.self_play)
    opponents = pool.current_opponents()

    env = make_train_env(cfg, opponents, str(ckman.logs_dir))
    if cfg.self_play.opponent_inference == "learner" and cfg.n_shards > 1:
        env.venv.set_opponent_evaluator(LearnerOpponentInference(device))
    env.env_method("set_opponent_resampler", OpponentSampler(pool))
    _sync_learner_inference(env, pool)

    model = build_model(env, cfg, str(ckman.tensorboard_dir), device)

    opponent_callback = OpponentResampleCallback(
        pool, every_n_rollouts=cfg.self_play.resample_every_n_rollouts, verbose=1,
    )
    max_rollouts_callback = MaxRolloutsCallback(max_rollouts, verbose=1)

    if max_rollouts is not None:
        print(f"[max-rollouts] capping this process to {max_rollouts} rollout(s), "
              f"then stopping (run remains resumable afterwards)")

    timesteps_done = 0
    if resume_from:
        checkpoint_dir = (
            ckman.get_checkpoint(resume_checkpoint) if resume_checkpoint
            else ckman.latest_checkpoint()
        )
        if checkpoint_dir is not None:
            loaded_model = ckman.load_model(MaskablePPO, checkpoint_dir, env=env, device=device,
                                            custom_objects={"rollout_buffer_class": PinnedMaskableDictRolloutBuffer})
            model = loaded_model
            timesteps_done = ckman.resolved_timesteps(checkpoint_dir)
            apply_ppo_hyperparams(model, cfg.ppo)
            print(f"Loaded checkpoint {checkpoint_dir.name} ({timesteps_done} timesteps)")
        else:
            print("No checkpoint found in this run yet — starting from scratch.")

    schedule_applied_idx = -1
    if schedule:
        schedule_applied_idx = apply_schedule_up_to(
            cfg, schedule, timesteps_done, schedule_applied_idx,
            model=model, env=env, ckman=ckman, verbose=True,
        )

    schedule_callback = ScheduleCallback(
        cfg, ckman, pool, schedule or [], applied_idx=schedule_applied_idx, verbose=1,
    )
    learn_callback = CallbackList([
        opponent_callback, SymmetryAugmentationCallback(cfg), max_rollouts_callback, schedule_callback,
    ])
    evaluator = BackgroundEvaluator(cfg, ckman)

    while timesteps_done < cfg.total_timesteps:
        chunk = min(cfg.save_every_timesteps, cfg.total_timesteps - timesteps_done)

        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="PPO",
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
                "eval_suite": None,
            },
        )
        print(f"Saved checkpoint at {timesteps_done} timesteps -> {ckpt_dir}")

        if cfg.self_play.enabled:
            pool.maybe_add_checkpoint(ckpt_dir, timesteps_done)

        if cfg.eval_every_save:
            test_opponents = pool.current_opponents()
            if cfg.eval_suite_background:
                evaluator.start(ckpt_dir, timesteps_done, test_opponents)
            else:
                evaluate_checkpoint(cfg, str(ckman.run_dir), str(ckpt_dir), timesteps_done, test_opponents, model=model)

        if max_rollouts_callback.limit_reached:
            print(f"[max-rollouts] hit the {max_rollouts}-rollout cap for this process "
                  f"at {timesteps_done} timesteps -- stopping here. "
                  f"cfg.total_timesteps ({cfg.total_timesteps}) not yet reached; "
                  f"resume this run (without --max-rollouts, or with a new one) to continue.")
            break

    evaluator.wait()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Run name or path to resume from")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint name to resume from (default: latest)")
    p.add_argument("--overrides-file", type=str, default=None,
                   help="Path to a JSON file of {'dotted.key': value} overrides -- the file-based "
                        "counterpart to --set, for when you don't want a long CLI command. On "
                        "--resume, a 'config_overrides.json' sitting in the run's own directory is "
                        "also applied automatically (no flag needed); this flag layers on top of "
                        "that, and --set layers on top of both.")
    p.add_argument("--schedule-file", type=str, default=None,
                   help="Path to a JSON file describing the curriculum schedule (see "
                        "training_schedule.py for the format and the built-in default) used to "
                        "anneal reward shaping, PPO hyperparameters, and the self-play opponent "
                        "mix as training progresses. Stages are keyed by an absolute count of "
                        "timesteps reached so far (not wall-clock time, not a fraction), and are applied "
                        "directly by this same process as it trains -- no second process or "
                        "on-disk live-overrides file needed, and --resume picks the schedule back "
                        "up at the right stage automatically. Defaults to the built-in curriculum; "
                        "pass --no-schedule to disable scheduling entirely.")
    p.add_argument("--no-schedule", action="store_true",
                   help="Disable the curriculum schedule entirely (cfg stays exactly as configured "
                        "/ overridden, for the whole run). Ignored if --schedule-file is also given.")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"],
                   help="Override cfg.device. Use 'cuda' to force GPU and hard-fail if unavailable.")
    p.add_argument("--n-envs", type=int, default=None, help="Override cfg.n_envs for this run.")
    p.add_argument("--n-shards", type=int, default=None,
                   help="Override cfg.n_shards -- split n_envs across this many subprocesses for real "
                        "multi-core usage (n_envs must be divisible by n_shards).")
    p.add_argument("--max-rollouts", type=int, default=None,
                   help="Stop this training PROCESS after at most this many PPO rollouts "
                        "(each cfg.ppo.n_steps * cfg.n_envs timesteps), then exit cleanly "
                        "(with a checkpoint saved) instead of continuing to cfg.total_timesteps. "
                        "This is a per-process cap only: it is never saved into "
                        "run_manifest.json, so it does not carry over across --resume "
                        "invocations, and on --resume it counts rollouts made by *this* "
                        "process from 0, ignoring however many rollouts earlier processes "
                        "already completed for the run. Useful for splitting a long run into "
                        "cluster jobs with a fixed wall-clock budget each.")
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
                        "Entries do NOT need to share the same total -- mixing different totals "
                        "(including 0, i.e. no opponents at all) is recommended so the agent "
                        "generalizes to coin-only / reduced-opponent play instead of collapsing "
                        "outside a fixed opponent count. Each entry's n_static + n_self_play must "
                        "still fit within the game's opponent seats (<= 3). Two accepted formats "
                        "-- shorthand (recommended on Windows/PowerShell, no quote characters "
                        "needed): semicolon-separated 'n_static,n_self_play,weight' triples, e.g. "
                        "'0,0,1;0,3,3;1,2,3;2,1,2;3,0,1'. Or JSON (needs careful quoting on "
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
    cfg.total_timesteps = cfg.ppo.n_steps * cfg.n_envs * 3
    cfg.save_every_timesteps = cfg.ppo.n_steps * cfg.n_envs
    cfg.eval_every_save = True
    cfg.self_play.enabled = False
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
        max_rollouts=args.max_rollouts,
        schedule=schedule,
    )