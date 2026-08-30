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
import multiprocessing as mp
import pathlib
from datetime import datetime
from multiprocessing import freeze_support

import numpy as np
import torch
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

    device = resolve_device(cfg.device)

    pool = OpponentPool(ckman, cfg.self_play)
    opponents = pool.current_opponents()

    env = make_train_env(cfg, opponents, str(ckman.logs_dir))
    model = build_model(env, cfg, str(ckman.tensorboard_dir), device)

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
    if args.device is not None:
        cfg = copy.deepcopy(cfg)
        cfg.device = args.device
    if args.n_envs is not None:
        cfg = copy.deepcopy(cfg)
        cfg.n_envs = args.n_envs
    if args.n_shards is not None:
        cfg = copy.deepcopy(cfg)
        cfg.n_shards = args.n_shards

    run(cfg, resume_from=args.resume, resume_checkpoint=args.checkpoint)