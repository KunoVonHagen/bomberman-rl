from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback
from typing import Optional

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

import settings as s
from .config import TrainingConfig
from .gym_environment import ACTION_INDICES, BombermanGymEnv, WorldArgs
from .symmetry import ACTION_PERM, N_SYMMETRIES, transform_masks, transform_observation
from .model import BombermanFeatureExtractor, InferenceOptimizer

RUN: str = "bomberman_ppo_20260919_021721"
CHECKPOINT: Optional[str] = "checkpoint_0139984896"
ENSEMBLE: list = []
TTA_SYMMETRIES: int = N_SYMMETRIES
DETERMINISTIC: bool = True
ENSEMBLE_FILE = "ensemble.txt"

AGENT_DIR = pathlib.Path(__file__).resolve().parent
ENV_PREFIX = "PPO_AGENT"
_ACTION_NAMES = {v: k for k, v in ACTION_INDICES.items() if k is not None}


def _setting(name: str, default):
    return os.environ.get(f"{ENV_PREFIX}_{name}", default)


def _member_specs(raw) -> list:
    entries = [e.strip() for e in raw.split(",")] if isinstance(raw, str) else list(raw)
    return [e for e in entries if e]


def _split_member(entry: str, default_run: str):
    run, sep, checkpoint = entry.rpartition("/")
    if not sep or run in ("", "."):
        return default_run, entry
    return run, checkpoint


def _resolve_run_dir(run: str) -> pathlib.Path:
    candidates = [
        AGENT_DIR / "runs" / run,
        AGENT_DIR / run,
        pathlib.Path(run),
        AGENT_DIR.parent.parent / "runs" / run,
    ]
    for run_dir in candidates:
        if (run_dir / "run_manifest.json").exists():
            return run_dir
    raise FileNotFoundError(
        f"ppo_agent: no run '{run}' with a run_manifest.json in any of "
        f"{[str(c) for c in candidates]} -- set RUN at the top of callbacks.py"
    )


def _read_ensemble_file(run_dir: pathlib.Path) -> list:
    pointer = run_dir / "checkpoints" / ENSEMBLE_FILE
    if not pointer.exists():
        raise FileNotFoundError(f"{pointer} not found (run evaluation.select_checkpoint --write-ensemble)")
    return [line.strip() for line in pointer.read_text().splitlines() if line.strip()]


def _symmetry_batch(obs: dict, masks: np.ndarray, n: int) -> tuple:
    views = [transform_observation(obs, k) for k in range(n)]
    batch = {"grid_tensor": np.stack([v["grid_tensor"] for v in views]), "features": np.stack([v["features"] for v in views])}
    return batch, np.stack([transform_masks(masks, k) for k in range(n)])


def _average_symmetries(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    return np.mean([values[k, ACTION_PERM[k]] for k in range(n)], axis=0)


def _resolve_checkpoint_dir(run_dir: pathlib.Path, checkpoint: Optional[str]) -> pathlib.Path:
    checkpoints = run_dir / "checkpoints"
    if checkpoint and checkpoint not in ("latest", "best"):
        checkpoint_dir = checkpoints / checkpoint
    else:
        pointer = checkpoints / f"{checkpoint or 'latest'}.txt"
        if not pointer.exists() and checkpoint == "best":
            print(f"{ENV_PREFIX.lower()}: {pointer} not found (run evaluation.select_checkpoint --write-best), "
                  "falling back to the latest checkpoint", file=sys.stderr)
            pointer = checkpoints / "latest.txt"
        if pointer.exists():
            checkpoint_dir = checkpoints / pointer.read_text().strip()
        else:
            found = sorted(p for p in checkpoints.glob("checkpoint_*") if p.is_dir())
            if not found:
                raise FileNotFoundError(f"ppo_agent: no checkpoints in '{checkpoints}'")
            checkpoint_dir = found[-1]
    if not (checkpoint_dir / "model.zip").exists():
        raise FileNotFoundError(
            f"ppo_agent: no model.zip in '{checkpoint_dir}' -- set CHECKPOINT at the top of "
            f"callbacks.py to a valid checkpoint name or None for the latest."
        )
    return checkpoint_dir


def _load_config(run_dir: pathlib.Path) -> TrainingConfig:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    return TrainingConfig.from_dict(manifest["config"])


def _load_model(checkpoint_dir: pathlib.Path, cfg: TrainingConfig) -> MaskablePPO:
    return MaskablePPO.load(
        checkpoint_dir / "model.zip",
        device="cpu",
        custom_objects={
            "n_envs": 1,
            "n_steps": 1,
            "rollout_buffer_class": MaskableDictRolloutBuffer,
            "policy_kwargs": dict(
                features_extractor_class=BombermanFeatureExtractor,
                features_extractor_kwargs=dict(dropout=cfg.ppo.dropout),
                share_features_extractor=cfg.ppo.share_features_extractor,
                optimizer_class=InferenceOptimizer,
            ),
        },
    )


def _get_dummy_env(cfg: TrainingConfig) -> BombermanGymEnv:
    env_cfg = cfg.env
    world_args = WorldArgs(
        scenario=env_cfg.scenario,
        seed=None,
        silence_errors=True,
        no_gui=True,
        make_video=False,
        save_replay=False,
        save_stats=False,
        turn_based=env_cfg.turn_based,
        update_interval=env_cfg.update_interval,
        log_dir=None,
        match_name=None,
        fps=env_cfg.fps,
        replay=False,
        continue_without_training=env_cfg.continue_without_training,
    )
    dummy_opponents = [((lambda handle: None), (lambda handle, state: "WAIT"))] * 3
    return BombermanGymEnv(
        world_args,
        opponents=dummy_opponents,
        layer_config=env_cfg.layer_config,
        env_version=env_cfg.env_version,
    )


def _make_warmup_state() -> dict:
    W, H = s.COLS, s.ROWS
    field = np.zeros((W, H), dtype=np.int8)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(W):
        for y in range(H):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    for x, y in ((3, 1), (1, 3), (7, 8), (8, 7), (9, 9)):
        field[x, y] = 1

    explosion_map = np.zeros((W, H))
    for x, y in ((5, 5), (5, 6), (5, 7)):
        explosion_map[x, y] = 1

    return {
        "round": 0,
        "step": 1,
        "field": field,
        "self": ("warmup", 0, False, (1, 1)),
        "others": [("warmup_opponent", 0, True, (W - 2, H - 2))],
        "bombs": [((1, 1), 3), ((W - 2, 1), 1)],
        "coins": [(1, 5), (11, 11)],
        "explosion_map": explosion_map,
        "user_input": None,
    }


def _choose(self, game_state: dict) -> str:
    obs_env = self._ppo_obs_env
    obs = obs_env.observation_from_game_state(game_state)
    action_masks = obs_env.action_masks()
    if len(self._ppo_models) == 1 and self._ppo_tta <= 1:
        action_idx, _ = self._ppo_model.predict(obs, deterministic=DETERMINISTIC, action_masks=action_masks)
        return _ACTION_NAMES[int(action_idx)]
    if self._ppo_tta > 1:
        batch, masks = _symmetry_batch(obs, np.asarray(action_masks[0]), self._ppo_tta)
    else:
        batch, masks = obs, action_masks
    probs = []
    with torch.no_grad():
        for policy in self._ppo_models:
            obs_t, _ = policy.obs_to_tensor(batch)
            p = policy.get_distribution(obs_t, action_masks=masks).distribution.probs.cpu().numpy()
            probs.append(_average_symmetries(p) if self._ppo_tta > 1 else p[0])
    mean = np.mean(probs, axis=0)
    mean = np.where(np.asarray(action_masks[0]).astype(bool), mean, 0.0)
    action_idx = int(np.argmax(mean)) if DETERMINISTIC else int(np.random.choice(len(mean), p=mean / mean.sum()))
    return _ACTION_NAMES[action_idx]


def _warm_up(self) -> None:
    time_start = time.time()
    state = _make_warmup_state()
    _choose(self, state)
    state["bombs"] = []
    state["explosion_map"] = np.zeros_like(state["explosion_map"])
    _choose(self, state)

    self.logger.info(f"ppo_agent.setup: warm-up finished in {time.time() - time_start:.2f}s")


def _fail(self, message: str) -> None:
    text = f"{message}\n{traceback.format_exc()}"
    self.logger.error(text)
    print(text, file=sys.stderr)


def setup(self):
    torch.set_num_threads(1)
    self._ppo_model = None
    self._ppo_models = []
    self._ppo_obs_env = None
    self._ppo_tta = max(1, min(N_SYMMETRIES, int(_setting("TTA_SYMMETRIES", TTA_SYMMETRIES))))
    try:
        run = _setting("RUN", RUN)
        members = _member_specs(_setting("ENSEMBLE", ENSEMBLE)) or [_setting("CHECKPOINT", CHECKPOINT) or "latest"]
        run_dir = _resolve_run_dir(run)
        if members == ["ensemble"]:
            members = _read_ensemble_file(run_dir)
        cfg = _load_config(run_dir)
        loaded = []
        for entry in members:
            member_run, checkpoint = _split_member(entry, run)
            member_run_dir = run_dir if member_run == run else _resolve_run_dir(member_run)
            member_cfg = cfg if member_run == run else _load_config(member_run_dir)
            checkpoint_dir = _resolve_checkpoint_dir(member_run_dir, checkpoint)
            policy = _load_model(checkpoint_dir, member_cfg).policy
            policy.requires_grad_(False)
            if loaded and policy.observation_space["features"].shape != loaded[0].observation_space["features"].shape:
                raise ValueError(f"ensemble member {checkpoint_dir} has a different observation space than {members[0]}")
            loaded.append(policy)
            self.logger.info(f"ppo_agent.setup: loaded {checkpoint_dir}")
        obs_env = _get_dummy_env(cfg)
        n_model = int(loaded[0].observation_space["features"].shape[0])
        if n_model != obs_env.n_features:
            raise ValueError(f"{members[0]} expects {n_model} features, run {run} (env_version {cfg.env.env_version}) "
                             f"produces {obs_env.n_features}")
        self._ppo_obs_env = obs_env
        self._ppo_models = loaded
        self._ppo_model = loaded[0]
        if len(loaded) > 1:
            self.logger.info(f"ppo_agent.setup: averaging the action probabilities of {len(loaded)} policies")
    except Exception:
        _fail(self, "ppo_agent.setup: could not load the model, every action will be WAIT")
        return

    try:
        _warm_up(self)
    except Exception:
        _fail(self, "ppo_agent.setup: warm-up failed, the first steps may exceed the time limit")


def act(self, game_state: dict) -> str:
    if self._ppo_model is None:
        return "WAIT"

    time_start = time.time()
    try:
        chosen_action = _choose(self, game_state)
    except Exception:
        _fail(self, f"ppo_agent.act: failed at step {game_state.get('step')}, returning WAIT")
        return "WAIT"

    self.logger.debug(
        f"ppo_agent.act: step={game_state['step']}, round={game_state['round']}, "
        f"action={chosen_action}, time={time.time() - time_start:.3f}s"
    )
    return chosen_action
