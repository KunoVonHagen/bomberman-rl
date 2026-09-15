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

import settings as s
from .config import TrainingConfig
from .gym_environment import ACTION_INDICES, BombermanGymEnv, WorldArgs
from .model import MaskableDQN

RUN: str = "run_20260901-120000"
CHECKPOINT: Optional[str] = None
DETERMINISTIC: bool = True

AGENT_DIR = pathlib.Path(__file__).resolve().parent
ENV_PREFIX = "DQN_AGENT"
_ACTION_NAMES = {v: k for k, v in ACTION_INDICES.items() if k is not None}


def _setting(name: str, default):
    return os.environ.get(f"{ENV_PREFIX}_{name}", default)


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
        f"dqn_agent: no run '{run}' with a run_manifest.json in any of "
        f"{[str(c) for c in candidates]} -- set RUN at the top of callbacks.py"
    )


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
                raise FileNotFoundError(f"dqn_agent: no checkpoints in '{checkpoints}'")
            checkpoint_dir = found[-1]
    if not (checkpoint_dir / "model.zip").exists():
        raise FileNotFoundError(
            f"dqn_agent: no model.zip in '{checkpoint_dir}' -- set CHECKPOINT at the top of "
            f"callbacks.py to a valid checkpoint name or None for the latest."
        )
    return checkpoint_dir


def _load_config(run_dir: pathlib.Path) -> TrainingConfig:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    return TrainingConfig.from_dict(manifest["config"])


def _load_model(checkpoint_dir: pathlib.Path) -> MaskableDQN:
    return MaskableDQN.load(checkpoint_dir / "model.zip", device="cpu")


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


def _warm_up(self) -> None:
    time_start = time.time()
    obs_env = self._dqn_obs_env

    state = _make_warmup_state()
    obs = obs_env.observation_from_game_state(state)
    self._dqn_model.predict(obs, deterministic=True, action_masks=obs_env.action_masks())

    state["bombs"] = []
    state["explosion_map"] = np.zeros_like(state["explosion_map"])
    obs = obs_env.observation_from_game_state(state)
    self._dqn_model.predict(obs, deterministic=True, action_masks=obs_env.action_masks())

    self.logger.info(f"dqn_agent.setup: warm-up finished in {time.time() - time_start:.2f}s")


def _fail(self, message: str) -> None:
    text = f"{message}\n{traceback.format_exc()}"
    self.logger.error(text)
    print(text, file=sys.stderr)


def setup(self):
    torch.set_num_threads(1)
    self._dqn_model = None
    self._dqn_obs_env = None

    try:
        run_dir = _resolve_run_dir(_setting("RUN", RUN))
        cfg = _load_config(run_dir)
        checkpoint_dir = _resolve_checkpoint_dir(run_dir, _setting("CHECKPOINT", CHECKPOINT))
        model = _load_model(checkpoint_dir)
        self._dqn_obs_env = _get_dummy_env(cfg)
        self._dqn_model = model
        self.logger.info(f"dqn_agent.setup: loaded {checkpoint_dir}")
    except Exception:
        _fail(self, "dqn_agent.setup: could not load the model, every action will be WAIT")
        return

    try:
        _warm_up(self)
    except Exception:
        _fail(self, "dqn_agent.setup: warm-up failed, the first steps may exceed the time limit")


def act(self, game_state: dict) -> str:
    if self._dqn_model is None:
        return "WAIT"

    time_start = time.time()
    try:
        obs_env = self._dqn_obs_env
        obs = obs_env.observation_from_game_state(game_state)
        action_masks = obs_env.action_masks()
        action_idx, _ = self._dqn_model.predict(
            obs,
            deterministic=DETERMINISTIC,
            action_masks=action_masks,
        )
        chosen_action = _ACTION_NAMES[int(action_idx)]
    except Exception:
        _fail(self, f"dqn_agent.act: failed at step {game_state.get('step')}, returning WAIT")
        return "WAIT"

    self.logger.debug(
        f"dqn_agent.act: step={game_state['step']}, round={game_state['round']}, "
        f"action={chosen_action}, time={time.time() - time_start:.3f}s"
    )
    return chosen_action
