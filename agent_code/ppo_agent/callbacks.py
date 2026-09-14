from __future__ import annotations

import pathlib
import time
from typing import Optional

import numpy as np
import torch
from sb3_contrib import MaskablePPO

import settings as s
from agent_code.ppo_agent.checkpoint_manager import CheckpointManager
from agent_code.ppo_agent.config import DEFAULT_CONFIG
from agent_code.ppo_agent.gym_environment import ACTION_INDICES, BombermanGymEnv, WorldArgs

RUN: str = "run_20260831-190704"
CHECKPOINT: Optional[str] = "checkpoint_0143654912"
DETERMINISTIC: bool = True

_ACTION_NAMES = {v: k for k, v in ACTION_INDICES.items() if k is not None}


def _resolve_run_dir() -> pathlib.Path:
    """Resolve RUN (a run name or a path to a run directory) to a run dir."""
    run_dir = pathlib.Path(RUN)
    if not run_dir.exists():
        run_dir = pathlib.Path(DEFAULT_CONFIG.runs_dir) / RUN
    if not run_dir.exists():
        raise FileNotFoundError(
            f"ppo_agent: no run found at '{RUN}' -- set RUN at the top of "
            f"callbacks.py to a valid run name or path."
        )
    return run_dir

def _load_model(run_dir: pathlib.Path, checkpoint: Optional[str] = None) -> MaskablePPO:
    """Load a MaskablePPO model from a run directory and optional checkpoint name."""
    manager = CheckpointManager.resume(str(run_dir))
    if checkpoint is None:
        checkpoint_path = manager.latest_checkpoint()
    else:
        checkpoint_path = manager.get_checkpoint(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ppo_agent: no checkpoint found at '{checkpoint_path}' -- set CHECKPOINT at the top of "
            f"callbacks.py to a valid checkpoint name or None for the latest."
        )
    return MaskablePPO.load(
        checkpoint_path / "model.zip",
        device="cpu",
        custom_objects={"n_envs": 1, "n_steps": 1}
    )

def _get_dummy_env(run_dir: pathlib.Path) -> BombermanGymEnv:
    """Get a dummy BombermanGymEnv for the given run directory."""
    manager = CheckpointManager.resume(run_dir)
    manifest = manager.read_manifest()
    env_cfg = manager.config.env

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
    obs_env = self._ppo_obs_env

    state = _make_warmup_state()
    obs = obs_env.observation_from_game_state(state)
    self._ppo_model.predict(obs, deterministic=True, action_masks=obs_env.action_masks())

    state["bombs"] = []
    state["explosion_map"] = np.zeros_like(state["explosion_map"])
    obs = obs_env.observation_from_game_state(state)
    self._ppo_model.predict(obs, deterministic=True, action_masks=obs_env.action_masks())

    self.logger.info(f"ppo_agent.setup: warm-up finished in {time.time() - time_start:.2f}s")


def setup(self):
    """
    Called once at the start of a match.
    Loads the trained MaskablePPO checkpoint and prepares a BombermanGymEnv for observation conversion.
    """
    torch.set_num_threads(1)

    _RUN_DIR = _resolve_run_dir()

    self._ppo_model = _load_model(_RUN_DIR, CHECKPOINT).policy
    self._ppo_obs_env = _get_dummy_env(_RUN_DIR)

    self._ppo_model.requires_grad_(False)

    try:
        _warm_up(self)
    except Exception as exc:
        self.logger.warning(f"ppo_agent.setup: warm-up failed ({exc!r}) the first steps may exceed the time limit")


def act(self, game_state: dict) -> str:
    """
    Called on every game step.
    Converts the game_state to an observation, applies action masks, and uses the trained MaskablePPO model to predict the next action.
    """
    time_start = time.time()

    obs_env = self._ppo_obs_env
    obs = obs_env.observation_from_game_state(game_state)

    action_masks = obs_env.action_masks()

    action_idx, _ = self._ppo_model.predict(
        obs,
        deterministic=DETERMINISTIC,
        action_masks=action_masks,
    )

    chosen_action = _ACTION_NAMES[int(action_idx)]

    self.logger.debug(
        f"ppo_agent.act: step={game_state['step']}, round={game_state['round']}, "
        f"action={chosen_action}, time={time.time() - time_start:.3f}s"
    )
    return chosen_action

