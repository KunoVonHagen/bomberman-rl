from __future__ import annotations

import pathlib
import time
from typing import Optional
from sb3_contrib import MaskablePPO

from agent_code.ppo_agent.checkpoint_manager import CheckpointManager
from agent_code.ppo_agent.config import DEFAULT_CONFIG
from agent_code.ppo_agent.gym_environment import ACTION_INDICES, BombermanGymEnv, WorldArgs

RUN: str = "run_20260831-190704"
CHECKPOINT: Optional[str] = "checkpoint_0253638656"
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


def setup(self):
    """
    Called once at the start of a match.
    Loads the trained MaskablePPO checkpoint and prepares a BombermanGymEnv for observation conversion.
    """

    run_dir = _resolve_run_dir()
    ckman = CheckpointManager.resume(str(run_dir))
    env_cfg = ckman.config.env

    ckpt_dir = (
        ckman.get_checkpoint(CHECKPOINT) if CHECKPOINT
        else ckman.latest_checkpoint()
    )
    if ckpt_dir is None:
        raise FileNotFoundError(f"ppo_agent: run '{run_dir}' has no saved checkpoints")

    self.logger.info(f"ppo_agent: loading {ckpt_dir} (run={run_dir.name})")

    model = MaskablePPO.load(
        str(ckpt_dir / "model.zip"),
        device="cpu",
        custom_objects={"n_envs": 1, "n_steps": 1},
    )

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
    obs_env = BombermanGymEnv(
        world_args,
        opponents=dummy_opponents,
        layer_config=env_cfg.layer_config,
    )

    self._ppo_model = model
    self._ppo_obs_env = obs_env


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

