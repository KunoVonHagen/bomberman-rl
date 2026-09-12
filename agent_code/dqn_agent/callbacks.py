from __future__ import annotations

import pathlib
import time
from typing import Optional

from environment import WorldArgs

from agent_code.dqn_agent.checkpoint_manager import CheckpointManager
from agent_code.dqn_agent.config import DEFAULT_CONFIG
from agent_code.dqn_agent.gym_environment import ACTION_INDICES, BombermanGymEnv
from agent_code.dqn_agent.model import MaskableDQN

RUN: str = "run_20260901-120000"
CHECKPOINT: Optional[str] = None
DETERMINISTIC: bool = True

_ACTION_NAMES = {v: k for k, v in ACTION_INDICES.items() if k is not None}


def _resolve_run_dir() -> pathlib.Path:
    """Resolve the configured run name or path to a directory."""
    run_dir = pathlib.Path(RUN)
    if not run_dir.exists():
        run_dir = pathlib.Path(DEFAULT_CONFIG.runs_dir) / RUN
    if not run_dir.exists():
        raise FileNotFoundError(
            f"dqn_agent: no run found at '{RUN}' -- set RUN at the top of "
            f"callbacks.py to a valid run name or path."
        )
    return run_dir


def _load_model(run_dir: pathlib.Path, checkpoint: Optional[str] = None) -> MaskableDQN:
    """Load a DQN model from a checkpoint."""
    manager = CheckpointManager.resume(str(run_dir))
    if checkpoint is None:
        checkpoint_path = manager.latest_checkpoint()
    else:
        checkpoint_path = manager.get_checkpoint(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"dqn_agent: no checkpoint found at '{checkpoint_path}' -- set CHECKPOINT at the top of "
            f"callbacks.py to a valid checkpoint name or None for the latest."
        )
    return MaskableDQN.load(checkpoint_path / "model.zip", device="cpu")


def _get_dummy_env(run_dir: pathlib.Path) -> BombermanGymEnv:
    """Create a minimal environment for observation conversion."""
    manager = CheckpointManager.resume(run_dir)
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
    )


def setup(self):
    """Load the trained model and prepare the observation environment."""
    run_dir = _resolve_run_dir()

    self._dqn_model = _load_model(run_dir, CHECKPOINT)
    self._dqn_obs_env = _get_dummy_env(run_dir)


def act(self, game_state: dict) -> str:
    """Convert the game state into a masked action prediction."""
    time_start = time.time()

    obs_env = self._dqn_obs_env
    obs = obs_env.observation_from_game_state(game_state)
    action_masks = obs_env.action_masks()

    action_idx, _ = self._dqn_model.predict(
        obs,
        deterministic=DETERMINISTIC,
        action_masks=action_masks,
    )

    chosen_action = _ACTION_NAMES[int(action_idx)]

    self.logger.debug(
        f"dqn_agent.act: step={game_state['step']}, round={game_state['round']}, "
        f"action={chosen_action}, time={time.time() - time_start:.3f}s"
    )
    return chosen_action