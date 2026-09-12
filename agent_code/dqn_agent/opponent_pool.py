from __future__ import annotations

import tempfile

from environment import WorldArgs
from agent_code.ppo_agent.opponent_pool import (
    OpponentPool as _BaseOpponentPool,
    _CheckpointOpponent,
    OpponentSampler,
)
from agent_code.ppo_agent.gym_environment import BombermanGymEnv, ACTION_INDICES
from .model import MaskableDQN

__all__ = ["OpponentPool", "OpponentSampler"]


class _DQNCheckpointOpponent(_CheckpointOpponent):
    """Wraps a MaskableDQN checkpoint as a self-play opponent."""

    def _ensure_ready(self):
        if self._model is None:
            self._model = MaskableDQN.load(self.model_path, device="cpu")

        if self._obs_env is None:
            log_dir = tempfile.mkdtemp(prefix="dqn_checkpoint_opponent_")
            world_args = WorldArgs(
                scenario=self.env_cfg.scenario,
                seed=None,
                silence_errors=True,
                no_gui=True,
                make_video=False,
                save_replay=False,
                save_stats=False,
                turn_based=self.env_cfg.turn_based,
                update_interval=self.env_cfg.update_interval,
                log_dir=log_dir,
                match_name=None,
                fps=self.env_cfg.fps,
                replay=False,
                continue_without_training=self.env_cfg.continue_without_training,
            )
            self._obs_env = BombermanGymEnv(
                world_args,
                opponents=[((lambda handle: None), (lambda handle, state: "WAIT"))] * 3,
                layer_config=self.env_cfg.layer_config,
            )
            self._action_names = {v: k for k, v in ACTION_INDICES.items() if k is not None}

    def act(self, agent, game_state):
        self._ensure_ready()
        obs_env = self._obs_env
        obs = obs_env.observation_from_game_state(game_state)
        action_masks = obs_env.action_masks()
        action_idx, _ = self._model.predict(obs, deterministic=False, action_masks=action_masks)
        return self._action_names[int(action_idx)]


class OpponentPool(_BaseOpponentPool):
    """Self-play pool that loads DQN checkpoints instead of PPO ones."""

    def _checkpoint_opponent(self, checkpoint_dir):
        key = str(checkpoint_dir.resolve())
        if key not in self._opponent_cache:
            model_path = str((checkpoint_dir / "model.zip").resolve())
            self._opponent_cache[key] = _DQNCheckpointOpponent(model_path, self.env_cfg).as_pair()
        return self._opponent_cache[key]