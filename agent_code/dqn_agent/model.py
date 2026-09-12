from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import DQN
from sb3_contrib.common.maskable.utils import get_action_masks

from agent_code.ppo_agent.model import BombermanFeatureExtractor

__all__ = ["BombermanFeatureExtractor", "MaskableDQN"]


class MaskableDQN(DQN):
    """DQN variant that applies action masks during exploration, exploitation, and warmup sampling."""

    def _sample_masked_actions(self, action_masks: np.ndarray) -> np.ndarray:
        """Draw one uniformly random valid action per row of a boolean mask matrix."""
        actions = np.empty(action_masks.shape[0], dtype=np.int64)
        for i, mask in enumerate(action_masks):
            actions[i] = np.random.choice(np.flatnonzero(mask))
        return actions

    def _predict_masked(self, observation, action_masks: np.ndarray) -> np.ndarray:
        """Return the greedy action per row after masking out invalid actions with -inf."""
        self.policy.set_training_mode(False)
        obs_tensor, _ = self.policy.obs_to_tensor(observation)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor).cpu().numpy()
        q_values = np.where(action_masks.astype(bool), q_values, -np.inf)
        return q_values.argmax(axis=1)

    def predict(self, observation, state=None, episode_start=None, deterministic=False, action_masks=None):
        if action_masks is None:
            return super().predict(observation, state, episode_start, deterministic)

        action_masks = np.asarray(action_masks)
        if action_masks.ndim == 1:
            action_masks = action_masks[None, :]

        vectorized = self.policy.is_vectorized_observation(observation)
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = self._sample_masked_actions(action_masks)
        else:
            action = self._predict_masked(observation, action_masks)

        if not vectorized:
            action = action[0]
        return action, state

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action_masks = get_action_masks(self.env)

        if self.num_timesteps < learning_starts:
            unscaled_action = self._sample_masked_actions(action_masks)
        else:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False, action_masks=action_masks)

        return unscaled_action, unscaled_action