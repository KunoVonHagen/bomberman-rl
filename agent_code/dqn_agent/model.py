from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.utils import get_action_masks

__all__ = ["BombermanFeatureExtractor", "MaskableDQN"]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class BombermanFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        grid_space = observation_space["grid_tensor"]
        feature_space = observation_space["features"]

        n_input_channels = grid_space.shape[0]
        h, w = grid_space.shape[1], grid_space.shape[2]

        stem_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, stem_channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(stem_channels),
            ResidualBlock(stem_channels),
        )

        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, h, w, dtype=torch.float32)
            n_cnn_flatten = self._conv_forward(sample).shape[1]

        self.grid_fc = nn.Sequential(
            nn.Linear(n_cnn_flatten, 128),
            nn.ReLU(),
        )

        self.features_preprocess_fc = nn.Sequential(
            nn.Linear(feature_space.shape[0], 64),
            nn.ReLU(),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
        )

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        return self.flatten(x)

    def forward(self, observations: dict) -> torch.Tensor:
        grid_tensor = observations["grid_tensor"].float()
        features = observations["features"].float()

        cnn_out = self._conv_forward(grid_tensor)
        cnn_fc_out = self.grid_fc(cnn_out)

        features_fc_out = self.features_preprocess_fc(features)

        combined = torch.cat([cnn_fc_out, features_fc_out], dim=1)
        return self.combined_fc(combined)


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