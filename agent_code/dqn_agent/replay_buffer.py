from __future__ import annotations

import time
from typing import Optional

import numpy as np


class DictReplayBuffer:
    """Circular buffer for dict observations with action masks."""

    def __init__(self, buffer_size: int, n_envs: int, observation_space, action_dim: int,
                 grid_dtype=np.float16):
        self.n_envs = max(1, int(n_envs))
        self.rows = max(2, int(buffer_size) // self.n_envs)
        self.action_dim = int(action_dim)
        self.grid_dtype = grid_dtype

        grid_shape = observation_space["grid_tensor"].shape
        feat_shape = observation_space["features"].shape
        self.grid_shape = grid_shape
        self.feat_shape = feat_shape

        shape = (self.rows, self.n_envs)
        self.grid = np.zeros(shape + grid_shape, dtype=grid_dtype)
        self.features = np.zeros(shape + feat_shape, dtype=np.float32)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.action_masks = np.ones(shape + (self.action_dim,), dtype=bool)

        self._pos = 0
        self._full = False

        self.last_sample_time_s: Optional[float] = None

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.rows * self.n_envs if self._full else self._pos * self.n_envs

    def add(self, obs, next_obs, actions, rewards, dones, action_masks, next_action_masks) -> None:
        """Store one transition for each active environment."""
        i = self._pos
        self.grid[i] = obs["grid_tensor"]
        self.features[i] = obs["features"]
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.action_masks[i] = action_masks

        self._pos += 1
        if self._pos == self.rows:
            self._pos = 0
            self._full = True

    def sample(self, batch_size: int) -> dict:
        """Sample a minibatch of transitions."""
        t0 = time.perf_counter()

        if self._full:
            row_idx = (self._pos + np.random.randint(0, self.rows - 1, size=batch_size)) % self.rows
        else:
            upper = max(self._pos - 1, 1)
            row_idx = np.random.randint(0, upper, size=batch_size)

        env_idx = np.random.randint(0, self.n_envs, size=batch_size)
        next_row_idx = (row_idx + 1) % self.rows

        batch = dict(
            obs={
                "grid_tensor": self.grid[row_idx, env_idx],
                "features": self.features[row_idx, env_idx],
            },
            next_obs={
                "grid_tensor": self.grid[next_row_idx, env_idx],
                "features": self.features[next_row_idx, env_idx],
            },
            actions=self.actions[row_idx, env_idx],
            rewards=self.rewards[row_idx, env_idx],
            dones=self.dones[row_idx, env_idx],
            action_masks=self.action_masks[row_idx, env_idx],
            next_action_masks=self.action_masks[next_row_idx, env_idx],
        )

        self.last_sample_time_s = time.perf_counter() - t0
        return batch