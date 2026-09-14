from __future__ import annotations

import os
import pathlib
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
        self._has_next = np.zeros(self.rows, dtype=bool)

        self._pos = 0
        self._full = False

        self._seam_row: Optional[int] = None

        self.last_sample_time_s: Optional[float] = None
        self.num_timesteps: int = 0

    @property
    def capacity(self) -> int:
        """Max. Stored transitions."""
        return self.rows * self.n_envs

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.rows * self.n_envs if self._full else self._pos * self.n_envs

    def n_sampleable_rows(self) -> int:
        return int(self._has_next.sum())

    def add(self, obs, next_obs, actions, rewards, dones, action_masks, next_action_masks) -> None:
        """Store one transition for each active environment."""
        i = self._pos
        self.grid[i] = obs["grid_tensor"]
        self.features[i] = obs["features"]
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.action_masks[i] = action_masks

        self._has_next[i] = False
        if i == self._seam_row:
            self._seam_row = None
        prev = (i - 1) % self.rows
        if (i > 0 or self._full) and prev != self._seam_row:
            self._has_next[prev] = True

        self._pos += 1
        if self._pos == self.rows:
            self._pos = 0
            self._full = True

    def sample(self, batch_size: int) -> dict:
        t0 = time.perf_counter()

        rows = np.flatnonzero(self._has_next)
        if len(rows) == 0:
            raise ValueError("replay buffer holds no transition with a stored successor yet")
        row_idx = rows[np.random.randint(0, len(rows), size=batch_size)]
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

    def _stored_rows(self) -> int:
        return self.rows if self._full else self._pos

    def save(self, path, max_rows: Optional[int] = None, num_timesteps: int = 0) -> int:
        n = self._stored_rows()
        if max_rows is not None:
            n = min(n, max(0, int(max_rows)))
        start = (self._pos - n) % self.rows
        idx = (start + np.arange(n)) % self.rows

        path = pathlib.Path(path)
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            with open(tmp_path, "wb") as f:
                np.savez(
                    f,
                    n_rows=np.int64(n),
                    n_envs=np.int64(self.n_envs),
                    num_timesteps=np.int64(num_timesteps),
                    grid=self.grid[idx],
                    features=self.features[idx],
                    actions=self.actions[idx],
                    rewards=self.rewards[idx],
                    dones=self.dones[idx],
                    action_masks=self.action_masks[idx],
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return n

    def load(self, path) -> int:
        with np.load(path) as data:
            n = int(data["n_rows"])
            if int(data["n_envs"]) != self.n_envs:
                raise ValueError(f"replay buffer was saved with n_envs={int(data['n_envs'])}, "
                                 f"this buffer has n_envs={self.n_envs}")
            expected = {
                "grid": (self.n_envs,) + tuple(self.grid_shape),
                "features": (self.n_envs,) + tuple(self.feat_shape),
                "action_masks": (self.n_envs, self.action_dim),
            }
            for key, shape in expected.items():
                if tuple(data[key].shape[1:]) != shape:
                    raise ValueError(f"replay buffer field '{key}' has shape {data[key].shape[1:]} "
                                     f"per row, expected {shape}")
            keep = min(n, self.rows)
            for key in ("grid", "features", "actions", "rewards", "dones", "action_masks"):
                getattr(self, key)[:keep] = data[key][n - keep:n]
            self.num_timesteps = int(data["num_timesteps"]) if "num_timesteps" in data else 0

        self._has_next[:] = False
        if keep > 1:
            self._has_next[:keep - 1] = True
        self._seam_row = keep - 1 if keep > 0 else None
        self._pos = keep % self.rows
        self._full = keep == self.rows
        return keep
