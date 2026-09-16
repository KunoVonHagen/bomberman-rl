from __future__ import annotations

import json
import os
import pathlib
import struct
import time
from typing import Optional

import numpy as np
import torch

REPLAY_MAGIC = b"BRB1"
REPLAY_FIELDS = ("grid", "features", "actions", "rewards", "dones", "action_masks")


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
        self._grid_flat = torch.from_numpy(self.grid.reshape((self.rows * self.n_envs,) + tuple(grid_shape)))
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
        torch.from_numpy(self.grid[i]).copy_(torch.from_numpy(np.ascontiguousarray(obs["grid_tensor"])))
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

    def sample(self, batch_size: int, out: Optional[dict] = None) -> dict:
        t0 = time.perf_counter()

        rows = np.flatnonzero(self._has_next)
        if len(rows) == 0:
            raise ValueError("replay buffer holds no transition with a stored successor yet")
        row_idx = rows[np.random.randint(0, len(rows), size=batch_size)]
        env_idx = np.random.randint(0, self.n_envs, size=batch_size)
        next_row_idx = (row_idx + 1) % self.rows

        flat = torch.from_numpy(row_idx * self.n_envs + env_idx)
        next_flat = torch.from_numpy(next_row_idx * self.n_envs + env_idx)
        grid = torch.index_select(self._grid_flat, 0, flat, out=None if out is None else out["grid"])
        next_grid = torch.index_select(self._grid_flat, 0, next_flat, out=None if out is None else out["next_grid"])

        batch = dict(
            obs={
                "grid_tensor": grid,
                "features": self.features[row_idx, env_idx],
            },
            next_obs={
                "grid_tensor": next_grid,
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

    def _ring_chunks(self, n: int) -> list:
        start = (self._pos - n) % self.rows
        first = min(n, self.rows - start)
        chunks = [(start, start + first)]
        if n > first:
            chunks.append((0, n - first))
        return chunks

    def save(self, path, max_rows: Optional[int] = None, num_timesteps: int = 0) -> int:
        n = self._stored_rows()
        if max_rows is not None:
            n = min(n, max(0, int(max_rows)))
        chunks = self._ring_chunks(n)

        fields, offset = {}, 0
        for name in REPLAY_FIELDS:
            arr = getattr(self, name)
            fields[name] = {"dtype": arr.dtype.str, "shape": list(arr.shape[1:]), "offset": offset}
            offset += n * arr[0].nbytes
        header = json.dumps({"n_rows": n, "n_envs": self.n_envs, "num_timesteps": int(num_timesteps),
                             "fields": fields}).encode("utf-8")

        path = pathlib.Path(path)
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(REPLAY_MAGIC)
                f.write(struct.pack("<Q", len(header)))
                f.write(header)
                for name in REPLAY_FIELDS:
                    arr = getattr(self, name)
                    for a, b in chunks:
                        f.write(memoryview(arr[a:b]))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return n

    def _check_layout(self, n_envs: int, shapes: dict) -> None:
        if n_envs != self.n_envs:
            raise ValueError(f"replay buffer was saved with n_envs={n_envs}, this buffer has n_envs={self.n_envs}")
        expected = {
            "grid": (self.n_envs,) + tuple(self.grid_shape),
            "features": (self.n_envs,) + tuple(self.feat_shape),
            "action_masks": (self.n_envs, self.action_dim),
        }
        for key, shape in expected.items():
            if tuple(shapes[key]) != shape:
                raise ValueError(f"replay buffer field '{key}' has shape {tuple(shapes[key])} per row, expected {shape}")

    def _finish_load(self, keep: int, num_timesteps: int) -> int:
        self.num_timesteps = int(num_timesteps)
        self._has_next[:] = False
        if keep > 1:
            self._has_next[:keep - 1] = True
        self._seam_row = keep - 1 if keep > 0 else None
        self._pos = keep % self.rows
        self._full = keep == self.rows
        return keep

    def load(self, path) -> int:
        path = pathlib.Path(path)
        with open(path, "rb") as f:
            if f.read(len(REPLAY_MAGIC)) != REPLAY_MAGIC:
                return self._load_npz(path)
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len).decode("utf-8"))
            data_start = f.tell()
            n = int(header["n_rows"])
            self._check_layout(int(header["n_envs"]), {k: v["shape"] for k, v in header["fields"].items()})
            keep = min(n, self.rows)
            for name in REPLAY_FIELDS:
                arr = getattr(self, name)
                spec = header["fields"][name]
                if spec["dtype"] != arr.dtype.str:
                    raise ValueError(f"replay buffer field '{name}' was saved as {spec['dtype']}, expected {arr.dtype.str}")
                row_bytes = arr[0].nbytes
                f.seek(data_start + int(spec["offset"]) + (n - keep) * row_bytes)
                target = memoryview(arr[:keep]).cast("B")
                if f.readinto(target) != len(target):
                    raise ValueError(f"replay buffer file {path} is truncated in field '{name}'")
        return self._finish_load(keep, header.get("num_timesteps", 0))

    def _load_npz(self, path) -> int:
        with np.load(path) as data:
            n = int(data["n_rows"])
            self._check_layout(int(data["n_envs"]), {k: data[k].shape[1:] for k in ("grid", "features", "action_masks")})
            keep = min(n, self.rows)
            for key in REPLAY_FIELDS:
                getattr(self, key)[:keep] = data[key][n - keep:n]
            num_timesteps = int(data["num_timesteps"]) if "num_timesteps" in data else 0
        return self._finish_load(keep, num_timesteps)


def read_replay_file(path) -> dict:
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        if f.read(len(REPLAY_MAGIC)) != REPLAY_MAGIC:
            with np.load(path) as data:
                out = {"n_rows": int(data["n_rows"]), "n_envs": int(data["n_envs"]),
                       "num_timesteps": int(data["num_timesteps"]) if "num_timesteps" in data else 0}
                for name in REPLAY_FIELDS:
                    out[name] = np.array(data[name])
                return out
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = f.tell()
        n = int(header["n_rows"])
        out = {"n_rows": n, "n_envs": int(header["n_envs"]), "num_timesteps": int(header.get("num_timesteps", 0))}
        for name in REPLAY_FIELDS:
            spec = header["fields"][name]
            f.seek(data_start + int(spec["offset"]))
            shape = (n,) + tuple(spec["shape"])
            out[name] = np.fromfile(f, dtype=np.dtype(spec["dtype"]), count=int(np.prod(shape))).reshape(shape)
    return out
