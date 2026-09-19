from __future__ import annotations

import json
import os
import pathlib
import struct
import time
from typing import Optional

import numpy as np
import torch
from numba import njit

REPLAY_MAGIC = b"BRB1"
GRID_SENTINEL_CODE = 255
GRID_MAX_CODE = 254


@njit(cache=True)
def _encode_grid_kernel(src, den, sentinel, out):
    for e in range(src.shape[0]):
        for l in range(src.shape[1]):
            d = den[l]
            flag = sentinel[l]
            for i in range(src.shape[2]):
                for j in range(src.shape[3]):
                    v = src[e, l, i, j]
                    if flag and v < 0:
                        out[e, l, i, j] = GRID_SENTINEL_CODE
                    else:
                        code = int(np.rint(v * d))
                        if code < 0:
                            code = 0
                        elif code > GRID_MAX_CODE:
                            code = GRID_MAX_CODE
                        out[e, l, i, j] = code


@njit(cache=True)
def _tree_set_kernel(tree, capacity, idx, values):
    for k in range(idx.shape[0]):
        node = idx[k] + capacity
        tree[node] = values[k]
        node >>= 1
        while node >= 1:
            tree[node] = tree[2 * node] + tree[2 * node + 1]
            node >>= 1


@njit(cache=True)
def _tree_build_kernel(tree, capacity):
    for node in range(capacity - 1, 0, -1):
        tree[node] = tree[2 * node] + tree[2 * node + 1]


@njit(cache=True)
def _tree_sample_kernel(tree, capacity, draws, out):
    for k in range(draws.shape[0]):
        node = 1
        s = draws[k]
        while node < capacity:
            left = 2 * node
            if s < tree[left]:
                node = left
            else:
                s -= tree[left]
                node = left + 1
        out[k] = node - capacity


REPLAY_FIELDS = ("grid", "features", "actions", "rewards", "dones", "action_masks")


class DictReplayBuffer:
    """Circular buffer for dict observations with action masks."""

    def __init__(self, buffer_size: int, n_envs: int, observation_space, action_dim: int,
                 grid_dtype=np.float16, grid_codec=None, priority_alpha: float = 0.0, priority_eps: float = 0.01,
                 priority_n_step: int = 1):
        self.n_envs = max(1, int(n_envs))
        self.rows = max(2, int(buffer_size) // self.n_envs)
        self.action_dim = int(action_dim)
        self.grid_codec = None
        if grid_codec is not None:
            den, sentinel = grid_codec
            self.grid_codec = (np.ascontiguousarray(den, dtype=np.float32), np.ascontiguousarray(sentinel, dtype=np.bool_))
            grid_dtype = np.uint8
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
        self._init_priorities(priority_alpha, priority_eps, priority_n_step)

    def _init_priorities(self, alpha: float, eps: float, n_step: int = 1) -> None:
        self.priority_alpha = 0.0
        self.priority_eps = max(0.0, float(eps))
        self._td_raw = np.ones((self.rows, self.n_envs), dtype=np.float32)
        self._max_td = 1.0
        self.priorities = np.ones((self.rows, self.n_envs), dtype=np.float64)
        self._max_priority = 1.0
        self._row_version = np.zeros(self.rows, dtype=np.int64)
        self._tree = None
        self._tree_capacity = 0
        self._tree_n_step = 1
        self.set_priority_alpha(alpha, eps, n_step)

    def _priority_of(self, td):
        return np.maximum((np.asarray(td, dtype=np.float64) + self.priority_eps) ** self.priority_alpha, 1e-12)

    def set_priority_alpha(self, alpha: float, eps: Optional[float] = None, n_step: Optional[int] = None) -> None:
        alpha = max(0.0, float(alpha))
        eps = self.priority_eps if eps is None else max(0.0, float(eps))
        n_step = self._tree_n_step if n_step is None else max(1, int(n_step))
        rescale = alpha != self.priority_alpha or eps != self.priority_eps
        regate = n_step != self._tree_n_step
        self.priority_alpha = alpha
        self.priority_eps = eps
        self._tree_n_step = n_step
        if alpha <= 0.0:
            self._tree = None
            self._tree_capacity = 0
            return
        if rescale:
            self.priorities[:] = self._priority_of(self._td_raw)
            self._max_priority = float(self._priority_of(self._max_td))
        if self._tree is None or rescale or regate:
            self._rebuild_tree()

    def _tree_ready(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        chain = (rows[:, None] + np.arange(self._tree_n_step, dtype=np.int64)[None, :]) % self.rows
        return self._has_next[chain].all(axis=1)

    def priority_stats(self) -> Optional[dict]:
        if self._tree is None:
            return None
        active = int(self._tree_ready(np.arange(self.rows)).sum()) * self.n_envs
        return {
            "priority_max": float(self._max_priority),
            "priority_mean": float(self._tree[1] / active) if active else 0.0,
        }

    def _rebuild_tree(self) -> None:
        capacity = 1
        while capacity < self.rows * self.n_envs:
            capacity *= 2
        tree = np.zeros(2 * capacity, dtype=np.float64)
        leaves = self.priorities * self._tree_ready(np.arange(self.rows))[:, None]
        tree[capacity:capacity + self.rows * self.n_envs] = leaves.reshape(-1)
        _tree_build_kernel(tree, capacity)
        self._tree = tree
        self._tree_capacity = capacity

    def _set_row_leaves(self, row: int, active: bool) -> None:
        if self._tree is None:
            return
        idx = row * self.n_envs + np.arange(self.n_envs, dtype=np.int64)
        values = self.priorities[row] if active else np.zeros(self.n_envs, dtype=np.float64)
        _tree_set_kernel(self._tree, self._tree_capacity, idx, np.ascontiguousarray(values, dtype=np.float64))

    def _after_write(self, i: int) -> None:
        self._row_version[i] += 1
        self._td_raw[i] = self._max_td
        self.priorities[i] = self._max_priority
        self._has_next[i] = False
        self._set_row_leaves(i, False)
        if i == self._seam_row:
            self._seam_row = None
        prev = (i - 1) % self.rows
        if (i > 0 or self._full) and prev != self._seam_row:
            self._has_next[prev] = True
            old = (i - self._tree_n_step) % self.rows
            if self._tree is not None and self._tree_ready(np.array([old]))[0]:
                self._set_row_leaves(old, True)

        self._pos += 1
        if self._pos == self.rows:
            self._pos = 0
            self._full = True

    def _draw_uniform(self, batch_size: int) -> tuple:
        rows = np.flatnonzero(self._has_next)
        if len(rows) == 0:
            raise ValueError("replay buffer holds no transition with a stored successor yet")
        row_idx = rows[np.random.randint(0, len(rows), size=batch_size)]
        env_idx = np.random.randint(0, self.n_envs, size=batch_size)
        return row_idx, env_idx

    def _draw_indices(self, batch_size: int) -> tuple:
        if self._tree is None or self._tree[1] <= 0.0:
            return self._draw_uniform(batch_size)
        total = self._tree[1]
        draws = (np.arange(batch_size, dtype=np.float64) + np.random.random(batch_size)) * (total / batch_size)
        flat = np.empty(batch_size, dtype=np.int64)
        _tree_sample_kernel(self._tree, self._tree_capacity, draws, flat)
        row_idx = flat // self.n_envs
        env_idx = flat - row_idx * self.n_envs
        bad = (flat >= self.rows * self.n_envs) | ~self._has_next[np.minimum(row_idx, self.rows - 1)]
        if bad.any():
            fix_rows, fix_envs = self._draw_uniform(int(bad.sum()))
            row_idx[bad] = fix_rows
            env_idx[bad] = fix_envs
        return row_idx, env_idx

    def update_priorities(self, indices: np.ndarray, versions: np.ndarray, td_abs: np.ndarray) -> None:
        if self._tree is None:
            return
        indices = np.asarray(indices, dtype=np.int64)
        rows = indices // self.n_envs
        fresh = np.asarray(versions, dtype=np.int64) == self._row_version[rows]
        if not fresh.any():
            return
        indices, rows = indices[fresh], rows[fresh]
        td = np.nan_to_num(np.asarray(td_abs, dtype=np.float64)[fresh], nan=0.0, posinf=0.0, neginf=0.0)
        values = self._priority_of(td)
        self._td_raw.reshape(-1)[indices] = td
        self.priorities.reshape(-1)[indices] = values
        if float(td.max()) > self._max_td:
            self._max_td = float(td.max())
            self._max_priority = float(self._priority_of(self._max_td))
        active = self._tree_ready(rows)
        if active.any():
            _tree_set_kernel(self._tree, self._tree_capacity, np.ascontiguousarray(indices[active]),
                             np.ascontiguousarray(values[active]))

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
        if self.grid_codec is not None:
            _encode_grid_kernel(np.ascontiguousarray(obs["grid_tensor"], dtype=np.float32),
                                self.grid_codec[0], self.grid_codec[1], self.grid[i])
        else:
            torch.from_numpy(self.grid[i]).copy_(torch.from_numpy(np.ascontiguousarray(obs["grid_tensor"])))
        self.features[i] = obs["features"]
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.action_masks[i] = action_masks
        self._after_write(i)

    def _n_step_targets(self, row_idx: np.ndarray, env_idx: np.ndarray, n_step: int, gamma: float,
                        rewards: np.ndarray, dones: np.ndarray) -> tuple:
        batch = row_idx.shape[0]
        returns = np.zeros(batch, dtype=np.float64)
        discount = np.ones(batch, dtype=np.float64)
        terminal = np.zeros(batch, dtype=bool)
        alive = np.ones(batch, dtype=bool)
        cur = row_idx.copy()
        boot = (row_idx + 1) % self.rows
        for j in range(max(1, int(n_step))):
            r = rewards[cur, env_idx].astype(np.float64)
            d = dones[cur, env_idx].astype(bool)
            returns += np.where(alive, discount * r, 0.0)
            discount = np.where(alive, discount * gamma, discount)
            terminal |= alive & d
            nxt = (cur + 1) % self.rows
            boot = np.where(alive, nxt, boot)
            alive &= ~d
            cont = alive & self._has_next[nxt] & (j + 1 < n_step)
            cur = np.where(cont, nxt, cur)
            alive = cont
            if not alive.any():
                break
        return boot, returns.astype(np.float32), terminal.astype(np.float32), discount.astype(np.float32)

    def sample(self, batch_size: int, out: Optional[dict] = None, n_step: int = 1, gamma: float = 0.99) -> dict:
        t0 = time.perf_counter()

        row_idx, env_idx = self._draw_indices(batch_size)
        next_row_idx, returns, terminal, discounts = self._n_step_targets(
            row_idx, env_idx, n_step, gamma, self.rewards, self.dones)

        flat_np = row_idx * self.n_envs + env_idx
        flat = torch.from_numpy(flat_np)
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
            rewards=returns,
            dones=terminal,
            discounts=discounts,
            action_masks=self.action_masks[row_idx, env_idx],
            next_action_masks=self.action_masks[next_row_idx, env_idx],
            indices=flat_np,
            versions=self._row_version[row_idx],
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
            dtype_str, row_shape, row_nbytes = self._field_spec(name)
            fields[name] = {"dtype": dtype_str, "shape": list(row_shape), "offset": offset}
            offset += n * row_nbytes
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
                    for a, b in chunks:
                        f.write(self._chunk_bytes(name, a, b))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return n

    def _field_spec(self, name: str) -> tuple:
        arr = getattr(self, name)
        return arr.dtype.str, tuple(arr.shape[1:]), arr[0].nbytes

    def _chunk_bytes(self, name: str, a: int, b: int):
        return memoryview(getattr(self, name)[a:b])

    def _read_rows(self, f, name: str, keep: int) -> None:
        arr = getattr(self, name)
        target = memoryview(arr[:keep]).cast("B")
        if f.readinto(target) != len(target):
            raise ValueError(f"replay buffer file is truncated in field '{name}'")

    def _store_rows(self, name: str, rows: np.ndarray) -> None:
        getattr(self, name)[:rows.shape[0]] = rows

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
        self._td_raw[:] = self._max_td
        self.priorities[:] = self._max_priority
        self._row_version += 1
        if self._tree is not None:
            self._rebuild_tree()
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
                dtype_str, _row_shape, row_bytes = self._field_spec(name)
                spec = header["fields"][name]
                if spec["dtype"] != dtype_str:
                    raise ValueError(f"replay buffer field '{name}' was saved as {spec['dtype']}, expected {dtype_str}")
                f.seek(data_start + int(spec["offset"]) + (n - keep) * row_bytes)
                self._read_rows(f, name, keep)
        return self._finish_load(keep, header.get("num_timesteps", 0))

    def _load_npz(self, path) -> int:
        with np.load(path) as data:
            n = int(data["n_rows"])
            self._check_layout(int(data["n_envs"]), {k: data[k].shape[1:] for k in ("grid", "features", "action_masks")})
            grid_dtype = self._field_spec("grid")[0]
            if data["grid"].dtype.str != grid_dtype:
                raise ValueError(f"replay buffer grid was saved as {data['grid'].dtype.str}, expected {grid_dtype}")
            keep = min(n, self.rows)
            for key in REPLAY_FIELDS:
                self._store_rows(key, data[key][n - keep:n])
            num_timesteps = int(data["num_timesteps"]) if "num_timesteps" in data else 0
        return self._finish_load(keep, num_timesteps)


class DeviceReplayBuffer(DictReplayBuffer):
    def __init__(self, buffer_size: int, n_envs: int, observation_space, action_dim: int, device,
                 grid_codec=None, priority_alpha: float = 0.0, priority_eps: float = 0.01, priority_n_step: int = 1):
        self.device = torch.device(device)
        self.n_envs = max(1, int(n_envs))
        self.rows = max(2, int(buffer_size) // self.n_envs)
        self.action_dim = int(action_dim)
        self.grid_codec = None
        grid_dtype = torch.float16
        if grid_codec is not None:
            den, sentinel = grid_codec
            self.grid_codec = (np.ascontiguousarray(den, dtype=np.float32), np.ascontiguousarray(sentinel, dtype=np.bool_))
            grid_dtype = torch.uint8
        self.grid_dtype = np.uint8 if grid_dtype == torch.uint8 else np.float16
        grid_shape = tuple(observation_space["grid_tensor"].shape)
        feat_shape = tuple(observation_space["features"].shape)
        self.grid_shape = grid_shape
        self.feat_shape = feat_shape

        shape = (self.rows, self.n_envs)
        dev = self.device
        self.grid = torch.zeros(shape + grid_shape, dtype=grid_dtype, device=dev)
        self._grid_flat = self.grid.view((self.rows * self.n_envs,) + grid_shape)
        self.features = torch.zeros(shape + feat_shape, dtype=torch.float32, device=dev)
        self.actions = torch.zeros(shape, dtype=torch.int64, device=dev)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.action_masks = torch.ones(shape + (self.action_dim,), dtype=torch.bool, device=dev)
        self._has_next = np.zeros(self.rows, dtype=bool)
        self._rewards_host = np.zeros(shape, dtype=np.float32)
        self._dones_host = np.zeros(shape, dtype=np.float32)

        pin = dev.type == "cuda"
        self._stage = {
            "grid": torch.empty((self.n_envs,) + grid_shape, dtype=grid_dtype, pin_memory=pin),
            "features": torch.empty((self.n_envs,) + feat_shape, dtype=torch.float32, pin_memory=pin),
            "actions": torch.empty((self.n_envs,), dtype=torch.int64, pin_memory=pin),
            "rewards": torch.empty((self.n_envs,), dtype=torch.float32, pin_memory=pin),
            "dones": torch.empty((self.n_envs,), dtype=torch.float32, pin_memory=pin),
            "action_masks": torch.empty((self.n_envs, self.action_dim), dtype=torch.bool, pin_memory=pin),
        }
        self._stage_np = {name: tensor.numpy() for name, tensor in self._stage.items()}
        self._stage_event = torch.cuda.Event() if pin else None

        self._pos = 0
        self._full = False
        self._seam_row: Optional[int] = None
        self.last_sample_time_s: Optional[float] = None
        self.num_timesteps: int = 0
        self._init_priorities(priority_alpha, priority_eps, priority_n_step)

    def add(self, obs, next_obs, actions, rewards, dones, action_masks, next_action_masks) -> None:
        i = self._pos
        if self._stage_event is not None:
            self._stage_event.synchronize()
        stage = self._stage_np
        if self.grid_codec is not None:
            _encode_grid_kernel(np.ascontiguousarray(obs["grid_tensor"], dtype=np.float32),
                                self.grid_codec[0], self.grid_codec[1], stage["grid"])
        else:
            self._stage["grid"].copy_(torch.from_numpy(np.ascontiguousarray(obs["grid_tensor"])))
        stage["features"][:] = obs["features"]
        stage["actions"][:] = actions
        stage["rewards"][:] = rewards
        stage["dones"][:] = dones
        stage["action_masks"][:] = action_masks
        self._rewards_host[i] = stage["rewards"]
        self._dones_host[i] = stage["dones"]
        non_blocking = self._stage_event is not None
        for name, tensor in self._stage.items():
            getattr(self, name)[i].copy_(tensor, non_blocking=non_blocking)
        if self._stage_event is not None:
            self._stage_event.record()
        self._after_write(i)

    def sample(self, batch_size: int, out: Optional[dict] = None, n_step: int = 1, gamma: float = 0.99) -> dict:
        t0 = time.perf_counter()
        if self._stage_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._stage_event)
        row_idx, env_idx = self._draw_indices(batch_size)
        next_row_idx, returns, terminal, discounts = self._n_step_targets(
            row_idx, env_idx, n_step, gamma, self._rewards_host, self._dones_host)
        non_blocking = self.device.type == "cuda"
        flat_np = row_idx * self.n_envs + env_idx
        flat = torch.from_numpy(flat_np).to(self.device, non_blocking=non_blocking)
        next_flat = torch.from_numpy(next_row_idx * self.n_envs + env_idx).to(self.device, non_blocking=non_blocking)
        total = self.rows * self.n_envs
        features = self.features.view(total, -1)
        masks = self.action_masks.view(total, -1)
        batch = dict(
            obs={
                "grid_tensor": torch.index_select(self._grid_flat, 0, flat),
                "features": torch.index_select(features, 0, flat),
            },
            next_obs={
                "grid_tensor": torch.index_select(self._grid_flat, 0, next_flat),
                "features": torch.index_select(features, 0, next_flat),
            },
            actions=torch.index_select(self.actions.view(total), 0, flat),
            rewards=torch.from_numpy(returns).to(self.device, non_blocking=non_blocking),
            dones=torch.from_numpy(terminal).to(self.device, non_blocking=non_blocking),
            discounts=torch.from_numpy(discounts).to(self.device, non_blocking=non_blocking),
            action_masks=torch.index_select(masks, 0, flat),
            next_action_masks=torch.index_select(masks, 0, next_flat),
            indices=flat_np,
            versions=self._row_version[row_idx],
        )
        self.last_sample_time_s = time.perf_counter() - t0
        return batch

    def _field_spec(self, name: str) -> tuple:
        arr = getattr(self, name)
        dtype_str = arr[:0].cpu().numpy().dtype.str
        return dtype_str, tuple(arr.shape[1:]), arr[0].numel() * arr.element_size()

    def _chunk_bytes(self, name: str, a: int, b: int):
        return memoryview(np.ascontiguousarray(getattr(self, name)[a:b].cpu().numpy()))

    def _read_rows(self, f, name: str, keep: int) -> None:
        arr = getattr(self, name)
        host = np.empty((keep,) + tuple(arr.shape[1:]), dtype=np.dtype(self._field_spec(name)[0]))
        target = memoryview(host).cast("B")
        if f.readinto(target) != len(target):
            raise ValueError(f"replay buffer file is truncated in field '{name}'")
        self._store_rows(name, host)

    def _store_rows(self, name: str, rows: np.ndarray) -> None:
        arr = getattr(self, name)
        arr[:rows.shape[0]].copy_(torch.from_numpy(np.ascontiguousarray(rows)).to(arr.dtype))
        if name == "rewards":
            self._rewards_host[:rows.shape[0]] = rows
        elif name == "dones":
            self._dones_host[:rows.shape[0]] = rows


def read_replay_file(path, fields=None) -> dict:
    path = pathlib.Path(path)
    fields = list(REPLAY_FIELDS) if fields is None else [name for name in REPLAY_FIELDS if name in set(fields)]
    with open(path, "rb") as f:
        if f.read(len(REPLAY_MAGIC)) != REPLAY_MAGIC:
            with np.load(path) as data:
                out = {"n_rows": int(data["n_rows"]), "n_envs": int(data["n_envs"]),
                       "num_timesteps": int(data["num_timesteps"]) if "num_timesteps" in data else 0}
                for name in fields:
                    out[name] = np.array(data[name])
                return out
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = f.tell()
        n = int(header["n_rows"])
        out = {"n_rows": n, "n_envs": int(header["n_envs"]), "num_timesteps": int(header.get("num_timesteps", 0))}
        for name in fields:
            spec = header["fields"][name]
            f.seek(data_start + int(spec["offset"]))
            shape = (n,) + tuple(spec["shape"])
            out[name] = np.fromfile(f, dtype=np.dtype(spec["dtype"]), count=int(np.prod(shape))).reshape(shape)
    return out
