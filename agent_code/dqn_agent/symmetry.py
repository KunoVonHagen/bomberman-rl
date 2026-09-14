from __future__ import annotations

import numpy as np
import torch

import settings as s
from .gym_environment import (
    ACTIONS,
    FEATURE_SAFE_DOWN,
    FEATURE_SAFE_LEFT,
    FEATURE_SAFE_RIGHT,
    FEATURE_SAFE_UP,
    FEATURE_SELF_X,
    FEATURE_SELF_Y,
)

N_SYMMETRIES = 8
_W, _H = s.COLS, s.ROWS
_MOVE_DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
SAFE_FEATURES = [FEATURE_SAFE_UP, FEATURE_SAFE_RIGHT, FEATURE_SAFE_DOWN, FEATURE_SAFE_LEFT]


def transform_array(array: np.ndarray, k: int) -> np.ndarray:
    out = np.rot90(array, k % 4, axes=(-2, -1))
    if k >= 4:
        out = np.flip(out, axis=-2)
    return np.ascontiguousarray(out)


def transform_tensor(tensor: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.rot90(tensor, k % 4, dims=(-2, -1))
    if k >= 4:
        out = torch.flip(out, dims=(-2,))
    return out


def _position_tables() -> np.ndarray:
    index = np.arange(_W * _H).reshape(_W, _H)
    tables = np.zeros((N_SYMMETRIES, _W, _H, 2), dtype=np.int64)
    for k in range(N_SYMMETRIES):
        moved = transform_array(index, k)
        for nx in range(_W):
            for ny in range(_H):
                ox, oy = divmod(int(moved[nx, ny]), _H)
                tables[k, ox, oy] = (nx, ny)
    return tables


POSITION = _position_tables()


def _action_permutations() -> np.ndarray:
    perms = np.arange(len(ACTIONS))[None, :].repeat(N_SYMMETRIES, axis=0)
    ox, oy = _W // 2, _H // 2
    for k in range(N_SYMMETRIES):
        for name, (dx, dy) in _MOVE_DELTAS.items():
            o = POSITION[k, ox, oy]
            p = POSITION[k, ox + dx, oy + dy]
            delta = (int(p[0] - o[0]), int(p[1] - o[1]))
            target = next(n for n, d in _MOVE_DELTAS.items() if d == delta)
            perms[k, ACTIONS.index(name)] = ACTIONS.index(target)
    return perms


ACTION_PERM = _action_permutations()
ACTION_PERM_INV = np.argsort(ACTION_PERM, axis=1)


def transform_position(x: int, y: int, k: int) -> tuple[int, int]:
    p = POSITION[k, x, y]
    return int(p[0]), int(p[1])


def transform_features(features: np.ndarray, k: int) -> np.ndarray:
    out = np.array(features, copy=True)
    xs = np.rint((out[..., FEATURE_SELF_X] + 1.0) * (_W - 1) / 2.0).astype(np.int64)
    ys = np.rint((out[..., FEATURE_SELF_Y] + 1.0) * (_H - 1) / 2.0).astype(np.int64)
    moved = POSITION[k, xs, ys]
    out[..., FEATURE_SELF_X] = moved[..., 0] * (2.0 / (_W - 1)) - 1.0
    out[..., FEATURE_SELF_Y] = moved[..., 1] * (2.0 / (_H - 1)) - 1.0
    out[..., SAFE_FEATURES] = out[..., SAFE_FEATURES][..., ACTION_PERM_INV[k, :4]]
    return out


def transform_actions(actions: np.ndarray, k: int) -> np.ndarray:
    return ACTION_PERM[k][np.asarray(actions).astype(np.int64)]


def transform_masks(masks: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(masks)[..., ACTION_PERM_INV[k]]


def transform_observation(obs: dict, k: int) -> dict:
    return {
        "grid_tensor": transform_array(np.asarray(obs["grid_tensor"]), k),
        "features": transform_features(np.asarray(obs["features"]), k),
    }


def transform_game_state(state: dict, k: int) -> dict:
    def pos(p):
        return transform_position(int(p[0]), int(p[1]), k)

    def agent(entry):
        return (entry[0], entry[1], entry[2], pos(entry[3]))

    out = dict(state)
    out["field"] = transform_array(np.asarray(state["field"]), k)
    out["explosion_map"] = transform_array(np.asarray(state["explosion_map"]), k)
    out["self"] = agent(state["self"])
    out["others"] = [agent(o) for o in state.get("others", [])]
    out["bombs"] = [(pos(p), t) for p, t in state.get("bombs", [])]
    out["coins"] = [pos(p) for p in state.get("coins", [])]
    return out


class TensorAugmenter:
    def __init__(self, device: torch.device):
        self.device = device
        self.perm = torch.as_tensor(ACTION_PERM, device=device)
        self.perm_inv = torch.as_tensor(ACTION_PERM_INV, device=device)
        self.position = torch.as_tensor(POSITION, device=device, dtype=torch.float32)
        self.safe_idx = torch.as_tensor(SAFE_FEATURES, device=device)

    def transform_features(self, f: torch.Tensor, k: int) -> torch.Tensor:
        f = f.clone()
        xs = torch.round((f[:, FEATURE_SELF_X] + 1.0) * (_W - 1) / 2.0).long()
        ys = torch.round((f[:, FEATURE_SELF_Y] + 1.0) * (_H - 1) / 2.0).long()
        moved = self.position[k, xs, ys]
        f[:, FEATURE_SELF_X] = moved[:, 0] * (2.0 / (_W - 1)) - 1.0
        f[:, FEATURE_SELF_Y] = moved[:, 1] * (2.0 / (_H - 1)) - 1.0
        f[:, self.safe_idx] = f[:, self.safe_idx][:, self.perm_inv[k, :4]]
        return f

    def augment(self, obs: dict, next_obs: dict, actions: torch.Tensor, next_masks: torch.Tensor) -> None:
        ks = torch.randint(0, N_SYMMETRIES, (actions.shape[0],), device=self.device)
        for k in range(1, N_SYMMETRIES):
            idx = torch.nonzero(ks == k).squeeze(1)
            if idx.numel() == 0:
                continue
            for d in (obs, next_obs):
                d["grid_tensor"][idx] = transform_tensor(d["grid_tensor"][idx], k)
                d["features"][idx] = self.transform_features(d["features"][idx], k)
            actions[idx] = self.perm[k][actions[idx]]
            next_masks[idx] = next_masks[idx][:, self.perm_inv[k]]
