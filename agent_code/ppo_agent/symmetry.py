from __future__ import annotations

import numpy as np
import torch

import settings as s
from .gym_environment import (
    ACTIONS,
    COIN_DIRECTION_FEATURES,
    CRATE_DIRECTION_FEATURES,
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
DIRECTION_FEATURES = [
    [FEATURE_SAFE_UP, FEATURE_SAFE_RIGHT, FEATURE_SAFE_DOWN, FEATURE_SAFE_LEFT],
    list(COIN_DIRECTION_FEATURES),
    list(CRATE_DIRECTION_FEATURES),
]


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
    for group in DIRECTION_FEATURES:
        if group[-1] < out.shape[-1]:
            out[..., group] = out[..., group][..., ACTION_PERM_INV[k, :4]]
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


_SPATIAL_PERM = torch.from_numpy(np.stack(
    [transform_array(np.arange(_W * _H).reshape(_W, _H), k).reshape(-1) for k in range(N_SYMMETRIES)]
).astype(np.int64))


def augment_grid(grid_flat: np.ndarray, ks: np.ndarray, chunk_rows: int = 2048) -> None:
    n, channels = grid_flat.shape[0], grid_flat.shape[1]
    view = torch.from_numpy(grid_flat).view(n, channels, _W * _H)
    ks_t = torch.from_numpy(np.ascontiguousarray(ks, dtype=np.int64))
    scratch = torch.empty((min(chunk_rows, n), channels, _W * _H), dtype=view.dtype)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        index = _SPATIAL_PERM[ks_t[start:end]].unsqueeze(1).expand(end - start, channels, _W * _H)
        out = scratch[:end - start]
        torch.gather(view[start:end], 2, index, out=out)
        view[start:end].copy_(out)


def augment_flat(features: np.ndarray, actions: np.ndarray, masks: np.ndarray, ks: np.ndarray) -> None:
    for k in range(1, N_SYMMETRIES):
        sel = np.flatnonzero(ks == k)
        if sel.size == 0:
            continue
        features[sel] = transform_features(features[sel], k)
        actions[sel] = transform_actions(actions[sel], k)
        masks[sel] = transform_masks(masks[sel], k)


def augment_arrays(grid: np.ndarray, features: np.ndarray, actions: np.ndarray,
                   masks: np.ndarray, ks: np.ndarray) -> None:
    for k in range(1, N_SYMMETRIES):
        sel = ks == k
        if not sel.any():
            continue
        grid[sel] = transform_array(grid[sel], k)
        features[sel] = transform_features(features[sel], k)
        actions[sel] = transform_actions(actions[sel], k)
        masks[sel] = transform_masks(masks[sel], k)


def augment_rollout_buffer(buffer, policy, batch_size: int) -> None:
    grid = buffer.observations["grid_tensor"]
    features = buffer.observations["features"]
    n_steps, n_envs = buffer.actions.shape[:2]
    ks = np.random.randint(0, N_SYMMETRIES, size=(n_steps, n_envs))

    flat_obs = {
        "grid_tensor": grid.reshape(-1, *grid.shape[2:]),
        "features": features.reshape(-1, features.shape[-1]),
    }
    masks = buffer.action_masks.reshape(-1, buffer.action_masks.shape[-1])
    augment_grid(flat_obs["grid_tensor"], ks.reshape(-1))
    augment_flat(flat_obs["features"], buffer.actions.reshape(-1, buffer.actions.shape[-1]), masks, ks.reshape(-1))
    actions = buffer.actions.reshape(-1).astype(np.int64)
    log_probs = np.empty(actions.shape[0], dtype=np.float32)
    values = np.empty_like(log_probs)
    with torch.no_grad():
        for start in range(0, actions.shape[0], batch_size):
            end = start + batch_size
            obs_t = {key: torch.as_tensor(value[start:end], device=policy.device) for key, value in flat_obs.items()}
            actions_t = torch.as_tensor(actions[start:end], device=policy.device)
            masks_t = torch.as_tensor(masks[start:end], device=policy.device)
            v, lp, _ = policy.evaluate_actions(obs_t, actions_t, action_masks=masks_t)
            values[start:end] = v.flatten().cpu().numpy()
            log_probs[start:end] = lp.cpu().numpy()
    buffer.log_probs[:] = log_probs.reshape(n_steps, n_envs)
    buffer.values[:] = values.reshape(n_steps, n_envs)

_DEVICE_TABLES: dict = {}


def _tables(device) -> dict:
    key = str(device)
    tables = _DEVICE_TABLES.get(key)
    if tables is None:
        tables = {
            "spatial": _SPATIAL_PERM.to(device),
            "position": torch.from_numpy(POSITION).to(device),
            "perm": torch.from_numpy(ACTION_PERM.astype(np.int64)).to(device),
            "inv": torch.from_numpy(ACTION_PERM_INV.astype(np.int64)).to(device),
            "groups": [
                torch.as_tensor(g, dtype=torch.long, device=device)
                for g in DIRECTION_FEATURES
            ],
        }
        _DEVICE_TABLES[key] = tables
    return tables


def action_perm_torch(ks: torch.Tensor) -> torch.Tensor:
    """(B, A) index tensor; row i maps an action `a` in the original frame to its index in the
    transformed frame (ACTION_PERM[k_i][a])."""
    return _tables(ks.device)["perm"][ks]


def transform_grid_torch(grid: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """grid: (B, C, W, H); ks: (B,) ints in [0, 8). Same result as transform_array per sample."""
    b, c = grid.shape[0], grid.shape[1]
    if grid.shape[-1] * grid.shape[-2] != _W * _H:
        raise ValueError(f"grid spatial shape {tuple(grid.shape[-2:])} != ({_W}, {_H})")
    flat = grid.reshape(b, c, _W * _H)
    index = _tables(grid.device)["spatial"][ks].unsqueeze(1).expand(b, c, _W * _H)
    return torch.gather(flat, 2, index).reshape(grid.shape)


def transform_features_torch(features: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """features: (B, F); ks: (B,). Same result as transform_features per sample."""
    t = _tables(features.device)
    out = features.clone()
    xs = torch.round((features[:, FEATURE_SELF_X] + 1.0) * (_W - 1) / 2.0).long().clamp_(0, _W - 1)
    ys = torch.round((features[:, FEATURE_SELF_Y] + 1.0) * (_H - 1) / 2.0).long().clamp_(0, _H - 1)
    moved = t["position"][ks, xs, ys]  # (B, 2)
    out[:, FEATURE_SELF_X] = moved[:, 0].to(out.dtype) * (2.0 / (_W - 1)) - 1.0
    out[:, FEATURE_SELF_Y] = moved[:, 1].to(out.dtype) * (2.0 / (_H - 1)) - 1.0
    move_inv = t["inv"][ks][:, :4]  # (B, 4)
    for group, cols in zip(DIRECTION_FEATURES, t["groups"]):
        if max(group) < features.shape[-1]:
            out[:, cols] = torch.gather(features[:, cols], 1, move_inv)
    return out


def transform_masks_torch(masks: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """masks: (B, A) bool/float. Returns a bool mask in the transformed frame."""
    inv = _tables(masks.device)["inv"][ks]
    return torch.gather(masks.to(torch.float32), 1, inv) > 0.5


def selftest(n: int = 512) -> None:
    """Checks the torch transforms against the numpy ones. Run from the repo root:
        python -m agent_code.ppo_agent.symmetry
    """
    rng = np.random.default_rng(0)
    ks = rng.integers(0, N_SYMMETRIES, size=n)
    ks_t = torch.from_numpy(ks)

    grid = rng.normal(size=(n, 5, _W, _H)).astype(np.float32)
    ref = np.stack([transform_array(grid[i], int(ks[i])) for i in range(n)])
    assert np.array_equal(transform_grid_torch(torch.from_numpy(grid), ks_t).numpy(), ref), "grid"

    n_feat = max(FEATURE_SELF_X, FEATURE_SELF_Y, *(i for g in DIRECTION_FEATURES for i in g)) + 3
    feats = rng.normal(size=(n, n_feat)).astype(np.float32)
    px, py = rng.integers(0, _W, n), rng.integers(0, _H, n)
    feats[:, FEATURE_SELF_X] = px * (2.0 / (_W - 1)) - 1.0
    feats[:, FEATURE_SELF_Y] = py * (2.0 / (_H - 1)) - 1.0
    ref = np.stack([transform_features(feats[i], int(ks[i])) for i in range(n)])
    got = transform_features_torch(torch.from_numpy(feats), ks_t).numpy()
    assert np.allclose(got, ref, atol=1e-6), "features"

    masks = (rng.random((n, len(ACTIONS))) > 0.3)
    ref = np.stack([transform_masks(masks[i], int(ks[i])) for i in range(n)])
    assert np.array_equal(transform_masks_torch(torch.from_numpy(masks), ks_t).numpy(), ref), "masks"

    perm = action_perm_torch(ks_t).numpy()
    acts = rng.integers(0, len(ACTIONS), n)
    ref = np.array([transform_actions(acts[i], int(ks[i])) for i in range(n)])
    assert np.array_equal(perm[np.arange(n), acts], ref), "actions"
    print("symmetry selftest OK")


if __name__ == "__main__":
    selftest()