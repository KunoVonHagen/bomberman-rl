import numpy as np
from sim import VecBomberman, FREE, UP, DOWN, LEFT, RIGHT, WAIT, BOMB


class RandomBot:
    def __call__(self, sim: VecBomberman, slots: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 6, size=slots.shape)


class FrozenPolicy:
    """Wraps a stable-baselines3 policy to freeze its weights and
    disable training-mode behavior (e.g. dropout, batchnorm) for inference.
    """

    def __init__(self, policy):
        import copy
        self.policy = copy.deepcopy(policy)
        self.policy.set_training_mode(False)

    def predict(self, obs, deterministic: bool = False):
        import torch
        with torch.no_grad():
            obs_t, _ = self.policy.obs_to_tensor(obs)
            actions, _, _ = self.policy(obs_t, deterministic=deterministic)
        return actions.cpu().numpy(), None


class SelfPlayOpponent:
    """
    Maintains a pool of frozen snapshots of the agent's own policy, and
    randomly selects one per environment to play against. This is a cheap
    way to implement self-play without the complexity of a full opponent
    training loop. The pool is updated by calling add_snapshot() with the
    current policy after each training iteration.
    """

    def __init__(self, pool_size: int = 5, fallback=None):
        from gym_env import encode_obs  # local import: avoids bots<->gym_env cycle
        self._encode_obs = encode_obs
        self.pool_size = pool_size
        self.policies: list[FrozenPolicy] = []
        self.fallback = fallback or RandomBot()

    def add_snapshot(self, policy):
        self.policies.append(FrozenPolicy(policy))
        if len(self.policies) > self.pool_size:
            self.policies.pop(0)

    def __call__(self, sim: VecBomberman, slots: np.ndarray) -> np.ndarray:
        N, k = slots.shape
        if not self.policies:
            return self.fallback(sim, slots)

        actions = np.zeros((N, k), dtype=np.int64)
        pool_idx = np.random.randint(0, len(self.policies), size=N)
        obs_per_slot = [self._encode_obs(sim, slots[:, j]) for j in range(k)]

        for p_i, policy in enumerate(self.policies):
            sel = pool_idx == p_i
            if not sel.any():
                continue

            batched_obs = np.concatenate([obs_per_slot[j][sel] for j in range(k)], axis=0)
            act, _ = policy.predict(batched_obs, deterministic=False)
            act = act.reshape(k, -1).T
            actions[sel] = act
        return actions


class RandomSafeBot:
    """
    RandomBot that avoids moving into walls or lethal tiles, and avoids
    dropping bombs if one is already active. This is a cheap way to implement
    a "safe" opponent without the complexity of a full opponent training loop.
    """

    def __call__(self, sim: VecBomberman, slots: np.ndarray) -> np.ndarray:
        N, k = slots.shape
        danger = sim.dangerous_mask()
        actions = np.random.randint(0, 6, size=(N, k))
        rows = np.arange(N)
        for j in range(k):
            slot = slots[:, j]
            x = sim.agent_xy[rows, slot, 0].astype(np.int32)
            y = sim.agent_xy[rows, slot, 1].astype(np.int32)
            for a, dx, dy in ((UP, 0, -1), (DOWN, 0, 1), (LEFT, -1, 0), (RIGHT, 1, 0)):
                sel = actions[:, j] == a
                if not sel.any():
                    continue
                nx, ny = np.clip(x + dx, 0, sim.C - 1), np.clip(y + dy, 0, sim.R - 1)
                bad = sel & ((sim.arena[rows, nx, ny] != FREE) | danger[rows, nx, ny])
                actions[bad, j] = WAIT
            drop = actions[:, j] == BOMB
            no_bomb = drop & sim.bomb_active[rows, slot]
            actions[no_bomb, j] = WAIT
        return actions