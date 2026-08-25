from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim import (
    VecBomberman, Settings, ACTION_NAMES, WALL, CRATE,
    EV_COIN_COLLECTED, EV_KILLED_OPPONENT, EV_KILLED_SELF, EV_GOT_KILLED,
    EV_INVALID, EV_CRATE_DESTROYED, EV_BOMB_DROPPED,
)

N_CHANNELS = 8  # walls, crates, coins, danger, bomb_timer, self, teammates(unused=0), opponents


def encode_obs(sim: VecBomberman, agent_slot: np.ndarray) -> np.ndarray:
    """
    sim: VecBomberman with N games in parallel
    agent_slot: (N,) which slot is the learning agent in each game
    returns: (N, N_CHANNELS, C, R) float32 in [0, 1] -- the observation for each game, for the learning agent's slot
    """
    N, C, R = sim.n, sim.C, sim.R
    rows = np.arange(N)
    obs = np.zeros((N, N_CHANNELS, C, R), dtype=np.float32)
    obs[:, 0] = (sim.arena == WALL)
    obs[:, 1] = (sim.arena == CRATE)

    coin_grid = np.zeros((N, C, R), dtype=np.float32)
    collectable = sim.coin_state == 1
    for k in range(sim.K):
        sel = collectable[:, k]
        if sel.any():
            coin_grid[sel, sim.coin_xy[sel, k, 0], sim.coin_xy[sel, k, 1]] = 1.0
    obs[:, 2] = coin_grid

    obs[:, 3] = sim.danger_map().astype(np.float32) / max(sim.s.explosion_timer, 1)

    bomb_grid = np.zeros((N, C, R), dtype=np.float32)
    for slot in range(4):
        active = sim.bomb_active[:, slot]
        if active.any():
            bomb_grid[active, sim.bomb_xy[active, slot, 0], sim.bomb_xy[active, slot, 1]] = \
                sim.bomb_timer[active, slot].astype(np.float32) / max(sim.s.bomb_timer, 1)
    obs[:, 4] = bomb_grid

    self_grid = np.zeros((N, C, R), dtype=np.float32)
    alive_self = sim.agent_alive[rows, agent_slot]
    self_grid[alive_self, sim.agent_xy[rows, agent_slot, 0][alive_self],
              sim.agent_xy[rows, agent_slot, 1][alive_self]] = 1.0
    obs[:, 5] = self_grid

    opp_grid = np.zeros((N, C, R), dtype=np.float32)
    for slot in range(4):
        is_other = (slot != agent_slot) & sim.agent_alive[:, slot]
        if is_other.any():
            opp_grid[is_other, sim.agent_xy[is_other, slot, 0], sim.agent_xy[is_other, slot, 1]] = 1.0
    obs[:, 6] = opp_grid

    obs[:, 7] = sim.step_no[:, None, None].astype(np.float32) / sim.s.max_steps
    return obs


def default_reward(events: np.ndarray) -> np.ndarray:
    """events: (..., ) uint16 bit flags -> shaped reward, same shape."""
    r = np.zeros(events.shape, dtype=np.float32)
    r += 1.0 * ((events & EV_COIN_COLLECTED) != 0)
    r += 5.0 * ((events & EV_KILLED_OPPONENT) != 0)
    r -= 5.0 * ((events & EV_KILLED_SELF) != 0)
    r -= 5.0 * ((events & EV_GOT_KILLED) != 0)
    r -= 0.01 * ((events & EV_INVALID) != 0)
    r += 0.1 * ((events & EV_CRATE_DESTROYED) != 0)
    return r


class RandomOpponent:
    """Fills in actions for non-learning slots. Swap in a trained policy
    (or a rule-based bot from bots.py) for stronger training partners /
    self-play."""

    def __call__(self, sim: VecBomberman, slots: np.ndarray) -> np.ndarray:
        # slots: (N, k) which slots need actions filled in
        return np.random.randint(0, 6, size=slots.shape)


class BombermanEnv(gym.Env):
    """Single-game Gymnasium Env. Learning agent is always slot 0."""

    metadata = {"render_modes": []}

    def __init__(self, settings: Settings | None = None, opponent=None, seed: int = 0):
        super().__init__()
        self.sim = VecBomberman(1, settings, seed=seed)
        self.opponent = opponent or RandomOpponent()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CHANNELS, self.sim.C, self.sim.R), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(np.array([0]))
        obs = encode_obs(self.sim, np.array([0]))[0]
        return obs, {}

    def step(self, action: int):
        actions = np.zeros((1, 4), dtype=np.int64)
        actions[0, 0] = action
        other_slots = np.array([[1, 2, 3]])
        actions[0, 1:4] = self.opponent(self.sim, other_slots)[0]
        events, done = self.sim.step(actions)
        obs = encode_obs(self.sim, np.array([0]))[0]
        reward = float(default_reward(events[0, 0]))
        terminated = bool(done[0])
        truncated = bool(self.sim.step_no[0] >= self.sim.s.max_steps and terminated)
        info = {"events": int(events[0, 0]), "score": int(self.sim.agent_score[0, 0])}
        return obs, reward, terminated, truncated, info


class BombermanVecEnv:
    """
    stable-baselines3 VecEnv-compatible wrapper. This is the high-throughput
    path: `num_envs` games are advanced in a single sim.step() call.
    Learning agent is slot 0 in every game; slots 1-3 use `opponent`.
    """

    def __init__(self, num_envs: int, settings: Settings | None = None, opponent=None, seed: int = 0):
        self.num_envs = num_envs
        self.sim = VecBomberman(num_envs, settings, seed=seed)
        self.opponent = opponent or RandomOpponent()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CHANNELS, self.sim.C, self.sim.R), dtype=np.float32)
        self._actions = None
        self.render_mode = None

    def reset(self):
        self.sim.reset_all()
        return encode_obs(self.sim, np.zeros(self.num_envs, dtype=np.int64))

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        full_actions = np.zeros((self.num_envs, 4), dtype=np.int64)
        full_actions[:, 0] = self._actions
        other_slots = np.tile(np.array([1, 2, 3]), (self.num_envs, 1))
        full_actions[:, 1:4] = self.opponent(self.sim, other_slots)

        events, done = self.sim.step(full_actions)
        rewards = default_reward(events[:, 0])
        infos = [{"score": int(self.sim.agent_score[n, 0])} for n in range(self.num_envs)]
        for n in np.nonzero(done)[0]:
            infos[n]["terminal_observation"] = None

            n_alive = int(self.sim.agent_alive[n].sum())
            if n_alive <= 1:
                infos[n]["won"] = bool(self.sim.agent_alive[n, 0])
            else:
                scores = self.sim.agent_score[n]
                infos[n]["won"] = bool(scores[0] == scores.max() and scores[0] > 0)

        self.sim.auto_reset_done(done)
        obs = encode_obs(self.sim, np.zeros(self.num_envs, dtype=np.int64))
        return obs, rewards, done, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [seed] * self.num_envs