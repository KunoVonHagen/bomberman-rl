from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

WALL, FREE, CRATE = -1, 0, 1

# Actions
UP, DOWN, LEFT, RIGHT, WAIT, BOMB = 0, 1, 2, 3, 4, 5
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]
DX = np.array([0, 0, -1, 1, 0, 0])
DY = np.array([-1, 1, 0, 0, 0, 0])

SCENARIOS = {
    "empty": dict(CRATE_DENSITY=0.0, COIN_COUNT=0),
    "coin-heaven": dict(CRATE_DENSITY=0.0, COIN_COUNT=50),
    "loot-crate": dict(CRATE_DENSITY=0.75, COIN_COUNT=50),
    "classic": dict(CRATE_DENSITY=0.75, COIN_COUNT=9),
}


@dataclass
class Settings:
    cols: int = 17
    rows: int = 17
    max_steps: int = 400
    bomb_timer: int = 4
    bomb_power: int = 3
    explosion_timer: int = 2
    smoke_timer: int = 2
    reward_kill: int = 5
    reward_coin: int = 1
    scenario: str = "classic"
    max_coins: int = field(default=None)

    def __post_init__(self):
        if self.max_coins is None:
            self.max_coins = SCENARIOS[self.scenario]["COIN_COUNT"]


START_POSITIONS = None


def _start_positions(s: Settings):
    return [(1, 1), (1, s.rows - 2), (s.cols - 2, 1), (s.cols - 2, s.rows - 2)]


EV_MOVED = 1 << 0
EV_INVALID = 1 << 1
EV_WAITED = 1 << 2
EV_BOMB_DROPPED = 1 << 3
EV_BOMB_EXPLODED = 1 << 4
EV_CRATE_DESTROYED = 1 << 5
EV_COIN_FOUND = 1 << 6
EV_COIN_COLLECTED = 1 << 7
EV_KILLED_SELF = 1 << 8
EV_KILLED_OPPONENT = 1 << 9
EV_GOT_KILLED = 1 << 10
EV_OPPONENT_ELIMINATED = 1 << 11
EV_SURVIVED_ROUND = 1 << 12


class VecBomberman:
    """
    Vectorized Bomberman simulator. Implements the same rules as the original
    Bomberman game, but in a fully vectorized NumPy implementation for high-throughput RL training.
    Supports multiple parallel environments (n_envs) and multiple agents per environment (4).
    """

    def __init__(self, n_envs: int, settings: Settings | None = None, seed: int = 0):
        self.n = n_envs
        self.s = settings or Settings()
        self.rng = np.random.default_rng(seed)
        C, R = self.s.cols, self.s.rows
        self.C, self.R = C, R
        K = self.s.max_coins
        self.K = K

        self.arena = np.zeros((self.n, C, R), dtype=np.int8)
        self.coin_xy = np.full((self.n, K, 2), -1, dtype=np.int16)
        self.coin_state = np.zeros((self.n, K), dtype=np.int8)
        self.agent_xy = np.zeros((self.n, 4, 2), dtype=np.int16)
        self.agent_alive = np.zeros((self.n, 4), dtype=bool)
        self.agent_score = np.zeros((self.n, 4), dtype=np.int32)
        self.bomb_active = np.zeros((self.n, 4), dtype=bool)
        self.bomb_xy = np.zeros((self.n, 4, 2), dtype=np.int16)
        self.bomb_timer = np.zeros((self.n, 4), dtype=np.int8)
        self.expl_active = np.zeros((self.n, 4), dtype=bool)
        self.expl_dangerous = np.zeros((self.n, 4), dtype=bool)
        self.expl_timer = np.zeros((self.n, 4), dtype=np.int8)
        self.expl_mask = np.zeros((self.n, 4, C, R), dtype=bool)
        self.step_no = np.zeros(self.n, dtype=np.int32)
        self.running = np.zeros(self.n, dtype=bool)

        self.reset_all()

    def reset_all(self):
        idx = np.arange(self.n)
        self.reset(idx)

    def reset(self, env_idx: np.ndarray):
        """
        env_idx: (M,) int array of which environments to reset (0 <= idx < n_envs)
        """
        s = self.s
        C, R = self.C, self.R
        starts = _start_positions(s)
        info = SCENARIOS[s.scenario]

        xs, ys = np.meshgrid(np.arange(C), np.arange(R), indexing="ij")
        checker = ((xs + 1) * (ys + 1)) % 2 == 1
        all_pos = np.stack((xs, ys), -1)
        border = np.zeros((C, R), dtype=bool)
        border[:1, :] = border[-1:, :] = border[:, :1] = border[:, -1:] = True
        fixed_walls = checker | border
        clear_positions = set()
        for (x, y) in starts:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                clear_positions.add((xx, yy))

        for n in env_idx:
            arena = np.zeros((C, R), dtype=np.int8)
            arena[self.rng.random((C, R)) < info["CRATE_DENSITY"]] = CRATE
            arena[fixed_walls] = WALL
            for (xx, yy) in clear_positions:
                if arena[xx, yy] == CRATE:
                    arena[xx, yy] = FREE

            crate_pos = self.rng.permutation(all_pos[arena == CRATE])
            free_pos = self.rng.permutation(all_pos[arena == FREE])
            coin_pos = np.concatenate([crate_pos, free_pos], axis=0)[: s.max_coins]

            self.arena[n] = arena
            self.coin_xy[n] = -1
            self.coin_state[n] = 0
            for k, (x, y) in enumerate(coin_pos):
                self.coin_xy[n, k] = (x, y)
                self.coin_state[n, k] = 1 if arena[x, y] == FREE else 0

            order = self.rng.permutation(4)
            for slot, start_i in enumerate(order):
                x, y = starts[start_i]
                self.agent_xy[n, slot] = (x, y)
            self.agent_alive[n] = True
            self.agent_score[n] = 0
            self.bomb_active[n] = False
            self.bomb_timer[n] = 0
            self.expl_active[n] = False
            self.expl_dangerous[n] = False
            self.expl_timer[n] = 0
            self.expl_mask[n] = False
            self.step_no[n] = 0
            self.running[n] = True

    def step(self, actions: np.ndarray):
        """
        actions: (N, 4) int array of action ids (one per agent slot).
        Dead / inactive agents' actions are ignored.

        Returns:
          events   (N, 4) uint16  bit-packed EV_* flags for this step
          done     (N,)   bool    whether the round ended this step
        """
        assert actions.shape == (self.n, 4)
        N, C, R = self.n, self.C, self.R
        events = np.zeros((N, 4), dtype=np.uint16)

        alive = self.agent_alive
        self.step_no += self.running.astype(np.int32)

        perm = np.argsort(self.rng.random((N, 4)), axis=1)
        occ_bomb = self.bomb_active.copy()

        for t in range(4):
            slot = perm[:, t]
            rows = np.arange(N)
            act = actions[rows, slot]
            active = alive[rows, slot] & self.running
            if not active.any():
                continue

            x = self.agent_xy[rows, slot, 0].astype(np.int32)
            y = self.agent_xy[rows, slot, 1].astype(np.int32)

            for a, dx, dy in ((UP, 0, -1), (DOWN, 0, 1), (LEFT, -1, 0), (RIGHT, 1, 0)):
                sel = active & (act == a)
                if not sel.any():
                    continue
                nx, ny = x + dx, y + dy
                free = self.arena[rows, nx, ny] == FREE
                occ_by_bomb = occ_bomb[rows] & (self.bomb_xy[rows, :, 0] == nx[:, None]) & \
                              (self.bomb_xy[rows, :, 1] == ny[:, None])
                occ_by_bomb = occ_by_bomb.any(axis=1)
                occ_by_agent = alive & (self.agent_xy[..., 0] == nx[:, None]) & \
                               (self.agent_xy[..., 1] == ny[:, None])
                occ_by_agent = occ_by_agent.any(axis=1)
                can_move = sel & free & ~occ_by_bomb & ~occ_by_agent
                self.agent_xy[rows[can_move], slot[can_move], 0] = nx[can_move]
                self.agent_xy[rows[can_move], slot[can_move], 1] = ny[can_move]
                events[rows[can_move], slot[can_move]] |= EV_MOVED
                invalid = sel & ~can_move
                events[rows[invalid], slot[invalid]] |= EV_INVALID

            sel = active & (act == WAIT)
            events[rows[sel], slot[sel]] |= EV_WAITED

            sel = active & (act == BOMB) & (~self.bomb_active[rows, slot])
            if sel.any():
                self.bomb_active[rows[sel], slot[sel]] = True
                self.bomb_xy[rows[sel], slot[sel], 0] = x[sel]
                self.bomb_xy[rows[sel], slot[sel], 1] = y[sel]
                self.bomb_timer[rows[sel], slot[sel]] = self.s.bomb_timer
                events[rows[sel], slot[sel]] |= EV_BOMB_DROPPED
                occ_bomb[rows[sel], slot[sel]] = True
            invalid_bomb = active & (act == BOMB) & self.bomb_active[rows, slot]
            events[rows[invalid_bomb], slot[invalid_bomb]] |= EV_INVALID

        for k in range(self.K):
            collectable = self.coin_state[:, k] == 1
            if not collectable.any():
                continue
            cx = self.coin_xy[:, k, 0]
            cy = self.coin_xy[:, k, 1]
            hit = alive & (self.agent_xy[..., 0] == cx[:, None]) & (self.agent_xy[..., 1] == cy[:, None])
            hit &= collectable[:, None]
            got = hit.any(axis=1)
            if got.any():
                self.coin_state[got, k] = 2
                first = np.argmax(hit[got], axis=1)
                envs = np.nonzero(got)[0]
                self.agent_score[envs, first] += self.s.reward_coin
                events[envs, first] |= EV_COIN_COLLECTED

        dec = self.expl_active & self.running[:, None]
        self.expl_timer[dec] -= 1
        flip = dec & (self.expl_timer <= 0) & self.expl_dangerous

        self.expl_dangerous[flip] = False
        self.expl_timer[flip] = self.s.smoke_timer
        expire = dec & (self.expl_timer <= 0) & ~self.expl_dangerous
        self.expl_active[expire] = False
        self.expl_mask[expire] = False

        due = self.bomb_active & self.running[:, None] & (self.bomb_timer <= 0)
        if due.any():
            env_ids, slots = np.nonzero(due)
            for n, slot in zip(env_ids, slots):
                bx, by = int(self.bomb_xy[n, slot, 0]), int(self.bomb_xy[n, slot, 1])
                mask = np.zeros((C, R), dtype=bool)
                mask[bx, by] = True
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = bx, by
                    for _ in range(self.s.bomb_power):
                        x, y = x + ddx, y + ddy
                        if self.arena[n, x, y] == WALL:
                            break
                        mask[x, y] = True

                crate_hit = mask & (self.arena[n] == CRATE)
                if crate_hit.any():
                    self.arena[n][crate_hit] = FREE
                    events[n, slot] |= EV_CRATE_DESTROYED
                    for k in range(self.K):
                        if self.coin_state[n, k] == 0:
                            cx, cy = self.coin_xy[n, k]
                            if cx >= 0 and crate_hit[cx, cy]:
                                self.coin_state[n, k] = 1
                                events[n, slot] |= EV_COIN_FOUND
                events[n, slot] |= EV_BOMB_EXPLODED
                self.bomb_active[n, slot] = False
                self.expl_active[n, slot] = True
                self.expl_dangerous[n, slot] = True
                self.expl_timer[n, slot] = self.s.explosion_timer
                self.expl_mask[n, slot] = mask
        not_due = self.bomb_active & self.running[:, None] & ~due
        self.bomb_timer[not_due] -= 1

        got_hit = np.zeros((N, 4), dtype=bool)
        for owner_slot in range(4):
            dangerous = self.expl_dangerous[:, owner_slot] & self.running
            if not dangerous.any():
                continue
            mask = self.expl_mask[:, owner_slot, :, :]
            for i in range(4):
                a_alive = alive[:, i] & dangerous
                if not a_alive.any():
                    continue
                ax = self.agent_xy[:, i, 0]
                ay = self.agent_xy[:, i, 1]
                hit = a_alive & mask[np.arange(N), ax, ay]
                if not hit.any():
                    continue
                got_hit[hit, i] = True
                self_kill = hit & (owner_slot == i)
                opp_kill = hit & ~self_kill
                events[self_kill, i] |= EV_KILLED_SELF
                if opp_kill.any():
                    self.agent_score[opp_kill, owner_slot] += self.s.reward_kill
                    events[opp_kill, owner_slot] |= EV_KILLED_OPPONENT

        if got_hit.any():
            self.agent_alive[got_hit] = False
            events[got_hit.any(axis=1)] |= 0
            envs_with_kill = np.nonzero(got_hit.any(axis=1))[0]
            for n in envs_with_kill:
                dead_here = np.nonzero(got_hit[n])[0]
                for d in dead_here:
                    events[n, d] |= EV_GOT_KILLED
                still_alive = np.nonzero(self.agent_alive[n])[0]
                for a in still_alive:
                    events[n, a] |= EV_OPPONENT_ELIMINATED

        n_alive = self.agent_alive.sum(axis=1)
        crates_left = (self.arena == CRATE).any(axis=(1, 2))
        coins_left = (self.coin_state == 1).any(axis=1)
        bombs_or_expl = self.bomb_active.any(axis=1) | self.expl_active.any(axis=1)
        stalemate = (n_alive == 1) & ~crates_left & ~coins_left & ~bombs_or_expl
        timeout = self.step_no >= self.s.max_steps
        done = self.running & ((n_alive == 0) | stalemate | timeout)

        if done.any():
            survivors = self.agent_alive & done[:, None]
            events[survivors] |= EV_SURVIVED_ROUND
            self.running[done] = False

        return events, done

    def auto_reset_done(self, done: np.ndarray):
        """Call after step() to immediately restart any finished envs
        (standard for high-throughput vectorized RL training)."""
        idx = np.nonzero(done)[0]
        if len(idx):
            self.reset(idx)

    def dangerous_mask(self) -> np.ndarray:
        """(N, C, R) bool -- tiles covered by a currently-lethal explosion."""
        out = np.zeros((self.n, self.C, self.R), dtype=bool)
        for slot in range(4):
            dangerous = self.expl_dangerous[:, slot]
            if not dangerous.any():
                continue
            out |= self.expl_mask[:, slot, :, :] & dangerous[:, None, None]
        return out

    def danger_map(self) -> np.ndarray:
        """(N, C, R) countdown grid matching the original's explosion_map
        (max over dangerous explosions covering each tile, value = timer-1)."""
        out = np.zeros((self.n, self.C, self.R), dtype=np.int8)
        for slot in range(4):
            dangerous = self.expl_dangerous[:, slot]
            if not dangerous.any():
                continue
            val = np.maximum(self.expl_timer[:, slot] - 1, 0)
            contrib = self.expl_mask[:, slot, :, :] & dangerous[:, None, None]
            out = np.where(contrib, np.maximum(out, val[:, None, None]), out)
        return out
