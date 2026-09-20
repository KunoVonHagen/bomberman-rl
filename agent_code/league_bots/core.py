from __future__ import annotations

import random
from collections import deque

import numpy as np

BOMB_POWER = 3
BOMB_TIMER = 4
HORIZON = 7
INF = 10 ** 6

MOVES = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
STEPS = [("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0)), ("WAIT", (0, 0))]


def blast_tiles(field, pos):
    """Tiles hit by a bomb at ``pos``: only stone walls stop the blast."""
    x, y = pos
    width, height = field.shape
    tiles = [(x, y)]
    for dx, dy in MOVES.values():
        for i in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < width and 0 <= ny < height) or field[nx, ny] == -1:
                break
            tiles.append((nx, ny))
    return tiles


def effective_countdowns(field, bombs):
    """Countdown of every bomb after chain reactions: a bomb inside another's blast goes off no later than that one."""
    bombs = list(bombs)
    blasts = [set(blast_tiles(field, pos)) for pos, _ in bombs]
    effective = [countdown for _, countdown in bombs]
    changed = True
    while changed:
        changed = False
        for i, (pos_i, _) in enumerate(bombs):
            for j in range(len(bombs)):
                if i != j and pos_i in blasts[j] and effective[j] < effective[i]:
                    effective[i] = effective[j]
                    changed = True
    return effective


def effective_countdowns(field, bombs):
    """Countdown of every bomb after chain reactions: a bomb inside another's blast goes off no later than that one."""
    bombs = list(bombs)
    blasts = [set(blast_tiles(field, pos)) for pos, _ in bombs]
    effective = [countdown for _, countdown in bombs]
    changed = True
    while changed:
        changed = False
        for i, (pos_i, _) in enumerate(bombs):
            for j in range(len(bombs)):
                if i != j and pos_i in blasts[j] and effective[j] < effective[i]:
                    effective[i] = effective[j]
                    changed = True
    return effective


def build_danger(field, bombs, explosion_map, extra_bombs=()):
    """danger[k, x, y] is True if standing on (x, y) after k of our actions is (conservatively) deadly."""
    danger = np.zeros((HORIZON + 1,) + field.shape, dtype=bool)
    for k in range(1, HORIZON + 1):
        danger[k] |= explosion_map >= k
    all_bombs = list(bombs) + list(extra_bombs)
    for (pos, _), countdown in zip(all_bombs, effective_countdowns(field, all_bombs)):
        for k in (countdown + 1, countdown + 2):
            if 1 <= k <= HORIZON:
                for tile in blast_tiles(field, pos):
                    danger[k][tile] = True
    return danger


def opponent_reach(shape, others):
    """reach[k, x, y] is True if some opponent could stand on (x, y) after k moves (manhattan, ignoring walls)."""
    reach = np.zeros((HORIZON + 1,) + tuple(shape), dtype=bool)
    xs, ys = np.indices(shape)
    for ox, oy in others:
        distance = np.abs(xs - ox) + np.abs(ys - oy)
        for k in range(1, HORIZON + 1):
            reach[k] |= distance <= k
    return reach


def can_survive(start, k0, passable, danger, reach=None):
    width, height = passable.shape
    frontier = {start}
    for k in range(k0 + 1, HORIZON + 1):
        nxt = set()
        for x, y in frontier:
            for _, (dx, dy) in STEPS:
                nx, ny = x + dx, y + dy
                if dx or dy:
                    if not (0 <= nx < width and 0 <= ny < height and passable[nx, ny]):
                        continue
                    if reach is not None and reach[k, nx, ny]:
                        continue
                if not danger[k, nx, ny]:
                    nxt.add((nx, ny))
        if not nxt:
            return False
        frontier = nxt
    return True


class Context:
    def __init__(self, state):
        self.field = np.asarray(state["field"])
        _, _, bombs_left, pos = state["self"]
        self.pos = (int(pos[0]), int(pos[1]))
        self.bombs_left = bool(bombs_left)
        self.others = [(int(o[3][0]), int(o[3][1])) for o in state["others"]]
        self.coins = [(int(c[0]), int(c[1])) for c in state["coins"]]
        self.bombs = [((int(b[0][0]), int(b[0][1])), int(b[1])) for b in state["bombs"]]
        self.explosion = np.asarray(state["explosion_map"])
        self.bomb_tiles = {b[0] for b in self.bombs}

        self.passable = self.field == 0
        for tile in self.bomb_tiles | set(self.others):
            self.passable[tile] = False
        self.danger = build_danger(self.field, self.bombs, self.explosion)
        self.reach = opponent_reach(self.field.shape, self.others)
        self.reach = opponent_reach(self.field.shape, self.others)

    def _next_pos(self, delta):
        return (self.pos[0] + delta[0], self.pos[1] + delta[1])

    def safe_actions(self):
        actions = []
        for name, delta in STEPS:
            nxt = self._next_pos(delta)
            if name != "WAIT" and not self.passable[nxt]:
                continue
            if self.danger[1][nxt] or not can_survive(nxt, 1, self.passable, self.danger, self.reach):
                continue
            actions.append((name, nxt))
        return actions

    def fallback_action(self):
        """No provably safe action: take any step that at least is not deadly right now."""
        options = [name for name, delta in STEPS
                   if (name == "WAIT" or self.passable[self._next_pos(delta)]) and not self.danger[1][self._next_pos(delta)]]
        return random.choice(options) if options else "WAIT"

    def can_bomb(self):
        return self.bombs_left and self.pos not in self.bomb_tiles

    def bomb_is_safe(self):
        danger = build_danger(self.field, self.bombs, self.explosion, extra_bombs=[(self.pos, BOMB_TIMER)])
        passable = self.passable.copy()
        passable[self.pos] = False
        return can_survive(self.pos, 1, passable, danger, self.reach)

    def opponent_in_blast(self):
        return any(tile in self.others for tile in blast_tiles(self.field, self.pos))

    def crates_in_blast(self):
        return sum(1 for tile in blast_tiles(self.field, self.pos) if self.field[tile] == 1)

    def crate_tiles(self):
        xs, ys = np.nonzero(self.field == 1)
        return list(zip(xs.tolist(), ys.tolist()))

    def distance_map(self, sources):
        """Multi-source BFS over walkable tiles; sources themselves need not be walkable."""
        dist = np.full(self.field.shape, INF, dtype=np.int64)
        queue = deque()
        for tile in sources:
            dist[tile] = 0
            queue.append(tile)
        width, height = self.field.shape
        while queue:
            x, y = queue.popleft()
            for dx, dy in MOVES.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and self.passable[nx, ny] and dist[nx, ny] == INF:
                    dist[nx, ny] = dist[x, y] + 1
                    queue.append((nx, ny))
        return dist

    def step_towards(self, safe, dist):
        """Among safe actions pick the one whose next tile is closest to the objective (ties: move, then random)."""
        best, best_key = None, None
        for name, nxt in safe:
            key = (dist[nxt], name == "WAIT", random.random())
            if best_key is None or key < best_key:
                best, best_key = name, key
        return best if best_key is not None and best_key[0] < INF else None

    def wander(self, safe):
        moves = [name for name, _ in safe if name != "WAIT"]
        return random.choice(moves) if moves else "WAIT"
