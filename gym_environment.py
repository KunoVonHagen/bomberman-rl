from collections import namedtuple, deque
from typing import List, Tuple, Callable, Optional, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful, correctness-preserving fallback
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        # No-op decorator so the exact same Python implementation below still
        # runs correctly (just without JIT compilation) if numba isn't installed.
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap

import settings as s
import events as e

from agent_code.my_agent.features import (
    get_features,
    FEATURES_DIM,
    EVENT_REWARDS,
    FEATURE_REWARDS,
    FEATURE_DIFF_REWARDS,
    SIMPLE_EVENT_REWARDS,
)

# Kept for drop-in compatibility with callers that construct WorldArgs(...).
WorldArgs = namedtuple(
    "WorldArgs",
    ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay",
     "make_video", "continue_without_training", "log_dir", "save_stats",
     "match_name", "seed", "silence_errors", "scenario"],
)


class _NullLogger:
    """No-op logger, API-compatible with `self.logger`."""
    __slots__ = ()

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


_NULL_LOGGER = _NullLogger()


class AgentHandle:
    """Minimal per-agent record: only what game logic, game_state, and
    reward computation actually use."""

    __slots__ = ("name", "train", "logger", "x", "y", "score", "total_score",
                 "bombs_left", "dead", "events")

    def __init__(self, name: str):
        self.name = name
        self.train = True
        self.logger = _NULL_LOGGER

        self.x = 0
        self.y = 0
        self.score = 0
        self.total_score = 0
        self.bombs_left = True
        self.dead = False
        self.events: List[str] = []

    def get_state(self):
        return self.name, self.score, self.bombs_left, (self.x, self.y)

    def add_event(self, event):
        self.events.append(event)

    def update_score(self, delta):
        self.score += delta
        self.total_score += delta

    def reset_game_events(self):
        self.events = []


ACTIONS = [
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
]

ACTION_INDICES = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
    "WAIT": 4,
    "BOMB": 5,
    None: 4,
}

_EXPLOSION_STAGE1_TICKS = 2

WALL_LAYER, CRATE_LAYER, COIN_LAYER, SELF_LAYER, SELF_BLAST_LAYER, OPPONENT_LAYER, OPPONENT_DANGER_LAYER, BOMBS_LEFT_LAYER = range(8)
_BASE_LAYERS = 8 + 2 * s.BOMB_TIMER + s.EXPLOSION_TIMER

DANGER_MAP_LAYERS = [_BASE_LAYERS + t for t in range(s.BOMB_TIMER + s.EXPLOSION_TIMER)]
OCCUPIED_MAP_LAYERS = [DANGER_MAP_LAYERS[-1] + 1 + t for t in range(s.BOMB_TIMER + s.EXPLOSION_TIMER + 1)]
SELF_DISTANCE_LAYER = OCCUPIED_MAP_LAYERS[-1] + 1
OPPONENTS_LEAST_DISTANCE_LAYER = SELF_DISTANCE_LAYER + 1
CRATE_POTENTIAL_LAYER = OPPONENTS_LEAST_DISTANCE_LAYER + 1
DANGER_ONSET_LAYER = CRATE_POTENTIAL_LAYER + 1
DANGER_CLEAR_LAYER = DANGER_ONSET_LAYER + 1
MOBILITY_LAYER = DANGER_CLEAR_LAYER + 1
CRATE_DISTANCE_LAYER = MOBILITY_LAYER + 1
COIN_DISTANCE_LAYER = CRATE_DISTANCE_LAYER + 1

NUM_LAYERS = COIN_DISTANCE_LAYER + 1

@njit(cache=True)
def _time_aware_bfs_kernel(starts, occ, W, H, T):
    dist = np.full((W, H), -1.0, dtype=np.float32)
    visited = np.zeros((W, H, T + 1), dtype=np.bool_)

    max_nodes = W * H * (T + 1)
    qx = np.empty(max_nodes, dtype=np.int32)
    qy = np.empty(max_nodes, dtype=np.int32)
    qt = np.empty(max_nodes, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(starts.shape[0]):
        x = starts[i, 0]
        y = starts[i, 1]
        if occ[0, x, y] > 0:
            continue
        if not visited[x, y, 0]:
            visited[x, y, 0] = True
            dist[x, y] = 0.0
            qx[tail] = x
            qy[tail] = y
            qt[tail] = 0
            tail += 1

    while head < tail:
        x = qx[head]
        y = qy[head]
        t = qt[head]
        head += 1

        next_t = t + 1
        if next_t > T:
            next_t = T
        occ_next = occ[next_t]

        for k in range(5):
            if k == 0:
                nx, ny = x - 1, y
            elif k == 1:
                nx, ny = x + 1, y
            elif k == 2:
                nx, ny = x, y - 1
            elif k == 3:
                nx, ny = x, y + 1
            else:
                nx, ny = x, y

            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if occ_next[nx, ny] > 0:
                continue
            if visited[nx, ny, next_t]:
                continue

            visited[nx, ny, next_t] = True
            if dist[nx, ny] == -1.0:
                dist[nx, ny] = next_t
            qx[tail] = nx
            qy[tail] = ny
            qt[tail] = next_t
            tail += 1

    return dist


@njit(cache=True)
def _multi_source_bfs_kernel(targets, occ, W, H):
    dist = np.full((W, H), -1.0, dtype=np.float32)
    qx = np.empty(W * H, dtype=np.int32)
    qy = np.empty(W * H, dtype=np.int32)
    head = 0
    tail = 0

    for x in range(W):
        for y in range(H):
            if targets[x, y]:
                dist[x, y] = 0.0
                qx[tail] = x
                qy[tail] = y
                tail += 1

    while head < tail:
        x = qx[head]
        y = qy[head]
        head += 1

        for k in range(4):
            if k == 0:
                nx, ny = x - 1, y
            elif k == 1:
                nx, ny = x + 1, y
            elif k == 2:
                nx, ny = x, y - 1
            else:
                nx, ny = x, y + 1

            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if occ[nx, ny] or dist[nx, ny] != -1.0:
                continue
            dist[nx, ny] = dist[x, y] + 1.0
            qx[tail] = nx
            qy[tail] = ny
            tail += 1

    return dist


class BombermanGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        args,
        opponents: List[Tuple[Callable[["AgentHandle"], None], Callable[["AgentHandle", dict], "Optional[str]"]]],
        reward_fn=None,
        render_mode=None,
    ):
        super().__init__()
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.render_mode = render_mode

        self.agent = AgentHandle("RLAgent")
        self.opponent_handles: List[AgentHandle] = []
        self.opponent_act_fns: List[Callable] = []
        for i, (setup_fn, act_fn) in enumerate(opponents):
            self.opponent_handles.append(AgentHandle(f"OpponentAgent{i}"))
            self.opponent_act_fns.append(act_fn)
        self.all_agents: List[AgentHandle] = [self.agent] + self.opponent_handles
        self.n_agents = len(self.all_agents)
        self.active_agents: List[AgentHandle] = list(self.all_agents)

        self.reward_fn = reward_fn or self.shaped_reward

        self.width, self.height = s.COLS, s.ROWS
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.n_observation_layers = NUM_LAYERS
        self._BT = s.BOMB_TIMER
        self._ET = s.EXPLOSION_TIMER

        self.grid_tensor = np.zeros((self.n_observation_layers, self.width, self.height), dtype=np.float32)
        self._centered_tensor = np.zeros_like(self.grid_tensor)
        self.features = np.zeros(FEATURES_DIM, dtype=np.float32)

        """
        self.observation_space = spaces.Dict({
            "grid_tensor": spaces.Box(
                low=-1,
                high=self.width * self.height,
                shape=(self.n_observation_layers, self.width, self.height),
                dtype=np.float32,
            ),
            "features": spaces.Box(
                low=-1,
                high=1,
                shape=(FEATURES_DIM,),
                dtype=np.float32,
            ),
        })
        """
        self.observation_space = spaces.Box(
            low=-1,
            high=self.width * self.height,
            shape=(self.n_observation_layers, self.width, self.height),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        wall_mask = self._build_wall_mask()
        self._wall_layer = np.where(wall_mask == -1, 1.0, 0.0).astype(np.float32)
        self._wall_bool = self._wall_layer.astype(bool)
        self._empty_map = np.zeros((self.width, self.height), dtype=np.float32)
        self._occ_scratch = np.zeros((self.width, self.height), dtype=bool)
        self.PRECOMPUTED_BLAST_COORDS: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.PRECOMPUTED_BLAST_MAP: Dict[Tuple[int, int], np.ndarray] = {}
        for x, y in np.argwhere(wall_mask != -1):
            x, y = int(x), int(y)
            coords = self._compute_blast_coords(x, y, wall_mask, s.BOMB_POWER)
            self.PRECOMPUTED_BLAST_COORDS[(x, y)] = coords
            bmap = np.zeros((self.width, self.height), dtype=np.float32)
            xs_, ys_ = zip(*coords)
            bmap[list(xs_), list(ys_)] = 1.0
            self.PRECOMPUTED_BLAST_MAP[(x, y)] = bmap

        self._blast_tensor = np.zeros((self.width, self.height, self.width, self.height), dtype=np.float32)
        for (bx, by), bmap in self.PRECOMPUTED_BLAST_MAP.items():
            self._blast_tensor[bx, by] = bmap

        self.round = 0
        self.step_count = 0
        self.arena = np.zeros((self.width, self.height), dtype=np.int8)
        self.coins_xy = np.zeros((0, 2), dtype=np.int64)
        self.coins_collectable = np.zeros((0,), dtype=bool)
        self.bombs: List[dict] = []
        self.explosions: List[dict] = []

        self.previous_visited_count = 1
        self.visited = np.zeros((self.width, self.height), dtype=bool)
        self.previous_features = None

        self.agent_actions = {}

        self.new_round()

    def _build_wall_mask(self) -> np.ndarray:
        """Wall portion of BombeRLeWorld.build_arena (RNG-independent)."""
        WALL = -1
        arena = np.zeros((s.COLS, s.ROWS), dtype=np.int8)
        arena[:1, :] = WALL
        arena[-1:, :] = WALL
        arena[:, :1] = WALL
        arena[:, -1:] = WALL
        xs, ys = np.meshgrid(np.arange(s.COLS), np.arange(s.ROWS), indexing="ij")
        arena[((xs + 1) * (ys + 1)) % 2 == 1] = WALL
        return arena

    @staticmethod
    def _compute_blast_coords(x, y, wall_mask, power) -> List[Tuple[int, int]]:
        """Reproduces items.Bomb.get_blast_coords without a Bomb object."""
        coords = [(x, y)]
        for i in range(1, power + 1):
            if wall_mask[x + i, y] == -1:
                break
            coords.append((x + i, y))
        for i in range(1, power + 1):
            if wall_mask[x - i, y] == -1:
                break
            coords.append((x - i, y))
        for i in range(1, power + 1):
            if wall_mask[x, y + i] == -1:
                break
            coords.append((x, y + i))
        for i in range(1, power + 1):
            if wall_mask[x, y - i] == -1:
                break
            coords.append((x, y - i))
        return coords

    def _generate_round_layout(self):
        """Reproduces BombeRLeWorld.build_arena (crates, coins, start positions)."""
        WALL, FREE, CRATE = -1, 0, 1
        arena = self._build_wall_mask().copy()

        scenario_info = s.SCENARIOS[self.args.scenario]
        crate_mask = self.rng.random((s.COLS, s.ROWS)) < scenario_info["CRATE_DENSITY"]
        arena[(arena != WALL) & crate_mask] = CRATE

        start_positions = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if arena[xx, yy] == CRATE:
                    arena[xx, yy] = FREE

        all_positions = np.stack(np.meshgrid(np.arange(s.COLS), np.arange(s.ROWS), indexing="ij"), -1)
        crate_positions = self.rng.permutation(all_positions[arena == CRATE])
        free_positions = self.rng.permutation(all_positions[arena == FREE])
        coin_positions = np.concatenate([crate_positions, free_positions], axis=0)[:scenario_info["COIN_COUNT"]]

        coins_xy = [(int(x), int(y)) for x, y in coin_positions]
        coins_collectable = [arena[x, y] == FREE for (x, y) in coins_xy]

        perm = self.rng.permutation(len(start_positions))
        positions = [start_positions[i] for i in perm[:self.n_agents]]

        return arena, coins_xy, coins_collectable, positions

    def new_round(self):
        self.round += 1
        self.step_count = 0
        self.bombs = []
        self.explosions = []

        arena, coins_xy, coins_collectable, positions = self._generate_round_layout()
        self.arena = arena
        self.coins_xy = np.array(coins_xy, dtype=np.int64) if coins_xy else np.zeros((0, 2), dtype=np.int64)
        self.coins_collectable = np.array(coins_collectable, dtype=bool) if coins_collectable else np.zeros((0,), dtype=bool)

        for handle, (x, y) in zip(self.all_agents, positions):
            handle.x, handle.y = int(x), int(y)
            handle.dead = False
            handle.score = 0
            handle.total_score = handle.total_score
            handle.bombs_left = True
            handle.events = []

        self.active_agents = list(self.all_agents)

        self._rebuild_full_grid_tensor()

        self.previous_visited_count = 1
        self.visited.fill(False)
        self.previous_features = None
        self.agent_actions = {}

    def _rebuild_full_grid_tensor(self):
        """Full rebuild -- only once per round, cost amortized over the round."""
        gt = self.grid_tensor
        gt.fill(0)
        gt[WALL_LAYER] = self._wall_layer
        gt[CRATE_LAYER] = np.where(self.arena == 1, 1.0, 0.0)
        collectable_idx = np.nonzero(self.coins_collectable)[0]
        for ci in collectable_idx:
            x, y = self.coins_xy[ci]
            gt[COIN_LAYER, x, y] = 1.0
        gt[SELF_LAYER, self.agent.x, self.agent.y] = 1.0
        self._refresh_dynamic_layers()
        self._init_crate_potential()
        self._refresh_forecast_layers()

    def _refresh_dynamic_layers(self):
        """Recomputes channels that are inherently collective (all enemies)
        or shift every tick (bomb/explosion timer channels). Walls/crates/
        coins/self position are maintained incrementally elsewhere."""
        gt = self.grid_tensor
        BT = self._BT

        gt[4:_BASE_LAYERS].fill(0)

        ax, ay = self.agent.x, self.agent.y
        gt[SELF_BLAST_LAYER] = self.PRECOMPUTED_BLAST_MAP.get((ax, ay), self._empty_map)

        for h in self.opponent_handles:
            ex_, ey_ = h.x, h.y
            gt[OPPONENT_LAYER, ex_, ey_] = 1.0
            gt[OPPONENT_DANGER_LAYER] += self.PRECOMPUTED_BLAST_MAP.get((ex_, ey_), self._empty_map)
            gt[BOMBS_LEFT_LAYER, ex_, ey_] = 1.0 if h.bombs_left else 0.0

        gt[OPPONENT_DANGER_LAYER] = np.where(gt[OPPONENT_DANGER_LAYER] > 0, 1.0, 0.0)
        gt[BOMBS_LEFT_LAYER, ax, ay] = 1.0 if self.agent.bombs_left else 0.0

        for b in self.bombs:
            pos_ch = 8 + b["timer"]
            danger_ch = 8 + BT + b["timer"]
            gt[pos_ch, b["x"], b["y"]] = 1.0
            gt[danger_ch] = self.PRECOMPUTED_BLAST_MAP.get((b["x"], b["y"]), self._empty_map)

        for ex in self.explosions:
            if ex["stage"] == 0:
                ch = 7 + 2 * BT + ex["timer"]
                for (x, y) in ex["coords"]:
                    gt[ch, x, y] = 1.0

    def _init_crate_potential(self):
        """Full recompute -- only once per round. Kept current afterwards by
        incremental subtraction in `_update_bombs`."""
        self.grid_tensor[CRATE_POTENTIAL_LAYER] = np.einsum(
            "xyij,ij->xy", self._blast_tensor, self.grid_tensor[CRATE_LAYER]
        )

    def _compute_danger_forecast(self):
        """danger[t] = cells dangerous t steps from now, assuming no new
        bombs. Iterates only over currently active bombs/explosions."""
        gt = self.grid_tensor
        T = self._BT + self._ET
        active_explosions = [ex for ex in self.explosions if ex["stage"] == 0]

        if not self.bombs and not active_explosions:
            gt[DANGER_MAP_LAYERS[0]:DANGER_MAP_LAYERS[-1] + 1] = 0.0
            return

        for t in range(T):
            danger = gt[DANGER_MAP_LAYERS[t]]
            danger.fill(0.0)
            for ex in active_explosions:
                if ex["timer"] - t > 0:
                    for (x, y) in ex["coords"]:
                        danger[x, y] = 1.0
            for b in self.bombs:
                if b["timer"] <= t < b["timer"] + self._ET:
                    np.maximum(danger, self.PRECOMPUTED_BLAST_MAP.get((b["x"], b["y"]), self._empty_map), out=danger)

    def _compute_occupancy_forecast(self):
        """occ[t] = walls | surviving crates | pending bombs | danger, at t
        steps from now. Needs danger[t] computed first."""
        gt = self.grid_tensor
        T = self._BT + self._ET

        if not self.bombs and not self.explosions:
            static = (self._wall_bool | gt[CRATE_LAYER].astype(bool)).astype(np.float32)
            for layer in OCCUPIED_MAP_LAYERS:
                gt[layer] = static
            return

        remaining_crates = gt[CRATE_LAYER].astype(bool)
        occ = self._occ_scratch
        for t in range(T):
            danger_t = gt[DANGER_MAP_LAYERS[t]].astype(bool)
            remaining_crates &= ~danger_t

            np.logical_or(self._wall_bool, remaining_crates, out=occ)
            for b in self.bombs:
                if b["timer"] >= t:
                    occ[b["x"], b["y"]] = True
            occ |= danger_t

            gt[OCCUPIED_MAP_LAYERS[t]] = occ

        gt[OCCUPIED_MAP_LAYERS[-1]] = (self._wall_bool | remaining_crates).astype(np.float32)

    def _time_aware_bfs(self, starts, out_layer):
        """Earliest arrival time at every cell, respecting the occupancy
        forecast (four moves + wait). Delegates to a JIT-compiled kernel that
        implements the identical FIFO/BFS algorithm."""
        T = len(OCCUPIED_MAP_LAYERS) - 1
        W, H = self.width, self.height
        gt = self.grid_tensor

        if starts:
            starts_arr = np.asarray(starts, dtype=np.int64)
        else:
            starts_arr = np.zeros((0, 2), dtype=np.int64)

        occ = gt[OCCUPIED_MAP_LAYERS[0]:OCCUPIED_MAP_LAYERS[-1] + 1]
        gt[out_layer] = _time_aware_bfs_kernel(starts_arr, occ, W, H, T)

    def _compute_danger_summary(self):
        """Onset/clear timestep of the danger forecast, per cell (-1 = never)."""
        gt = self.grid_tensor
        stack = gt[DANGER_MAP_LAYERS[0]:DANGER_MAP_LAYERS[-1] + 1]
        ever = stack.any(axis=0)

        onset = np.argmax(stack, axis=0)
        gt[DANGER_ONSET_LAYER] = np.where(ever, onset, -1)

        T = stack.shape[0]
        last_from_end = np.argmax(stack[::-1], axis=0)
        gt[DANGER_CLEAR_LAYER] = np.where(ever, T - last_from_end, -1)

    def _compute_mobility(self):
        """Free 4-neighbor count under final occupancy."""
        gt = self.grid_tensor
        free = 1.0 - gt[OCCUPIED_MAP_LAYERS[-1]]
        m = np.zeros_like(free)
        m[1:, :] += free[:-1, :]
        m[:-1, :] += free[1:, :]
        m[:, 1:] += free[:, :-1]
        m[:, :-1] += free[:, 1:]
        gt[MOBILITY_LAYER] = m

    def _multi_source_bfs(self, targets: np.ndarray, occ: np.ndarray) -> np.ndarray:
        """Static (non-time-aware) BFS distance from every cell to the
        nearest True cell in `targets`, under fixed occupancy `occ`.
        Delegates to a JIT-compiled kernel implementing the identical
        FIFO/BFS algorithm (distances are order-independent, so this is a
        drop-in replacement)."""
        W, H = self.width, self.height
        return _multi_source_bfs_kernel(
            np.ascontiguousarray(targets, dtype=np.bool_),
            np.ascontiguousarray(occ, dtype=np.bool_),
            W, H,
        )

    def _compute_distance_fields(self):
        gt = self.grid_tensor
        occ_now = gt[OCCUPIED_MAP_LAYERS[0]].astype(bool)

        crate_targets = (gt[CRATE_POTENTIAL_LAYER] > 0) & ~occ_now
        gt[CRATE_DISTANCE_LAYER] = self._multi_source_bfs(crate_targets, occ_now)

        coin_targets = gt[COIN_LAYER].astype(bool)
        gt[COIN_DISTANCE_LAYER] = self._multi_source_bfs(coin_targets, occ_now)

    def _refresh_forecast_layers(self):
        self._compute_danger_forecast()
        self._compute_occupancy_forecast()
        self._time_aware_bfs([(self.agent.x, self.agent.y)], SELF_DISTANCE_LAYER)
        opponent_positions = [(h.x, h.y) for h in self.opponent_handles if not h.dead]
        self._time_aware_bfs(opponent_positions, OPPONENTS_LEAST_DISTANCE_LAYER)
        self._compute_danger_summary()
        self._compute_mobility()
        self._compute_distance_fields()

    def _get_centered_tensor(self) -> np.ndarray:
        dx = self.center_x - self.agent.x
        dy = self.center_y - self.agent.y

        src_x0 = max(0, -dx)
        src_x1 = min(self.width, self.width - dx)
        src_y0 = max(0, -dy)
        src_y1 = min(self.height, self.height - dy)

        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        self._centered_tensor.fill(0)
        self._centered_tensor[:, dst_x0:dst_x1, dst_y0:dst_y1] = \
            self.grid_tensor[:, src_x0:src_x1, src_y0:src_y1]
        return self._centered_tensor

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_actions = {}

        self.new_round()

        grid_tensor = self._get_centered_tensor()
        #self.features = get_features(grid_tensor)
        #obs = {"grid_tensor": grid_tensor, "features": self.features}
        info = self._get_info()
        return grid_tensor, info

    def step(self, action):
        self.previous_features = self.features
        self.step_count += 1

        shared = self._build_shared_state()
        actions = {}
        for handle, act_fn in zip(self.opponent_handles, self.opponent_act_fns):
            handle.reset_game_events()
            if handle.dead:
                continue
            state = self._agent_state_dict(handle, shared)
            actions[handle] = act_fn(handle, state)

        self.agent.reset_game_events()
        actions[self.agent] = ACTIONS[action]
        self.agent_actions = actions

        order = self.rng.permutation(len(self.active_agents))
        for i in order:
            a = self.active_agents[i]
            act = actions.get(a, "WAIT")
            self._perform_agent_action(a, act)

        self._collect_coins()
        self._update_explosions()
        self._update_bombs()
        self._evaluate_explosions()

        self._refresh_dynamic_layers()
        self._refresh_forecast_layers()

        grid_tensor = self._get_centered_tensor()
        #self.features = get_features(grid_tensor)
        #obs = {"grid_tensor": grid_tensor, "features": self.features}

        self.visited[self.agent.x, self.agent.y] = True

        reward = self.reward_fn()

        terminated = self.agent.dead
        truncated = self.step_count >= s.MAX_STEPS

        info = self._get_info()

        return grid_tensor, reward, terminated, truncated, info

    def _build_shared_state(self) -> Dict[str, Any]:
        """Parts of game_state identical for every agent, built once per step."""
        field = np.array(self.arena)

        explosion_map = np.zeros(self.arena.shape, dtype=np.float64)
        for ex in self.explosions:
            if ex["stage"] == 0:
                t = ex["timer"] - 1
                for (x, y) in ex["coords"]:
                    if t > explosion_map[x, y]:
                        explosion_map[x, y] = t

        bombs_state = [((b["x"], b["y"]), b["timer"]) for b in self.bombs]
        coins_state = [
            (int(x), int(y))
            for (x, y), collectable in zip(self.coins_xy, self.coins_collectable)
            if collectable
        ]

        return {
            "field": field,
            "explosion_map": explosion_map,
            "bombs": bombs_state,
            "coins": coins_state,
        }

    def _agent_state_dict(self, handle: AgentHandle, shared: Dict[str, Any]) -> dict:
        return {
            "round": self.round,
            "step": self.step_count,
            "field": shared["field"],
            "self": handle.get_state(),
            "others": [o.get_state() for o in self.active_agents if o is not handle],
            "bombs": shared["bombs"],
            "coins": shared["coins"],
            "user_input": "WAIT",
            "explosion_map": shared["explosion_map"],
        }

    def get_state_for_agent(self, handle: AgentHandle) -> dict:
        """Public single-agent accessor, API parity with BombeRLeWorld."""
        return self._agent_state_dict(handle, self._build_shared_state())

    def _tile_is_free(self, x, y) -> bool:
        if self.arena[x, y] != 0:
            return False
        for b in self.bombs:
            if b["x"] == x and b["y"] == y:
                return False
        for a in self.active_agents:
            if a.x == x and a.y == y:
                return False
        return True

    def _move_agent(self, handle: AgentHandle, new_x, new_y):
        if handle is self.agent:
            self.grid_tensor[SELF_LAYER, handle.x, handle.y] = 0.0
            self.grid_tensor[SELF_LAYER, new_x, new_y] = 1.0
        handle.x, handle.y = new_x, new_y

    def _place_bomb(self, agent: AgentHandle):
        self.bombs.append({"x": agent.x, "y": agent.y, "timer": s.BOMB_TIMER, "owner": agent})
        agent.bombs_left = False

    def _perform_agent_action(self, agent: AgentHandle, action):
        x, y = agent.x, agent.y
        if action == "UP" and self._tile_is_free(x, y - 1):
            self._move_agent(agent, x, y - 1)
            agent.add_event(e.MOVED_UP)
        elif action == "DOWN" and self._tile_is_free(x, y + 1):
            self._move_agent(agent, x, y + 1)
            agent.add_event(e.MOVED_DOWN)
        elif action == "LEFT" and self._tile_is_free(x - 1, y):
            self._move_agent(agent, x - 1, y)
            agent.add_event(e.MOVED_LEFT)
        elif action == "RIGHT" and self._tile_is_free(x + 1, y):
            self._move_agent(agent, x + 1, y)
            agent.add_event(e.MOVED_RIGHT)
        elif action == "BOMB" and agent.bombs_left:
            self._place_bomb(agent)
            agent.add_event(e.BOMB_DROPPED)
        elif action == "WAIT":
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def _collect_coins(self):
        collectable_idx = np.nonzero(self.coins_collectable)[0]
        if len(collectable_idx) == 0 or not self.active_agents:
            return
        active_pos = np.array([[a.x, a.y] for a in self.active_agents], dtype=np.int64)
        for ci in collectable_idx:
            cx, cy = self.coins_xy[ci]
            matches = np.nonzero((active_pos[:, 0] == cx) & (active_pos[:, 1] == cy))[0]
            if len(matches):
                handle = self.active_agents[int(matches[0])]
                self.coins_collectable[ci] = False
                self.grid_tensor[COIN_LAYER, cx, cy] = 0.0
                handle.update_score(s.REWARD_COIN)
                handle.add_event(e.COIN_COLLECTED)

    def _update_explosions(self):
        if not self.explosions:
            return
        remaining = []
        for ex in self.explosions:
            ex["timer"] -= 1
            if ex["timer"] <= 0:
                ex["stage"] += 1
                if ex["stage"] == 1:
                    ex["timer"] = _EXPLOSION_STAGE1_TICKS
                    ex["owner"].bombs_left = True
                else:
                    ex["stage"] = None
            if ex["stage"] is not None:
                remaining.append(ex)
        self.explosions = remaining

    def _update_bombs(self):
        if not self.bombs:
            return
        remaining = []
        for b in self.bombs:
            if b["timer"] <= 0:
                owner = b["owner"]
                owner.add_event(e.BOMB_EXPLODED)
                blast = self.PRECOMPUTED_BLAST_COORDS[(b["x"], b["y"])]

                for (x, y) in blast:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        self.grid_tensor[CRATE_LAYER, x, y] = 0.0
                        self.grid_tensor[CRATE_POTENTIAL_LAYER] -= self._blast_tensor[:, :, x, y]
                        owner.add_event(e.CRATE_DESTROYED)
                        if len(self.coins_xy):
                            coin_matches = np.nonzero(
                                (self.coins_xy[:, 0] == x) & (self.coins_xy[:, 1] == y)
                            )[0]
                            for ci in coin_matches:
                                if not self.coins_collectable[ci]:
                                    self.coins_collectable[ci] = True
                                    self.grid_tensor[COIN_LAYER, x, y] = 1.0
                                    owner.add_event(e.COIN_FOUND)

                self.explosions.append({
                    "coords": blast,
                    "coords_set": set(blast),
                    "owner": owner,
                    "timer": s.EXPLOSION_TIMER,
                    "stage": 0,
                })
            else:
                b["timer"] -= 1
                remaining.append(b)
        self.bombs = remaining

    def _evaluate_explosions(self):
        if not self.explosions or not self.active_agents:
            return
        hit = set()
        for ex in self.explosions:
            if ex["stage"] != 0:
                continue
            owner = ex["owner"]
            coords_set = ex["coords_set"]
            for a in self.active_agents:
                if (not a.dead) and (a.x, a.y) in coords_set:
                    hit.add(a)
                    if a is owner:
                        a.add_event(e.KILLED_SELF)
                    else:
                        owner.update_score(s.REWARD_KILL)
                        owner.add_event(e.KILLED_OPPONENT)

        for a in hit:
            a.dead = True
            self.active_agents.remove(a)
            a.add_event(e.GOT_KILLED)
            for other in self.active_agents:
                other.add_event(e.OPPONENT_ELIMINATED)

    def check_game_state_conversion_accuracy(self, obs):
        from agent_code.my_agent.input_processing import observation_to_game_state

        correct_game_state = self.get_state_for_agent(self.agent)
        computed_game_state = observation_to_game_state(obs)

        for important_key in ["field", "explosion_map"]:
            if not np.array_equal(correct_game_state[important_key], computed_game_state[important_key]):
                raise ValueError(f"Game state computation differs from correct game state in field '{important_key}')")

        for important_key in ["bombs", "coins"]:
            if not set(correct_game_state[important_key]) == set(computed_game_state[important_key]):
                raise ValueError(f"Game state computation differs from correct game state in field '{important_key}')")

        for important_index in [2, 3]:
            if not correct_game_state["self"][important_index] == computed_game_state["self"][important_index]:
                raise ValueError(f"Game state computation differs from correct game state in field 'self[{important_index}]'")

        if not len(correct_game_state["others"]) == len(computed_game_state["others"]):
            raise ValueError("Game state computation differs from correct game state in field 'others' (length mismatch)")

        for opponent_index in range(len(correct_game_state["others"])):
            for important_index in [2, 3]:
                if not correct_game_state["others"][opponent_index][important_index] == computed_game_state["others"][opponent_index][important_index]:
                    raise ValueError(f"Game state computation differs from correct game state in field 'others [{opponent_index}][{important_index}]'")

    def _get_info(self):
        return {
            "step": self.step_count,
            "score": self.agent.score,
            "alive": not self.agent.dead,
        }

    def default_reward(self):
        reward = 0
        for event in self.agent.events:
            reward += SIMPLE_EVENT_REWARDS.get(event, 0)
        return reward

    def action_masks(self):
        mask = np.ones(6, dtype=bool)
        x, y = self.agent.x, self.agent.y
        mask[0] = self._tile_is_free(x, y - 1)   # UP
        mask[1] = self._tile_is_free(x + 1, y)   # RIGHT
        mask[2] = self._tile_is_free(x, y + 1)   # DOWN
        mask[3] = self._tile_is_free(x - 1, y)   # LEFT
        mask[4] = True                           # WAIT always valid
        mask[5] = self.agent.bombs_left          # BOMB
        return mask

    def is_walkable(self, x, y):
        if self.arena[x, y] != 0:
            return False
        if any(b["x"] == x and b["y"] == y for b in self.bombs):
            return False
        if any(a.x == x and a.y == y and not a.dead for a in self.all_agents):
            return False
        return True

    def shaped_reward(self):
        reward = 0


        for event in self.agent.events:
            reward += EVENT_REWARDS.get(event, 0)

        visited_count = np.sum(self.visited)
        new_visited = visited_count - self.previous_visited_count
        self.previous_visited_count = visited_count

        if new_visited > 0:
            reward += 0.02

        return reward

    def render(self):
        pass

    def close(self):
        pass