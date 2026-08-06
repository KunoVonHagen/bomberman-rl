import pickle
from collections import namedtuple, deque
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any, Iterable, Set

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
from rewards import (
    SIMPLE_EVENT_REWARDS,
    EVENT_REWARDS,
    CRATE_SHAPING_COEF,
    COIN_SHAPING_COEF,
    ESCAPE_BONUS_COEF,
    DANGER_PENALTY_COEF
)

# Kept for drop-in compatibility with callers that construct WorldArgs(...).
WorldArgs = namedtuple(
    "WorldArgs",
    ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay",
     "make_video", "continue_without_training", "log_dir", "save_stats",
     "match_name", "seed", "silence_errors", "scenario"],
)


class _NullLogger:
    """
    Minimal logger that does nothing, for use in the gym environment when
    no logging is desired. This avoids the overhead of checking for a logger
    in the main loop and allows the environment to be used in contexts where
    logging is not set up or desired.
    """
    __slots__ = ()

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


_NULL_LOGGER = _NullLogger()


class AgentHandle:
    """
    A handle for an agent in the environment, used to track its state and
    interactions with the environment. This class is used to represent both
    the learning agent and its opponents, providing a consistent interface
    for managing their state, score, and events during the game.
    """

    __slots__ = ("name", "train", "logger", "x", "y", "score", "total_score",
                 "bombs_left", "dead", "events", "__dict__")

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

MAX_OPPONENTS = 3

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

_DANGER_SLICE = slice(DANGER_MAP_LAYERS[0], DANGER_MAP_LAYERS[-1] + 1)
_OCC_SLICE = slice(OCCUPIED_MAP_LAYERS[0], OCCUPIED_MAP_LAYERS[-1] + 1)

LAYER_GROUPS: Dict[str, List[int]] = {
    "base": [WALL_LAYER, CRATE_LAYER, COIN_LAYER, SELF_LAYER, SELF_BLAST_LAYER,
              OPPONENT_LAYER, OPPONENT_DANGER_LAYER, BOMBS_LEFT_LAYER],
    "timer_channels": list(range(8, _BASE_LAYERS)),
    "forecast": list(DANGER_MAP_LAYERS) + list(OCCUPIED_MAP_LAYERS),
    "self_distance": [SELF_DISTANCE_LAYER],
    "opponent_distance": [OPPONENTS_LEAST_DISTANCE_LAYER],
    "crate_potential": [CRATE_POTENTIAL_LAYER],
    "danger_summary": [DANGER_ONSET_LAYER, DANGER_CLEAR_LAYER],
    "mobility": [MOBILITY_LAYER],
    "crate_distance": [CRATE_DISTANCE_LAYER],
    "coin_distance": [COIN_DISTANCE_LAYER],
}

LAYER_GROUP_DEPENDENCIES: Dict[str, List[str]] = {
    "self_distance": ["forecast"],
    "opponent_distance": ["forecast"],
    "danger_summary": ["forecast"],
    "mobility": ["forecast"],
    "crate_distance": ["forecast", "crate_potential"],
    "coin_distance": ["forecast"],
}

ALL_LAYER_GROUPS: Tuple[str, ...] = tuple(LAYER_GROUPS.keys())

(
    FEATURE_SELF_X,
    FEATURE_SELF_Y,
    FEATURE_BOMBS_LEFT,
    FEATURE_STEP_PROGRESS,
    FEATURE_COIN_DISTANCE,
    FEATURE_CRATE_DISTANCE,
    FEATURE_OPPONENT_DISTANCE,
    FEATURE_BOMB_DANGER,
    FEATURE_MOBILITY,
    FEATURE_OPPONENTS_ALIVE,
    FEATURE_COINS_REMAINING,
    FEATURE_CRATES_REMAINING,
    FEATURE_SAFE_UP,
    FEATURE_SAFE_RIGHT,
    FEATURE_SAFE_DOWN,
    FEATURE_SAFE_LEFT,
    FEATURE_SAFE_WAIT,
    FEATURE_SAFE_BOMB,
    FEATURE_BOMB_TARGET_VALUE,
    FEATURE_TRAPPED_OPPONENT_DISTANCE,
) = range(20)

FEATURE_NAMES: Tuple[str, ...] = (
    "self_x",
    "self_y",
    "bombs_left",
    "step_progress",
    "coin_distance",
    "crate_distance",
    "opponent_distance",
    "bomb_danger",
    "mobility",
    "opponents_alive",
    "coins_remaining",
    "crates_remaining",
    "safe_up",
    "safe_right",
    "safe_down",
    "safe_left",
    "safe_wait",
    "safe_bomb",
    "bomb_target_value",
    "trapped_opponent_distance",
)
NUM_FEATURES = len(FEATURE_NAMES)


def resolve_layer_groups(requested: Optional[Iterable[str]]) -> Set[str]:
    """
    Given a requested set of layer groups, resolve it to the full set of groups that should be enabled, including dependencies and the base group.
    """
    if requested is None:
        return set(LAYER_GROUPS.keys())

    requested = set(requested)
    unknown = requested - set(LAYER_GROUPS.keys())
    if unknown:
        raise ValueError(
            f"Unknown layer group(s): {sorted(unknown)}. "
            f"Available groups: {sorted(LAYER_GROUPS.keys())}"
        )

    requested.add("base")
    resolved: Set[str] = set()

    def _add(group: str):
        if group in resolved:
            return
        resolved.add(group)
        for dep in LAYER_GROUP_DEPENDENCIES.get(group, ()):
            _add(dep)

    for g in requested:
        _add(g)
    return resolved


@njit(cache=True)
def _time_aware_bfs_kernel(starts, occ, W, H, T, dist, visited, qx, qy, qt):
    """
    Time-aware BFS kernel to compute the shortest distance from any of the starting cells to all other cells in the grid, considering occupied cells as obstacles and time steps.
    """
    dist[:, :] = -1.0
    visited[:, :, :] = False
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


@njit(cache=True)
def _multi_source_bfs_kernel(targets, occ, W, H, dist, qx, qy):
    """
    Multi-source BFS kernel to compute the shortest distance from any of the target cells to all other cells in the grid, considering occupied cells as obstacles.
    """
    dist[:, :] = -1.0
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


@njit(cache=True)
def _forecast_kernel(bomb_x, bomb_y, bomb_timer, blast_tensor,
                      exp_x, exp_y, exp_timer,
                      wall, crate, T, ET, danger_out, occ_out):
    """
    Fused danger-forecast + occupancy-forecast computation.
    """
    W, H = wall.shape
    remaining_crates = crate.copy()

    for t in range(T):
        d = danger_out[t]
        d[:, :] = 0.0
        for i in range(exp_x.shape[0]):
            if exp_timer[i] - t > 0:
                d[exp_x[i], exp_y[i]] = 1.0
        for i in range(bomb_x.shape[0]):
            bt = bomb_timer[i]
            if bt <= t < bt + ET:
                blast = blast_tensor[bomb_x[i], bomb_y[i]]
                for xx in range(W):
                    for yy in range(H):
                        if blast[xx, yy] > d[xx, yy]:
                            d[xx, yy] = blast[xx, yy]

        for xx in range(W):
            for yy in range(H):
                if d[xx, yy] > 0.0:
                    remaining_crates[xx, yy] = False

        o = occ_out[t]
        for xx in range(W):
            for yy in range(H):
                v = wall[xx, yy] or remaining_crates[xx, yy]
                o[xx, yy] = v or d[xx, yy] > 0.0
        for i in range(bomb_x.shape[0]):
            if bomb_timer[i] >= t:
                o[bomb_x[i], bomb_y[i]] = True

    of = occ_out[T]
    for xx in range(W):
        for yy in range(H):
            of[xx, yy] = wall[xx, yy] or remaining_crates[xx, yy]


class BombermanGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        args,
        opponents: List[Tuple[Callable[["AgentHandle"], None], Callable[["AgentHandle", dict], "Optional[str]"]]],
        reward_fn=None,
        render_mode=None,
        layer_config: Optional[Iterable[str]] = None,
    ):
        """
        Gymnasium environment for the Bomberman game, designed for reinforcement learning.

        Args:
            args: Configuration arguments for the environment.
            opponents: A list of tuples, each containing a setup function and an action function for an opponent agent.
            reward_fn: Optional custom reward function. If None, a default shaped reward function is used.
            render_mode: Optional rendering mode. If None, no rendering is performed.
            layer_config: Optional iterable of layer group names to include in the observation. If None, all layers are included.
        """
        super().__init__()
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.render_mode = render_mode

        if len(opponents) > MAX_OPPONENTS:
            raise ValueError(
                f"got {len(opponents)} opponents, but this environment only "
                f"supports up to MAX_OPPONENTS={MAX_OPPONENTS}"
            )

        self.agent = AgentHandle("RLAgent")

        self.opponent_handles: List[AgentHandle] = [
            AgentHandle(f"OpponentAgent{i}") for i in range(MAX_OPPONENTS)
        ]
        self.opponent_act_fns: List[Callable] = [act_fn for (_setup_fn, act_fn) in opponents]
        self.n_real_opponents = len(opponents)

        for handle, (setup_fn, _act_fn) in zip(self.opponent_handles, opponents):
            setup_fn(handle)

        self.all_agents: List[AgentHandle] = [self.agent] + self.opponent_handles
        self.n_agents = 1 + self.n_real_opponents
        self.active_agents: List[AgentHandle] = list(self.all_agents[: self.n_agents])

        self.reward_fn = reward_fn or self.shaped_reward

        self.width, self.height = s.COLS, s.ROWS
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.enabled_groups: Set[str] = resolve_layer_groups(layer_config)

        self._enable_timer_channels = "timer_channels" in self.enabled_groups
        self._enable_forecast = "forecast" in self.enabled_groups
        self._enable_self_distance = "self_distance" in self.enabled_groups
        self._enable_opponent_distance = "opponent_distance" in self.enabled_groups
        self._enable_crate_potential = "crate_potential" in self.enabled_groups
        self._enable_danger_summary = "danger_summary" in self.enabled_groups
        self._enable_mobility = "mobility" in self.enabled_groups
        self._enable_crate_distance = "crate_distance" in self.enabled_groups
        self._enable_coin_distance = "coin_distance" in self.enabled_groups

        output_indices = sorted(
            idx for g in self.enabled_groups for idx in LAYER_GROUPS[g]
        )
        self._full_output = len(output_indices) == NUM_LAYERS
        self._output_layer_indices = (
            None if self._full_output else np.array(output_indices, dtype=np.int64)
        )

        self.n_observation_layers = NUM_LAYERS
        self.n_output_layers = len(output_indices)
        self._BT = s.BOMB_TIMER
        self._ET = s.EXPLOSION_TIMER


        self._T_HORIZON = float(self._BT + self._ET)
        self._CRATE_POTENTIAL_MAX = float(4 * s.BOMB_POWER)
        self._DIST_MAX = float(self.width * self.height)

        self.grid_tensor = np.zeros((self.n_observation_layers, self.width, self.height), dtype=np.float32)
        self._centered_tensor = np.zeros_like(self.grid_tensor)

        T_bfs = self._BT + self._ET
        self._ta_bfs_visited = np.zeros((self.width, self.height, T_bfs + 1), dtype=np.bool_)
        max_nodes_ta = self.width * self.height * (T_bfs + 1)
        self._ta_bfs_qx = np.empty(max_nodes_ta, dtype=np.int32)
        self._ta_bfs_qy = np.empty(max_nodes_ta, dtype=np.int32)
        self._ta_bfs_qt = np.empty(max_nodes_ta, dtype=np.int32)

        max_nodes_ms = self.width * self.height
        self._ms_bfs_qx = np.empty(max_nodes_ms, dtype=np.int32)
        self._ms_bfs_qy = np.empty(max_nodes_ms, dtype=np.int32)

        self.observation_space = spaces.Dict({
            "grid_tensor": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.n_output_layers, self.width, self.height),
                dtype=np.float32,
            ),
            "features": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(NUM_FEATURES,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(len(ACTIONS))

        wall_mask = self._build_wall_mask()
        self._wall_layer = np.where(wall_mask == -1, 1.0, 0.0).astype(np.float32)
        self._wall_bool = self._wall_layer.astype(bool)
        self.PRECOMPUTED_BLAST_COORDS: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._blast_tensor = np.zeros((self.width, self.height, self.width, self.height), dtype=np.float32)
        for x, y in np.argwhere(wall_mask != -1):
            x, y = int(x), int(y)
            coords = self._compute_blast_coords(x, y, wall_mask, s.BOMB_POWER)
            self.PRECOMPUTED_BLAST_COORDS[(x, y)] = coords
            xs_, ys_ = zip(*coords)
            self._blast_tensor[x, y, list(xs_), list(ys_)] = 1.0

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
        self._features = np.zeros(NUM_FEATURES, dtype=np.float32)
        self._initial_crate_count = 0
        self._initial_coin_count = 0
        self._initial_n_opponents = len(self.opponent_handles)

        self._prev_coin_dist: Optional[float] = None
        self._prev_crate_dist: Optional[float] = None
        self._prev_bomb_danger: float = 0.0

        self.agent_actions = {}

        self._replay: Optional[Dict[str, Any]] = None

        self.new_round()

    @staticmethod
    def available_layer_groups() -> Dict[str, List[int]]:
        """
        Introspection helper: group name -> internal layer indices.
        """
        return dict(LAYER_GROUPS)

    @staticmethod
    def feature_names() -> Tuple[str, ...]:
        """
        Introspection helper: returns the names of the features in the order they are returned by `compute_features`.
        """
        return FEATURE_NAMES

    def _build_wall_mask(self) -> np.ndarray:
        """
        Wall portion of BombeRLeWorld.build_arena (RNG-independent).
        """
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
        """
        Reproduces items.Bomb.get_blast_coords without a Bomb object.
        """
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
        """
        Reproduces BombeRLeWorld.build_arena (crates, coins, start positions).
        """
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

        self._initial_crate_count = int(np.sum(arena == 1))
        self._initial_coin_count = int(self.coins_collectable.sum())

        real_agents = [self.agent] + self.opponent_handles[: self.n_real_opponents]
        for handle, (x, y) in zip(real_agents, positions):
            handle.x, handle.y = int(x), int(y)
            handle.dead = False
            handle.score = 0
            handle.total_score = handle.total_score
            handle.bombs_left = True
            handle.events = []

        for handle in self.opponent_handles[self.n_real_opponents:]:
            handle.dead = True

        self.active_agents = list(real_agents)

        self._rebuild_full_grid_tensor()

        self.previous_visited_count = 1
        self.visited.fill(False)
        self.previous_features = None
        self.agent_actions = {}

        self._prev_coin_dist = self._coin_distance_now()
        self._prev_crate_dist = self._crate_distance_now() if self.agent.bombs_left else None
        self._prev_bomb_danger = self._bomb_danger_now()

        if self._replay is not None:
            self._finalize_replay_and_save()

        self._start_replay_recording()

    def _start_replay_recording(self) -> None:
        """
        Begin buffering everything `replay.ReplayWorld` needs to replay this round, if `self.args.save_replay` is set. No-op otherwise.
        """
        if not getattr(self.args, "save_replay", False):
            self._replay = None
            return

        self._replay = {
            "round": self.round,
            "arena": self.arena.copy(),
            "coins": [(int(x), int(y)) for (x, y) in self.coins_xy.tolist()],
            "agents": [
                (h.name, h.score, h.bombs_left, (h.x, h.y))
                for h in self.active_agents
            ],
            "actions": {h.name: [] for h in self.active_agents},
            "permutations": [],
            "n_steps": 0,
        }

    def _rebuild_full_grid_tensor(self):
        """
        Full rebuild -- only once per round, cost amortized over the round.
        """
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
        if self._enable_crate_potential:
            self._init_crate_potential()
        self._refresh_forecast_layers()

    def _refresh_dynamic_layers(self):
        """
        Recomputes channels that are inherently collective (all enemies) or shift every tick (bomb/explosion timer channels).
        Walls/crates/coins/self position are maintained incrementally elsewhere.
        """
        gt = self.grid_tensor
        BT = self._BT

        gt[4:_BASE_LAYERS].fill(0)

        ax, ay = self.agent.x, self.agent.y
        gt[SELF_BLAST_LAYER] = self._blast_tensor[ax, ay]

        for h in self.opponent_handles:
            if h.dead:
                continue
            ex_, ey_ = h.x, h.y
            gt[OPPONENT_LAYER, ex_, ey_] = 1.0
            gt[OPPONENT_DANGER_LAYER] += self._blast_tensor[ex_, ey_]
            gt[BOMBS_LEFT_LAYER, ex_, ey_] = 1.0 if h.bombs_left else 0.0

        gt[OPPONENT_DANGER_LAYER] = np.where(gt[OPPONENT_DANGER_LAYER] > 0, 1.0, 0.0)
        gt[BOMBS_LEFT_LAYER, ax, ay] = 1.0 if self.agent.bombs_left else 0.0

        if self._enable_timer_channels:
            for b in self.bombs:
                pos_ch = 8 + b["timer"]
                danger_ch = 8 + BT + b["timer"]
                gt[pos_ch, b["x"], b["y"]] = 1.0
                gt[danger_ch] = self._blast_tensor[b["x"], b["y"]]

            for ex in self.explosions:
                if ex["stage"] == 0:
                    ch = 7 + 2 * BT + ex["timer"]
                    for (x, y) in ex["coords"]:
                        gt[ch, x, y] = 1.0

    def _init_crate_potential(self):
        """
        Full recompute -- only once per round. Kept current afterwards by incremental subtraction in `_update_bombs`.
        """
        self.grid_tensor[CRATE_POTENTIAL_LAYER] = np.einsum(
            "xyij,ij->xy", self._blast_tensor, self.grid_tensor[CRATE_LAYER]
        )

    def _compute_forecasts(self):
        """
        Fused replacement for the former _compute_danger_forecast + _compute_occupancy_forecast pair.
        """
        gt = self.grid_tensor
        T = self._BT + self._ET
        active_explosions = [ex for ex in self.explosions if ex["stage"] == 0]

        if not self.bombs and not active_explosions:
            gt[_DANGER_SLICE] = 0.0
            static = (self._wall_bool | gt[CRATE_LAYER].astype(bool)).astype(np.float32)
            gt[_OCC_SLICE] = static
            return

        if self.bombs:
            bomb_x = np.fromiter((b["x"] for b in self.bombs), dtype=np.int64, count=len(self.bombs))
            bomb_y = np.fromiter((b["y"] for b in self.bombs), dtype=np.int64, count=len(self.bombs))
            bomb_timer = np.fromiter((b["timer"] for b in self.bombs), dtype=np.int64, count=len(self.bombs))
        else:
            bomb_x = bomb_y = bomb_timer = np.zeros(0, dtype=np.int64)

        if active_explosions:
            exp_x_list, exp_y_list, exp_timer_list = [], [], []
            for ex in active_explosions:
                for (x, y) in ex["coords"]:
                    exp_x_list.append(x)
                    exp_y_list.append(y)
                    exp_timer_list.append(ex["timer"])
            exp_x = np.array(exp_x_list, dtype=np.int64)
            exp_y = np.array(exp_y_list, dtype=np.int64)
            exp_timer = np.array(exp_timer_list, dtype=np.int64)
        else:
            exp_x = exp_y = exp_timer = np.zeros(0, dtype=np.int64)

        danger_out = gt[_DANGER_SLICE]
        occ_out = gt[_OCC_SLICE]

        _forecast_kernel(
            bomb_x, bomb_y, bomb_timer, self._blast_tensor,
            exp_x, exp_y, exp_timer,
            self._wall_bool, gt[CRATE_LAYER].astype(bool),
            T, self._ET, danger_out, occ_out,
        )

    def _time_aware_bfs(self, starts, out_layer):
        """
        Earliest arrival time at every cell, respecting the occupancy forecast (four moves + wait).
        Delegates to a JIT-compiled kernel that implements the identical FIFO/BFS algorithm.
        """
        T = len(OCCUPIED_MAP_LAYERS) - 1
        W, H = self.width, self.height
        gt = self.grid_tensor

        if starts:
            starts_arr = np.asarray(starts, dtype=np.int64)
        else:
            starts_arr = np.zeros((0, 2), dtype=np.int64)

        occ = gt[_OCC_SLICE]
        _time_aware_bfs_kernel(
            starts_arr, occ, W, H, T,
            gt[out_layer], self._ta_bfs_visited,
            self._ta_bfs_qx, self._ta_bfs_qy, self._ta_bfs_qt,
        )

    def _compute_danger_summary(self):
        """
        Onset/clear timestep of the danger forecast, per cell (-1 = never).
        """
        gt = self.grid_tensor
        stack = gt[_DANGER_SLICE]
        ever = stack.any(axis=0)

        onset = np.argmax(stack, axis=0)
        gt[DANGER_ONSET_LAYER] = np.where(ever, onset, -1)

        T = stack.shape[0]
        last_from_end = np.argmax(stack[::-1], axis=0)
        gt[DANGER_CLEAR_LAYER] = np.where(ever, T - last_from_end, -1)

    def _compute_mobility(self):
        """
        Free 4-neighbor count under final occupancy.
        """
        gt = self.grid_tensor
        free = 1.0 - gt[OCCUPIED_MAP_LAYERS[-1]]
        m = np.zeros_like(free)
        m[1:, :] += free[:-1, :]
        m[:-1, :] += free[1:, :]
        m[:, 1:] += free[:, :-1]
        m[:, :-1] += free[:, 1:]
        gt[MOBILITY_LAYER] = m

    def _multi_source_bfs(self, targets: np.ndarray, occ: np.ndarray, out: np.ndarray) -> None:
        """
        BFS from multiple target cells, respecting the occupancy map (four moves only).
        Fills `out` with the distance to the nearest target for each cell, or -1 if unreachable.
        """
        W, H = self.width, self.height
        _multi_source_bfs_kernel(
            np.ascontiguousarray(targets, dtype=np.bool_),
            np.ascontiguousarray(occ, dtype=np.bool_),
            W, H, out, self._ms_bfs_qx, self._ms_bfs_qy,
        )

    def _compute_distance_fields(self):
        """
        Compute static BFS distance to nearest crate and coin, if enabled.
        """
        if not (self._enable_crate_distance or self._enable_coin_distance):
            return
        gt = self.grid_tensor
        occ_now = gt[OCCUPIED_MAP_LAYERS[0]].astype(bool)

        if self._enable_crate_distance:
            crate_targets = (gt[CRATE_POTENTIAL_LAYER] > 0) & ~occ_now
            self._multi_source_bfs(crate_targets, occ_now, gt[CRATE_DISTANCE_LAYER])

        if self._enable_coin_distance:
            coin_targets = gt[COIN_LAYER].astype(bool)
            self._multi_source_bfs(coin_targets, occ_now, gt[COIN_DISTANCE_LAYER])

    def _refresh_forecast_layers(self):
        """
        Recompute all forecast-related layers (danger forecast, occupancy forecast, self distance, opponent distance, danger summary, mobility, static distances).
        """
        if self._enable_forecast:
            self._compute_forecasts()
        if self._enable_self_distance:
            self._time_aware_bfs([(self.agent.x, self.agent.y)], SELF_DISTANCE_LAYER)
        if self._enable_opponent_distance:
            opponent_positions = [(h.x, h.y) for h in self.opponent_handles if not h.dead]
            self._time_aware_bfs(opponent_positions, OPPONENTS_LEAST_DISTANCE_LAYER)
        if self._enable_danger_summary:
            self._compute_danger_summary()
        if self._enable_mobility:
            self._compute_mobility()
        self._compute_distance_fields()

    def _get_centered_tensor(self) -> np.ndarray:
        """
        Return a copy of the grid tensor, centered on the RL agent's position.
        The returned tensor has the same shape as self.grid_tensor, but the agent's position is at the center of the tensor.
        """
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

    def _normalize_observation(self, tensor: np.ndarray) -> np.ndarray:
        """
        Normalize the observation tensor to the range [-1, 1] for each layer.
        The normalization is done based on the maximum values defined for each layer.
        """
        T = self._T_HORIZON

        for layer in (SELF_DISTANCE_LAYER, OPPONENTS_LEAST_DISTANCE_LAYER,
                      DANGER_ONSET_LAYER, DANGER_CLEAR_LAYER):
            raw = tensor[layer]
            tensor[layer] = np.where(raw < 0, -1.0, raw / T)

        tensor[CRATE_POTENTIAL_LAYER] = np.clip(
            tensor[CRATE_POTENTIAL_LAYER] / self._CRATE_POTENTIAL_MAX, 0.0, 1.0
        )

        tensor[MOBILITY_LAYER] = tensor[MOBILITY_LAYER] / 4.0

        for layer in (CRATE_DISTANCE_LAYER, COIN_DISTANCE_LAYER):
            raw = tensor[layer]
            tensor[layer] = np.where(
                raw < 0, -1.0, np.clip(raw / self._DIST_MAX, 0.0, 1.0)
            )

        return tensor

    def _select_output_layers(self, tensor: np.ndarray) -> np.ndarray:
        """
        Slice down to just the enabled groups' layers. Returns the original tensor unchanged (no copy) when every group is enabled.
        """
        if self._output_layer_indices is None:
            return tensor
        return tensor[self._output_layer_indices]

    def set_opponents(self, opponents):
        if len(opponents) > MAX_OPPONENTS:
            raise ValueError(...)
        self.opponent_act_fns = [act_fn for (_setup_fn, act_fn) in opponents]
        self.n_real_opponents = len(opponents)
        for handle, (setup_fn, _act_fn) in zip(self.opponent_handles, opponents):
            setup_fn(handle)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new round. This method initializes the game state, generates a new round layout, and returns the initial observation and info.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_actions = {}

        self.new_round()

        grid_tensor = self._normalize_observation(self._get_centered_tensor())
        grid_tensor = self._select_output_layers(grid_tensor)
        features = self._compute_global_features()
        obs = {"grid_tensor": grid_tensor, "features": features}
        info = self._get_info()
        return obs, info

    def step(self, action):
        """
        Perform a single step in the environment using the given action for the RL agent.
        This method updates the game state, processes actions for all agents, and returns the new observation, reward, termination status, truncation status, and additional info.
        """
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

        self._record_replay_step(order, actions)

        self._collect_coins()
        self._update_explosions()
        self._update_bombs()
        self._evaluate_explosions()

        self._refresh_dynamic_layers()
        self._refresh_forecast_layers()

        grid_tensor = self._normalize_observation(self._get_centered_tensor())
        grid_tensor = self._select_output_layers(grid_tensor)
        features = self._compute_global_features()
        obs = {"grid_tensor": grid_tensor, "features": features}

        self.visited[self.agent.x, self.agent.y] = True

        reward = self.reward_fn()

        terminated = self.agent.dead
        truncated = self.step_count >= s.MAX_STEPS

        if terminated or truncated:
            self._finalize_replay_and_save()

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _record_replay_step(self, order: np.ndarray, actions: Dict["AgentHandle", str]) -> None:
        """
        If replay recording is enabled, append the current step's agent order and actions to the replay buffer.
        """
        if self._replay is None:
            return
        self._replay["permutations"].append([int(i) for i in order])
        for a in self.active_agents:
            self._replay["actions"][a.name].append(actions.get(a, "WAIT"))

    def _replay_save_path(self) -> str:
        """
        Determine the file path for saving the replay of the current round.
        If a custom replay path is provided in the arguments, it will be used.
        Otherwise, the replay will be saved in the "replays" directory with a filename based on the match name and round number.
        """
        replay_path = getattr(self.args, "replay", None)
        if replay_path:
            return str(replay_path)

        round_no = self._replay.get("round", self.round) if self._replay else self.round
        match_name = getattr(self.args, "match_name", None) or "match"
        replays_dir = Path("replays")
        replays_dir.mkdir(parents=True, exist_ok=True)
        return str(replays_dir / f"{match_name}_round{round_no:03d}.pkl")

    def _finalize_replay_and_save(self) -> None:
        """
        If replay recording is enabled and there are recorded steps, finalize the replay data and save it to a file.
        The replay data includes the round number, arena layout, coin positions, agent states, actions, permutations of agent order, and the total number of steps.
        The replay is saved as a pickle file at the determined replay save path. After saving, the replay buffer is cleared.
        """
        if self._replay is None:
            return
        if not self._replay["permutations"]:
            self._replay = None
            return
        self._replay["n_steps"] = self.step_count
        path = self._replay_save_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(self._replay, f)
        finally:
            self._replay = None

    def _build_shared_state(self) -> Dict[str, Any]:
        """
        Parts of game_state identical for every agent, built once per step.
        """
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
        """
        Internal single-agent accessor, used by `get_state_for_agent` and by opponent act_fns.
        Returns a dict in the same shape as BombeRLeWorld.get_state_for_agent, with keys: "round", "step", "field", "self", "others", "bombs", "coins", "user_input", "explosion_map".
        """
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
        """
        Public single-agent accessor, API parity with BombeRLeWorld.
        """
        return self._agent_state_dict(handle, self._build_shared_state())

    def _load_game_state(self, game_state: dict) -> None:
        """
        Load a BombeRLeWorld-style `game_state` dict into this env's internal snapshot fields.
        """
        self.round = game_state.get("round", self.round)
        self.step_count = game_state.get("step", self.step_count)

        self.arena = np.asarray(game_state["field"], dtype=np.int8).copy()

        _, self_score, self_bombs_left, (sx, sy) = game_state["self"]
        self.agent.x, self.agent.y = int(sx), int(sy)
        self.agent.bombs_left = bool(self_bombs_left)
        self.agent.score = self_score
        self.agent.dead = False

        others = game_state.get("others", [])
        by_name = {h.name: h for h in self.opponent_handles}
        unused_handles = [h for h in self.opponent_handles]
        for h in self.opponent_handles:
            h.dead = True
        for (name, score, bombs_left, (ox, oy)) in others:
            h = by_name.get(name)
            if h is None and unused_handles:
                h = unused_handles[0]
            if h is None:
                continue  # more opponents than this env has handles for
            if h in unused_handles:
                unused_handles.remove(h)
            h.x, h.y = int(ox), int(oy)
            h.bombs_left = bool(bombs_left)
            h.score = score
            h.dead = False

        bombs = game_state.get("bombs", [])
        self.bombs = [
            {"x": int(x), "y": int(y), "timer": int(timer), "owner": None}
            for ((x, y), timer) in bombs
        ]

        coins = game_state.get("coins", [])
        self.coins_xy = (
            np.array(coins, dtype=np.int64) if coins else np.zeros((0, 2), dtype=np.int64)
        )
        self.coins_collectable = np.ones(len(coins), dtype=bool)

        explosion_map = np.asarray(
            game_state.get("explosion_map", np.zeros(self.arena.shape))
        )
        cells_by_timer: Dict[int, List[Tuple[int, int]]] = {}
        for x, y in np.argwhere(explosion_map > 0):
            t = int(explosion_map[x, y])
            cells_by_timer.setdefault(t, []).append((int(x), int(y)))
        self.explosions = [
            {
                "coords": coords,
                "coords_set": set(coords),
                "owner": None,
                "timer": t + 1,
                "stage": 0,
            }
            for t, coords in cells_by_timer.items()
        ]

    def observation_from_game_state(self, game_state: dict) -> dict:
        """
        Given a BombeRLeWorld-style `game_state` dict, return the corresponding observation dict for this environment.
        """
        self._load_game_state(game_state)
        self._rebuild_full_grid_tensor()

        grid_tensor = self._normalize_observation(self._get_centered_tensor())
        grid_tensor = self._select_output_layers(grid_tensor)
        features = self._compute_global_features()
        return {"grid_tensor": grid_tensor, "features": features}

    def _tile_is_free(self, x, y) -> bool:
        """
        Check if the tile at (x, y) is free for an agent to move into.
        A tile is considered free if it is not a wall or crate, and there are no bombs or agents occupying that tile.
        """
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
        """
        Move the specified agent to a new position (new_x, new_y) on the grid.
        Updates the agent's position and the corresponding layer in the grid tensor to reflect the move.
        """
        if handle is self.agent:
            self.grid_tensor[SELF_LAYER, handle.x, handle.y] = 0.0
            self.grid_tensor[SELF_LAYER, new_x, new_y] = 1.0
        handle.x, handle.y = new_x, new_y

    def _place_bomb(self, agent: AgentHandle):
        """
        Place a bomb at the agent's current position. The bomb will have a timer set to the predefined BOMB_TIMER value.
        The agent's bombs_left attribute is set to False, indicating that the agent cannot place another bomb until the current one explodes.
        """
        self.bombs.append({"x": agent.x, "y": agent.y, "timer": s.BOMB_TIMER, "owner": agent})
        agent.bombs_left = False

    def _perform_agent_action(self, agent: AgentHandle, action):
        """
        Perform the specified action for the given agent. The action can be one of the following:
        - "UP": Move the agent up if the tile above is free.
        - "DOWN": Move the agent down if the tile below is free.
        - "LEFT": Move the agent left if the tile to the left is free.
        - "RIGHT": Move the agent right if the tile to the right is free.
        - "BOMB": Place a bomb at the agent's current position if the agent has bombs left.
        - "WAIT": The agent does nothing for this step.
        If the action is invalid (e.g., moving into a wall or crate, or placing a bomb when none are left), the agent will receive an INVALID_ACTION event.
        """
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
        """
        Check if any active agents are on the same position as collectable coins.
        If an agent is on a coin, the coin is collected, removed from the grid, and the agent's score is updated with the coin reward.
        """
        collectable_idx = np.nonzero(self.coins_collectable)[0]
        if len(collectable_idx) == 0 or not self.active_agents:
            return
        active_pos = np.array([[a.x, a.y] for a in self.active_agents], dtype=np.int64)
        coin_pos = self.coins_xy[collectable_idx]
        eq = (coin_pos[:, None, :] == active_pos[None, :, :]).all(-1)
        hit_coin, hit_agent = np.nonzero(eq)
        seen = set()
        for ci_local, ai in zip(hit_coin, hit_agent):
            ci = int(collectable_idx[ci_local])
            if ci in seen:
                continue
            seen.add(ci)
            cx, cy = self.coins_xy[ci]
            handle = self.active_agents[int(ai)]
            self.coins_collectable[ci] = False
            self.grid_tensor[COIN_LAYER, cx, cy] = 0.0
            handle.update_score(s.REWARD_COIN)
            handle.add_event(e.COIN_COLLECTED)

    def _update_explosions(self):
        """
        Update the state of active explosions. Each explosion has a timer that counts down each step.
        When the timer reaches zero, the explosion progresses to the next stage.
        If the explosion is in stage 0, it will transition to stage 1 and reset its timer.
        If the explosion is in stage 1, it will be removed from the active explosions list.
        The owner of the explosion will have their bombs_left attribute set to True when the explosion transitions from stage 0 to stage 1, allowing them to place another bomb.
        """
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
        """
        Update the state of active bombs. Each bomb has a timer that counts down each step.
        When the timer reaches zero, the bomb explodes, affecting the surrounding area based on its blast radius.
        The explosion can destroy crates and make coins collectable. The owner of the bomb will receive events for bomb explosion, crate destruction, and coin collection as appropriate.
        """
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
                        if self._enable_crate_potential:
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
        """
        Check if any active agents are within the blast radius of any active explosions.
        If an agent is hit by an explosion, they are marked as dead and removed from the active agents list.
        The owner of the explosion receives score updates and events based on whether they killed themselves or an opponent.
        """
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

    def _get_info(self):
        """
        Return a dictionary containing information about the current state of the environment, including the current step count, the RL agent's score, and whether the RL agent is alive or dead.
        """
        return {
            "step": self.step_count,
            "score": self.agent.score,
            "alive": not self.agent.dead,
        }

    def default_reward(self):
        """
        Compute the default reward for the RL agent based on the events that occurred during the current step.
        The reward is calculated by summing the rewards associated with each event in the agent's event list, using the SIMPLE_EVENT_REWARDS dictionary to look up the reward values for each event.
        """
        reward = 0
        for event in self.agent.events:
            reward += SIMPLE_EVENT_REWARDS.get(event, 0)
        return reward

    def action_masks(self):
        """
        Return a boolean array of shape (6,) indicating which actions are valid for the RL agent at the current step.
        The actions correspond to the following indices:
        0: UP
        1: DOWN
        2: LEFT
        3: RIGHT
        4: WAIT
        5: BOMB
        An action is considered valid if the corresponding tile is free for movement (for UP, DOWN, LEFT, RIGHT) or if the agent has bombs left (for BOMB).
        The WAIT action is always valid.
        The returned array can be used to mask out invalid actions during action selection.
        """
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

    def _nearest_coin_distance(self) -> float:
        """Manhattan distance from the agent to the nearest collectable
        coin, ignoring obstacles. O(#coins). -1.0 if none remain."""
        idx = np.nonzero(self.coins_collectable)[0]
        if len(idx) == 0:
            return -1.0
        coin_xy = self.coins_xy[idx]
        d = np.abs(coin_xy[:, 0] - self.agent.x) + np.abs(coin_xy[:, 1] - self.agent.y)
        return float(d.min())

    def _nearest_crate_distance(self) -> float:
        """Manhattan distance from the agent to the nearest crate,
        ignoring obstacles. O(#crates). -1.0 if none remain."""
        xs, ys = np.nonzero(self.arena == 1)
        if len(xs) == 0:
            return -1.0
        d = np.abs(xs - self.agent.x) + np.abs(ys - self.agent.y)
        return float(d.min())

    def _coin_distance_now(self) -> float:
        """Distance from the agent to the nearest coin. Uses the
        already-computed occupancy-aware BFS layer when available (its
        cost is already paid for the observation); otherwise falls back
        to the cheap Manhattan estimate above. -1.0 if no coins remain."""
        if self._enable_coin_distance:
            d = float(self.grid_tensor[COIN_DISTANCE_LAYER, self.agent.x, self.agent.y])
            if d >= 0:
                return d
        return self._nearest_coin_distance()

    def _crate_distance_now(self) -> float:
        """Distance from the agent to the nearest (still-useful) crate,
        same BFS-layer-first / Manhattan-fallback strategy as coins."""
        if self._enable_crate_distance:
            d = float(self.grid_tensor[CRATE_DISTANCE_LAYER, self.agent.x, self.agent.y])
            if d >= 0:
                return d
        return self._nearest_crate_distance()

    def _bomb_danger_now(self) -> float:
        """Danger score in [0, 1] for the agent's current tile: the max,
        over active bombs whose blast already covers this tile, of how
        close that bomb is to detonating (1.0 = about to explode).
        0.0 if the tile is safe. O(#bombs) via the precomputed blast
        tensor -- no BFS or forecast layers needed, so this is always
        available regardless of `layer_config`."""
        if not self.bombs:
            return 0.0
        ax, ay = self.agent.x, self.agent.y
        worst = 0.0
        for b in self.bombs:
            if self._blast_tensor[b["x"], b["y"], ax, ay] > 0:
                urgency = (s.BOMB_TIMER - b["timer"] + 1) / (s.BOMB_TIMER + 1)
                if urgency > worst:
                    worst = urgency
        return worst

    def _mobility_now(self) -> float:
        """Count (0-4) of the agent's immediately-walkable orthogonal
        neighbour tiles. O(1) -- checks the four neighbours directly via
        `_tile_is_free`; does not require the (forecast-dependent, only
        available when the 'mobility' group is enabled) MOBILITY_LAYER."""
        x, y = self.agent.x, self.agent.y
        count = 0
        for (nx, ny) in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if self._tile_is_free(nx, ny):
                count += 1
        return float(count)

    def _nearest_opponent_distance(self) -> float:
        """Manhattan distance from the agent to the nearest living
        opponent, ignoring obstacles. O(#opponents). -1.0 if none are
        alive (including games with no opponents at all)."""
        alive = [h for h in self.opponent_handles if not h.dead]
        if not alive:
            return -1.0
        d = min(abs(h.x - self.agent.x) + abs(h.y - self.agent.y) for h in alive)
        return float(d)

    def _danger_at_tick(self, x: int, y: int, t: int) -> bool:
        """Whether (x, y) is on fire at tick `t` from now, computed
        directly from `self.bombs`/`self.explosions` (not from the
        forecast grid layers), using the same "explosion persists while
        its timer - t > 0" / "bomb blasts while bt <= t < bt + ET" rules
        as `_forecast_kernel`. Only ever queried for t in {0, 1, 2}, so
        this stays O(#bombs + #explosions) and needs none of the
        (optional, layer_config-gated) forecast layers -- matching this
        file's convention that `features` never hard-depends on a
        disable-able layer group."""
        for ex in self.explosions:
            if ex["stage"] == 0 and (ex["timer"] - t) > 0 and (x, y) in ex["coords_set"]:
                return True
        for b in self.bombs:
            bt = b["timer"]
            if bt <= t < bt + self._ET:
                if self._blast_tensor[b["x"], b["y"], x, y] > 0:
                    return True
        return False

    def _action_safety_now(self) -> Tuple[float, float, float, float, float, float]:
        """One-tick-lookahead stand-in for the rule-based agent's
        `get_legal_actions` + `is_action_safe` filtering (objectives.py),
        which it runs before *any* other decision-making. For each of the
        six actions: 0.0 if it's illegal (wall/crate/bomb/other agent in
        the way) or walks onto a tile that's already on fire or about to
        be next tick; 1.0 otherwise. For BOMB specifically, a hypothetical
        bomb is placed at the agent's own tile and 1.0 is only returned if
        at least one neighbouring tile (or staying put) would still be
        clear of blast for the following two ticks -- i.e. "don't bomb
        yourself into a corner". This is deliberately a shallow
        approximation of the rule-based agent's full permanently-safe BFS
        (`get_safe_square_action`) -- exact enough to flag immediately
        suicidal actions, cheap enough to run every RL step.
        Returned in ACTIONS order: (UP, RIGHT, DOWN, LEFT, WAIT, BOMB).
        """
        ax, ay = self.agent.x, self.agent.y
        deltas = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))  # UP, RIGHT, DOWN, LEFT, WAIT

        safety = []
        for (dx, dy) in deltas:
            nx, ny = ax + dx, ay + dy
            if (dx, dy) != (0, 0) and not self._tile_is_free(nx, ny):
                safety.append(0.0)
                continue
            safety.append(0.0 if (self._danger_at_tick(nx, ny, 0) or self._danger_at_tick(nx, ny, 1)) else 1.0)

        if not self.agent.bombs_left:
            safety.append(0.0)
        else:
            hypothetical = {"x": ax, "y": ay, "timer": self._BT, "owner": self.agent}
            self.bombs.append(hypothetical)
            escape = False
            for (dx, dy) in deltas:
                nx, ny = ax + dx, ay + dy
                if (dx, dy) != (0, 0) and not self._tile_is_free(nx, ny):
                    continue
                if not self._danger_at_tick(nx, ny, 1) and not self._danger_at_tick(nx, ny, 2):
                    escape = True
                    break
            self.bombs.pop()
            safety.append(1.0 if escape else 0.0)

        return tuple(safety)

    def _bomb_target_value_now(self) -> float:
        """
        Heuristic value in [0, 1] for the agent's current tile as a
        bombing target: 0.0 if the agent has no bombs left, otherwise a
        weighted sum of
        (0.5 * fraction of remaining crates that would be hit by a bomb here)
        + (0.5 * 1.0 if any living opponent would be hit by a bomb here, else 0.0).
        """
        if not self.agent.bombs_left:
            return 0.0
        ax, ay = self.agent.x, self.agent.y
        blast_coords = self.PRECOMPUTED_BLAST_COORDS.get((ax, ay), [(ax, ay)])
        blast_set = set(blast_coords)
        crates_hit = sum(1 for (bx, by) in blast_coords if self.arena[bx, by] == 1)
        opponents_hit = sum(1 for h in self.opponent_handles if not h.dead and (h.x, h.y) in blast_set)

        crate_frac = min(1.0, crates_hit / self._CRATE_POTENTIAL_MAX) if self._CRATE_POTENTIAL_MAX > 0 else 0.0
        value = 0.5 * crate_frac + 0.5 * (1.0 if opponents_hit > 0 else 0.0)
        return float(np.clip(value, 0.0, 1.0))

    def _nearest_trapped_opponent_distance(self) -> float:
        """Manhattan distance to the nearest living opponent with at most
        one open orthogonal neighbour -- the same cornered-opponent
        signal `find_trap_targets` (strategy.py) uses to pick hunting
        targets for the rule-based agent. O(#opponents). -1.0 if no
        opponent currently qualifies (including no opponents left)."""
        ax, ay = self.agent.x, self.agent.y
        best = None
        for h in self.opponent_handles:
            if h.dead:
                continue
            free = 0
            for (nx, ny) in ((h.x, h.y - 1), (h.x, h.y + 1), (h.x - 1, h.y), (h.x + 1, h.y)):
                if self._tile_is_free(nx, ny):
                    free += 1
            if free <= 1:
                d = abs(h.x - ax) + abs(h.y - ay)
                if best is None or d < best:
                    best = d
        return -1.0 if best is None else float(best)

    def _compute_global_features(self) -> np.ndarray:
        """Cheap, always-on numeric summary of global game state.

        Every entry is O(1) or O(#agents / #bombs / #coins / #crates) --
        never a full-grid BFS or forecast sweep -- so this method returns a
        complete, meaningful feature vector regardless of `layer_config`
        (including `layer_config=[]`, where every high-complexity spatial
        layer is disabled). A couple of entries (coin/crate distance)
        opportunistically reuse an already-computed time-aware BFS layer
        when its group happens to be enabled, but fall back to an equally
        cheap raw Manhattan estimate otherwise -- so they're never a hard
        dependency on the expensive layers, just occasionally correlated
        with them.

        Every value is scaled to sit inside [-1, 1] for easy consumption by
        a neural net: fixed-range quantities (position, elapsed time,
        danger, mobility, remaining coins/crates/opponents) are simple
        fractions in [0, 1]; open-ended distances use the same "-1 means
        not applicable / none left, otherwise a clipped fraction of the
        board size in [0, 1]" sentinel convention already used for the
        distance-like grid layers in `_normalize_observation`.

        The trailing block (safe_up/right/down/left/wait/bomb,
        bomb_target_value, trapped_opponent_distance) is new: it distills
        the decision checks the rule-based agent (callbacks.py) always
        runs before choosing a move -- "which actions won't get me
        killed", "is bombing here worth it", "is any opponent cornered
        right now" -- into features, since the RL agent previously had no
        direct signal for any of that beyond the single-tile bomb_danger
        value. Like the rest of this method, they're all O(small) and
        independent of `layer_config`.
        """
        f = self._features
        W1 = max(self.width - 1, 1)
        H1 = max(self.height - 1, 1)

        f[FEATURE_SELF_X] = 2.0 * self.agent.x / W1 - 1.0
        f[FEATURE_SELF_Y] = 2.0 * self.agent.y / H1 - 1.0

        f[FEATURE_BOMBS_LEFT] = 1.0 if self.agent.bombs_left else 0.0

        f[FEATURE_STEP_PROGRESS] = np.clip(self.step_count / s.MAX_STEPS, 0.0, 1.0)

        coin_dist = self._coin_distance_now()
        f[FEATURE_COIN_DISTANCE] = (
            -1.0 if coin_dist < 0 else np.clip(coin_dist / self._DIST_MAX, 0.0, 1.0)
        )

        crate_dist = self._crate_distance_now()
        f[FEATURE_CRATE_DISTANCE] = (
            -1.0 if crate_dist < 0 else np.clip(crate_dist / self._DIST_MAX, 0.0, 1.0)
        )

        opp_dist = self._nearest_opponent_distance()
        f[FEATURE_OPPONENT_DISTANCE] = (
            -1.0 if opp_dist < 0 else np.clip(opp_dist / self._DIST_MAX, 0.0, 1.0)
        )

        f[FEATURE_BOMB_DANGER] = self._bomb_danger_now()

        f[FEATURE_MOBILITY] = self._mobility_now() / 4.0

        if self._initial_n_opponents > 0:
            n_alive = sum(1 for h in self.opponent_handles if not h.dead)
            f[FEATURE_OPPONENTS_ALIVE] = n_alive / self._initial_n_opponents
        else:
            f[FEATURE_OPPONENTS_ALIVE] = 0.0

        if self._initial_coin_count > 0:
            f[FEATURE_COINS_REMAINING] = (
                float(self.coins_collectable.sum()) / self._initial_coin_count
            )
        else:
            f[FEATURE_COINS_REMAINING] = 0.0

        if self._initial_crate_count > 0:
            f[FEATURE_CRATES_REMAINING] = (
                float(np.sum(self.arena == 1)) / self._initial_crate_count
            )
        else:
            f[FEATURE_CRATES_REMAINING] = 0.0

        safe_up, safe_right, safe_down, safe_left, safe_wait, safe_bomb = self._action_safety_now()
        f[FEATURE_SAFE_UP] = safe_up
        f[FEATURE_SAFE_RIGHT] = safe_right
        f[FEATURE_SAFE_DOWN] = safe_down
        f[FEATURE_SAFE_LEFT] = safe_left
        f[FEATURE_SAFE_WAIT] = safe_wait
        f[FEATURE_SAFE_BOMB] = safe_bomb

        f[FEATURE_BOMB_TARGET_VALUE] = self._bomb_target_value_now()

        trapped_dist = self._nearest_trapped_opponent_distance()
        f[FEATURE_TRAPPED_OPPONENT_DISTANCE] = (
            -1.0 if trapped_dist < 0 else np.clip(trapped_dist / self._DIST_MAX, 0.0, 1.0)
        )

        self.previous_features = f.copy()
        return f

    def shaped_reward(self):
        reward = 0.0

        for event in self.agent.events:
            reward += EVENT_REWARDS.get(event, 0)

        visited_count = np.sum(self.visited)
        new_visited = visited_count - self.previous_visited_count
        self.previous_visited_count = visited_count

        if new_visited > 0:
            reward += 0.02

        if self.agent.dead:
            return reward

        coin_dist = self._coin_distance_now()
        just_collected = e.COIN_COLLECTED in self.agent.events
        if (not just_collected) and coin_dist >= 0 and self._prev_coin_dist is not None and self._prev_coin_dist >= 0:
            reward += COIN_SHAPING_COEF * (self._prev_coin_dist - coin_dist)
        self._prev_coin_dist = coin_dist

        if self.agent.bombs_left:
            crate_dist = self._crate_distance_now()
            just_destroyed = e.CRATE_DESTROYED in self.agent.events
            if (not just_destroyed) and crate_dist >= 0 and self._prev_crate_dist is not None and self._prev_crate_dist >= 0:
                reward += CRATE_SHAPING_COEF * (self._prev_crate_dist - crate_dist)
            self._prev_crate_dist = crate_dist
        else:
            self._prev_crate_dist = None

        danger = self._bomb_danger_now()
        if danger > 0:
            reward -= DANGER_PENALTY_COEF * danger
        if self._prev_bomb_danger > danger:
            reward += ESCAPE_BONUS_COEF * (self._prev_bomb_danger - danger)
        self._prev_bomb_danger = danger

        return reward

    def render(self):
        pass

    def close(self):
        # Flush an in-progress replay recording rather than losing it if the
        # env is closed mid-episode.
        self._finalize_replay_and_save()