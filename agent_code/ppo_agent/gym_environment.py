import pickle
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any, Iterable, Set

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import njit

import settings as s
import events as e
from agent_code.ppo_agent.rewards import (
    EVENT_REWARDS,
    CRATE_SHAPING_COEF,
    COIN_SHAPING_COEF,
    ESCAPE_BONUS_COEF,
    DANGER_PENALTY_COEF,
    TRAP_SHAPING_COEF,
    build_event_rewards,
)
from agent_code.ppo_agent.config import RewardConfig

WorldArgs = namedtuple(
    "WorldArgs",
    ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay",
     "make_video", "continue_without_training", "log_dir", "save_stats",
     "match_name", "seed", "silence_errors", "scenario"],
)


class _NullLogger:
    """
    A logger that does nothing. Used as a default logger for AgentHandle instances
    when no logger is provided. This prevents the need for null checks before logging.
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
    Represents an agent in the Bomberman environment, tracking its state, score, and events.
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
    Resolves a set of requested layer groups, including their dependencies.
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
def _time_aware_bfs_kernel(starts, start_counts, occ, danger, W, H, T, dist, visited, qx, qy, qt, fixes):
    """
    Batched time-aware BFS. starts/occ/dist/visited: (n_envs, W, H, T+1).
    """
    n_envs = starts.shape[0]
    for env in range(n_envs):
        d = dist[env]
        vis = visited[env]
        d[:, :] = -1.0
        vis[:, :, :] = False
        head = 0
        tail = 0

        for i in range(start_counts[env]):
            x = starts[env, i, 0]
            y = starts[env, i, 1]
            if not fixes and occ[env, 0, x, y] > 0:
                continue
            if not vis[x, y, 0]:
                vis[x, y, 0] = True
                d[x, y] = 0.0
                qx[env, tail] = x
                qy[env, tail] = y
                qt[env, tail] = 0
                tail += 1

        while head < tail:
            x = qx[env, head]
            y = qy[env, head]
            t = qt[env, head]
            head += 1

            next_t = t + 1
            if next_t > T:
                next_t = T
            if fixes:
                land_t = t if t < T else T
                occ_next = occ[env, land_t]
                danger_t = land_t if land_t < T else T - 1
                any_danger = t < T
            else:
                occ_next = occ[env, next_t]
                danger_t = 0
                any_danger = False

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
                if fixes and k == 4:
                    if any_danger and danger[env, danger_t, nx, ny] > 0:
                        continue
                elif occ_next[nx, ny] > 0:
                    continue
                if vis[nx, ny, next_t]:
                    continue

                vis[nx, ny, next_t] = True
                if d[nx, ny] == -1.0:
                    d[nx, ny] = next_t
                qx[env, tail] = nx
                qy[env, tail] = ny
                qt[env, tail] = next_t
                tail += 1


@njit(cache=True)
def _multi_source_bfs_kernel(targets, occ, W, H, dist, qx, qy):
    """
    Batched multi-source BFS. targets/occ/dist: (n_envs, W, H).
    """
    n_envs = targets.shape[0]
    for env in range(n_envs):
        d = dist[env]
        d[:, :] = -1.0
        head = 0
        tail = 0
        tg = targets[env]
        oc = occ[env]

        for x in range(W):
            for y in range(H):
                if tg[x, y]:
                    d[x, y] = 0.0
                    qx[env, tail] = x
                    qy[env, tail] = y
                    tail += 1

        while head < tail:
            x = qx[env, head]
            y = qy[env, head]
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
                if oc[nx, ny] or d[nx, ny] != -1.0:
                    continue
                d[nx, ny] = d[x, y] + 1.0
                qx[env, tail] = nx
                qy[env, tail] = ny
                tail += 1


@njit(cache=True)
def _forecast_kernel(bomb_x, bomb_y, bomb_timer, bomb_counts, blast_tensor,
                     exp_x, exp_y, exp_timer, exp_counts,
                     wall, crate, T, ET, danger_out, occ_out, fixes):
    """
    Batched forecast of danger and occupied maps. All inputs/outputs are (n_envs, ...) arrays.
    """
    n_envs = bomb_counts.shape[0]
    width, height = wall.shape
    exp_offset = 1 if fixes else 0

    for env in range(n_envs):
        nb = bomb_counts[env]
        ne = exp_counts[env]
        remaining_crates = crate[env].copy()

        for t in range(T):
            d = danger_out[env, t]
            d[:, :] = 0.0
            for i in range(ne):
                if exp_timer[env, i] - exp_offset - t > 0:
                    d[exp_x[env, i], exp_y[env, i]] = 1.0
            for i in range(nb):
                bt = bomb_timer[env, i]
                if bt <= t < bt + ET:
                    blast = blast_tensor[bomb_x[env, i], bomb_y[env, i]]
                    for xx in range(width):
                        for yy in range(height):
                            if blast[xx, yy] > d[xx, yy]:
                                d[xx, yy] = blast[xx, yy]

            for xx in range(width):
                for yy in range(height):
                    if d[xx, yy] > 0.0:
                        remaining_crates[xx, yy] = False

            o = occ_out[env, t]
            for xx in range(width):
                for yy in range(height):
                    v = wall[xx, yy] or remaining_crates[xx, yy]
                    o[xx, yy] = v or d[xx, yy] > 0.0
            for i in range(nb):
                if bomb_timer[env, i] >= t:
                    o[bomb_x[env, i], bomb_y[env, i]] = True

        of = occ_out[env, T]
        for xx in range(width):
            for yy in range(height):
                of[xx, yy] = wall[xx, yy] or remaining_crates[xx, yy]

class BombermanGymEnv(gym.Env):
    """
    A Gymnasium environment for the Bomberman game, supporting multiple parallel games and various observation layers.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        args,
        opponents: List[Tuple[Callable[["AgentHandle"], None], Callable[["AgentHandle", dict], "Optional[str]"]]],
        reward_fn=None,
        reward_config: Optional[RewardConfig] = None,
        render_mode=None,
        layer_config: Optional[Iterable[str]] = None,
        n_envs: int = 1,
        auto_reset: bool = True,
        env_version: int = 1,
    ):
        super().__init__()
        self.args = args
        self.set_reward_config(reward_config)
        self.n_envs = int(n_envs)
        self.env_version = int(env_version)
        self._fixes = self.env_version >= 2
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.auto_reset = auto_reset
        self.rng = np.random.default_rng(args.seed)
        self.render_mode = render_mode

        if len(opponents) > MAX_OPPONENTS:
            raise ValueError(
                f"got {len(opponents)} opponents, but this environment only "
                f"supports up to MAX_OPPONENTS={MAX_OPPONENTS}"
            )

        self.agents: List[AgentHandle] = [AgentHandle("RLAgent") for _ in range(self.n_envs)]
        self.opponent_handles: List[List[AgentHandle]] = [
            [AgentHandle(f"OpponentAgent{i}") for i in range(MAX_OPPONENTS)]
            for _ in range(self.n_envs)
        ]
        self.opponent_act_fns: List[List[Callable]] = [
            [act_fn for (_setup_fn, act_fn) in opponents] for _ in range(self.n_envs)
        ]
        self.n_real_opponents: List[int] = [len(opponents) for _ in range(self.n_envs)]
        self.env_scenarios: List[str] = [args.scenario for _ in range(self.n_envs)]
        self._opponent_resampler = None

        for handles in self.opponent_handles:
            for handle, (setup_fn, _act_fn) in zip(handles, opponents):
                setup_fn(handle)

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

        output_indices = sorted(idx for g in self.enabled_groups for idx in LAYER_GROUPS[g])
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

        E = self.n_envs
        W, H = self.width, self.height

        self.grid_tensor = np.zeros((E, self.n_observation_layers, W, H), dtype=np.float32)
        self._centered_tensor = np.zeros_like(self.grid_tensor)
        self._features = np.zeros((E, NUM_FEATURES), dtype=np.float32)

        self.rounds = np.zeros(E, dtype=np.int64)
        self.step_counts = np.zeros(E, dtype=np.int64)
        self.arena = np.zeros((E, W, H), dtype=np.int8)

        max_coins = max(int(info["COIN_COUNT"]) for info in s.SCENARIOS.values())
        self.coins_xy = np.zeros((E, max_coins, 2), dtype=np.int64)
        self.coins_collectable = np.zeros((E, max_coins), dtype=bool)
        self.n_coins = np.zeros(E, dtype=np.int64)

        self.bombs: List[List[dict]] = [[] for _ in range(E)]
        self.explosions: List[List[dict]] = [[] for _ in range(E)]
        self.active_agents: List[List[AgentHandle]] = [[] for _ in range(E)]

        self.previous_visited_count = np.ones(E, dtype=np.int64)
        self.visited = np.zeros((E, W, H), dtype=bool)
        self.previous_features = np.zeros((E, NUM_FEATURES), dtype=np.float32)
        self._has_previous_features = np.zeros(E, dtype=bool)
        self._initial_crate_count = np.zeros(E, dtype=np.int64)
        self._initial_coin_count = np.zeros(E, dtype=np.int64)
        self._initial_n_opponents = MAX_OPPONENTS

        self._prev_coin_dist = np.full(E, np.nan)
        self._prev_crate_dist = np.full(E, np.nan)
        self._prev_bomb_danger = np.zeros(E)
        self._prev_trap_dist = np.zeros(E)

        self.agent_actions: List[Dict[Any, str]] = [{} for _ in range(E)]
        self._replays: List[Optional[Dict[str, Any]]] = [None] * E

        T_bfs = self._BT + self._ET
        self._ta_bfs_visited = np.zeros((E, W, H, T_bfs + 1), dtype=np.bool_)
        max_nodes_ta = W * H * (T_bfs + 1)
        self._ta_bfs_qx = np.empty((E, max_nodes_ta), dtype=np.int32)
        self._ta_bfs_qy = np.empty((E, max_nodes_ta), dtype=np.int32)
        self._ta_bfs_qt = np.empty((E, max_nodes_ta), dtype=np.int32)
        self._ta_starts = np.zeros((E, 1 + MAX_OPPONENTS, 2), dtype=np.int64)
        self._ta_start_counts = np.zeros(E, dtype=np.int64)

        max_nodes_ms = W * H
        self._ms_bfs_qx = np.empty((E, max_nodes_ms), dtype=np.int32)
        self._ms_bfs_qy = np.empty((E, max_nodes_ms), dtype=np.int32)

        self._MAX_BOMBS = 1 + MAX_OPPONENTS
        self._MAX_EXPLOSION_CELLS = self._MAX_BOMBS * (1 + 4 * s.BOMB_POWER)
        self._fc_bomb_x = np.zeros((E, self._MAX_BOMBS), dtype=np.int64)
        self._fc_bomb_y = np.zeros((E, self._MAX_BOMBS), dtype=np.int64)
        self._fc_bomb_timer = np.zeros((E, self._MAX_BOMBS), dtype=np.int64)
        self._fc_bomb_counts = np.zeros(E, dtype=np.int64)
        self._fc_exp_x = np.zeros((E, self._MAX_EXPLOSION_CELLS), dtype=np.int64)
        self._fc_exp_y = np.zeros((E, self._MAX_EXPLOSION_CELLS), dtype=np.int64)
        self._fc_exp_timer = np.zeros((E, self._MAX_EXPLOSION_CELLS), dtype=np.int64)
        self._fc_exp_counts = np.zeros(E, dtype=np.int64)

        self.single_observation_space = spaces.Dict({
            "grid_tensor": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.n_output_layers, W, H), dtype=np.float32,
            ),
            "features": spaces.Box(
                low=-1.0, high=1.0, shape=(NUM_FEATURES,), dtype=np.float32,
            ),
        })
        self.observation_space = spaces.Dict({
            "grid_tensor": spaces.Box(
                low=-1.0, high=1.0,
                shape=(E, self.n_output_layers, W, H), dtype=np.float32,
            ),
            "features": spaces.Box(
                low=-1.0, high=1.0, shape=(E, NUM_FEATURES), dtype=np.float32,
            ),
        })
        self.single_action_space = spaces.Discrete(len(ACTIONS))
        self.action_space = spaces.MultiDiscrete(np.full(E, len(ACTIONS), dtype=np.int64))

        wall_mask = self._build_wall_mask()
        self._wall_layer = np.where(wall_mask == -1, 1.0, 0.0).astype(np.float32)
        self._wall_bool = self._wall_layer.astype(bool)
        self.PRECOMPUTED_BLAST_COORDS: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._blast_tensor = np.zeros((W, H, W, H), dtype=np.float32)
        for x, y in np.argwhere(wall_mask != -1):
            x, y = int(x), int(y)
            coords = self._compute_blast_coords(x, y, wall_mask, s.BOMB_POWER)
            self.PRECOMPUTED_BLAST_COORDS[(x, y)] = coords
            xs_, ys_ = zip(*coords)
            self._blast_tensor[x, y, list(xs_), list(ys_)] = 1.0


    @staticmethod
    def available_layer_groups() -> Dict[str, List[int]]:
        return dict(LAYER_GROUPS)

    @staticmethod
    def feature_names() -> Tuple[str, ...]:
        return FEATURE_NAMES

    @staticmethod
    def _build_wall_mask() -> np.ndarray:
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

    def _generate_round_layout(self, env: int):
        WALL, FREE, CRATE = -1, 0, 1
        arena = self._build_wall_mask().copy()

        scenario_info = s.SCENARIOS[self.env_scenarios[env]]
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

        n_agents = 1 + self.n_real_opponents[env]
        perm = self.rng.permutation(len(start_positions))
        positions = [start_positions[i] for i in perm[:n_agents]]

        return arena, coins_xy, coins_collectable, positions

    def _reroll_opponents_and_scenario(self, env: int) -> None:
        """
        Rerolls the opponents and scenario for a given environment.
        """
        resampler = self._opponent_resampler
        if resampler is None:
            return

        opponents, scenario, _descriptions = resampler.sample_one()
        if scenario not in s.SCENARIOS:
            raise ValueError(
                f"Opponent resampler produced unknown scenario {scenario!r}. "
                f"Available: {sorted(s.SCENARIOS)}"
            )

        self.env_scenarios[env] = scenario
        self.n_real_opponents[env] = len(opponents)
        self.opponent_act_fns[env] = [act_fn for (_setup_fn, act_fn) in opponents]
        for handle, (setup_fn, _act_fn) in zip(self.opponent_handles[env], opponents):
            setup_fn(handle)

    def _new_round(self, env: int):
        """Reset one env's game state (per-env, cheap; heavy layers are refreshed batched)."""
        self._reroll_opponents_and_scenario(env)

        self.rounds[env] += 1
        self.step_counts[env] = 0
        self.bombs[env] = []
        self.explosions[env] = []

        arena, coins_xy, coins_collectable, positions = self._generate_round_layout(env)
        self.arena[env] = arena

        n = len(coins_xy)
        self.n_coins[env] = n
        if n:
            self.coins_xy[env, :n] = coins_xy
            self.coins_collectable[env, :n] = coins_collectable
        self.coins_xy[env, n:] = 0
        self.coins_collectable[env, n:] = False

        self._initial_crate_count[env] = int(np.sum(arena == 1))
        self._initial_coin_count[env] = int(sum(coins_collectable)) if n else 0

        n_real_opponents = self.n_real_opponents[env]
        real_agents = [self.agents[env]] + self.opponent_handles[env][:n_real_opponents]
        for handle, (x, y) in zip(real_agents, positions):
            handle.x, handle.y = int(x), int(y)
            handle.dead = False
            handle.score = 0
            handle.bombs_left = True
            handle.events = []

        for handle in self.opponent_handles[env][n_real_opponents:]:
            handle.dead = True

        self.active_agents[env] = list(real_agents)

        self._rebuild_static_layers(env)

        self.previous_visited_count[env] = 1
        self.visited[env].fill(False)
        self.previous_features[env].fill(0)
        self._has_previous_features[env] = False
        self.agent_actions[env] = {}

        if self._replays[env] is not None:
            self._finalize_replay_and_save(env)
        self._start_replay_recording(env)

    def _rebuild_static_layers(self, env: int):
        """Walls/crates/coins/self position for one env. Only at round start."""
        g = self.grid_tensor[env]
        g.fill(0)
        g[WALL_LAYER] = self._wall_layer
        g[CRATE_LAYER] = np.where(self.arena[env] == 1, 1.0, 0.0)
        n = self.n_coins[env]
        for ci in range(n):
            if self.coins_collectable[env, ci]:
                x, y = self.coins_xy[env, ci]
                g[COIN_LAYER, x, y] = 1.0
        g[SELF_LAYER, self.agents[env].x, self.agents[env].y] = 1.0

    def _init_crate_potential(self, envs=None):
        if envs is None:
            envs = range(self.n_envs)
        for env in envs:
            self.grid_tensor[env, CRATE_POTENTIAL_LAYER] = np.einsum(
                "xyij,ij->xy", self._blast_tensor, self.grid_tensor[env, CRATE_LAYER]
            )

    def _update_prev_helpers(self, envs=None):
        if envs is None:
            envs = range(self.n_envs)
        for env in envs:
            cd = self._coin_distance_now(env)
            self._prev_coin_dist[env] = np.nan if cd is None else cd
            kd = self._crate_distance_now(env) if self.agents[env].bombs_left else None
            self._prev_crate_dist[env] = np.nan if kd is None else kd
            self._prev_bomb_danger[env] = self._bomb_danger_now(env)
            self._prev_trap_dist[env] = self._trapped_opponent_distance_now(env)


    def _refresh_dynamic_layers(self):
        gt = self.grid_tensor
        BT = self._BT
        gt[:, 4:_BASE_LAYERS].fill(0)

        for env in range(self.n_envs):
            g = gt[env]
            agent = self.agents[env]
            ax, ay = agent.x, agent.y
            g[SELF_BLAST_LAYER] = self._blast_tensor[ax, ay]

            for h in self.opponent_handles[env]:
                if h.dead:
                    continue
                ex_, ey_ = h.x, h.y
                g[OPPONENT_LAYER, ex_, ey_] = 1.0
                g[OPPONENT_DANGER_LAYER] += self._blast_tensor[ex_, ey_]
                g[BOMBS_LEFT_LAYER, ex_, ey_] = 1.0 if h.bombs_left else 0.0

            g[OPPONENT_DANGER_LAYER] = np.where(g[OPPONENT_DANGER_LAYER] > 0, 1.0, 0.0)
            g[BOMBS_LEFT_LAYER, ax, ay] = 1.0 if agent.bombs_left else 0.0

            if self._enable_timer_channels:
                for b in self.bombs[env]:
                    pos_ch = 8 + b["timer"]
                    danger_ch = 8 + BT + b["timer"]
                    g[pos_ch, b["x"], b["y"]] = 1.0
                    g[danger_ch] = self._blast_tensor[b["x"], b["y"]]

                for ex in self.explosions[env]:
                    if ex["stage"] == 0:
                        if self._fixes and ex["timer"] <= 1:
                            continue
                        ch = 7 + 2 * BT + ex["timer"]
                        for (x, y) in ex["coords"]:
                            g[ch, x, y] = 1.0

    def _compute_forecasts(self):
        """Batched fused danger/occupancy forecast for all envs at once."""
        gt = self.grid_tensor
        T = self._BT + self._ET

        bx, by, bt, bc = (self._fc_bomb_x, self._fc_bomb_y,
                          self._fc_bomb_timer, self._fc_bomb_counts)
        exx, exy, ext, exc = (self._fc_exp_x, self._fc_exp_y,
                              self._fc_exp_timer, self._fc_exp_counts)

        any_active = False
        for env in range(self.n_envs):
            bombs = self.bombs[env]
            nb = len(bombs)
            if nb > self._MAX_BOMBS:
                raise RuntimeError(
                    f"env {env} has {nb} bombs, more than the padded maximum "
                    f"{self._MAX_BOMBS}"
                )
            bc[env] = nb
            for i, b in enumerate(bombs):
                bx[env, i] = b["x"]
                by[env, i] = b["y"]
                bt[env, i] = b["timer"]

            k = 0
            for ex in self.explosions[env]:
                if ex["stage"] != 0:
                    continue
                for (x, y) in ex["coords"]:
                    if k >= self._MAX_EXPLOSION_CELLS:
                        raise RuntimeError("explosion padding exceeded")
                    exx[env, k] = x
                    exy[env, k] = y
                    ext[env, k] = ex["timer"]
                    k += 1
            exc[env] = k

            if nb or k:
                any_active = True

        if not any_active:
            gt[:, _DANGER_SLICE] = 0.0
            static = (self._wall_bool[None, :, :] | gt[:, CRATE_LAYER].astype(bool)).astype(np.float32)
            gt[:, _OCC_SLICE] = static[:, None, :, :]
            return

        _forecast_kernel(
            bx, by, bt, bc, self._blast_tensor,
            exx, exy, ext, exc,
            self._wall_bool, gt[:, CRATE_LAYER].astype(bool),
            T, self._ET, gt[:, _DANGER_SLICE], gt[:, _OCC_SLICE], self._fixes,
        )

    def _time_aware_bfs(self, starts_per_env, out_layer):
        """Batched time-aware BFS. `starts_per_env` is a list (len n_envs) of start lists."""
        T = len(OCCUPIED_MAP_LAYERS) - 1
        W, H = self.width, self.height
        starts = self._ta_starts
        counts = self._ta_start_counts
        for env, lst in enumerate(starts_per_env):
            counts[env] = len(lst)
            for i, (x, y) in enumerate(lst):
                starts[env, i, 0] = x
                starts[env, i, 1] = y

        _time_aware_bfs_kernel(
            starts, counts, self.grid_tensor[:, _OCC_SLICE], self.grid_tensor[:, _DANGER_SLICE], W, H, T,
            self.grid_tensor[:, out_layer], self._ta_bfs_visited,
            self._ta_bfs_qx, self._ta_bfs_qy, self._ta_bfs_qt, self._fixes,
        )

    def _compute_danger_summary(self):
        gt = self.grid_tensor
        stack = gt[:, _DANGER_SLICE]
        ever = stack.any(axis=1)

        onset = np.argmax(stack, axis=1)
        gt[:, DANGER_ONSET_LAYER] = np.where(ever, onset, -1)

        T = stack.shape[1]
        last_from_end = np.argmax(stack[:, ::-1], axis=1)
        gt[:, DANGER_CLEAR_LAYER] = np.where(ever, T - last_from_end, -1)

    def _compute_mobility(self):
        gt = self.grid_tensor
        free = 1.0 - gt[:, OCCUPIED_MAP_LAYERS[-1]]
        m = np.zeros_like(free)
        m[:, 1:, :] += free[:, :-1, :]
        m[:, :-1, :] += free[:, 1:, :]
        if not self._fixes:
            m[:, 1:, :] += free[:, :-1, :]
            m[:, :-1, :] += free[:, 1:, :]
        m[:, :, 1:] += free[:, :, :-1]
        m[:, :, :-1] += free[:, :, 1:]
        gt[:, MOBILITY_LAYER] = m

    def _multi_source_bfs(self, targets: np.ndarray, occ: np.ndarray, out: np.ndarray) -> None:
        _multi_source_bfs_kernel(
            np.ascontiguousarray(targets, dtype=np.bool_),
            np.ascontiguousarray(occ, dtype=np.bool_),
            self.width, self.height, out,
            self._ms_bfs_qx, self._ms_bfs_qy,
        )

    def _compute_distance_fields(self):
        if not (self._enable_crate_distance or self._enable_coin_distance):
            return
        gt = self.grid_tensor
        occ_now = gt[:, OCCUPIED_MAP_LAYERS[0]].astype(bool)
        blocked_targets = occ_now
        if self._fixes:
            blocked_targets = occ_now.copy()
            for env, agent in enumerate(self.agents):
                occ_now[env, agent.x, agent.y] = False

        if self._enable_crate_distance:
            crate_targets = (gt[:, CRATE_POTENTIAL_LAYER] > 0) & ~blocked_targets
            self._multi_source_bfs(crate_targets, occ_now, gt[:, CRATE_DISTANCE_LAYER])

        if self._enable_coin_distance:
            coin_targets = gt[:, COIN_LAYER].astype(bool)
            self._multi_source_bfs(coin_targets, occ_now, gt[:, COIN_DISTANCE_LAYER])

    def _refresh_forecast_layers(self):
        """One batched pass of all forecast-related layers for all envs."""
        if self._enable_forecast:
            self._compute_forecasts()
        if self._enable_self_distance:
            self._time_aware_bfs(
                [[(self.agents[env].x, self.agents[env].y)] for env in range(self.n_envs)],
                SELF_DISTANCE_LAYER,
            )
        if self._enable_opponent_distance:
            opp_starts = [
                [(h.x, h.y) for h in self.opponent_handles[env] if not h.dead]
                for env in range(self.n_envs)
            ]
            self._time_aware_bfs(opp_starts, OPPONENTS_LEAST_DISTANCE_LAYER)
        if self._enable_danger_summary:
            self._compute_danger_summary()
        if self._enable_mobility:
            self._compute_mobility()
        self._compute_distance_fields()


    def _get_centered_tensor(self) -> np.ndarray:
        """Batched centering: (n_envs, L, W, H) with each env centered on its agent."""
        out = self._centered_tensor
        out.fill(0)
        for env, agent in enumerate(self.agents):
            dx = self.center_x - agent.x
            dy = self.center_y - agent.y

            src_x0 = max(0, -dx)
            src_x1 = min(self.width, self.width - dx)
            src_y0 = max(0, -dy)
            src_y1 = min(self.height, self.height - dy)

            dst_x0 = max(0, dx)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y0 = max(0, dy)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            out[env, :, dst_x0:dst_x1, dst_y0:dst_y1] = \
                self.grid_tensor[env, :, src_x0:src_x1, src_y0:src_y1]
        return out

    def _normalize_observation(self, tensor: np.ndarray) -> np.ndarray:
        T = self._T_HORIZON

        for layer in (SELF_DISTANCE_LAYER, OPPONENTS_LEAST_DISTANCE_LAYER,
                      DANGER_ONSET_LAYER, DANGER_CLEAR_LAYER):
            raw = tensor[:, layer]
            tensor[:, layer] = np.where(raw < 0, -1.0, raw / T)

        tensor[:, CRATE_POTENTIAL_LAYER] = np.clip(
            tensor[:, CRATE_POTENTIAL_LAYER] / self._CRATE_POTENTIAL_MAX, 0.0, 1.0
        )

        tensor[:, MOBILITY_LAYER] = tensor[:, MOBILITY_LAYER] / 4.0

        for layer in (CRATE_DISTANCE_LAYER, COIN_DISTANCE_LAYER):
            raw = tensor[:, layer]
            tensor[:, layer] = np.where(
                raw < 0, -1.0, np.clip(raw / self._DIST_MAX, 0.0, 1.0)
            )

        return tensor

    def _select_output_layers(self, tensor: np.ndarray) -> np.ndarray:
        if self._output_layer_indices is None:
            return tensor
        return tensor[:, self._output_layer_indices]

    def _compute_global_features(self) -> np.ndarray:
        """RECONSTRUCTED: see porting notes -- verify against your original."""
        E = self.n_envs
        idx = np.arange(E)
        gt = self.grid_tensor
        f = self._features
        W, H = self.width, self.height
        agents = self.agents

        ax = np.fromiter((a.x for a in agents), dtype=np.int64, count=E)
        ay = np.fromiter((a.y for a in agents), dtype=np.int64, count=E)

        f[:, FEATURE_SELF_X] = ax * (2.0 / max(W - 1, 1)) - 1.0
        f[:, FEATURE_SELF_Y] = ay * (2.0 / max(H - 1, 1)) - 1.0
        f[:, FEATURE_BOMBS_LEFT] = np.fromiter(
            (1.0 if a.bombs_left else 0.0 for a in agents), np.float32, E)
        f[:, FEATURE_STEP_PROGRESS] = self.step_counts / float(s.MAX_STEPS)

        coin_d = gt[idx, COIN_DISTANCE_LAYER, ax, ay]
        f[:, FEATURE_COIN_DISTANCE] = np.where(coin_d < 0, -1.0, coin_d / self._DIST_MAX)
        crate_d = gt[idx, CRATE_DISTANCE_LAYER, ax, ay]
        f[:, FEATURE_CRATE_DISTANCE] = np.where(crate_d < 0, -1.0, crate_d / self._DIST_MAX)
        opp_d = gt[idx, OPPONENTS_LEAST_DISTANCE_LAYER, ax, ay]
        f[:, FEATURE_OPPONENT_DISTANCE] = np.where(opp_d < 0, -1.0, opp_d / self._T_HORIZON)

        onset = gt[idx, DANGER_ONSET_LAYER, ax, ay]
        f[:, FEATURE_BOMB_DANGER] = np.where(onset < 0, 0.0, 1.0 / (1.0 + np.maximum(onset, 0.0)))
        f[:, FEATURE_MOBILITY] = gt[idx, MOBILITY_LAYER, ax, ay] / 4.0

        alive = np.fromiter(
            (sum(1 for h in self.opponent_handles[env] if not h.dead)
             for env in range(E)), np.float32, E)
        f[:, FEATURE_OPPONENTS_ALIVE] = alive / float(MAX_OPPONENTS)

        coins_left = self.coins_collectable.sum(axis=1)
        if self._fixes:
            total_coins = np.fromiter(
                (s.SCENARIOS[self.env_scenarios[env]]["COIN_COUNT"] for env in range(E)), np.float32, E)
            total_coins = np.maximum(total_coins, self._initial_coin_count)
            f[:, FEATURE_COINS_REMAINING] = np.minimum(coins_left / np.maximum(total_coins, 1), 1.0)
        else:
            f[:, FEATURE_COINS_REMAINING] = coins_left / np.maximum(self._initial_coin_count, 1)
        crates_left = (self.arena == 1).sum(axis=(1, 2))
        f[:, FEATURE_CRATES_REMAINING] = crates_left / np.maximum(self._initial_crate_count, 1)

        step_idx = 0 if self._fixes else 1
        occ1, dng1 = OCCUPIED_MAP_LAYERS[step_idx], DANGER_MAP_LAYERS[step_idx]

        def _safe_dir(tx, ty):
            valid = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
            tx_c = np.clip(tx, 0, W - 1)
            ty_c = np.clip(ty, 0, H - 1)
            blocked = gt[idx, occ1, tx_c, ty_c] > 0
            deadly = gt[idx, dng1, tx_c, ty_c] > 0
            return (valid & ~blocked & ~deadly).astype(np.float32)

        f[:, FEATURE_SAFE_UP] = _safe_dir(ax, ay - 1)
        f[:, FEATURE_SAFE_RIGHT] = _safe_dir(ax + 1, ay)
        f[:, FEATURE_SAFE_DOWN] = _safe_dir(ax, ay + 1)
        f[:, FEATURE_SAFE_LEFT] = _safe_dir(ax - 1, ay)
        if self._fixes:
            f[:, FEATURE_SAFE_WAIT] = (gt[idx, dng1, ax, ay] <= 0).astype(np.float32)
        else:
            f[:, FEATURE_SAFE_WAIT] = _safe_dir(ax, ay)

        f[:, FEATURE_SAFE_BOMB] = np.maximum(
            np.maximum(f[:, FEATURE_SAFE_UP], f[:, FEATURE_SAFE_RIGHT]),
            np.maximum(f[:, FEATURE_SAFE_DOWN], f[:, FEATURE_SAFE_LEFT]),
        ) * f[:, FEATURE_BOMBS_LEFT]

        f[:, FEATURE_BOMB_TARGET_VALUE] = np.clip(
            gt[idx, CRATE_POTENTIAL_LAYER, ax, ay] / self._CRATE_POTENTIAL_MAX, 0.0, 1.0)
        trap = gt[idx, OPPONENTS_LEAST_DISTANCE_LAYER, ax, ay]
        f[:, FEATURE_TRAPPED_OPPONENT_DISTANCE] = np.where(
            trap < 0, -1.0, trap / self._T_HORIZON)
        return f

    def _build_observation(self) -> Dict[str, np.ndarray]:
        grid_tensor = self._normalize_observation(self._get_centered_tensor())
        grid_tensor = self._select_output_layers(grid_tensor)
        features = self._compute_global_features()
        return {
            "grid_tensor": grid_tensor.copy() if self._full_output else grid_tensor,
            "features": features.copy(),
        }

    def set_scenario(self, scenario: str) -> None:
        """
        Set the scenario for the environment. This will reset the environment and apply the new scenario.
        """
        if scenario not in s.SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario!r}. Available: {sorted(s.SCENARIOS)}"
            )
        try:
            self.args = self.args._replace(scenario=scenario)
        except AttributeError:
            self.args.scenario = scenario
        self.env_scenarios = [scenario for _ in range(self.n_envs)]

    def set_reward_config(self, reward_config: Optional[RewardConfig]) -> None:
        """
        Set the reward configuration for the environment. This will reset the environment and apply the new reward configuration.
        """
        if reward_config is None:
            self._event_rewards = EVENT_REWARDS
            self._coin_shaping_coef = COIN_SHAPING_COEF
            self._crate_shaping_coef = CRATE_SHAPING_COEF
            self._danger_penalty_coef = DANGER_PENALTY_COEF
            self._escape_bonus_coef = ESCAPE_BONUS_COEF
            self._trap_shaping_coef = TRAP_SHAPING_COEF
        else:
            self._event_rewards = build_event_rewards(reward_config)
            self._coin_shaping_coef = reward_config.coin_shaping_coef
            self._crate_shaping_coef = reward_config.crate_shaping_coef
            self._danger_penalty_coef = reward_config.danger_penalty_coef
            self._escape_bonus_coef = reward_config.escape_bonus_coef
            self._trap_shaping_coef = reward_config.trap_shaping_coef

    def set_opponents(self, opponents):
        """
        Set the opponents for the environment. This will reset the environment and apply the new opponents.
        """
        if len(opponents) > MAX_OPPONENTS:
            raise ValueError(
                f"got {len(opponents)} opponents, but this environment only "
                f"supports up to MAX_OPPONENTS={MAX_OPPONENTS}"
            )
        act_fns = [act_fn for (_setup_fn, act_fn) in opponents]
        self.opponent_act_fns = [list(act_fns) for _ in range(self.n_envs)]
        self.n_real_opponents = [len(opponents) for _ in range(self.n_envs)]
        for handles in self.opponent_handles:
            for handle, (setup_fn, _act_fn) in zip(handles, opponents):
                setup_fn(handle)

    def set_opponent_resampler(self, resampler) -> None:
        """
        Set the opponent resampler for the environment. This will reset the environment and apply the new opponent resampler.
        """
        self._opponent_resampler = resampler

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        for env in range(self.n_envs):
            self.agent_actions[env] = {}
            self._new_round(env)

        self._refresh_dynamic_layers()
        if self._enable_crate_potential:
            self._init_crate_potential()
        self._refresh_forecast_layers()
        self._update_prev_helpers()

        obs = self._build_observation()
        infos = [self._get_info(env) for env in range(self.n_envs)]
        return obs, infos

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.size == 1 and self.n_envs > 1:
            actions = np.repeat(actions, self.n_envs)
        if actions.size != self.n_envs:
            raise ValueError(f"expected {self.n_envs} actions, got {actions.size}")
        if np.any(actions < 0) or np.any(actions >= len(ACTIONS)):
            raise ValueError("action index out of range")

        for env in range(self.n_envs):
            self._advance(env, int(actions[env]))

        self._refresh_dynamic_layers()
        self._refresh_forecast_layers()
        obs = self._build_observation()

        if self._fixes:
            for env in np.nonzero(self.step_counts >= s.MAX_STEPS)[0]:
                if not self.agents[int(env)].dead:
                    self.agents[int(env)].add_event(e.SURVIVED_ROUND)

        rewards = np.zeros(self.n_envs, dtype=np.float32)
        for env in range(self.n_envs):
            self.visited[env, self.agents[env].x, self.agents[env].y] = True
            rewards[env] = self.reward_fn(env)

        terminated = np.fromiter((a.dead for a in self.agents), dtype=bool, count=self.n_envs)
        truncated = self.step_counts >= s.MAX_STEPS
        done = terminated | truncated

        infos = [self._get_info(env) for env in range(self.n_envs)]
        done_idx = np.nonzero(done)[0]
        for env in done_idx:
            self._finalize_replay_and_save(int(env))

        if self.auto_reset and done_idx.size:
            for env in done_idx:
                infos[int(env)]["terminal_observation"] = {
                    "grid_tensor": obs["grid_tensor"][int(env)].copy(),
                    "features": obs["features"][int(env)].copy(),
                }
            for env in done_idx:
                self._new_round(int(env))
            self._refresh_dynamic_layers()
            if self._enable_crate_potential:
                self._init_crate_potential(done_idx)
            self._refresh_forecast_layers()
            self._update_prev_helpers(done_idx)
            obs = self._build_observation()

        return obs, rewards, terminated, truncated, infos

    def _advance(self, env: int, action: int):
        """Game logic for one env (the Python-heavy, but cheap, part)."""
        self.step_counts[env] += 1

        shared = self._build_shared_state(env)
        actions = {}
        for handle, act_fn in zip(self.opponent_handles[env], self.opponent_act_fns[env]):
            handle.reset_game_events()
            if handle.dead:
                continue
            state = self._agent_state_dict(env, handle, shared)
            actions[handle] = act_fn(handle, state)

        agent = self.agents[env]
        agent.reset_game_events()
        actions[agent] = ACTIONS[action]
        self.agent_actions[env] = actions

        order = self.rng.permutation(len(self.active_agents[env]))
        for i in order:
            a = self.active_agents[env][i]
            act = actions.get(a, "WAIT")
            self._perform_agent_action(env, a, act)

        self._record_replay_step(env, order, actions)

        self._collect_coins(env)
        self._update_explosions(env)
        self._update_bombs(env)
        self._evaluate_explosions(env)

    def _start_replay_recording(self, env: int) -> None:
        if not getattr(self.args, "save_replay", False):
            self._replays[env] = None
            return
        self._replays[env] = {
            "round": int(self.rounds[env]),
            "arena": self.arena[env].copy(),
            "coins": [(int(x), int(y))
                      for (x, y) in self.coins_xy[env, : self.n_coins[env]].tolist()],
            "agents": [
                (h.name, h.score, h.bombs_left, (h.x, h.y))
                for h in self.active_agents[env]
            ],
            "actions": {h.name: [] for h in self.active_agents[env]},
            "permutations": [],
            "n_steps": 0,
        }

    def _record_replay_step(self, env: int, order: np.ndarray, actions: Dict["AgentHandle", str]) -> None:
        replay = self._replays[env]
        if replay is None:
            return
        replay["permutations"].append([int(i) for i in order])
        for a in self.active_agents[env]:
            replay["actions"][a.name].append(actions.get(a, "WAIT"))

    def _replay_save_path(self, env: int) -> str:
        replay_path = getattr(self.args, "replay", None)
        if replay_path:
            p = Path(str(replay_path))
            if self.n_envs > 1:
                p = p.with_name(f"{p.stem}_env{env:02d}{p.suffix}")
            return str(p)

        round_no = (self._replays[env].get("round", self.rounds[env])
                    if self._replays[env] else self.rounds[env])
        match_name = getattr(self.args, "match_name", None) or "match"
        replays_dir = Path("replays")
        replays_dir.mkdir(parents=True, exist_ok=True)
        if self.n_envs > 1:
            return str(replays_dir / f"{match_name}_env{env:02d}_round{round_no:03d}.pkl")
        return str(replays_dir / f"{match_name}_round{round_no:03d}.pkl")

    def _finalize_replay_and_save(self, env: int) -> None:
        replay = self._replays[env]
        if replay is None:
            return
        if not replay["permutations"]:
            self._replays[env] = None
            return
        replay["n_steps"] = int(self.step_counts[env])
        path = self._replay_save_path(env)
        try:
            with open(path, "wb") as f:
                pickle.dump(replay, f)
        finally:
            self._replays[env] = None

    def _build_shared_state(self, env: int) -> Dict[str, Any]:
        field = np.array(self.arena[env])

        explosion_map = np.zeros((self.width, self.height), dtype=np.float64)
        for ex in self.explosions[env]:
            if ex["stage"] == 0:
                t = ex["timer"] - 1
                for (x, y) in ex["coords"]:
                    if t > explosion_map[x, y]:
                        explosion_map[x, y] = t

        bombs_state = [((b["x"], b["y"]), b["timer"]) for b in self.bombs[env]]
        coins_state = [
            (int(self.coins_xy[env, ci, 0]), int(self.coins_xy[env, ci, 1]))
            for ci in range(self.n_coins[env]) if self.coins_collectable[env, ci]
        ]

        return {
            "field": field,
            "explosion_map": explosion_map,
            "bombs": bombs_state,
            "coins": coins_state,
        }

    def _agent_state_dict(self, env: int, handle: AgentHandle, shared: Dict[str, Any]) -> dict:
        return {
            "round": int(self.rounds[env]),
            "step": int(self.step_counts[env]),
            "field": shared["field"],
            "self": handle.get_state(),
            "others": [o.get_state() for o in self.active_agents[env] if o is not handle],
            "bombs": shared["bombs"],
            "coins": shared["coins"],
            "user_input": "WAIT",
            "explosion_map": shared["explosion_map"],
        }

    def get_state_for_agent(self, handle: AgentHandle) -> dict:
        for env in range(self.n_envs):
            if handle is self.agents[env] or handle in self.opponent_handles[env]:
                return self._agent_state_dict(env, handle, self._build_shared_state(env))
        raise ValueError("handle does not belong to this environment")

    def _ensure_coin_capacity(self, min_coins: int) -> None:
        """
        Ensure that the coin arrays have enough capacity for at least `min_coins` coins.
        If the current capacity is less than `min_coins`, the arrays are resized to accommodate
        the new capacity.
        """
        current = self.coins_xy.shape[1]
        if min_coins <= current:
            return
        E = self.n_envs
        new_xy = np.zeros((E, min_coins, 2), dtype=self.coins_xy.dtype)
        new_xy[:, :current] = self.coins_xy
        self.coins_xy = new_xy

        new_collectable = np.zeros((E, min_coins), dtype=bool)
        new_collectable[:, :current] = self.coins_collectable
        self.coins_collectable = new_collectable

    def _load_game_state(self, game_state: dict, env: int = 0) -> None:
        prev_round = int(self.rounds[env])
        new_round = int(game_state.get("round", prev_round))
        new_step = int(game_state.get("step", self.step_counts[env]))

        self.rounds[env] = new_round
        self.step_counts[env] = new_step

        self.arena[env] = np.asarray(game_state["field"], dtype=np.int8)

        _, self_score, self_bombs_left, (sx, sy) = game_state["self"]
        agent = self.agents[env]
        agent.x, agent.y = int(sx), int(sy)
        agent.bombs_left = bool(self_bombs_left)
        agent.score = self_score
        agent.dead = False

        others = game_state.get("others", [])
        by_name = {h.name: h for h in self.opponent_handles[env]}
        unused_handles = [h for h in self.opponent_handles[env]]
        for h in self.opponent_handles[env]:
            h.dead = True
        for (name, score, bombs_left, (ox, oy)) in others:
            h = by_name.get(name)
            if h is None and unused_handles:
                h = unused_handles[0]
            if h is None:
                continue
            if h in unused_handles:
                unused_handles.remove(h)
            h.x, h.y = int(ox), int(oy)
            h.bombs_left = bool(bombs_left)
            h.score = score
            h.dead = False

        self.active_agents[env] = [agent] + [h for h in self.opponent_handles[env] if not h.dead]

        bombs = game_state.get("bombs", [])
        self.bombs[env] = [
            {"x": int(x), "y": int(y), "timer": int(timer), "owner": None}
            for ((x, y), timer) in bombs
        ]

        coins = game_state.get("coins", [])
        self._ensure_coin_capacity(len(coins))
        self.n_coins[env] = len(coins)
        if coins:
            self.coins_xy[env, :len(coins)] = coins
            self.coins_collectable[env, :len(coins)] = True
        self.coins_collectable[env, len(coins):] = False

        is_new_round = (new_round != prev_round) or (new_step <= 1)
        if is_new_round:
            self._initial_crate_count[env] = int(np.sum(self.arena[env] == 1))
            self._initial_coin_count[env] = len(coins)

        explosion_map = np.asarray(
            game_state.get("explosion_map", np.zeros((self.width, self.height)))
        )
        cells_by_timer: Dict[int, List[Tuple[int, int]]] = {}
        for x, y in np.argwhere(explosion_map > 0):
            t = int(explosion_map[x, y])
            cells_by_timer.setdefault(t, []).append((int(x), int(y)))
        self.explosions[env] = [
            {
                "coords": coords,
                "coords_set": set(coords),
                "owner": None,
                "timer": t + 1,
                "stage": 0,
            }
            for t, coords in cells_by_timer.items()
        ]

    def observation_from_game_state(self, game_state: dict, env: int = 0) -> dict:
        """Single-env utility: returns the un-batched observation for `env`."""
        self._load_game_state(game_state, env)
        self._rebuild_static_layers(env)
        self._refresh_dynamic_layers()
        if self._enable_crate_potential:
            self._init_crate_potential([env])
        self._refresh_forecast_layers()

        obs = self._build_observation()
        return {
            "grid_tensor": obs["grid_tensor"][env],
            "features": obs["features"][env],
        }

    def _tile_is_free(self, env: int, x, y) -> bool:
        if self.arena[env, x, y] != 0:
            return False
        for b in self.bombs[env]:
            if b["x"] == x and b["y"] == y:
                return False
        for a in self.active_agents[env]:
            if a.x == x and a.y == y:
                return False
        return True

    def _move_agent(self, env: int, handle: AgentHandle, new_x, new_y):
        if handle is self.agents[env]:
            self.grid_tensor[env, SELF_LAYER, handle.x, handle.y] = 0.0
            self.grid_tensor[env, SELF_LAYER, new_x, new_y] = 1.0
        handle.x, handle.y = new_x, new_y

    def _place_bomb(self, env: int, agent: AgentHandle):
        self.bombs[env].append(
            {"x": agent.x, "y": agent.y, "timer": s.BOMB_TIMER, "owner": agent}
        )
        agent.bombs_left = False

    def _perform_agent_action(self, env: int, agent: AgentHandle, action):
        x, y = agent.x, agent.y
        if action == "UP" and self._tile_is_free(env, x, y - 1):
            self._move_agent(env, agent, x, y - 1)
            agent.add_event(e.MOVED_UP)
        elif action == "DOWN" and self._tile_is_free(env, x, y + 1):
            self._move_agent(env, agent, x, y + 1)
            agent.add_event(e.MOVED_DOWN)
        elif action == "LEFT" and self._tile_is_free(env, x - 1, y):
            self._move_agent(env, agent, x - 1, y)
            agent.add_event(e.MOVED_LEFT)
        elif action == "RIGHT" and self._tile_is_free(env, x + 1, y):
            self._move_agent(env, agent, x + 1, y)
            agent.add_event(e.MOVED_RIGHT)
        elif action == "BOMB" and agent.bombs_left:
            self._place_bomb(env, agent)
            agent.add_event(e.BOMB_DROPPED)
        elif action == "WAIT":
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def _collect_coins(self, env: int):
        n = self.n_coins[env]
        collectable_idx = np.nonzero(self.coins_collectable[env, :n])[0]
        if len(collectable_idx) == 0 or not self.active_agents[env]:
            return
        active_pos = np.array([[a.x, a.y] for a in self.active_agents[env]], dtype=np.int64)
        coin_pos = self.coins_xy[env, collectable_idx]
        eq = (coin_pos[:, None, :] == active_pos[None, :, :]).all(-1)
        hit_coin, hit_agent = np.nonzero(eq)
        seen = set()
        for ci_local, ai in zip(hit_coin, hit_agent):
            ci = int(collectable_idx[ci_local])
            if ci in seen:
                continue
            seen.add(ci)
            cx, cy = self.coins_xy[env, ci]
            handle = self.active_agents[env][int(ai)]
            self.coins_collectable[env, ci] = False
            self.grid_tensor[env, COIN_LAYER, cx, cy] = 0.0
            handle.update_score(s.REWARD_COIN)
            handle.add_event(e.COIN_COLLECTED)

    def _update_explosions(self, env: int):
        if not self.explosions[env]:
            return
        remaining = []
        for ex in self.explosions[env]:
            ex["timer"] -= 1
            if ex["timer"] <= 0:
                ex["stage"] += 1
                if ex["stage"] == 1:
                    ex["timer"] = _EXPLOSION_STAGE1_TICKS
                    if ex["owner"] is not None:
                        ex["owner"].bombs_left = True
                else:
                    ex["stage"] = None
            if ex["stage"] is not None:
                remaining.append(ex)
        self.explosions[env] = remaining

    def _update_bombs(self, env: int):
        """RECONSTRUCTED tail (your file was cut off inside this method)."""
        bombs = self.bombs[env]
        if not bombs:
            return
        remaining = []
        g = self.grid_tensor[env]
        for b in bombs:
            if b["timer"] <= 0:
                owner = b["owner"]
                owner.add_event(e.BOMB_EXPLODED)
                blast = self.PRECOMPUTED_BLAST_COORDS[(b["x"], b["y"])]
                for (x, y) in blast:
                    if self.arena[env, x, y] == 1:
                        self.arena[env, x, y] = 0
                        g[CRATE_LAYER, x, y] = 0.0
                        owner.add_event(e.CRATE_DESTROYED)
                        if self._enable_crate_potential:
                            g[CRATE_POTENTIAL_LAYER] -= self._blast_tensor[:, :, x, y]
                        for ci in range(self.n_coins[env]):
                            if (not self.coins_collectable[env, ci]
                                    and self.coins_xy[env, ci, 0] == x
                                    and self.coins_xy[env, ci, 1] == y):
                                self.coins_collectable[env, ci] = True
                                g[COIN_LAYER, x, y] = 1.0
                                owner.add_event(e.COIN_FOUND)
                self.explosions[env].append({
                    "coords": blast,
                    "coords_set": set(blast),
                    "owner": owner,
                    "timer": s.EXPLOSION_TIMER,
                    "stage": 0,
                })
            else:
                b["timer"] -= 1
                remaining.append(b)
        self.bombs[env] = remaining

    def _evaluate_explosions(self, env: int):
        """RECONSTRUCTED (was beyond the truncation point)."""
        explosions = self.explosions[env]
        if not explosions:
            return
        active = self.active_agents[env]
        kill_reward = getattr(s, "REWARD_KILL", 0)
        if self._fixes:
            hit = []
            for ex in explosions:
                if ex["stage"] != 0:
                    continue
                cells = ex["coords_set"]
                owner = ex["owner"]
                for a in active:
                    if (a.x, a.y) in cells:
                        if a not in hit:
                            hit.append(a)
                        if a is owner:
                            a.add_event(e.KILLED_SELF)
                        elif owner is not None:
                            owner.add_event(e.KILLED_OPPONENT)
                            if kill_reward:
                                owner.update_score(kill_reward)
            for a in hit:
                a.dead = True
                a.add_event(e.GOT_KILLED)
            self.active_agents[env] = [a for a in active if not a.dead]
            if hit:
                for a in self.active_agents[env]:
                    for _ in hit:
                        a.add_event(e.OPPONENT_ELIMINATED)
            return
        for ex in explosions:
            if ex["stage"] != 0:
                continue
            cells = ex["coords_set"]
            owner = ex["owner"]
            for a in active:
                if a.dead:
                    continue
                if (a.x, a.y) in cells:
                    a.dead = True
                    a.add_event(e.GOT_KILLED)
                    if a is owner:
                        suicide = getattr(e, "KILLED_SELF", None)
                        if suicide is not None:
                            a.add_event(suicide)
                    else:
                        owner.add_event(e.KILLED_OPPONENT)
                        if kill_reward:
                            owner.update_score(kill_reward)
        self.active_agents[env] = [a for a in active if not a.dead]

    def shaped_reward(self, env: int) -> float:
        """
        Default reward. Event rewards and shaping coefficients come from
        self._event_rewards / self._*_coef (see set_reward_config), not
        module constants, so they can be changed live mid-run.
        """
        agent = self.agents[env]
        total = 0.0
        for ev in agent.events:
            total += self._event_rewards.get(ev, 0.0)

        coin_now = self._coin_distance_now(env)
        prev = self._prev_coin_dist[env]
        if prev == prev and coin_now is not None:
            total += self._coin_shaping_coef * (prev - coin_now)
        self._prev_coin_dist[env] = np.nan if coin_now is None else coin_now

        crate_now = self._crate_distance_now(env) if agent.bombs_left else None
        prev = self._prev_crate_dist[env]
        if prev == prev and crate_now is not None:
            total += self._crate_shaping_coef * (prev - crate_now)
        self._prev_crate_dist[env] = np.nan if crate_now is None else crate_now

        danger_now = self._bomb_danger_now(env)
        total += self._escape_bonus_coef * (self._prev_bomb_danger[env] - danger_now)
        total -= self._danger_penalty_coef * danger_now
        self._prev_bomb_danger[env] = danger_now

        trap_now = self._trapped_opponent_distance_now(env)
        total += self._trap_shaping_coef * (self._prev_trap_dist[env] - trap_now)
        self._prev_trap_dist[env] = trap_now
        return total

    def _coin_distance_now(self, env: int) -> Optional[float]:
        a = self.agents[env]
        d = self.grid_tensor[env, COIN_DISTANCE_LAYER, a.x, a.y]
        return None if d < 0 else float(d)

    def _crate_distance_now(self, env: int) -> Optional[float]:
        a = self.agents[env]
        d = self.grid_tensor[env, CRATE_DISTANCE_LAYER, a.x, a.y]
        return None if d < 0 else float(d)

    def _bomb_danger_now(self, env: int) -> float:
        a = self.agents[env]
        onset = self.grid_tensor[env, DANGER_ONSET_LAYER, a.x, a.y]
        if onset < 0:
            return 0.0
        return 1.0 / (1.0 + float(onset))

    def _trapped_opponent_distance_now(self, env: int) -> float:
        a = self.agents[env]
        d = self.grid_tensor[env, OPPONENTS_LEAST_DISTANCE_LAYER, a.x, a.y]
        return float(d) if d >= 0 else self._T_HORIZON

    def _get_info(self, env: int) -> Dict[str, Any]:
        """RECONSTRUCTED -- adapt to whatever your original returned."""
        return {
            "env_id": env,
            "round": int(self.rounds[env]),
            "step": int(self.step_counts[env]),
            "score": self.agents[env].score,
            "events": list(self.agents[env].events),
        }

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean array of shape (n_envs, 6) indicating which actions
        are valid. MaskablePPO will use this to prevent illegal moves.
        """
        E = self.n_envs
        W, H = self.width, self.height
        gt = self.grid_tensor

        masks = np.ones((E, len(ACTIONS)), dtype=np.bool_)

        ax = np.fromiter((a.x for a in self.agents), dtype=np.int64, count=E)
        ay = np.fromiter((a.y for a in self.agents), dtype=np.int64, count=E)

        occ_layer = gt[:, OCCUPIED_MAP_LAYERS[0]] > 0
        dng_layer = gt[:, DANGER_MAP_LAYERS[0]] > 0

        def is_blocked(dx, dy):
            tx = np.clip(ax + dx, 0, W - 1)
            ty = np.clip(ay + dy, 0, H - 1)
            oob = (ax + dx < 0) | (ax + dx >= W) | (ay + dy < 0) | (ay + dy >= H)
            return oob | occ_layer[np.arange(E), tx, ty] | dng_layer[np.arange(E), tx, ty]

        masks[:, ACTION_INDICES["UP"]] = ~is_blocked(0, -1)
        masks[:, ACTION_INDICES["DOWN"]] = ~is_blocked(0, 1)
        masks[:, ACTION_INDICES["LEFT"]] = ~is_blocked(-1, 0)
        masks[:, ACTION_INDICES["RIGHT"]] = ~is_blocked(1, 0)

        bombs_left = np.fromiter((a.bombs_left for a in self.agents), dtype=np.bool_, count=E)
        masks[:, ACTION_INDICES["BOMB"]] = bombs_left

        return masks