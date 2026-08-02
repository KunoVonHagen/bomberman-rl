"""
Fast, self-contained Gymnasium environment for training Bomberman RL agents.

THIS MODULE WAS FULLY GENERATED USING CLAUDE SONNET 5. SINCE ALL

This module intentionally does NOT depend on environment.py / agents.py / items.py.
Those modules exist to support the full game (GUI rendering via pygame, replay
recording, per-agent subprocess/multiprocessing backends, file logging, ...),
none of which is needed -- or wanted -- in a tight training loop. Pulling them
in costs real time and memory for every environment instance (pygame image
loads, log file handles, defaultdict bookkeeping, replay dict growth) and,
more importantly, the original gym_environment.py re-derived the *entire*
observation tensor and a fresh copy of the game-state dict for every single
agent on every single step, even though almost none of that data changes
between one agent's turn and the next.

Design of this rewrite:

  * No pygame, no logging, no multiprocessing, no replay recording. Agents
    are represented by a tiny `AgentHandle` (a handful of plain attributes)
    instead of the full `agents.Agent`, which used to load two 30x30 sprite
    images per agent just to sit unused in headless training.

  * Game *logic* (movement rules, bomb/explosion timing, coin collection,
    scoring, kill attribution, round-end conditions) is a straight,
    behavior-preserving port of GenericWorld/BombeRLeWorld from
    environment.py. Every quirk of the original (e.g. an agent overlapped by
    two simultaneous explosions gets scored against twice; bomb-danger
    channels are overwritten rather than merged when two bombs share a
    countdown value; dead opponents keep occupying their last cell on the
    observation tensor because the original code never filtered them out)
    is preserved on purpose.

  * The `game_state` dict handed to agent callbacks has the exact same shape
    as `BombeRLeWorld.get_state_for_agent`. The expensive parts of building
    it (`field`, `explosion_map`, `bombs`, `coins`) are computed *once per
    step* and shared across every agent's dict instead of being rebuilt from
    scratch per agent.

  * The observation tensor is a persistent buffer, not reallocated every
    step. Layers that rarely change (walls -- never; crates and coins --
    only at the handful of cells that are actually destroyed/collected;
    the acting agent's own position) are updated incrementally with O(1)
    point writes instead of being rebuilt via a full-array scan. Layers that
    are inherently "shift every tick" (bomb position/danger channels change
    index every time a bomb's timer ticks down; explosion channels the same)
    or that combine several small entities (enemy positions/union danger
    zone/can-place-bomb) are still recomputed each step, but only over the
    handful of active bombs/explosions/agents -- never via `np.where` over
    the whole arena, and never after a `tensor.fill(0)` of all 19+ layers.

  * Blast propagation only ever depends on the (static) wall layout, never
    on crates, so the blast-coordinate/blast-map lookup tables are computed
    exactly once, from the wall pattern alone, before the first round is
    even generated -- and reused for the lifetime of the environment.

NOTE on returned buffers: `obs["grid_tensor"]` and `obs["features"]` are
views into buffers owned by this environment and are mutated in place on the
next `step()`/`reset()` call. This avoids an allocation + full copy every
single step. Downstream code that needs to retain an observation across
steps (e.g. for logging) should copy it explicitly (`obs["grid_tensor"].copy()`).
Standard RL libraries (SB3, etc.) already copy observations into their own
rollout/replay buffers immediately, so this is safe for normal training use.
"""

from collections import namedtuple
from typing import List, Tuple, Callable, Optional, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

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

# Kept for drop-in compatibility with callers that construct
# `WorldArgs(...)` for this environment. Deliberately *not* imported from
# environment.py, since importing that module pulls in pygame/agents/items.
WorldArgs = namedtuple(
    "WorldArgs",
    ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay",
     "make_video", "continue_without_training", "log_dir", "save_stats",
     "match_name", "seed", "silence_errors", "scenario"],
)


# ---------------------------------------------------------------------------
# Lightweight stand-ins for agents.Agent / agents.RLAgent and
# items.Coin / items.Bomb / items.Explosion.
# ---------------------------------------------------------------------------

class _NullLogger:
    """No-op logger, API-compatible with the `self.logger` opponent callback
    code expects (mirrors agents.RLAgentLogger, minus the RLAgent/Agent
    machinery around it)."""
    __slots__ = ()

    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


_NULL_LOGGER = _NullLogger()


class AgentHandle:
    """Minimal per-agent record. Carries only what the game logic, the
    game_state dict, and the reward computation actually use -- no sprites,
    no per-agent log files, no lifetime statistics dict."""

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


class BombermanGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

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

    # Explosion stage-1 ("smoke", no-longer-dangerous) duration. In the
    # original items.Explosion this comes from `len(Explosion.ASSETS[1])`
    # (2 animation frames) -- a rendering constant that the original game
    # loop nonetheless used to drive real explosion lifetime. Reproduced
    # here as an explicit constant since we no longer load any sprites.
    _EXPLOSION_STAGE1_TICKS = 2

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

        # -- Agents ----------------------------------------------------
        self.agent = AgentHandle("RLAgent")
        self.opponent_handles: List[AgentHandle] = []
        self.opponent_act_fns: List[Callable] = []
        for i, (setup_fn, act_fn) in enumerate(opponents):
            # NOTE: mirrors the original gym_environment.py, which stored
            # `setup_fn` but never called it. Preserved as-is rather than
            # silently changing opponent-initialization behavior.
            self.opponent_handles.append(AgentHandle(f"OpponentAgent{i}"))
            self.opponent_act_fns.append(act_fn)
        self.all_agents: List[AgentHandle] = [self.agent] + self.opponent_handles
        self.n_agents = len(self.all_agents)
        self.active_agents: List[AgentHandle] = list(self.all_agents)

        self.reward_fn = reward_fn or self.shaped_reward

        # -- Board dimensions -------------------------------------------
        self.width, self.height = s.COLS, s.ROWS
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.n_observation_layers = 8 + 2 * s.BOMB_TIMER + s.EXPLOSION_TIMER
        self._BT = s.BOMB_TIMER
        self._ET = s.EXPLOSION_TIMER

        # Persistent buffers -- allocated once, mutated in place forever.
        self.grid_tensor = np.zeros((self.n_observation_layers, self.width, self.height), dtype=np.float32)
        self._centered_tensor = np.zeros_like(self.grid_tensor)
        self.features = np.zeros(FEATURES_DIM, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "grid_tensor": spaces.Box(
                low=0,
                high=1,
                shape=(self.n_observation_layers, 17, 17),
                dtype=np.float32,
            ),
            "features": spaces.Box(
                low=-1,
                high=1,
                shape=(FEATURES_DIM,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # -- Static blast lookup tables -----------------------------------
        # Blast propagation only stops at walls, never at crates, so this
        # only depends on the (deterministic, RNG-independent) wall layout
        # and can be computed once, before any round exists, and reused
        # forever -- crate destruction never invalidates it.
        wall_mask = self._build_wall_mask()
        self._wall_layer = np.where(wall_mask == -1, 1.0, 0.0).astype(np.float32)
        self._empty_map = np.zeros((self.width, self.height), dtype=np.float32)
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

        # -- Round-scoped state, (re)initialised in new_round() -----------
        self.round = 0
        self.step_count = 0
        self.arena = np.zeros((self.width, self.height), dtype=np.int8)
        self.coins_xy = np.zeros((0, 2), dtype=np.int64)
        self.coins_collectable = np.zeros((0,), dtype=bool)
        self.bombs: List[dict] = []
        self.explosions: List[dict] = []

        # Reward-shaping auxiliaries.
        self.previous_visited_count = 1
        self.visited = np.zeros((self.width, self.height), dtype=bool)
        self.previous_features = None

        self.agent_actions = {}

        self.new_round()

    # ------------------------------------------------------------------
    # Static-layout / blast precomputation helpers
    # ------------------------------------------------------------------

    def _build_wall_mask(self) -> np.ndarray:
        """Reproduces the wall portion of BombeRLeWorld.build_arena. Walls
        never depend on the RNG or on crate placement, so this is computed
        once and never touched again."""
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
        """Reproduces items.Bomb.get_blast_coords without needing a Bomb
        object (and therefore without needing to import items.py / pygame)."""
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

    # ------------------------------------------------------------------
    # Round setup
    # ------------------------------------------------------------------

    def _generate_round_layout(self):
        """Reproduces BombeRLeWorld.build_arena (crate placement, coin
        placement, start-position clearing, start-position assignment)."""
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
            handle.total_score = handle.total_score  # persists across rounds, like Agent.total_score
            handle.bombs_left = True
            handle.events = []

        self.active_agents = list(self.all_agents)

        self._rebuild_full_grid_tensor()

        self.previous_visited_count = 1
        self.visited.fill(False)
        self.previous_features = None
        self.agent_actions = {}

    # ------------------------------------------------------------------
    # Observation tensor maintenance
    # ------------------------------------------------------------------

    def _rebuild_full_grid_tensor(self):
        """Full rebuild -- only ever called once per round (not once per
        step), so its cost is amortized over the whole round."""
        gt = self.grid_tensor
        gt.fill(0)
        gt[0] = self._wall_layer
        gt[1] = np.where(self.arena == 1, 1.0, 0.0)
        collectable_idx = np.nonzero(self.coins_collectable)[0]
        for ci in collectable_idx:
            x, y = self.coins_xy[ci]
            gt[2, x, y] = 1.0
        gt[3, self.agent.x, self.agent.y] = 1.0
        self._refresh_dynamic_layers()

    def _refresh_dynamic_layers(self):
        """Recomputes only the channels whose content is inherently
        collective (all enemies) or inherently shifts every tick (bomb
        timer channels, explosion timer channels). Walls/crates/coins/self
        position (layers 0-3) are maintained incrementally elsewhere and
        are *not* touched here."""
        gt = self.grid_tensor
        BT = self._BT

        gt[4:].fill(0)

        ax, ay = self.agent.x, self.agent.y
        gt[4] = self.PRECOMPUTED_BLAST_MAP.get((ax, ay), self._empty_map)

        for h in self.opponent_handles:
            ex_, ey_ = h.x, h.y
            gt[5, ex_, ey_] = 1.0
            gt[6] += self.PRECOMPUTED_BLAST_MAP.get((ex_, ey_), self._empty_map)
            gt[7, ex_, ey_] = 1.0 if h.bombs_left else 0.0

        gt[6] = np.where(gt[6] > 0, 1.0, 0.0)
        gt[7, ax, ay] = 1.0 if self.agent.bombs_left else 0.0

        for b in self.bombs:
            pos_ch = 8 + b["timer"]
            danger_ch = 8 + BT + b["timer"]
            gt[pos_ch, b["x"], b["y"]] = 1.0
            # Overwrite (not accumulate), matching the original: if two
            # bombs share a countdown value, only the last one processed
            # is reflected in that channel.
            gt[danger_ch] = self.PRECOMPUTED_BLAST_MAP.get((b["x"], b["y"]), self._empty_map)

        for ex in self.explosions:
            if ex["stage"] == 0:
                ch = 7 + 2 * BT + ex["timer"]
                for (x, y) in ex["coords"]:
                    gt[ch, x, y] = 1.0

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

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_actions = {}

        self.new_round()

        grid_tensor = self._get_centered_tensor()
        self.features = get_features(grid_tensor)
        obs = {"grid_tensor": grid_tensor, "features": self.features}
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.previous_features = self.features
        self.step_count += 1

        # -- 1) Decide every agent's action using the PRE-step state ------
        shared = self._build_shared_state()
        actions = {}
        for handle, act_fn in zip(self.opponent_handles, self.opponent_act_fns):
            handle.reset_game_events()
            if handle.dead:
                # Dead agents' actions are never applied -- skip the call
                # entirely rather than computing and discarding it.
                continue
            state = self._agent_state_dict(handle, shared)
            actions[handle] = act_fn(handle, state)

        self.agent.reset_game_events()
        actions[self.agent] = self.ACTIONS[action]
        self.agent_actions = actions

        # -- 2) Apply actions in random turn order -------------------------
        order = self.rng.permutation(len(self.active_agents))
        for i in order:
            a = self.active_agents[i]
            act = actions.get(a, "WAIT")
            self._perform_agent_action(a, act)

        # -- 3) Progress world elements (same order as step_world) --------
        self._collect_coins()
        self._update_explosions()
        self._update_bombs()
        self._evaluate_explosions()

        # -- 4) Refresh only the channels that can have changed ------------
        self._refresh_dynamic_layers()

        grid_tensor = self._get_centered_tensor()
        self.features = get_features(grid_tensor)
        obs = {"grid_tensor": grid_tensor, "features": self.features}

        self.visited[self.agent.x, self.agent.y] = True

        reward = self.reward_fn()

        terminated = self.agent.dead
        truncated = self.step_count >= s.MAX_STEPS

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Game-state dict construction (shared across agents per step)
    # ------------------------------------------------------------------

    def _build_shared_state(self) -> Dict[str, Any]:
        """Computes the parts of the game_state dict that are identical for
        every agent exactly once per step, instead of once per agent."""
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
        """Public single-agent accessor, kept for API parity with
        BombeRLeWorld.get_state_for_agent (e.g. used by
        check_game_state_conversion_accuracy)."""
        return self._agent_state_dict(handle, self._build_shared_state())

    # ------------------------------------------------------------------
    # Core game logic (behavior-preserving port of GenericWorld)
    # ------------------------------------------------------------------

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
            self.grid_tensor[3, handle.x, handle.y] = 0.0
            self.grid_tensor[3, new_x, new_y] = 1.0
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
                self.grid_tensor[2, cx, cy] = 0.0
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
                    ex["timer"] = self._EXPLOSION_STAGE1_TICKS
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
                        self.grid_tensor[1, x, y] = 0.0
                        owner.add_event(e.CRATE_DESTROYED)
                        if len(self.coins_xy):
                            coin_matches = np.nonzero(
                                (self.coins_xy[:, 0] == x) & (self.coins_xy[:, 1] == y)
                            )[0]
                            for ci in coin_matches:
                                if not self.coins_collectable[ci]:
                                    self.coins_collectable[ci] = True
                                    self.grid_tensor[2, x, y] = 1.0
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

    # ------------------------------------------------------------------
    # Misc helpers / API parity with the original gym_environment.py
    # ------------------------------------------------------------------

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

        disable_coin_away_movement_penalty = False

        for event in self.agent.events:
            if event == e.COIN_COLLECTED:
                disable_coin_away_movement_penalty = True
            reward += EVENT_REWARDS.get(event, 0)

        visited_count = np.sum(self.visited)
        new_visited = visited_count - self.previous_visited_count
        self.previous_visited_count = visited_count

        if new_visited > 0:
            reward += 0.02

        feature_diff = np.sign(self.features - self.previous_features)

        for feature_index, reward_function in FEATURE_REWARDS.items():
            reward += reward_function(self.features[feature_index])

        for feature_diff_index, diff_reward in FEATURE_DIFF_REWARDS.items():
            if feature_diff_index == 2 and disable_coin_away_movement_penalty:
                reward += abs(diff_reward)
                continue
            reward += diff_reward * feature_diff[feature_diff_index]

        return reward

    def render(self):
        pass

    def close(self):
        pass