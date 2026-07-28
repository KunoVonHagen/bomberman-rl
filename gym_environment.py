from typing import List, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from agents import RLAgent
from events import WAITED, GOT_KILLED, INVALID_ACTION, KILLED_OPPONENT, KILLED_SELF, OPPONENT_ELIMINATED, SURVIVED_ROUND, \
    COIN_COLLECTED, CRATE_DESTROYED, BOMB_DROPPED, BOMB_EXPLODED, MOVED_LEFT, MOVED_UP, MOVED_DOWN, MOVED_RIGHT, COIN_FOUND
import settings as s
from environment import BombeRLeWorld, WorldArgs, Trophy
from items import Bomb

from agent_code.my_agent.features import get_features, FEATURES_DIM, cell_attributes, \
    get_closest_target_directions_and_distances

EVENT_REWARDS = {
    MOVED_LEFT: 0,
    MOVED_RIGHT: 0,
    MOVED_UP: 0,
    MOVED_DOWN: 0,
    WAITED: -0.2,
    INVALID_ACTION: -10,

    BOMB_DROPPED: 0,
    BOMB_EXPLODED: 0,

    CRATE_DESTROYED: 1,
    COIN_FOUND: 0,
    COIN_COLLECTED: 1,

    KILLED_OPPONENT: 15,
    KILLED_SELF: -5,

    GOT_KILLED: -5,
    OPPONENT_ELIMINATED: 0,
    SURVIVED_ROUND: 20,
}

MAX_REWARD = max(EVENT_REWARDS.values())
EVENT_REWARDS = {k: np.float32(v/MAX_REWARD) for k, v in EVENT_REWARDS.items()}


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

    def __init__(
        self,
        args: WorldArgs,
        opponents: List[Tuple[str, bool]],
        reward_fn=None,
        render_mode=None,
    ):
        super().__init__()

        self.agent = RLAgent("RLAgent")
        self.world = BombeRLeWorld(args, [self.agent] + opponents)
        self.opponents = [a for a in self.world.agents if a != self.agent]
        self.render_mode = render_mode

        self.reward_fn = reward_fn or self.default_reward

        self.world.new_round()

        H, W = self.world.arena.shape

        self.n_observation_layers = 7 + 2 * s.BOMB_TIMER + s.EXPLOSION_TIMER

        self.grid_tensor = np.zeros((self.n_observation_layers, H, W), dtype=np.float32)
        self.features = []

        # Observation Space:
        # Map Layers:
        #   walls
        #   crates
        #   coins
        #   self
        #   self_danger_zone
        #   opponents
        #   opponents_danger_zone
        #   bombs (s.BOMB_TIMER)
        #   bombs_danger_zone (s.BOMB_TIMER)
        #   explosions (s.EXPLOSION_TIMER)
        # Distilled Features
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
                dtype=np.float32
            )
        })

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.agent_actions = {}

        non_wall_cells = self.world.arena != -1

        precomputed_blast_coords = {}
        for x,y in np.argwhere(non_wall_cells):
            bomb = Bomb(pos=(x,y), owner=None, timer=0, power=s.BOMB_POWER, bomb_sprite=None)
            precomputed_blast_coords[(x, y)] = bomb.get_blast_coords(self.world.arena)

        self.PRECOMPUTED_BLAST_MAP = {}
        for k, v in precomputed_blast_coords.items():
            blast_map = np.zeros_like(self.world.arena, dtype=np.int8)
            for bx, by in v:
                blast_map[bx, by] = 1
            self.PRECOMPUTED_BLAST_MAP[k] = blast_map


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_actions = {}

        if seed is not None:
            self.world.rng = np.random.default_rng(seed)

        self.world.new_round()

        grid_tensor = self._get_grid_tensor()
        features = get_features(grid_tensor)
        obs = {"grid_tensor": grid_tensor, "features": features}

        info = self._get_info()

        return obs, info

    def get_opponents_actions(self):
        for a in self.opponents:
            state = self.world.get_state_for_agent(a)
            a.store_game_state(state)
            a.reset_game_events()
            if a.available_think_time > 0:
                a.act(state)

            # Give agents time to decide

        for i in range(len(self.opponents)):
            a = self.opponents[i]
            if a.available_think_time > 0:
                try:
                    action, think_time = a.wait_for_act()
                except KeyboardInterrupt:
                    # Stop the game
                    raise
                except:
                    if not self.world.args.silence_errors:
                        raise
                    # Agents with errors cannot continue
                    action = "ERROR"
                    think_time = float("inf")

                if think_time > a.available_think_time:
                    next_think_time = a.base_timeout - (think_time - a.available_think_time)
                    action = "WAIT"
                    a.trophies.append(Trophy.time_trophy)
                    a.available_think_time = next_think_time
                else:
                    a.available_think_time = a.base_timeout
            else:
                a.available_think_time += a.base_timeout
                action = "WAIT"

            self.agent_actions[a] = action


    def step(self, action):
        self.get_opponents_actions()
        self.agent.reset_game_events()
        self.agent_actions[self.agent] = self.ACTIONS[action]

        self.world.step_world(self.agent_actions)

        grid_tensor = self._get_grid_tensor()
        self.features = get_features(self.grid_tensor)
        obs = {"grid_tensor": grid_tensor, "features": self.features}
        

        reward = self.reward_fn()

        terminated = self.agent.dead
        truncated = self.world.step >= s.MAX_STEPS

        info = self._get_info()

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def _get_grid_tensor(self):
        self.grid_tensor.fill(0)

        # walls
        self.grid_tensor[0] = np.where(self.world.arena == -1, 1, 0)

        # crates
        self.grid_tensor[1] = np.where(self.world.arena == 1, 1, 0)

        # coins
        for coin in self.world.coins:
            self.grid_tensor[2, coin.x, coin.y] = 1

        # self
        self.grid_tensor[3, self.agent.x, self.agent.y] = 1

        # self danger zone
        self.grid_tensor[4] = self.PRECOMPUTED_BLAST_MAP.get((self.agent.x, self.agent.y), np.zeros_like(self.world.arena, dtype=np.int8))

        # enemies
        for enemy in self.opponents:
            # positions
            self.grid_tensor[5, enemy.x, enemy.y] = 1
            # danger zones
            self.grid_tensor[6] += self.PRECOMPUTED_BLAST_MAP.get((enemy.x, enemy.y), np.zeros_like(self.world.arena, dtype=np.int8))

        self.grid_tensor[6] = np.where(self.grid_tensor[6] > 0, 1, 0)

        # bombs
        for bomb in self.world.bombs:
            # locations
            self.grid_tensor[6 + bomb.timer + 1, bomb.x, bomb.y] = 1
            #
            self.grid_tensor[6 + s.BOMB_TIMER + bomb.timer + 1] = self.PRECOMPUTED_BLAST_MAP.get((bomb.x, bomb.y), np.zeros_like(self.world.arena, dtype=np.int8))

        # explosions
        for explosion in self.world.explosions:
            for x, y in explosion.blast_coords:
                self.grid_tensor[6 + 2*s.BOMB_TIMER + explosion.timer, x, y] = 1

        return self.grid_tensor

    def _get_info(self):

        return {
            "step": self.world.step,
            "score": self.agent.score,
            "alive": not self.agent.dead,
        }

    def default_reward(self):
        reward = 0

        for event in self.agent.events:
            reward += EVENT_REWARDS.get(event, 0)

        if not self.agent.dead:
            reward += 0.001/MAX_REWARD

        is_safe = cell_attributes((self.agent.x, self.agent.y), self.grid_tensor)[10] == 0
        (_,_), closest_coin_distance = get_closest_target_directions_and_distances(
            (self.agent.x, self.agent.y),
            (self.grid_tensor[2] == 1)[None],
            self.grid_tensor[0] + self.grid_tensor[1] > 0
        )[0]

        safety_reward = 0.005/MAX_REWARD if is_safe else 0
        coin_closeness_reward = 0.01 * (1/(closest_coin_distance + 1)) / MAX_REWARD if closest_coin_distance > 0 else 0

        reward += safety_reward + coin_closeness_reward

        return reward

    def render(self):
        pass

    def close(self):
        self.world.end()