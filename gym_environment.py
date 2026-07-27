from typing import List, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from agents import RLAgent
from events import WAITED, GOT_KILLED, INVALID_ACTION, KILLED_OPPONENT, KILLED_SELF, OPPONENT_ELIMINATED, SURVIVED_ROUND, \
    COIN_COLLECTED, CRATE_DESTROYED, BOMB_DROPPED, BOMB_EXPLODED, MOVED_LEFT, MOVED_UP, MOVED_DOWN, MOVED_RIGHT, COIN_FOUND
import settings as s
from environment import BombeRLeWorld, WorldArgs, Trophy


EVENT_REWARDS = {
    MOVED_LEFT: 0,
    MOVED_RIGHT: 0,
    MOVED_UP: 0,
    MOVED_DOWN: 0,
    WAITED: -0.1,
    INVALID_ACTION: -1,

    BOMB_DROPPED: 0,
    BOMB_EXPLODED: 0,

    CRATE_DESTROYED: 1,
    COIN_FOUND: 1,
    COIN_COLLECTED: 5,

    KILLED_OPPONENT: 5,
    KILLED_SELF: -10,

    GOT_KILLED: -5,
    OPPONENT_ELIMINATED: 5,
    SURVIVED_ROUND: 1,
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

        # Example observation:
        # channels:
        #   walls
        #   crates
        #   coins
        #   bombs
        #   explosions
        #   self
        #   opponents
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(7, H, W),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        self.agent_actions = {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_actions = {}

        if seed is not None:
            self.world.rng = np.random.default_rng(seed)

        self.world.new_round()

        obs = self._get_obs()
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
        perm = self.world.rng.permutation(len(self.opponents))
        self.world.replay['permutations'].append(perm)
        for i in perm:
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

            self.world.replay['actions'][a.name].append(action)
            self.agent_actions[a] = action


    def step(self, action):
        self.agent_actions[self.agent] = self.ACTIONS[action]

        self.world.step_world(self.agent_actions)

        obs = self._get_obs()

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

    def _get_obs(self):
        state = self.world.get_state_for_agent(self.agent)

        arena = state["field"]

        H, W = arena.shape

        obs = np.zeros((7, H, W), dtype=np.float32)

        # walls
        obs[0] = np.where(arena == -1, 1, 0)

        # crates
        obs[1] = np.where(arena == 1, 1, 0)

        # coins
        for x, y in state["coins"]:
            obs[2, x, y] = 1

        # bombs
        for (x, y), timer in state["bombs"]:
            obs[3, x, y] = timer

        # explosions
        obs[4] = state["explosion_map"]

        # self
        sx, sy = state["self"][3]
        obs[5, sx, sy] = 1

        # enemies
        for enemy in state["others"]:
            ex, ey = enemy[3]
            obs[6, ex, ey] = 1

        return obs

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

        return reward

    def render(self):
        pass

    def close(self):
        self.world.end()