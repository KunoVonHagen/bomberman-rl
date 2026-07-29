from typing import List, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from agents import RLAgent
import settings as s
from environment import BombeRLeWorld, WorldArgs, Trophy
from items import Bomb

from agent_code.my_agent.features import get_features, FEATURES_DIM, EVENT_REWARDS, FEATURE_REWARDS, FEATURE_DIFF_REWARDS, SIMPLE_EVENT_REWARDS


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
        None: 4
    }

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

        self.reward_fn = reward_fn or self.shaped_reward

        self.world.new_round()

        H, W = self.world.arena.shape

        self.n_observation_layers = 8 + 2 * s.BOMB_TIMER + s.EXPLOSION_TIMER

        self.grid_tensor = np.zeros((self.n_observation_layers, H, W), dtype=np.float32)
        self.features = np.array([])

        # Observation Space:
        # Map Layers:
        #   walls
        #   crates
        #   coins
        #   self
        #   self_danger_zone
        #   opponents
        #   opponents_danger_zone
        #   can_place_bomb
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

        # Auxiliaries for rewards
        self.previous_visited_count = 1
        self.visited = np.zeros_like(self.world.arena, dtype=np.bool)
        self.previous_features = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_actions = {}

        self.previous_visited_count = 1
        self.visited.fill(False)
        self.previous_features = None

        if seed is not None:
            self.world.rng = np.random.default_rng(seed)

        self.world.new_round()

        grid_tensor = self._get_grid_tensor()
        self.features = get_features(grid_tensor)
        obs = {"grid_tensor": grid_tensor, "features": self.features}

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
        self.previous_features = self.features

        self.get_opponents_actions()
        self.agent.reset_game_events()
        self.agent_actions[self.agent] = self.ACTIONS[action]

        self.world.step_world(self.agent_actions)

        grid_tensor = self._get_grid_tensor()
        self.features = get_features(self.grid_tensor)
        obs = {"grid_tensor": grid_tensor, "features": self.features}

        self.visited[(self.agent.x, self.agent.y)] = True

        reward = self.reward_fn()

        terminated = self.agent.dead
        truncated = self.world.step >= s.MAX_STEPS

        info = self._get_info()

        #self.check_game_state_conversion_accuracy(obs)

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def check_game_state_conversion_accuracy(self, obs):
        from agent_code.my_agent.input_processing import observation_to_game_state

        correct_game_state = self.world.get_state_for_agent(self.agent)
        computed_game_state = observation_to_game_state(obs)

        for important_key in ["field", "explosion_map"]:
            if not np.array_equal(correct_game_state[important_key], computed_game_state[important_key]):
                print(f"Correct game state[{important_key}]")
                print(correct_game_state[important_key])
                print()
                print()
                print(f"Computed game_state[{important_key}]")
                print(computed_game_state[important_key])

                raise ValueError(f"Game state computation differs from correct game state in field '{important_key}')")

        for important_key in ["bombs", "coins"]:
            if not set(correct_game_state[important_key]) == set(computed_game_state[important_key]):
                print(f"Correct game state[{important_key}]")
                print(correct_game_state[important_key])
                print()
                print()
                print(f"Computed game_state[{important_key}]")
                print(computed_game_state[important_key])

                raise ValueError(f"Game state computation differs from correct game state in field '{important_key}')")

        for important_index in [2, 3]: # Can place bomb, position
            if not correct_game_state["self"][important_index] == computed_game_state["self"][important_index]:
                print(f"Correct game state[self][{important_index}]")
                print(correct_game_state["self"][important_index])
                print()
                print()
                print(f"Computed game_state[self][{important_index}]")
                print(computed_game_state["self"][important_index])

                raise ValueError(f"Game state computation differs from correct game state in field 'self[{important_index}]'")

        if not len(correct_game_state["others"]) == len(computed_game_state["others"]):
            print(f"Correct game state [others]")
            print(correct_game_state["others"])
            print()
            print()
            print(f"Computed game_state[others]")
            print(computed_game_state["others"])

            raise ValueError(f"Game state computation differs from correct game state in field 'others' (length mismatch)")

        for opponent_index in range(len(correct_game_state["others"])):
            for important_index in [2, 3]: # Can place bomb, position
                if not correct_game_state["others"][opponent_index][important_index] == computed_game_state["others"][opponent_index][important_index]:
                    print(f"Correct game state[others][{opponent_index}][{important_index}]")
                    print(correct_game_state["others"][opponent_index][important_index])
                    print()
                    print()
                    print(f"Computed game_state[others][{opponent_index}][{important_index}]")
                    print(computed_game_state["others"][opponent_index][important_index])

                    raise ValueError(f"Game state computation differs from correct game state in field 'others [{opponent_index}][{important_index}]'")

    def _get_grid_tensor(self):
        self.grid_tensor.fill(0)

        # walls
        self.grid_tensor[0] = np.where(self.world.arena == -1, 1, 0)

        # crates
        self.grid_tensor[1] = np.where(self.world.arena == 1, 1, 0)

        # coins
        for coin in self.world.coins:
            if coin.collectable:
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

            # can place bomb
            self.grid_tensor[7, enemy.x, enemy.y] = 1 if enemy.bombs_left else 0


        self.grid_tensor[6] = np.where(self.grid_tensor[6] > 0, 1, 0)
        self.grid_tensor[7, self.agent.x, self.agent.y] = 1 if self.agent.bombs_left else 0

        # bombs
        for bomb in self.world.bombs:
            #print("b", bomb.timer)
            # locations
            self.grid_tensor[7 + bomb.timer + 1, bomb.x, bomb.y] = 1
            # danger
            self.grid_tensor[7 + s.BOMB_TIMER + bomb.timer + 1] = self.PRECOMPUTED_BLAST_MAP.get((bomb.x, bomb.y), np.zeros_like(self.world.arena, dtype=np.int8))

        # explosions
        for explosion in self.world.explosions:
            if explosion.is_dangerous():
                for x, y in explosion.blast_coords:
                    self.grid_tensor[7 + 2*s.BOMB_TIMER + explosion.timer, x, y] = 1

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
            reward += SIMPLE_EVENT_REWARDS.get(event, 0)

        return reward


    def shaped_reward(self):
        reward = 0

        for event in self.agent.events:
            reward += EVENT_REWARDS.get(event, 0)

        if not self.agent.dead:
            reward -= 0.001 # Incentivize ending the game

        visited_count = np.sum(self.visited)
        new_visited = visited_count - self.previous_visited_count
        self.previous_visited_count = visited_count

        feature_diff = np.sign(self.features - self.previous_features)

        for feature_index, reward_function in FEATURE_REWARDS.items():
            reward += reward_function(self.features[feature_index])

        for feature_diff_index, reward in FEATURE_DIFF_REWARDS.items():
            reward += reward * feature_diff[feature_diff_index]

        if new_visited > 0:
            reward += 0.001

        return reward

    def render(self):
        pass

    def close(self):
        self.world.end()