import os
import pickle
import random
import time

from .pathfinding import get_obstacles
from .objectives import get_closest_coin, is_action_safe, get_legal_actions, get_safe_square_action

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    start = time.time()

    field = game_state["field"]
    coins = game_state["coins"]
    others_positions = {player[3] for player in game_state["others"]}
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    own_position = game_state['self'][3]

    field_shape = field.shape


    obstacles = get_obstacles(game_state)
    coins = game_state['coins']
    obstacles = get_obstacles(game_state)
    legal_actions = get_legal_actions(game_state)

    safe_actions = [a for a in legal_actions if is_action_safe(a, game_state)]

    if safe_actions:
        best_action = random.choice(safe_actions)
    else:
        # Emergency fallback only: no move passes is_action_safe, so head
        # for the nearest tile that will eventually be safe.
        escape_action, _ = get_safe_square_action(
            own_position, field, bombs, explosion_map, others_positions
        )
        best_action = escape_action if escape_action in legal_actions else (
            legal_actions[0] if legal_actions else 'WAIT'
        )

    coin_action, coin_distance = get_closest_coin(own_position, coins, obstacles)

    if coin_action is not None and coin_action in safe_actions:
        best_action = coin_action

    self.logger.info(f"Time taken for act: {time.time() - start:.6f} seconds")

    return best_action

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)