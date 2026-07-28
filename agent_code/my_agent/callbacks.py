import os
import pickle
import random
import time

from .pathfinding import get_obstacles
from .objectives import (
    get_closest_coin,
    get_action_toward_target,
    is_action_safe,
    is_currently_safe,
    get_legal_actions,
    get_safe_square_action,
    get_least_bad_action,
)

from .strategy import evaluate_bomb_placement, find_trap_targets, get_best_crate_targets

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
    crates = list(zip(*np.where(field == 1)))
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    own_position = game_state['self'][3]
    others_positions = {player[3] for player in game_state["others"]}

    obstacles = get_obstacles(game_state)
    legal_actions = get_legal_actions(game_state)
    safe_actions = [a for a in legal_actions if is_action_safe(a, game_state)]
    movement_actions = [a for a in safe_actions if a != 'BOMB']

    bombs_active = len(bombs) > 0
    in_immediate_danger = bombs_active and not is_currently_safe(own_position, field, bombs, explosion_map)

    if bombs_active:
        self.logger.info(
            f"Step-Diagnose: pos={own_position} bombs={bombs} others={list(others_positions)} "
            f"legal_actions={legal_actions} safe_actions={safe_actions} in_immediate_danger={in_immediate_danger}"
        )

    if not safe_actions:
        escape_action, _ = get_safe_square_action(
            own_position, field, bombs, explosion_map, others_positions, avoid_contested=True
        )
        if escape_action is None or escape_action not in legal_actions:
            escape_action, _ = get_safe_square_action(own_position, field, bombs, explosion_map, others_positions)
        if escape_action in legal_actions:
            best_action = escape_action
        elif legal_actions:
            fallback_actions = [a for a in legal_actions if a != 'BOMB'] or legal_actions
            best_action = get_least_bad_action(game_state, fallback_actions)
        else:
            best_action = 'WAIT'
    elif in_immediate_danger:
        escape_action, _ = get_safe_square_action(
            own_position, field, bombs, explosion_map, others_positions, avoid_contested=True
        )
        if escape_action is None or escape_action not in movement_actions:
            escape_action, _ = get_safe_square_action(own_position, field, bombs, explosion_map, others_positions)
        if escape_action is not None and escape_action in movement_actions:
            best_action = escape_action
        elif movement_actions:
            best_action = random.choice(movement_actions)
        else:
            best_action = 'WAIT'
    else:
        best_action = random.choice(movement_actions) if movement_actions else 'WAIT'

    coin_action, coin_distance = get_closest_coin(own_position, coins, obstacles)

    crate_targets = get_best_crate_targets(crates, field)
    crate_action, crate_distance = get_action_toward_target(own_position, crate_targets, obstacles)

    trap_targets = find_trap_targets(game_state, max_escape_routes=1)
    trapped_positions = {t['position'] for t in trap_targets}
    hunt_action, hunt_distance = get_action_toward_target(
        own_position, list(trapped_positions), obstacles
    )
    HUNT_RANGE = 6

    if not in_immediate_danger:
        if (hunt_action is not None and hunt_action in movement_actions
                and hunt_distance is not None and hunt_distance <= HUNT_RANGE):
            best_action = hunt_action
        elif coin_action is not None and coin_action in movement_actions:
            best_action = coin_action
        elif crate_action is not None and crate_action in movement_actions:
            best_action = crate_action

    if 'BOMB' in safe_actions:
        bomb_eval = evaluate_bomb_placement(game_state, trapped_positions=trapped_positions)
        if bomb_eval['should_bomb'] and (bomb_eval['opponents_hit'] or coin_action is None or coin_distance > 3):
            best_action = 'BOMB'

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