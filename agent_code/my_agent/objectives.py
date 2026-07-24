from typing import Tuple, List
import numpy as np

from .pathfinding import A_star_manhattan, manhattan_distance


def get_closest_coin(own_position: Tuple[int, int], coins: List[Tuple[int, int]], obstacles: np.typing.NDArray[np.bool_]) -> Tuple[int|None, int]:
    """
    Find the closest coin to the agent's current position using A* pathfinding.
    Apply iterative deepening to reduce computation time.

    :param own_position: The agent's current position as a tuple (x, y).
    :param coins: A list of coin positions as tuples [(x1, y1), (x2, y2), ...].
    :param obstacles: A 2D array representing the game board where True indicates an obstacle.
    :return: A tuple containing the best action to take and the distance to the closest coin.
             Returns (None, float('inf')) if no path to any coin is found.
    """

    MAX_SEARCH_DEPTH = 20

    best_action = None
    min_coin_distance = float("inf")

    coin_manhattan_distances = {coin: manhattan_distance(own_position, coin) for coin in coins}

    for search_depth in range(MAX_SEARCH_DEPTH):
        for coin, distance in coin_manhattan_distances.items():
            if distance > search_depth:
                continue  # Skip coins that are further than the current search depth

            result = A_star_manhattan(own_position, coin, obstacles, max_depth=search_depth)
            if result is not None:
                action, distance = result
                if distance < min_coin_distance:
                    min_coin_distance = distance
                    best_action = action

        # If a path to a coin was found at this depth, no need to search deeper
        if best_action is not None:
            break

    return best_action, min_coin_distance