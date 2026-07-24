from typing import Tuple, List
import numpy as np
from collections import deque

from .pathfinding import A_star_manhattan, manhattan_distance
from .prediction import predict_explosions


def get_closest_coin(
        own_position: Tuple[int, int],
        coins: List[Tuple[int, int]],
        obstacles: np.typing.NDArray[np.bool_]
) -> Tuple[str|None, int|None]:
    """
    Find the closest coin to the agent's current position using the flood fill algorithm.

    :param own_position: The agent's current position as a tuple (x, y).
    :param coins: A list of coin positions as tuples [(x1, y1), (x2, y2), ...].
    :param obstacles: A 2D array representing the game board where True indicates an obstacle.
    :return: A tuple containing the best action to take and the distance to the closest coin.
             Returns (None, None) if no path to any coin is found.
    """

    ACTION_MOVEMENT_MAPPING = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    visited = np.zeros_like(obstacles, dtype=bool)
    visited[own_position] = True
    queue = deque([(own_position, 0, None)])

    while queue:
        current_position, distance, first_move = queue.popleft()

        if current_position in coins:
            return first_move, distance

        elif obstacles[current_position]:
            continue

        x, y = current_position
        for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dx, dy = ACTION_MOVEMENT_MAPPING[action]
            neighbor = (x + dx, y + dy)

            if (0 <= neighbor[0] < obstacles.shape[0] and
                0 <= neighbor[1] < obstacles.shape[1] and
                not visited[neighbor]):

                visited[neighbor] = True
                queue.append((neighbor, distance + 1, first_move if first_move is not None else action))

    return None, None


def get_closest_safe_square(
        own_position: Tuple[int, int],
        field: np.typing.NDArray[np.int_],
        bombs: np.typing.NDArray[np.int_],
        explosion_map: np.typing.NDArray[np.int_],
        delta_t: int
) -> Tuple[int, int]|None:
    """
    Find the closest safe square to the agent's current position a certain amount of time steps into the future.
    :param own_position: The agent's current position as a tuple (x, y).
    :param field: A 2D array representing the game board where True indicates an obstacle.
    :param bombs: A 2D array representing the bombs on the board.
    :param explosion_map: A 2D array representing the explosions on the board.
    :param delta_t: The number of time steps to predict into the future.
    :return: A tuple containing the coordinates of the closest safe square.
    """

    predicted_field, predicted_bombs, predicted_explosions = predict_explosions(field, bombs, explosion_map, delta_t)

    obstacles = (predicted_field == -1) | (predicted_field == 1) | (predicted_explosions > 0) | (bombs > 0)

    visited = np.zeros_like(obstacles, dtype=bool)
    visited[own_position] = True
    queue = deque([(own_position, 0)])

    while queue:
        current_position, distance = queue.popleft()

        if not obstacles[current_position]:
            return current_position

        x, y = current_position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)

            if (0 <= neighbor[0] < obstacles.shape[0] and
                0 <= neighbor[1] < obstacles.shape[1] and
                not visited[neighbor]):

                visited[neighbor] = True
                queue.append((neighbor, distance + 1))

    return None



