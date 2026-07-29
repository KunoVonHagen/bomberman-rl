from typing import Tuple, Dict, Any
import numpy as np
from collections import deque

def get_obstacles(game_state: Dict[str, Any]) -> np.typing.NDArray[np.bool_]:
    """
    Extracts the obstacles from the game state and returns a dictionary of (x,y) -> True if obstacle present at position (x,y), False otherwise.

    :param game_state: The current game state.
    :return: A dictionary of obstacles.
    """
    field = game_state['field']
    explosion_map = game_state['explosion_map']

    obstacles = np.zeros_like(field, dtype=bool)

    obstacles |= field == -1 # Walls
    obstacles |= field == 1 # Crates
    obstacles |= explosion_map > 0 # Active Explosions

    for bomb in game_state['bombs']:
        bomb_x, bomb_y = bomb[0]
        obstacles[(bomb_x, bomb_y)] = True

    for other_agent in game_state['others']:
        other_x, other_y = other_agent[3]
        obstacles[(other_x, other_y)] = True

    return obstacles


def count_open_neighbors(position: Tuple[int, int], field: np.typing.NDArray[np.int_]) -> int:

    x, y = position
    count = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0:
            count += 1
    return count


def connected_cell_distances(position: Tuple[int, int], obstacles: np.typing.NDArray[np.bool_]) -> np.typing.NDArray[np.bool_]:
    """
    Returns an array of the same shape as obstacles, where the value indicates the distance from the given position to
    each cell, or -1 if the cell is not reachable.

    :param position: The starting position (x, y).
    :param obstacles: A 2D array representing the game board where True indicates an obstacle.
    :return: A 2D array of distances from the starting position to each cell, or -1 if not reachable.
    """
    height, width = obstacles.shape
    distance = np.full_like(obstacles, -1, dtype=int)
    visited = np.zeros_like(obstacles, dtype=bool)
    visited[position] = True
    queue = deque([(position, 0)])

    while queue:
        current_position, current_distance = queue.popleft()
        x, y = current_position
        distance[x, y] = current_distance

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor
            if (0 <= nx < width) and (0 <= ny < height) and not obstacles[nx, ny] and not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, current_distance + 1))
    return distance
