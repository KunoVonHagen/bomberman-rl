from typing import Tuple, Dict, List, Any
import numpy as np


def A_star_manhattan(position: Tuple[int, int], target: Tuple[int, int], obstacles: np.typing.NDArray[np.bool_], max_depth: int = float("inf")) -> Tuple[str, int] | None:
    """
    A* pathfinding algorithm using Manhattan distance as the heuristic.
    :param position: starting position (x,y)
    :param target: target position (x,y)
    :param obstacles: obstacles dictionary of (x,y) -> True if obstacle present at position (x,y), False otherwise
    :param max_depth: maximum depth to reach from start to target

    :return: The first move towards the target as a string ('UP', 'DOWN', 'LEFT', 'RIGHT') and the number of steps to
     reach the target, or None if no path is found.
    """

    # Define the possible moves and their corresponding directions
    moves = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0)
    }

    # Initialize the open and closed sets
    open_set = {position}
    closed_set = set()

    # Initialize the g_score and f_score dictionaries
    g_score = {position: 0}
    f_score = {position: manhattan_distance(position, target)}

    # Initialize the came_from dictionary to reconstruct the path later
    came_from = {}

    depth = 0

    while open_set and depth <= max_depth:
        # Get the node in open_set with the lowest f_score
        current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))

        # If we reached the target, reconstruct the path and return the first move
        if current == target:
            return reconstruct_path(came_from, current)

        # Move current from open_set to closed_set
        open_set.remove(current)
        closed_set.add(current)

        # Explore neighbors
        for direction, move in moves.items():
            neighbor = (current[0] + move[0], current[1] + move[1])

            # Skip if neighbor is an obstacle or already evaluated
            if obstacles[neighbor] or neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            # This path is the best until now. Record it!
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, target)

    return None  # No path found

def manhattan_distance(position: Tuple[int, int], target: Tuple[int, int]) -> int:
    return abs(position[0] - target[0]) + abs(position[1] - target[1])

def reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> Tuple[str, int] | None:
    """
    Reconstruct the path from the start to the target using the came_from dictionary.

    :param came_from: The came_from dictionary.
    :param current: The current position.

    :return: A tuple containing the first move direction as a string and the number of steps to reach the target,
    or None if no move is needed.
    """
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()  # Reverse to get the path from start to target

    if len(total_path) < 2:
        return None  # No move needed, already at target

    first_move = (total_path[1][0] - total_path[0][0], total_path[1][1] - total_path[0][1])
    direction_map = {
        (0, -1): 'UP',
        (0, 1): 'DOWN',
        (-1, 0): 'LEFT',
        (1, 0): 'RIGHT'
    }
    first_move_direction = direction_map.get(first_move)

    return first_move_direction, len(total_path) - 1

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