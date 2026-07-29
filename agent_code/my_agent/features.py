from typing import Tuple, List
import numpy as np
from collections import deque

import events as e

MAX_DISTANCE = 16+16

def get_closest_target_directions_and_distances(
        start_cell: Tuple[int, int],
        target_planes: np.typing.NDArray[np.bool_],
        obstacles_plane: np.typing.NDArray[np.bool_]
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Find the x and y direction and total distance (number of steps) to the closest instance of all the targets in the
    target planes.

    :param start_cell: The starting cell as a tuple (x, y).
    :param target_planes: A 3D boolean array of shape (T, W, H) where the first index identifies the target plane and
     the second and third indices identify the cell in the plane. True indicates a target is present at that cell.
    :param obstacles_plane: A 2D boolean array of shape (W, H) where True indicates an obstacle is present at that cell.
    :return: A tuple containing the (first step) direction to the closest target as (dx, dy) and the distance to the closest target
     or ((0, 0), 0) if no target can be found or the start_cell already contains the target.
    """

    n_target_planes = target_planes.shape[0]
    found_targets = np.zeros(n_target_planes, dtype=bool)
    results = [((0, 0), MAX_DISTANCE)] * n_target_planes

    queue = deque([(start_cell, [((None, None), 0)] * n_target_planes)])
    visited = set()
    visited.add(start_cell)

    while queue and not found_targets.all():
        current_cell, path_to_targets = queue.popleft()
        x, y = current_cell

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)

            if (0 <= neighbor[0] < obstacles_plane.shape[0] and
                    0 <= neighbor[1] < obstacles_plane.shape[1] and
                    not obstacles_plane[neighbor] and
                    neighbor not in visited):

                visited.add(neighbor)
                new_path_to_targets = path_to_targets.copy()

                for target_index in range(n_target_planes):
                    if not found_targets[target_index]:
                        if target_planes[target_index][neighbor]:
                            found_targets[target_index] = True
                            (old_x_dir, old_y_dir), old_distance  = path_to_targets[target_index]
                            new_x_dir, new_y_dir = old_x_dir or dx, old_y_dir or dy

                            new_path_to_targets[target_index] = ((new_x_dir, new_y_dir), old_distance + 1)
                            results[target_index] = new_path_to_targets[target_index]
                        else:
                            (old_x_dir, old_y_dir), old_distance = path_to_targets[target_index]
                            new_x_dir, new_y_dir = old_x_dir or dx, old_y_dir or dy
                            new_path_to_targets[target_index] = ((new_x_dir, new_y_dir), old_distance + 1)

                queue.append((neighbor, new_path_to_targets))

    for target_index in range(n_target_planes):
        if not found_targets[target_index]:
            results[target_index] = ((0, 0), MAX_DISTANCE)

        (dx, dy), d = results[target_index]
        results[target_index] = ((dx, dy), d / MAX_DISTANCE)

    return results


def cell_attributes(
        cell: Tuple[int, int],
        grid_tensor: np.typing.NDArray[np.int_]
) -> List[int]:
    """
    Extract attributes for a specific cell from the observation array.

    :param cell: The cell coordinates as a tuple (x, y).
    :param grid_tensor: A 3D array of shape (n_observation_layers, W, H) representing the observation.
    :return: A dictionary containing the cell attributes.
    """
    x, y = cell

    explosion_future_step = 1 if grid_tensor[11, x, y] + grid_tensor[12, x, y] + grid_tensor[13, x, y] + grid_tensor[14, x, y] > 0 else 0
    is_occupied_or_certain_death = 1 if grid_tensor[0, x, y] + grid_tensor[1, x, y] + grid_tensor[5, x, y] + grid_tensor[11, x, y] + grid_tensor[16, x, y] > 0 else 0
    is_occupied_or_certain_danger = 1 if is_occupied_or_certain_death + explosion_future_step > 0 else 0

    attributes = [
        grid_tensor[0, x, y], # is_wall
        grid_tensor[1, x, y], # is_crate
        grid_tensor[2, x, y], # is_coin
        grid_tensor[5, x, y], # is_enemy
        grid_tensor[6, x, y], # enemy_danger
        grid_tensor[11, x, y], # explosion_next_step
        explosion_future_step, # explosion_future_step
        grid_tensor[15, x, y], # explosion_will_vanish
        grid_tensor[16, x, y], # explosion_one_step_left
        is_occupied_or_certain_death, # is_occupied_or_certain_death
        is_occupied_or_certain_danger, # is_occupied_or_certain_danger
        1 if is_occupied_or_certain_danger + grid_tensor[6, x, y] > 0 else 0, # is_occupied_or_certain_danger_or_enemy_danger
    ]
    return attributes




FEATURES_DIM = 3 * 4 + 12 * 5  # 3 targets (coin, crate, enemy, safe) with direction and distance + 12 attributes for self and each neighbor
def get_features(grid_tensor: np.typing.NDArray[np.int_]) -> np.typing.NDArray[np.float32]:
    """
    Extract features from the observation array.

    :param grid_tensor: A 3D array of shape (n_observation_layers, W, H) representing the observation.
    :return: A dictionary containing extracted features.
    """

    features = []

    walls = grid_tensor[0]
    crates = grid_tensor[1]
    coins = grid_tensor[2]
    self_position = tuple(np.argwhere(grid_tensor[3] == 1)[0])
    self_danger_zone = grid_tensor[4]
    enemies = grid_tensor[5]
    enemy_danger_zone = grid_tensor[6]
    bombs1 = grid_tensor[7]
    bombs2 = grid_tensor[8]
    bombs3 = grid_tensor[9]
    bombs4 = grid_tensor[10]
    bombs1_danger_zone = grid_tensor[11]
    bombs2_danger_zone = grid_tensor[12]
    bombs3_danger_zone = grid_tensor[13]
    bombs4_danger_zone = grid_tensor[14]
    explosions1_map = grid_tensor[15]
    explosions2_map = grid_tensor[16]

    coin_plane = coins.astype(bool)[None]
    crate_plane = crates.astype(bool)[None]
    enemy_plane = enemies.astype(bool)[None]

    safe_plane = ((
            bombs1_danger_zone +
            bombs2_danger_zone +
            bombs3_danger_zone +
            bombs4_danger_zone +
            explosions2_map
    ) == 0)[None]

    direction_and_distance_features = get_closest_target_directions_and_distances(
        self_position,
        np.concatenate([
            coin_plane,
            crate_plane,
            enemy_plane,
            safe_plane
        ]),
        (walls + crates > 0),
    )

    for (x,y), d in direction_and_distance_features:
        features.append(x)
        features.append(y)
        features.append(d)

    features.extend(cell_attributes(self_position, grid_tensor))

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor_cell = (self_position[0] + dx, self_position[1] + dy)
        features.extend(cell_attributes(neighbor_cell, grid_tensor))

    return np.array(features, dtype=np.float32)


EVENT_REWARDS = {
    e.MOVED_LEFT: 0,
    e.MOVED_RIGHT: 0,
    e.MOVED_UP: 0,
    e.MOVED_DOWN: 0,
    e.WAITED: -0.01,
    e.INVALID_ACTION: -0.2,

    e.BOMB_DROPPED: 0.05,
    e.BOMB_EXPLODED: 0,

    e.CRATE_DESTROYED: 0.2,
    e.COIN_FOUND: 0.3,
    e.COIN_COLLECTED: 1.0,

    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -8.0,

    e.GOT_KILLED: -5.0,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0.0,
}

FEATURE_REWARDS = {
    # Coin distance
    2: lambda x:  1/(MAX_DISTANCE*x) * 0.02,

    # Crate distance
    5: lambda x: 1/(MAX_DISTANCE*x) * 0.005,

    # Safe tile distance
    11: lambda x: 1/(MAX_DISTANCE*x) * 0.05,

    # Standing in danger
    22: lambda x: x * (-0.05),
}

FEATURE_DIFF_REWARDS = {
    # Movement towards coin (negative -> closer)
    2: -0.01,

    # Movement towards crate
    5: -0.001,

    # Movement towards safe tile
    11: -0.02,
}

SIMPLE_EVENT_REWARDS = {
    e.MOVED_LEFT: 0,
    e.MOVED_RIGHT: 0,
    e.MOVED_UP: 0,
    e.MOVED_DOWN: 0,
    e.WAITED: 0,
    e.INVALID_ACTION: -1,

    e.BOMB_DROPPED: 0,
    e.BOMB_EXPLODED: 0,

    e.CRATE_DESTROYED: 0,
    e.COIN_FOUND: 0,
    e.COIN_COLLECTED: 1,

    e.KILLED_OPPONENT: 5,
    e.KILLED_SELF: 0,

    e.GOT_KILLED: 0,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0,
}
