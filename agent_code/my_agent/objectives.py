from typing import Tuple, List
import numpy as np
from collections import deque

from .pathfinding import A_star_manhattan, manhattan_distance
from .prediction import predict_danger_over_time, get_bomb_timer_array
from .settings import BOMB_TIMER, EXPLOSION_TIMER

ACTIONS_MOVE = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0), 'WAIT': (0, 0)}

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


def get_safe_square_action(
        own_position: Tuple[int, int],
        field: np.typing.NDArray[np.int_],
        bombs: list,
        explosion_map: np.typing.NDArray[np.int_],
        other_positions: List[Tuple[int, int]],
        max_horizon: int = None
) -> Tuple[str | None, int | None]:
    """
    BFS toward the closest tile that is permanently safe from the tick
    the agent would arrive there. Returns (first_move, ticks) or (None, None).
    """
    if max_horizon is None:
        max_horizon = BOMB_TIMER + EXPLOSION_TIMER

    bomb_timer_array = get_bomb_timer_array(bombs, field.shape)
    danger_by_t = predict_danger_over_time(field, bomb_timer_array, explosion_map, max_horizon)

    obstacles = (field != 0)
    bomb_positions = {xy for xy, t in bombs}
    for bx, by in bomb_positions:
        obstacles[bx, by] = True
    for ox, oy in other_positions:
        obstacles[ox, oy] = True

    def permanently_safe_from(x, y, t):
        return not any(danger_by_t[tt][x, y] for tt in range(t, max_horizon + 1))

    if permanently_safe_from(*own_position, 0):
        return 'WAIT', 0

    visited = {(own_position, 0)}
    queue = deque([(own_position, 0, None)])

    while queue:
        (x, y), t, first_move = queue.popleft()
        if t >= max_horizon:
            continue
        for action, (dx, dy) in ACTIONS_MOVE.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if obstacles[nx, ny]:
                continue
            nt = t + 1
            if danger_by_t[nt][nx, ny]:
                continue
            state = ((nx, ny), nt)
            if state in visited:
                continue
            visited.add(state)
            move = first_move if first_move is not None else action
            if permanently_safe_from(nx, ny, nt):
                return move, nt
            queue.append((state[0], nt, move))

    return None, None


def get_least_bad_action(game_state: dict, legal_actions: list, max_horizon: int = None) -> str | None:
    """
    Used only when no legal action passes is_action_safe
    AND get_safe_square_action finds no reachable safe tile either. Ranks the remaining legal actions by how
    many ticks pass before that tile catches fire, and picks the one that
    buys the most time.
    """
    if max_horizon is None:
        max_horizon = BOMB_TIMER + EXPLOSION_TIMER

    field = game_state['field']
    explosion_map = game_state['explosion_map']
    own_x, own_y = game_state['self'][3]
    bombs = list(game_state['bombs'])

    bomb_timer_array = get_bomb_timer_array(bombs, field.shape)
    danger_by_t = predict_danger_over_time(field, bomb_timer_array, explosion_map, max_horizon)

    def ticks_until_hit(pos):
        for t in range(max_horizon + 1):
            if danger_by_t[t][pos]:
                return t
        return max_horizon + 1  # never hit within the horizon we simulated

    best_action, best_ticks = None, -1
    for action in legal_actions:
        if action == 'BOMB':
            target = (own_x, own_y)
        else:
            dx, dy = ACTIONS_MOVE[action]
            target = (own_x + dx, own_y + dy)
        ticks = ticks_until_hit(target)
        if ticks > best_ticks:
            best_ticks, best_action = ticks, action

    return best_action


def is_action_safe(action: str, game_state: dict, max_horizon: int = None) -> bool:
    """
    Checks whether, after taking `action`, an escape route still exists
    that gets the agent out of every future explosion in time.
    """
    if max_horizon is None:
        max_horizon = BOMB_TIMER + EXPLOSION_TIMER

    field = game_state['field']
    explosion_map = game_state['explosion_map']
    own_x, own_y = game_state['self'][3]
    bombs = list(game_state['bombs'])
    other_positions = {p[3] for p in game_state['others']}
    bomb_positions = {xy for xy, t in bombs}

    if action == 'BOMB':
        start_pos = (own_x, own_y)
        bombs = bombs + [((own_x, own_y), BOMB_TIMER)]
    else:
        dx, dy = ACTIONS_MOVE[action]
        start_pos = (own_x + dx, own_y + dy)
        if field[start_pos] != 0:
            return False
        if start_pos in bomb_positions or start_pos in other_positions:
            return False  # can't walk onto a bomb tile or another agent

    bomb_timer_array = get_bomb_timer_array(bombs, field.shape)

    # Obstacles for traversal during the BFS: walls, crates, bomb tiles, other agents.
    obstacles = (field != 0)
    for (bx, by) in bomb_positions:
        obstacles[bx, by] = True
    for (ox, oy) in other_positions:
        obstacles[ox, oy] = True

    danger_by_t = predict_danger_over_time(field, bomb_timer_array, explosion_map, max_horizon)

    if danger_by_t[0][start_pos]:
        return False  # already lethal right now

    def permanently_safe_from(x, y, t):
        return not any(danger_by_t[tt][x, y] for tt in range(t, max_horizon + 1))

    if danger_by_t[1][start_pos]:
        return False

    if permanently_safe_from(*start_pos, 1):
        return True

    visited = {(start_pos, 1)}
    queue = deque([(start_pos, 1)])

    while queue:
        (x, y), t = queue.popleft()
        if t >= max_horizon:
            continue
        for ddx, ddy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + ddx, y + ddy
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if obstacles[nx, ny]:
                continue
            nt = t + 1
            if danger_by_t[nt][nx, ny]:
                continue
            state = ((nx, ny), nt)
            if state in visited:
                continue
            visited.add(state)
            if permanently_safe_from(nx, ny, nt):
                return True
            queue.append(state)

    return False

def get_legal_actions(game_state: dict) -> list:
    """
    Returns the subset of the 6 actions that are legal right now, i.e.
    don't walk into a wall, crate, bomb tile, active explosion, or another
    agent, and respect whether the agent currently has a bomb available.
    """
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    own_x, own_y = game_state['self'][3]
    bombs_left = game_state['self'][2]
    others = [p[3] for p in game_state['others']]
    bomb_positions = [xy for xy, t in game_state['bombs']]

    legal = []
    for action, (dx, dy) in ACTIONS_MOVE.items():
        target = (own_x + dx, own_y + dy)
        if (field[target] == 0 and target not in others and target not in bomb_positions
                and explosion_map[target] == 0):
            legal.append(action)
    if bombs_left > 0:
        legal.append('BOMB')
    return legal