import numpy as np
from typing import List, Tuple

from settings import BOMB_TIMER, BOMB_POWER, EXPLOSION_TIMER

_DANGER_CACHE: dict = {}
_DANGER_CACHE_LIMIT = 256


def predict_danger_over_time(field, bombs, explosion_map, max_horizon):
    key = (field.tobytes(), bombs.tobytes(), np.asarray(explosion_map).tobytes(), int(max_horizon))
    cached = _DANGER_CACHE.get(key)
    if cached is not None:
        return cached
    result = _predict_danger_over_time_uncached(field, bombs, explosion_map, max_horizon)
    if len(_DANGER_CACHE) >= _DANGER_CACHE_LIMIT:
        _DANGER_CACHE.clear()
    _DANGER_CACHE[key] = result
    return result


def _predict_danger_over_time_uncached(
        field: np.typing.NDArray[np.int_],
        bombs: np.typing.NDArray[np.int_],
        explosion_map: np.typing.NDArray[np.int_],
        max_horizon: int
) -> List[np.typing.NDArray[np.bool_]]:
    """
    Single-pass simulation returning the danger map for every tick t = 0..max_horizon.

    Check-before-decrement order, matching environment.py's update_bombs:
        if bomb.timer <= 0: explode NOW
        else: bomb.timer -= 1
    This makes t=0 (a bomb whose reported timer is already 0, exploding this
    very tick) a natural first iteration of the loop rather than a special case.

    :param field: -1 wall / 1 crate / 0 empty.
    :param bombs: from get_bomb_timer_array (-1 = no bomb, else timer).
    :param explosion_map: currently active explosions (remaining duration).
    :param max_horizon: how many ticks ahead to simulate.
    :return: list of bool arrays, danger[t][x, y] = True if (x, y) is on fire at tick t.
    """
    bomb_timers = bombs.copy()
    active_explosions = explosion_map.copy()
    danger_snapshots = []

    for t in range(max_horizon + 1):
        # Danger right now, at tick t: explosions already running + bombs at timer 0
        danger_now = active_explosions > 0
        zero_cells = np.argwhere(bomb_timers == 0)
        for x, y in zero_cells:
            _spread_blast(field, int(x), int(y), danger_now)
        danger_snapshots.append(danger_now)

        if t == max_horizon:
            break

        # Advance state by one tick for the next snapshot
        new_explosions = np.zeros_like(active_explosions, dtype=bool)
        for x, y in zero_cells:
            x, y = int(x), int(y)
            _spread_blast(field, x, y, new_explosions)
            bomb_timers[x, y] = -1  # bomb consumed

        active_explosions[active_explosions > 0] -= 1
        active_explosions[new_explosions] = EXPLOSION_TIMER
        bomb_timers[bomb_timers > 0] -= 1

    return danger_snapshots


def get_bomb_timer_array(
        bombs: List[Tuple[Tuple[int, int], int]],
        field_shape: Tuple[int, int]
) -> np.typing.NDArray[np.int_]:
    """Converts game_state['bombs'] into an array. -1 = no bomb, else timer."""
    arr = np.full(field_shape, -1, dtype=int)
    for (x, y), t in bombs:
        arr[x, y] = t
    return arr


def _spread_blast(field, x, y, danger):
    """Marks the blast of a bomb at (x,y) into `danger` in-place."""
    danger[x, y] = True
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for power in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * power, y + dy * power
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == -1:
                    break
                danger[nx, ny] = True
