from collections import deque

from .constants import ACTION_DELTA
from .blast import get_blast_coords


def compute_danger_map(field, bombs, explosion_map, power, horizon):
    """Return danger[t] = set of tiles that are lethal at future step `t`, for t in 0..horizon."""
    danger = [set() for _ in range(horizon + 1)]

    now_dangerous = {(x, y) for x in range(field.shape[0]) for y in range(field.shape[1])
                      if explosion_map[x, y] > 0}
    for t in (0, 1):
        if t <= horizon:
            danger[t] |= now_dangerous

    for bomb_pos, bomb_timer in bombs:
        blast = get_blast_coords(field, bomb_pos, power)
        explode_at = bomb_timer + 1
        for t in (explode_at, explode_at + 1):
            if 0 <= t <= horizon:
                danger[t] |= blast

    return danger


def _extend_danger_map(base_danger_map, field, bomb_pos, bomb_timer, power, horizon):
    """Return base_danger_map plus the blast contribution of one extra bomb."""
    blast = get_blast_coords(field, bomb_pos, power)
    explode_at = bomb_timer + 1

    extended = list(base_danger_map)
    for t in (explode_at, explode_at + 1):
        if 0 <= t <= horizon:
            extended[t] = base_danger_map[t] | blast
    return extended


def _time_expanded_search(field, occupied, danger_map, start, horizon):
    """BFS over (position, time) states to find a first action leading to permanent safety.

    Returns (first_action_to_full_safety, first_action_of_longest_survival_path); either may be None.
    """
    visited = {(start, 0)}
    queue = deque([(start, 0, None)])

    best_survival = (0, None)

    while queue:
        pos, t, first_action = queue.popleft()

        if t > best_survival[0]:
            best_survival = (t, first_action)

        if t > 0 and all(pos not in danger_map[tt] for tt in range(t, horizon + 1)):
            return first_action, best_survival[1]

        if t >= horizon:
            continue

        x, y = pos
        candidates = [('WAIT', pos)]
        candidates += [(action, (x + dx, y + dy)) for action, (dx, dy) in ACTION_DELTA.items()]

        for action, (nx, ny) in candidates:
            if (nx, ny) != pos and field[nx, ny] != 0:
                continue
            if (nx, ny) in occupied and (nx, ny) != start:
                continue
            nt = t + 1
            if (nx, ny) in danger_map[min(nt, horizon)]:
                continue
            state = ((nx, ny), nt)
            if state in visited:
                continue
            visited.add(state)
            queue.append(((nx, ny), nt, first_action or action))

    return None, best_survival[1]


def find_escape_action(field, bombs, danger_map, start, horizon, opponent_positions=frozenset()):
    """Return the best action to take from `start` to escape current danger, preferring full safety."""
    occupied = {b_pos for b_pos, _ in bombs} | opponent_positions
    full_safe, fallback = _time_expanded_search(field, occupied, danger_map, start, horizon)
    return full_safe if full_safe is not None else fallback


def has_escape_route(field, bombs, explosion_map, bomb_pos, bomb_timer, power, horizon, start, occupied,
                      base_danger_map=None):
    """Return True if a path to full safety exists from `start`, assuming a bomb is placed at `bomb_pos`."""
    if base_danger_map is not None:
        danger_map = _extend_danger_map(base_danger_map, field, bomb_pos, bomb_timer, power, horizon)
    else:
        hypothetical_bombs = list(bombs) + [(bomb_pos, bomb_timer)]
        danger_map = compute_danger_map(field, hypothetical_bombs, explosion_map, power, horizon)
    full_safe, _ = _time_expanded_search(field, occupied, danger_map, start, horizon)
    return full_safe is not None


def can_escape_own_bomb(field, bombs, explosion_map, pos, power, timer, horizon, opponent_positions=frozenset(),
                         base_danger_map=None):
    """Return True if the agent could survive dropping a bomb at `pos` right now."""
    occupied = {b_pos for b_pos, _ in bombs} | opponent_positions
    occupied.add(pos)
    return has_escape_route(field, bombs, explosion_map, pos, timer, power, horizon, pos, occupied,
                             base_danger_map=base_danger_map)