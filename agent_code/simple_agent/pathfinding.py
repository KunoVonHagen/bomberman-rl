from collections import deque

from .constants import ACTION_DELTA, DIRECTIONS


def bfs_next_step(field, start, targets, avoid=frozenset()):
    """Return the first action to take from `start` on a shortest path to any tile in `targets`.

    Returns None if `start` is already a target, `targets` is empty, or no target is reachable.
    """
    if not targets or start in targets:
        return None

    targets = set(targets)
    visited = {start}
    queue = deque()
    for action, (dx, dy) in ACTION_DELTA.items():
        nx, ny = start[0] + dx, start[1] + dy
        if field[nx, ny] == 0 and (nx, ny) not in avoid:
            visited.add((nx, ny))
            queue.append(((nx, ny), action))

    while queue:
        (cx, cy), first_action = queue.popleft()
        if (cx, cy) in targets:
            return first_action
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if field[nx, ny] == 0 and (nx, ny) not in avoid and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), first_action))

    return None


def find_reachable_tiles(field, start, avoid=frozenset()):
    """Return {tile: distance} for every free tile reachable from `start`, via BFS."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if field[nx, ny] == 0 and (nx, ny) not in avoid and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                queue.append((nx, ny))
    return dist


def bfs_distance(field, start, target, avoid=frozenset()):
    """Return the shortest-path distance from `start` to `target`, or None if unreachable."""
    if start == target:
        return 0
    return find_reachable_tiles(field, start, avoid=avoid).get(target)
