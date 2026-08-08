from .constants import DIRECTIONS
from .pathfinding import find_reachable_tiles


def get_blast_coords(field, pos, power):
    """Return the set of tiles a bomb at `pos` with the given `power` would hit, stopping at walls."""
    x, y = pos
    blast = {(x, y)}
    for dx, dy in DIRECTIONS:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:
                break
            blast.add((nx, ny))
    return blast


def territory_gained(field, pos, power):
    """Return how many additional tiles become reachable from `pos` if the crates in its blast radius were cleared."""
    blast = get_blast_coords(field, pos, power)
    crates_in_blast = [(x, y) for (x, y) in blast if field[x, y] == 1]
    if not crates_in_blast:
        return 0

    before = len(find_reachable_tiles(field, pos))

    opened_field = field.copy()
    for (x, y) in crates_in_blast:
        opened_field[x, y] = 0
    after = len(find_reachable_tiles(opened_field, pos))

    return after - before


def project_cleared_field(field, bombs, power):
    """Return a copy of `field` with crates removed wherever the given bombs would eventually hit."""
    if not bombs:
        return field
    projected = field.copy()
    for bomb_pos, _ in bombs:
        for (x, y) in get_blast_coords(field, bomb_pos, power):
            if projected[x, y] == 1:
                projected[x, y] = 0
    return projected
