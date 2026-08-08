from .constants import CRATE_SCORE, TERRITORY_SCORE, DISTANCE_PENALTY, OPPONENT_ADVANTAGE_PENALTY
from .pathfinding import find_reachable_tiles, bfs_distance
from .blast import get_blast_coords, territory_gained
from .danger import can_escape_own_bomb


def fastest_opponent_to(field, tile, others, exclude=frozenset()):
    """Return the shortest distance from any opponent (not in `exclude`) to `tile`, or None."""
    best = None
    for _, _, _, opp_pos in others:
        if opp_pos in exclude:
            continue
        d = bfs_distance(field, opp_pos, tile)
        if d is not None and (best is None or d < best):
            best = d
    return best


def best_bomb_spot(field, bombs, explosion_map, avoid, pos, power, timer, horizon, others, opponent_positions=frozenset()):
    """Score reachable tiles as bomb-placement candidates and return the best (position, score), or None.

    Only considers spots that hit at least one crate and from which the agent could escape its own bomb.
    """
    reachable = find_reachable_tiles(field, pos, avoid=avoid)

    best = None
    for cand_pos, dist in reachable.items():
        blast = get_blast_coords(field, cand_pos, power)
        crates = sum(1 for (x, y) in blast if field[x, y] == 1)
        if crates == 0:
            continue
        if not can_escape_own_bomb(field, bombs, explosion_map, cand_pos, power, timer, horizon, opponent_positions):
            continue

        territory = territory_gained(field, cand_pos, power)
        score = CRATE_SCORE * crates + TERRITORY_SCORE * territory - DISTANCE_PENALTY * dist

        opp_dist = fastest_opponent_to(field, cand_pos, others)
        if opp_dist is not None and opp_dist < dist:
            score -= OPPONENT_ADVANTAGE_PENALTY * (dist - opp_dist)

        if best is None or score > best[1]:
            best = (cand_pos, score)

    return best
