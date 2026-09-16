from .constants import CRATE_SCORE, TERRITORY_SCORE, DISTANCE_PENALTY, OPPONENT_ADVANTAGE_PENALTY
from .pathfinding import find_reachable_tiles
from .blast import get_blast_coords, territory_gained
from .danger import can_escape_own_bomb


def build_opponent_distance_maps(field, others):
    """Precompute one full reachable-distance map per opponent: {opp_pos: {tile: dist}}.

    Call this ONCE per act() call and reuse the result everywhere a
    distance-from-an-opponent is needed. The old code called
    fastest_opponent_to() (which ran a fresh BFS per opponent) up to four
    times per decision, plus once per candidate tile inside
    best_bomb_spot() -- ~20 fresh BFS runs per step. This turns all of
    that into O(#opponents) BFS runs total, with everything after that a
    dict lookup.
    """
    return {opp_pos: find_reachable_tiles(field, opp_pos) for _, _, _, opp_pos in others}


def fastest_opponent_to(opp_dist_maps, tile, exclude=frozenset()):
    """Return the shortest distance from any opponent (not in `exclude`) to `tile`, or None.

    `opp_dist_maps` is the dict returned by build_opponent_distance_maps().
    """
    best = None
    for opp_pos, dist_map in opp_dist_maps.items():
        if opp_pos in exclude:
            continue
        d = dist_map.get(tile)
        if d is not None and (best is None or d < best):
            best = d
    return best


def best_bomb_spot(field, bombs, explosion_map, avoid, pos, power, timer, horizon,
                    opp_dist_maps, opponent_positions=frozenset(), reachable=None,
                    escape_cache=None, blast_cache=None, base_danger_map=None):
    """Score reachable tiles as bomb-placement candidates and return the best (position, score), or None.

    Only considers spots that hit at least one crate and from which the agent could escape its own bomb.

    `opp_dist_maps` comes from build_opponent_distance_maps() -- avoids a
    fresh per-candidate BFS to every opponent.

    `escape_cache` and `blast_cache`, if given, are plain dicts keyed by
    position, shared with the caller across this call and any other call
    sites in the same act() invocation (e.g. callbacks.py steps 4/5, and
    the second best_bomb_spot() attempt with a wider avoid-set).

    `base_danger_map`, if given, is the caller's already-computed danger
    map for the CURRENT `bombs` -- passed straight through to
    can_escape_own_bomb() so it can add just the candidate bomb's blast
    instead of recomputing danger from scratch for every candidate (see
    danger.py: has_escape_route/can_escape_own_bomb). This is the
    dominant remaining cost, so passing it in matters a lot more than the
    two caches above.

    Pass `reachable` in if the caller already has it (e.g. computed once
    earlier in act()) to skip a duplicate BFS from `pos`.
    """
    if reachable is None:
        reachable = find_reachable_tiles(field, pos, avoid=avoid)
    if escape_cache is None:
        escape_cache = {}
    if blast_cache is None:
        blast_cache = {}

    best = None
    for cand_pos, dist in reachable.items():
        if cand_pos not in blast_cache:
            blast_cache[cand_pos] = get_blast_coords(field, cand_pos, power)
        blast = blast_cache[cand_pos]

        crates = sum(1 for (x, y) in blast if field[x, y] == 1)
        if crates == 0:
            continue

        if TERRITORY_SCORE == 0 and best is not None and CRATE_SCORE * crates - DISTANCE_PENALTY * dist <= best[1]:
            continue

        if cand_pos not in escape_cache:
            escape_cache[cand_pos] = can_escape_own_bomb(
                field, bombs, explosion_map, cand_pos, power, timer, horizon, opponent_positions,
                base_danger_map=base_danger_map)
        if not escape_cache[cand_pos]:
            continue

        territory = territory_gained(field, cand_pos, power) if TERRITORY_SCORE != 0 else 0
        score = CRATE_SCORE * crates + TERRITORY_SCORE * territory - DISTANCE_PENALTY * dist

        opp_dist = fastest_opponent_to(opp_dist_maps, cand_pos)
        if opp_dist is not None and opp_dist < dist:
            score -= OPPONENT_ADVANTAGE_PENALTY * (dist - opp_dist)

        if best is None or score > best[1]:
            best = (cand_pos, score)

    return best