from typing import Tuple, List, Dict
import numpy as np

from .objectives import is_action_safe
from .pathfinding import connected_cell_distances
from settings import BOMB_TIMER, BOMB_POWER, EXPLOSION_TIMER

MOVE_DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']


def get_hypothetical_blast_coords(field: np.typing.NDArray, position: Tuple[int, int]) -> List[Tuple[int, int]]:

    x, y = position
    coords = [(x, y)]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for power in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * power, y + dy * power
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def _make_game_state_with_bomb(game_state: dict, bomb_position: Tuple[int, int]) -> dict:

    new_state = dict(game_state)
    new_state['bombs'] = list(game_state['bombs']) + [(bomb_position, BOMB_TIMER)]
    return new_state


def _make_game_state_as(game_state: dict, position: Tuple[int, int], bombs_left: int = 1) -> dict:

    new_state = dict(game_state)
    new_state['self'] = ('hypothetical', 0, bombs_left, position)
    return new_state


def count_escape_directions(
        game_state: dict,
        position: Tuple[int, int] = None,
        bombs_left: int = 1,
        require_non_pocket: bool = True
) -> int:

    gs = game_state if position is None else _make_game_state_as(game_state, position, bombs_left)
    return sum(
        1 for a in MOVE_DIRECTIONS
        if is_action_safe(a, gs, require_non_pocket_escape=require_non_pocket)
    )


def count_uncontested_escape_directions(
        game_state: dict,
        position: Tuple[int, int] = None,
        bombs_left: int = 1,
        require_non_pocket: bool = True
) -> int:

    gs = game_state if position is None else _make_game_state_as(game_state, position, bombs_left)
    return sum(
        1 for a in MOVE_DIRECTIONS
        if is_action_safe(a, gs, avoid_contested=True, tick_offset=1, require_non_pocket_escape=require_non_pocket)
    )


def count_contested_tiles_we_win(
        own_position: Tuple[int, int],
        tiles: List[Tuple[int, int]],
        others: List[Tuple[int, int]],
        obstacles: np.typing.NDArray[np.bool_],
        own_delay: int = 0
) -> int:

    if not tiles:
        return 0

    own_dist = connected_cell_distances(own_position, obstacles)
    others_dist = [connected_cell_distances(pos, obstacles) for pos in others]

    won = 0
    for (tx, ty) in tiles:
        d_own = own_dist[tx, ty]
        if d_own < 0:
            continue  # we can't even reach it
        d_own += own_delay
        is_fastest = True
        for od in others_dist:
            d_other = od[tx, ty]
            if 0 <= d_other <= d_own:
                is_fastest = False
                break
        if is_fastest:
            won += 1
    return won


def fastest_opponent_threat_tick(
        own_position: Tuple[int, int],
        field: np.typing.NDArray,
        bombs: list,
        others: List[Tuple[int, int]],
) -> int | None:

    threat_tiles = set(get_hypothetical_blast_coords(field, own_position))
    threat_tiles.discard(own_position)
    if not threat_tiles or not others:
        return None

    obstacles = (field != 0)
    for xy, _ in bombs:
        obstacles[xy] = True

    best = None
    for opp_pos in others:
        dist_map = connected_cell_distances(opp_pos, obstacles)
        reachable = [int(dist_map[t]) for t in threat_tiles if dist_map[t] >= 0]
        if not reachable:
            continue
        lethal_at = min(reachable) + BOMB_TIMER
        if best is None or lethal_at < best:
            best = lethal_at
    return best


def evaluate_bomb_placement(
        game_state: dict,
        min_escape_routes: int = 2,
        crate_weight: float = 6.0,
        crate_weight_contested: float = 1.5,
        opponent_kill_bonus: float = 60.0,
        trapped_kill_bonus: float = 40.0,
        trapped_positions: set = None,
        min_score_to_bomb: float = 3.0,
) -> Dict:

    field = game_state['field']
    own_position = game_state['self'][3]
    bombs_left = game_state['self'][2]
    others = [p[3] for p in game_state['others']]
    trapped_positions = trapped_positions or set()

    result = {
        'should_bomb': False, 'score': 0.0, 'crates_hit': [], 'opponents_hit': [],
        'trapped_opponents_hit': [], 'escape_routes': 0, 'uncontested_escape_routes': 0,
        'contested_tiles_won': 0,
    }

    if bombs_left <= 0:
        return result

    blast_coords = get_hypothetical_blast_coords(field, own_position)
    crates_hit = [c for c in blast_coords if field[c] == 1]
    opponents_hit = [o for o in others if o in blast_coords]
    trapped_opponents_hit = [o for o in opponents_hit if o in trapped_positions]

    result['crates_hit'] = crates_hit
    result['opponents_hit'] = opponents_hit
    result['trapped_opponents_hit'] = trapped_opponents_hit

    if not crates_hit and not opponents_hit:
        return result  # nothing to gain, don't waste the bomb / risk

    hypothetical_state = _make_game_state_with_bomb(game_state, own_position)
    threat_tick = fastest_opponent_threat_tick(own_position, field, game_state['bombs'], others)
    no_realistic_threat = threat_tick is None or threat_tick > BOMB_TIMER + EXPLOSION_TIMER + 1

    escape_routes = count_escape_directions(hypothetical_state, require_non_pocket=not no_realistic_threat)
    result['escape_routes'] = escape_routes

    required_escape_routes = 1 if no_realistic_threat else min_escape_routes
    if escape_routes < required_escape_routes:
        return result

    if no_realistic_threat:
        uncontested_escape_routes = escape_routes
    else:
        uncontested_escape_routes = count_uncontested_escape_directions(hypothetical_state)
        if uncontested_escape_routes < 1:
            return result  # every escape could be cut off by an opponent moving into it
    result['uncontested_escape_routes'] = uncontested_escape_routes

    obstacles = (field != 0)
    for o in others:
        obstacles[o] = True
    contested_won = count_contested_tiles_we_win(own_position, crates_hit, others, obstacles)
    contested_lost = len(crates_hit) - contested_won
    result['contested_tiles_won'] = contested_won

    score = (
        crate_weight * contested_won
        + crate_weight_contested * contested_lost
        + opponent_kill_bonus * len(opponents_hit)
        + trapped_kill_bonus * len(trapped_opponents_hit)
    )

    result['score'] = score
    result['should_bomb'] = score >= min_score_to_bomb
    return result


def find_trap_targets(game_state: dict, max_escape_routes: int = 1) -> List[Dict]:

    others = game_state['others']
    targets = []
    for p in others:
        pos = p[3]
        routes = count_escape_directions(game_state, position=pos, bombs_left=p[2])
        if routes <= max_escape_routes:
            targets.append({'position': pos, 'escape_routes': routes})
    targets.sort(key=lambda t: t['escape_routes'])
    return targets


def get_best_crate_targets(
        crates: List[Tuple[int, int]],
        field: np.typing.NDArray,
        top_fraction: float = 0.3,
) -> List[Tuple[int, int]]:

    if not crates:
        return []

    crate_set = set(crates)
    scored = []
    for pos in crates:
        blast = get_hypothetical_blast_coords(field, pos)
        hit_crates = sum(1 for c in blast if c != pos and c in crate_set)
        scored.append((pos, hit_crates))

    scored.sort(key=lambda t: t[1], reverse=True)
    cutoff = max(1, int(len(scored) * top_fraction))
    return [pos for pos, _ in scored[:cutoff]]