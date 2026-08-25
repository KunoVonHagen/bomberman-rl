import json
import os
import random

import numpy as np

import settings as s
from .constants import ACTION_DELTA, DIRECTIONS, HORIZON
from .pathfinding import bfs_next_step, find_reachable_tiles, bfs_distance
from .blast import get_blast_coords, project_cleared_field
from .traps import find_traps
from .danger import compute_danger_map, find_escape_action, has_escape_route, can_escape_own_bomb
from .targeting import fastest_opponent_to, best_bomb_spot


REPLAY_LOG_PATH = os.path.join(os.path.dirname(__file__),
    "simple_agent_thoughts.json",
)

SAVE_REPLAY = getattr(s, "SAVE_REPLAY", False)


def _jsonable(value):
    """Convert numpy/sets/tuples and other game-state values to JSON-safe data."""
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)

    return value


def _write_replay_frame(self, game_state, action, reason, danger_map=None, pockets=None):
    """Append one visualization frame to the replay JSONL file.

    This records decision-relevant state/features, not private hidden chain-of-thought.
    The supplied replay viewer consumes the fields written here.
    """
    field = game_state["field"]
    pos = game_state["self"][3]
    bombs = game_state["bombs"]
    coins = game_state["coins"]
    explosion_map = game_state["explosion_map"]
    others = game_state.get("others", [])

    opponent_positions = {opp_pos for _, _, _, opp_pos in others}
    bomb_positions = {b_pos for b_pos, _ in bombs}

    if danger_map is None:
        danger_map = compute_danger_map(
            field, bombs, explosion_map, s.BOMB_POWER, HORIZON
        )
    danger_zone = set().union(*danger_map)

    reachable = find_reachable_tiles(
        field,
        pos,
        avoid=danger_zone | bomb_positions | opponent_positions,
    )

    if pockets is None:
        pockets = find_traps(field, s.BOMB_POWER) if others else {}
    trap_tiles = set()
    chokepoints = set()
    for choke, pocket, _dist_to_choke, _lethal in pockets.values():
        chokepoints.add(choke)
        trap_tiles.update(pocket)

    frame = {
        "field": _jsonable(field),
        "danger_map": _jsonable(danger_map),
        "reachable_tiles": [_jsonable(t) for t in reachable],
        "traps": _jsonable(trap_tiles),
        "chokepoints": _jsonable(chokepoints),
        "coins": _jsonable(coins),
        "opponents": _jsonable(opponent_positions),
        "position": _jsonable(pos),
        "bombs": _jsonable(bombs),
        "explosion_map": _jsonable(explosion_map),
        "action": action,
        "decision_reason": reason,
    }

    os.makedirs(os.path.dirname(REPLAY_LOG_PATH), exist_ok=True)
    with open(REPLAY_LOG_PATH, "a", encoding="utf-8") as replay_file:
        replay_file.write(json.dumps(frame, separators=(",", ":")) + "\n")


def _finish(self, game_state, action, reason, danger_map=None, pockets=None):
    """Record the chosen action, then return it to the game engine."""
    if SAVE_REPLAY:
        _write_replay_frame(self, game_state, action, reason, danger_map, pockets)
    return action


def setup(self):
    """One-time agent initialization and replay-file reset."""
    self.own_bomb_pos = None
    self._replay_initialized = True

    if SAVE_REPLAY:
        os.makedirs(os.path.dirname(REPLAY_LOG_PATH), exist_ok=True)
        with open(REPLAY_LOG_PATH, "w", encoding="utf-8"):
            pass


def act(self, game_state: dict):
    """Choose this step's action and append a replay frame for the viewer."""
    """Choose this step's action: dodge danger, spring/avoid traps, strike opponents, collect coins, or bomb crates."""
    field = game_state['field']
    pos = game_state['self'][3]
    bombs_left = game_state['self'][2]
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    others = game_state.get('others', [])
    opponent_positions = {opp_pos for _, _, _, opp_pos in others}
    bomb_positions = {b_pos for b_pos, _ in bombs}
    if not hasattr(self, 'own_bomb_pos'):
        self.own_bomb_pos = None
    if bombs_left:
        self.own_bomb_pos = None

    danger_map = compute_danger_map(field, bombs, explosion_map, s.BOMB_POWER, HORIZON)
    danger_zone = set().union(*danger_map)

    # 1. If currently standing in a future blast, escape first.
    currently_unsafe = any(pos in danger_map[t] for t in range(HORIZON + 1))
    if currently_unsafe:
        escape = find_escape_action(field, bombs, danger_map, pos, HORIZON, opponent_positions)
        if escape is not None:
            return _finish(self, game_state, escape, "escape danger", danger_map)
        x, y = pos
        occupied = {b_pos for b_pos, _ in bombs} | opponent_positions
        moves = [move for move, (dx, dy) in ACTION_DELTA.items()
                 if field[x + dx, y + dy] == 0 and (x + dx, y + dy) not in occupied]
        return _finish(self, game_state, random.choice(moves) if moves else 'WAIT', "emergency safe move", danger_map)

    # 2. Identify dead-end pockets, both now and after pending bombs clear crates.
    pockets = find_traps(field, s.BOMB_POWER) if others else {}
    future_pockets = {}
    if others and bombs:
        projected_field = project_cleared_field(field, bombs, s.BOMB_POWER)
        future_pockets = find_traps(projected_field, s.BOMB_POWER)

    # 3. If we're trapped and an opponent could seal us in, flee toward the chokepoint.
    if pos in pockets:
        choke, pocket, _dist_to_choke, _lethal = pockets[pos]
        our_dist = bfs_distance(field, pos, choke, avoid=danger_zone | bomb_positions | opponent_positions)
        if our_dist is not None:
            opp_dist = fastest_opponent_to(field, choke, others, exclude=pocket)
            if opp_dist is not None and opp_dist <= our_dist + s.BOMB_TIMER:
                move = bfs_next_step(field, pos, {choke}, avoid=danger_zone | bomb_positions | opponent_positions)
                if move is not None:
                    return _finish(self, game_state, move, "move toward tactical target", danger_map, pockets)

    # 4. If an opponent is trapped and we can beat them to the chokepoint, seal or bomb it.
    if bombs_left:
        for _, _, _, opp_pos in others:
            info = pockets.get(opp_pos) or future_pockets.get(opp_pos)
            if info is None:
                continue
            choke, pocket, _dist_to_choke, lethal = info
            if field[choke] != 0:
                continue
            their_pocket = find_reachable_tiles(field, opp_pos, avoid={choke})
            if pos in their_pocket and pos != choke:
                continue
            our_dist = bfs_distance(field, pos, choke, avoid=danger_zone | bomb_positions | opponent_positions)
            their_dist = bfs_distance(field, opp_pos, choke)
            if our_dist is None or their_dist is None or our_dist > their_dist:
                continue
            if pos == choke:
                if lethal and can_escape_own_bomb(field, bombs, explosion_map, choke, s.BOMB_POWER, s.BOMB_TIMER, HORIZON, opponent_positions):
                    self.own_bomb_pos = pos
                    return _finish(self, game_state, 'BOMB', "place a tactical bomb", danger_map, pockets)
                return _finish(self, game_state, 'WAIT', "wait for a tactical opening", danger_map, pockets)
            else:
                move = bfs_next_step(field, pos, {choke}, avoid=danger_zone | bomb_positions | opponent_positions)
                if move is not None:
                    return _finish(self, game_state, move, "move toward tactical target", danger_map, pockets)

    # 5. Look for a bomb spot that would hit a reachable, inescapable opponent.
    if bombs_left and others:
        reachable = find_reachable_tiles(field, pos, avoid=danger_zone | bomb_positions | opponent_positions)
        best_strike = None
        for _, _, _, opp_pos in others:
            for spot in get_blast_coords(field, opp_pos, s.BOMB_POWER):
                dist = reachable.get(spot)
                if dist is None or dist > s.BOMB_TIMER:
                    continue
                if best_strike is not None and dist >= best_strike[0]:
                    continue
                opp_occupied = ({b_pos for b_pos, _ in bombs} | opponent_positions | {spot}) - {opp_pos}
                if has_escape_route(field, bombs, explosion_map, spot, s.BOMB_TIMER, s.BOMB_POWER, HORIZON, opp_pos, opp_occupied):
                    continue
                if not can_escape_own_bomb(field, bombs, explosion_map, spot, s.BOMB_POWER, s.BOMB_TIMER, HORIZON, opponent_positions):
                    continue
                best_strike = (dist, spot)
        if best_strike is not None:
            _, target_pos = best_strike
            if target_pos == pos:
                self.own_bomb_pos = pos
                return _finish(self, game_state, 'BOMB', "place a tactical bomb", danger_map, pockets)
            move = bfs_next_step(field, pos, {target_pos}, avoid=danger_zone | bomb_positions | opponent_positions)
            if move is not None:
                return _finish(self, game_state, move, "move toward tactical target", danger_map, pockets)

    # 6. Mark tiles that are only risky because an opponent could trap us there.
    risky = set()
    seen_pockets = set()
    for choke, pocket, dist_to_choke, _lethal in pockets.values():
        if pocket in seen_pockets:
            continue
        seen_pockets.add(pocket)
        opp_dist = fastest_opponent_to(field, choke, others, exclude=pocket)
        if opp_dist is None:
            continue
        for t in pocket:
            if opp_dist <= dist_to_choke.get(t, 0) + s.BOMB_TIMER:
                risky.add(t)

    # 7. Go collect the nearest safe coin.
    coin_action = bfs_next_step(field, pos, coins, avoid=danger_zone | risky | bomb_positions | opponent_positions)
    if coin_action is None:
        coin_action = bfs_next_step(field, pos, coins, avoid=danger_zone | bomb_positions | opponent_positions)

    # 8. No coin reachable: head toward the best crate-clearing bomb spot instead.
    if coin_action is None and bombs_left:
        avoid_for_spot = danger_zone | risky | bomb_positions | opponent_positions
        target = best_bomb_spot(field, bombs, explosion_map, avoid_for_spot, pos, s.BOMB_POWER, s.BOMB_TIMER, HORIZON, others, opponent_positions)
        if target is None:
            avoid_for_spot = danger_zone | bomb_positions | opponent_positions
            target = best_bomb_spot(field, bombs, explosion_map, avoid_for_spot, pos, s.BOMB_POWER, s.BOMB_TIMER, HORIZON, others, opponent_positions)
        if target is not None:
            target_pos, _ = target
            if target_pos == pos:
                self.own_bomb_pos = pos
                return _finish(self, game_state, 'BOMB', "place a tactical bomb", danger_map, pockets)
            move = bfs_next_step(field, pos, {target_pos}, avoid=avoid_for_spot)
            if move is not None:
                return _finish(self, game_state, move, "move toward tactical target", danger_map, pockets)

    if coin_action is not None:
        return _finish(self, game_state, coin_action, "collect nearest safe coin", danger_map, pockets)

    # 9. Retreat toward our own just-placed bomb's safe zone while it counts down.
    if self.own_bomb_pos is not None:
        dist_to_bomb = find_reachable_tiles(field, self.own_bomb_pos, avoid=bomb_positions - {self.own_bomb_pos})
        reachable_safe = find_reachable_tiles(field, pos, avoid=danger_zone | bomb_positions | opponent_positions)
        candidates = [t for t in reachable_safe if t in dist_to_bomb]
        if candidates:
            best_tile = min(candidates, key=lambda t: (dist_to_bomb[t], reachable_safe[t]))
            if best_tile == pos:
                return _finish(self, game_state, 'WAIT', "wait for a tactical opening", danger_map, pockets)
            move = bfs_next_step(field, pos, {best_tile}, avoid=danger_zone | bomb_positions | opponent_positions)
            if move is not None:
                return _finish(self, game_state, move, "move toward tactical target", danger_map, pockets)

    # 10. Nothing better to do: hunt opponents, favoring chokepoints we can beat them to.
    if others:
        hunt_targets = set()
        for opp_x, opp_y in opponent_positions:
            for dx, dy in DIRECTIONS:
                nx, ny = opp_x + dx, opp_y + dy
                if field[nx, ny] == 0:
                    hunt_targets.add((nx, ny))
        hunt_targets -= opponent_positions

        if bombs_left:
            seen_chokes = set()
            for choke, pocket, _dist_to_choke, _lethal in pockets.values():
                if choke in seen_chokes or field[choke] != 0:
                    continue
                seen_chokes.add(choke)
                opp_dist = fastest_opponent_to(field, choke, others, exclude=pocket)
                our_dist = bfs_distance(field, pos, choke, avoid=danger_zone | bomb_positions | opponent_positions)
                if opp_dist is None or our_dist is None:
                    continue
                if our_dist <= opp_dist + s.BOMB_TIMER:
                    hunt_targets.add(choke)

        hunt_action = bfs_next_step(field, pos, hunt_targets, avoid=danger_zone | risky | bomb_positions | opponent_positions)
        if hunt_action is None:
            hunt_action = bfs_next_step(field, pos, hunt_targets, avoid=danger_zone | bomb_positions | opponent_positions)
        if hunt_action is not None:
            return _finish(self, game_state, hunt_action, "hunt opponent or approach chokepoint", danger_map, pockets)

    # 11. Fallback: take any safe move, preferring non-risky tiles.
    x, y = pos
    occupied = {b_pos for b_pos, _ in bombs} | opponent_positions
    valid = [move for move, (dx, dy) in ACTION_DELTA.items()
             if field[x + dx, y + dy] == 0
             and (x + dx, y + dy) not in occupied
             and (x + dx, y + dy) not in danger_zone]
    preferred = [m for m in valid
                 if (x + ACTION_DELTA[m][0], y + ACTION_DELTA[m][1]) not in risky]
    if preferred:
        valid = preferred
    valid.append('WAIT')

    return _finish(self, game_state, random.choice(valid), "fallback safe action", danger_map, pockets)