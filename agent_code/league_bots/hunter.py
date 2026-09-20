from __future__ import annotations

import random

from agent_code.league_bots.core import Context

COIN_RADIUS = 5
CRATE_BOMB_PROB = 0.5


def setup(self):
    pass


def act(self, game_state):
    ctx = Context(game_state)
    safe = ctx.safe_actions()
    if not safe:
        return ctx.fallback_action()

    coin_dist = ctx.distance_map(ctx.coins) if ctx.coins else None
    near_coin = coin_dist is not None and min(coin_dist[nxt] for _, nxt in safe) <= COIN_RADIUS

    if ctx.can_bomb():
        want = ctx.opponent_in_blast()
        if not want and not near_coin and ctx.crates_in_blast() and random.random() < CRATE_BOMB_PROB:
            want = True
        if want and ctx.bomb_is_safe():
            return "BOMB"

    if near_coin:
        action = ctx.step_towards(safe, coin_dist)
        if action:
            return action
    if ctx.others:
        action = ctx.step_towards(safe, ctx.distance_map(ctx.others))
        if action:
            return action
    crates = ctx.crate_tiles()
    if crates:
        action = ctx.step_towards(safe, ctx.distance_map(crates))
        if action:
            return action
    return ctx.wander(safe)
