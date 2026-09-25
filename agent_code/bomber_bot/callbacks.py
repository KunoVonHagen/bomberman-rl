from __future__ import annotations

import random

from agent_code.league_bots.core import Context

COIN_RADIUS = 3
BOMB_PROB = 0.85
RECKLESS_PROB = 0.04


def setup(self):
    pass


def act(self, game_state):
    ctx = Context(game_state)
    safe = ctx.safe_actions()
    if not safe:
        return ctx.fallback_action()

    if ctx.can_bomb():
        useful = ctx.opponent_in_blast() or ctx.crates_in_blast() > 0
        if useful and random.random() < BOMB_PROB and ctx.bomb_is_safe():
            return "BOMB"
        if random.random() < RECKLESS_PROB:
            return "BOMB"

    if ctx.coins:
        coin_dist = ctx.distance_map(ctx.coins)
        if min(coin_dist[nxt] for _, nxt in safe) <= COIN_RADIUS:
            action = ctx.step_towards(safe, coin_dist)
            if action:
                return action
    crates = ctx.crate_tiles()
    if crates:
        action = ctx.step_towards(safe, ctx.distance_map(crates))
        if action:
            return action
    if ctx.others:
        action = ctx.step_towards(safe, ctx.distance_map(ctx.others))
        if action:
            return action
    return ctx.wander(safe)