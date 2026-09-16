import events as e

RIVAL_KILLED_OPPONENT = "RIVAL_KILLED_OPPONENT"

SIMPLE_EVENT_REWARDS = {
    e.MOVED_LEFT: 0,
    e.MOVED_RIGHT: 0,
    e.MOVED_UP: 0,
    e.MOVED_DOWN: 0,
    e.WAITED: 0,
    e.INVALID_ACTION: -1,

    e.BOMB_DROPPED: 0,
    e.BOMB_EXPLODED: 0,

    e.CRATE_DESTROYED: 0,
    e.COIN_FOUND: 0,
    e.COIN_COLLECTED: 1,

    e.KILLED_OPPONENT: 5,
    e.KILLED_SELF: 0,

    e.GOT_KILLED: 0,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0,

    RIVAL_KILLED_OPPONENT: 0,
}

EVENT_REWARDS = {
    e.MOVED_LEFT: 0,
    e.MOVED_RIGHT: 0,
    e.MOVED_UP: 0,
    e.MOVED_DOWN: 0,
    e.WAITED: -0.01,
    e.INVALID_ACTION: -0.2,

    e.BOMB_DROPPED: 0,
    e.BOMB_EXPLODED: 0.,

    e.CRATE_DESTROYED: 0.3,
    e.COIN_FOUND: 0,
    e.COIN_COLLECTED: 1.0,

    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -1.0,

    e.GOT_KILLED: -5.0,
    e.OPPONENT_ELIMINATED: 0,
    e.SURVIVED_ROUND: 0.0,

    RIVAL_KILLED_OPPONENT: -2.0,
}

COIN_SHAPING_COEF = 0.05
CRATE_SHAPING_COEF = 0.02
DANGER_PENALTY_COEF = 0.05
ESCAPE_BONUS_COEF = 0.05
TRAP_SHAPING_COEF = 0.1


def build_event_rewards(cfg) -> dict:
    """
    Build a dictionary of event rewards based on the provided configuration.
    """
    return {
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: cfg.waited,
        e.INVALID_ACTION: cfg.invalid_action,

        e.BOMB_DROPPED: cfg.bomb_dropped,
        e.BOMB_EXPLODED: cfg.bomb_exploded,

        e.CRATE_DESTROYED: cfg.crate_destroyed,
        e.COIN_FOUND: cfg.coin_found,
        e.COIN_COLLECTED: cfg.coin_collected,

        e.KILLED_OPPONENT: cfg.killed_opponent,
        e.KILLED_SELF: cfg.killed_self,

        e.GOT_KILLED: cfg.got_killed,
        e.OPPONENT_ELIMINATED: cfg.opponent_eliminated,
        e.SURVIVED_ROUND: cfg.survived_round,

        RIVAL_KILLED_OPPONENT: cfg.rival_killed_opponent,
    }