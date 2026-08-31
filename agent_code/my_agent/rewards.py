import events as e

# Naive rewards matching actual scoring system
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
}


# Shaped rewards based on game events
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
}



# Dense shaping
COIN_SHAPING_COEF = 0.05
CRATE_SHAPING_COEF = 0.02
DANGER_PENALTY_COEF = 0.05
ESCAPE_BONUS_COEF = 0.05
TRAP_SHAPING_COEF = 0.1