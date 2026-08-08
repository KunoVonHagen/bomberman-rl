import settings as s

#: Maps each action name to its (dx, dy) grid offset.
ACTION_DELTA = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
}

#: The four movement offsets, without action labels.
DIRECTIONS = list(ACTION_DELTA.values())

#: How many future steps to reason about when planning around bomb danger.
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1

# Weights used to score candidate bombing spots.
CRATE_SCORE = 2.0
TERRITORY_SCORE = 0.0
DISTANCE_PENALTY = 0.5
OPPONENT_ADVANTAGE_PENALTY = 0.5
