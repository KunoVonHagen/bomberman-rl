import numpy as np
import settings as s
from features import get_features


def observation_to_game_state(obs):
    grid = obs["grid_tensor"]

    field = np.zeros((17, 17), dtype=np.int8)
    field[grid[0] == 1] = -1
    field[grid[1] == 1] = 1

    # self
    sx, sy = np.argwhere(grid[3] == 1)[0]
    bombs_left = bool(grid[7, sx, sy])

    self_state = (
        None,          # name unknown
        0,             # score unknown
        bombs_left,
        (int(sx), int(sy)),
    )

    # opponents
    others = []

    opponent_positions = np.argwhere(grid[5] == 1)

    for x, y in opponent_positions:
        others.append((
            None,                  # name unknown
            0,                     # score unknown
            bool(grid[7, x, y]),
            (int(x), int(y)),
        ))

    # bombs
    bombs = []

    for timer in range(s.BOMB_TIMER):
        layer = 8 + timer
        for x, y in np.argwhere(grid[layer] == 1):
            bombs.append(((int(x), int(y)), timer))

    # visible coins
    coins = [
        (int(x), int(y))
        for x, y in np.argwhere(grid[2] == 1)
    ]

    # explosion map
    explosion_map = np.zeros((17, 17), dtype=np.int8)

    first_layer = 7 + 2 * s.BOMB_TIMER
    for timer in range(s.EXPLOSION_TIMER):
        layer = first_layer + timer + 1
        explosion_map[grid[layer] == 1] = timer

    return {
        "round": None,
        "step": None,
        "field": field,
        "self": self_state,
        "others": others,
        "bombs": bombs,
        "coins": coins,
        "user_input": None,
        "explosion_map": explosion_map,
    }


def game_state_to_observation(game_state, precomputed_blast_map):
    H, W = game_state["field"].shape

    n_layers = 8 + 2 * s.BOMB_TIMER + s.EXPLOSION_TIMER

    grid = np.zeros((n_layers, H, W), dtype=np.float32)

    field = game_state["field"]

    # walls
    grid[0] = field == -1

    # crates
    grid[1] = field == 1

    # coins
    for x, y in game_state["coins"]:
        grid[2, x, y] = 1

    # self
    _, _, bombs_left, (x, y) = game_state["self"]

    grid[3, x, y] = 1
    grid[4] = precomputed_blast_map[(x, y)]
    grid[7, x, y] = float(bombs_left)

    # opponents
    for _, _, can_bomb, (x, y) in game_state["others"]:
        grid[5, x, y] = 1
        grid[6] += precomputed_blast_map[(x, y)]
        grid[7, x, y] = float(can_bomb)

    grid[6] = (grid[6] > 0).astype(np.float32)

    # bombs
    for (x, y), timer in game_state["bombs"]:
        grid[8 + timer, x, y] = 1
        grid[8 + s.BOMB_TIMER + timer] = precomputed_blast_map[(x, y)]

    # explosions
    first = 8 + 2 * s.BOMB_TIMER

    explosion_map = game_state["explosion_map"]

    for timer in range(s.EXPLOSION_TIMER):
        grid[first + timer] = (explosion_map == timer).astype(np.float32)

    return {
        "grid_tensor": grid,
        "features": get_features(grid),
    }