import numpy as np
from typing import List, Tuple

from .settings import BOMB_TIMER, BOMB_POWER, EXPLOSION_TIMER

def predict_explosions(
        field: np.typing.NDArray[np.int_],
        bombs: np.typing.NDArray[np.int_],
        explosion_map: np.typing.NDArray[np.int_],
        delta_t: int
) -> Tuple[np.typing.NDArray[np.int_], np.typing.NDArray[np.int_], np.typing.NDArray[np.int_]]:

    """
    Forecasts explosions into the future.

    :param field: The game field as a 2D numpy array. -1 for walls, 1 for crates, 0 for empty cells.
    :param bombs: A 2D numpy array of the same shape as field, where each cell contains the timer of a bomb if present,
     or 0 otherwise.
    :param explosion_map: A 2D numpy array of the same shape as field, where each cell contains the remaining time an
     explosion will last, or 0 if no explosion is present.
    :param delta_t: The number of time steps to predict into the future.

    :return: A tuple of three 2D numpy arrays (predicted_field, predicted_bombs, predicted_explosions).
    """

    # Create copies of the input arrays to avoid modifying them
    predicted_field = field.copy()
    predicted_bombs = bombs.copy()
    predicted_explosions = explosion_map.copy()

    for t in range(delta_t):
        # Decrease bomb timers
        predicted_bombs[predicted_bombs > 0] -= 1

        # Handle explosions
        exploding_cells = np.where(predicted_bombs == 0)
        for x, y in zip(*exploding_cells):
            # Set explosion timer
            predicted_explosions[x, y] = EXPLOSION_TIMER

            # Propagate explosion in all four directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for power in range(1, BOMB_POWER + 1):
                    nx, ny = x + dx * power, y + dy * power
                    if 0 <= nx < predicted_field.shape[0] and 0 <= ny < predicted_field.shape[1]:
                        if predicted_field[nx, ny] == -1:  # Wall
                            break
                        predicted_explosions[nx, ny] = EXPLOSION_TIMER
                        if predicted_field[nx, ny] == 1:  # Crate
                            break

        # Decrease explosion timers
        predicted_explosions[predicted_explosions > 0] -= 1

    return predicted_field, predicted_bombs, predicted_explosions
