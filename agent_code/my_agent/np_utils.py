import numpy as np
from typing import List, Tuple


def get_wall_array(field: np.typing.NDArray[np.int_]) -> np.typing.NDArray[np.bool_]:
    """
    :param field: np.ndarray with shape [height, width] and entries -1 for wall, 1 for crate and 0 for empty cell

    :return: np.ndarray with shape [height, width] and entries True for wall, False otherwise
    """

    return field == -1


def get_crate_array(field: np.typing.NDArray[np.int_]) -> np.typing.NDArray[np.bool_]:
    """
    :param field: np.ndarray with shape [height, width] and entries -1 for wall, 1 for crate and 0 for empty cell

    :return: np.ndarray with shape [height, width] and entries True for crate, False otherwise
    """
    return field == 1


def get_coin_array(coins: List[Tuple[int, int]], field_shape: Tuple[int, int]) -> np.typing.NDArray[np.bool_]:
    """
    :param coins: List of tuples with coordinates of coins
    :param field_shape: Tuple of (height, width) of the game field

    :return: np.ndarray with shape [height, width] and entries True for coin, False otherwise
    """
    coin_array = np.zeros(field_shape, dtype=bool)
    for coin in coins:
        coin_array[coin] = True
    return coin_array


def get_bombs_array(bombs: List[Tuple[Tuple[int, int], int]], field_shape: Tuple[int, int]) -> np.typing.NDArray[np.bool_]:
    """
    :param bombs: List of tuples with coordinates of bombs and their timers
    :param field_shape: Tuple of (height, width) of the game field

    :return: np.ndarray with shape [height, width] and entries True for bomb, False otherwise
    """
    bombs_array = np.zeros(field_shape, dtype=bool)
    for bomb in bombs:
        bombs_array[bomb[0]] = True
    return bombs_array


def get_other_agents_array(agents: List[Tuple[int, int]], field_shape: Tuple[int, int]) -> np.typing.NDArray[np.bool_]:
    """
    :param agents: List of tuples with coordinates of other agents
    :param field_shape: Tuple of (height, width) of the game field

    :return: np.ndarray with shape [height, width] and entries True for other agent, False otherwise
    """
    agents_array = np.zeros(field_shape, dtype=bool)
    for agent in agents:
        agents_array[agent] = True
    return agents_array

