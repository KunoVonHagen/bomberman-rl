import pathlib
import pickle
from gym_environment import BombermanGymEnv
import numpy as np

PICKLE_PATH = pathlib.Path(__file__).parent / "expert_transitions.pkl"

transitions = pickle.load(open(PICKLE_PATH, "rb"))

print(transitions.obs)

unique, counts = np.unique(transitions.acts, return_counts=True)

for u, c in zip(unique, counts):
    print(BombermanGymEnv.ACTIONS[u], c / len(transitions.acts))

# TODO: Filter out transitions that are not interesting for learning and remove them from the dataset
def filter_transitions_uninteresting_transitions(transitions):
    MIN_CRATES_LEFT = 15

    filtered_indices = []
    for i in range(len(transitions.acts)):
        obs = transitions.obs[i]
        field = obs["grid_tensor"][1]
        crates_left = np.sum(field == 1)

        if crates_left >= MIN_CRATES_LEFT:
            filtered_indices.append(i)

    return filtered_indices

# TODO: Balance the dataset along the different actions
