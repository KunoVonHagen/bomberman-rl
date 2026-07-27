from gym_environment import BombermanGymEnv
from environment import WorldArgs

import os
import pathlib
import torch.nn as nn
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env


N_STEPS = 1024
N_ENVS = 8
TOTAL_EPOCHS = 50_000_000 // (N_STEPS * N_ENVS)

class BombermanCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256):

        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        w, h = observation_space.shape[1], observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


classic_env_args = WorldArgs(
    scenario="classic",
    seed=42,
    silence_errors=True,
    no_gui=True,
    make_video=False,
    save_replay=False,
    save_stats=True,
    turn_based=False,
    update_interval=0.1,
    log_dir=str(pathlib.Path(__file__).parent / "logs"),
    match_name=None,
    fps=60,
    replay=False,
    continue_without_training=False
)
opponents = []


env = make_vec_env(
    lambda: BombermanGymEnv(classic_env_args, opponents=opponents),
    n_envs=N_ENVS
)
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True
)

policy_kwargs = dict(
    features_extractor_class=BombermanCNN,
)

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./tensorboard_log",
    verbose=1,
    n_steps=N_STEPS,
    batch_size=256,
    learning_rate=3e-4,
    ent_coef=0.01,
)


def play_test_game(model):
    os.makedirs(pathlib.Path(__file__).parent / "replays", exist_ok=True)
    os.makedirs(pathlib.Path(__file__).parent / "logs" / "test", exist_ok=True)
    test_env_args = WorldArgs(
    scenario="coin-heaven",
    seed=42,
    silence_errors=True,
    no_gui=True,
    make_video=False,
    save_replay=True,
    save_stats=True,
    turn_based=False,
    update_interval=0.1,
    log_dir=str(pathlib.Path(__file__).parent / "logs" / "test"),
    match_name=None,
    fps=60,
    replay=False,
    continue_without_training=False
)
    test_env = BombermanGymEnv(test_env_args, opponents=opponents)
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        test_env.render()
    test_env.close()


for epoch in range(TOTAL_EPOCHS):
    if epoch > 0:
        play_test_game(model)

    model.learn(
        total_timesteps=N_STEPS * N_ENVS+1,
        reset_num_timesteps=False
    )

    if epoch > 0:
        model.save(f"models/ppo_bomberman_{epoch * N_STEPS * N_ENVS}")





