import time

from gym_environment import BombermanGymEnv
from environment import WorldArgs

import os
import pathlib
import torch.nn as nn
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import freeze_support
from datetime import datetime


class BombermanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256):

        super().__init__(observation_space, features_dim)

        grid_space = observation_space["grid_tensor"]
        feature_space = observation_space["features"]

        n_input_channels = grid_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.ReLU(),

            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.zeros(1, *grid_space.shape, dtype=torch.float32)
            n_cnn_flatten = self.cnn(sample).shape[1]

        self.features_preprocess_fc = nn.Sequential(
            nn.Linear(feature_space.shape[0], 64),
            nn.ReLU()
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(n_cnn_flatten + 64, 512),
            nn.ReLU(),

            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        grid_tensor = observations["grid_tensor"].float()
        features = observations["features"].float()

        cnn_output = self.cnn(grid_tensor)
        features_output = self.features_preprocess_fc(features)

        combined_input = torch.cat((cnn_output, features_output), dim=1)
        return self.combined_fc(combined_input)

classic_env_args = WorldArgs(
    scenario="classic",
    seed=None,
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


def get_model(N_STEPS, BATCH_SIZE, env):
    policy_kwargs = dict(
        features_extractor_class=BombermanCNN,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_log",
        verbose=1,
        n_epochs=4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=lambda progress: progress * 3e-4 + (1-progress) * 1e-4,
        ent_coef=0.005,
        gamma=0.999
    )

    return model


def get_env(N_ENVS, opponents):
    env = make_vec_env(
        lambda: BombermanGymEnv(classic_env_args, opponents=opponents),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv
    )
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True
    )

    return env

def play_test_game(model, opponents):
    os.makedirs(pathlib.Path(__file__).parent / "replays", exist_ok=True)
    os.makedirs(pathlib.Path(__file__).parent / "logs" / "test", exist_ok=True)
    test_env_args = WorldArgs(
        scenario="classic",
        seed=None,
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
    grid_tensor, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(grid_tensor, deterministic=True)
        grid_tensor, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        test_env.render()
    test_env.close()


def run_epoch(N_STEPS, N_ENVS, epoch, model, opponents, training_start=None):
    model.learn(
        total_timesteps=N_STEPS * N_ENVS + 1,
        reset_num_timesteps=False,
        tb_log_name=f"PPO_{training_start}" if training_start else "PPO"
    )

    play_test_game(model, opponents)

    model.save(f"models/ppo_bomberman_{(epoch + 1) * N_STEPS * N_ENVS}")


def env_step_test(TEST_ROUNDS):
    env = BombermanGymEnv(classic_env_args, opponents=[])

    total_start_time = time.time()

    total_steps = 0

    for round_num in range(TEST_ROUNDS):
        round_steps = 0

        round_start_time = time.time()
        grid_tensor = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            grid_tensor, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            round_steps += 1

        round_end_time = time.time()
        round_duration = round_end_time - round_start_time

        print(f"Round {round_num + 1}: {round_steps/round_duration:.2f} fps")

        total_steps += round_steps

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"Total steps: {total_steps}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average steps per second: {total_steps / total_duration:.2f}")



def main(N_ENVS, N_STEPS, BATCH_SIZE, TOTAL_EPOCHS, opponents):
    freeze_support()
    env = get_env(N_ENVS, opponents)
    model = get_model(N_STEPS, BATCH_SIZE, env)

    training_start = f"{datetime.now():%Y%m%d-%H%M%S}"

    for epoch in range(TOTAL_EPOCHS):
        run_epoch(N_STEPS, N_ENVS, epoch, model, opponents, training_start)





if __name__ == "__main__":
    N_STEPS = 512
    N_ENVS = 12
    BATCH_SIZE = 64
    TOTAL_EPOCHS = 1 + 50_000_000 // (N_STEPS * N_ENVS)

    opponents = []

    env_step_test(10)

    main(N_ENVS, N_STEPS, BATCH_SIZE, TOTAL_EPOCHS, opponents)



