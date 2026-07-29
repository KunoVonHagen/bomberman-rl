from gym_environment import BombermanGymEnv
from environment import WorldArgs

import time
import numpy as np
import os
import torch.nn as nn
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import freeze_support
from datetime import datetime
from imitation.data.types import Transitions, DictObs
from imitation.algorithms import bc
from input_processing import observation_to_game_state
from agent_code.my_agent.callbacks import act as expert_act, setup as expert_setup
import tqdm
import pickle
import pathlib

class ExpertPolicy:
    def __init__(self, env):
        self.env = env
        expert_setup(env.agent)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        game_state = observation_to_game_state(observation)
        action = expert_act(self.env.agent, game_state)
        action = BombermanGymEnv.ACTION_INDICES[action]
        return action, state

def dagger_collect(
    env: BombermanGymEnv,
    policy,
    n_episodes: int,
):
    """
    Collect demonstrations using the current policy for exploration while
    labeling every visited state with the expert action (DAgger).
    """

    if expert_setup is not None:
        expert_setup(env.agent)

    all_obs = []
    all_next_obs = []
    all_actions = []
    all_dones = []

    for _ in tqdm.tqdm(range(n_episodes), desc="Collecting DAgger data"):

        obs, _ = env.reset()
        done = False

        while not done:

            # Current policy decides where we go
            policy_action, _ = policy.predict(obs, deterministic=True)

            # Expert labels this state
            game_state = observation_to_game_state(obs)
            expert_action = expert_act(env.agent, game_state)
            expert_action = BombermanGymEnv.ACTION_INDICES[expert_action]

            next_obs, reward, terminated, truncated, _ = env.step(policy_action)
            done = terminated or truncated

            all_obs.append(obs)
            all_next_obs.append(next_obs)
            all_actions.append(expert_action)
            all_dones.append(done)

            obs = next_obs

    obs = DictObs({
        "grid_tensor": np.stack([o["grid_tensor"] for o in all_obs]),
        "features": np.stack([o["features"] for o in all_obs]),
    })

    next_obs = DictObs({
        "grid_tensor": np.stack([o["grid_tensor"] for o in all_next_obs]),
        "features": np.stack([o["features"] for o in all_next_obs]),
    })

    return Transitions(
        obs=obs,
        acts=np.array(all_actions),
        next_obs=next_obs,
        dones=np.array(all_dones, dtype=bool),
        infos=np.array([{}] * len(all_actions), dtype=object),
    )

def merge_transitions(old, new):
    if old is None:
        return new

    obs = DictObs({
        "grid_tensor": np.concatenate([
            old.obs._d["grid_tensor"],
            new.obs._d["grid_tensor"],
        ]),
        "features": np.concatenate([
            old.obs._d["features"],
            new.obs._d["features"],
        ]),
    })

    next_obs = DictObs({
        "grid_tensor": np.concatenate([
            old.next_obs._d["grid_tensor"],
            new.next_obs._d["grid_tensor"],
        ]),
        "features": np.concatenate([
            old.next_obs._d["features"],
            new.next_obs._d["features"],
        ]),
    })

    return Transitions(
        obs=obs,
        acts=np.concatenate([old.acts, new.acts]),
        next_obs=next_obs,
        dones=np.concatenate([old.dones, new.dones]),
        infos=np.concatenate([old.infos, new.infos]),
    )
    
        

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

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.zeros(1, *grid_space.shape, dtype=torch.float32)
            n_cnn_flatten = self.cnn(sample).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(n_cnn_flatten, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.features_preprocess_fc = nn.Sequential(
            nn.Linear(feature_space.shape[0], 64),
            nn.ReLU()
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(n_cnn_flatten + 64, 256),
            nn.ReLU(),

            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        grid_tensor = observations["grid_tensor"].float()
        #features = observations["features"].float()

        cnn_output = self.cnn(grid_tensor)
        #features_output = self.features_preprocess_fc(features)

        #combined_input = torch.cat((cnn_output, features_output), dim=1)
        #return self.combined_fc(combined_input)

        return self.cnn_fc(cnn_output)

CLASSIC_ENV_ARGS = WorldArgs(
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


def get_model(env, PPO_PARAMS):
    policy_kwargs = dict(
        features_extractor_class=BombermanCNN,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_log",
        verbose=1,
        **PPO_PARAMS
    )

    return model


def get_env(N_ENVS, opponents):
    env = make_vec_env(
        lambda: BombermanGymEnv(CLASSIC_ENV_ARGS, opponents=opponents),
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
    env = BombermanGymEnv(CLASSIC_ENV_ARGS, opponents=[])

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


def get_demo_environment(opponents):
    env = BombermanGymEnv(CLASSIC_ENV_ARGS, opponents=opponents)
    return env


def main(N_ENVS, TOTAL_EPOCHS, N_DEMONSTRATION_EPISODES, opponents, PPO_PARAMS):
    freeze_support()
    env = get_env(N_ENVS, opponents)
    model = get_model(env, PPO_PARAMS)

    training_start = f"{datetime.now():%Y%m%d-%H%M%S}"

    demo_env = get_demo_environment(opponents)

    dataset = None

    for epoch in range(TOTAL_EPOCHS):
        # PPO improvement
        run_epoch(
            PPO_PARAMS["n_steps"],
            N_ENVS,
            epoch,
            model,
            opponents,
            training_start,
        )

        # Collect states visited by the improved policy
        new_data = dagger_collect(
            demo_env,
            model,
            N_DEMONSTRATION_EPISODES,
        )

        dataset = merge_transitions(dataset, new_data)

        bc_trainer = bc.BC(
            observation_space=demo_env.observation_space,
            action_space=demo_env.action_space,
            demonstrations=dataset,
            policy=model.policy,
            rng=np.random.default_rng(),
        )

        bc_trainer.train(
            n_epochs=2,
        )


if __name__ == "__main__":
    PPO_PARAMS = {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "target_kl": 0.03,
    }

    N_DEMONSTRATION_EPISODES = 10

    N_ENVS = 24
    TOTAL_EPOCHS = 1 + 50_000_000 // (PPO_PARAMS["n_steps"] * N_ENVS)
    opponents = []

    main(N_ENVS, TOTAL_EPOCHS, N_DEMONSTRATION_EPISODES, opponents, PPO_PARAMS)



