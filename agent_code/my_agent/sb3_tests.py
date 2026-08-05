from agent_code.my_agent.gym_environment import BombermanGymEnv
from environment import WorldArgs

import time
import numpy as np
import os
import copy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import freeze_support
from datetime import datetime
from imitation.data.types import Transitions, DictObs
from agent_code.my_agent.callbacks import act as expert_act, setup as expert_setup
import tqdm
import pathlib
from model import BombermanFeatureExtractor


LAYER_CONFIG = [
    "base",
    "timer_channels",
    "forecast",
    "self_distance",
    "opponent_distance",
    "crate_potential",
    "danger_summary",
    "mobility",
    "crate_distance",
    "coin_distance"
]

def dagger_collect(
    env: BombermanGymEnv,
    policy,
    n_episodes: int,
    cutoff_step: float = 400.
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

        while not done and env.world.step < cutoff_step:

            # Current policy decides where we go
            policy_action, _ = policy.predict(obs, deterministic=True)

            # Expert labels this state
            game_state = env.world.get_state_for_agent(env.agent)
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
        features_extractor_class=BombermanFeatureExtractor,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_log",
        verbose=1,
        **PPO_PARAMS
    )

    return model

def mask_fn(env):
    return env.action_masks()

def get_env(N_ENVS, opponents):
    env = make_vec_env(
        lambda: ActionMasker(BombermanGymEnv(
            CLASSIC_ENV_ARGS,
            opponents=opponents,
            layer_config=LAYER_CONFIG
        ), mask_fn),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env

def play_test_game(model, opponents, obs_rms, ret_rms):
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

    test_env = make_vec_env(
        lambda: ActionMasker(BombermanGymEnv(
            test_env_args,
            opponents=opponents,
            layer_config=LAYER_CONFIG
        ), mask_fn),
        n_envs=1,
        vec_env_cls=SubprocVecEnv
    )
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=True)
    test_env.obs_rms = obs_rms
    test_env.ret_rms = ret_rms

    obs = test_env.reset()
    done = False
    while not done:
        action_masks = get_action_masks(test_env)
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, dones, info = test_env.step(action)
        done = dones[0]
        test_env.render()
    test_env.close()


def run_epoch(N_STEPS, N_ENVS, epoch, model, opponents, training_start=None):
    model.learn(
        total_timesteps=N_STEPS * N_ENVS * 16,
        reset_num_timesteps=False,
        tb_log_name=f"PPO_{training_start}" if training_start else "PPO"
    )


    model.save(f"models/ppo_bomberman_{(epoch + 1) * N_STEPS * N_ENVS}")
    train_vec_norm_env = model.get_vec_normalize_env()
    train_vec_norm_env.save(f"models/ppo_bomberman_{(epoch + 1) * N_STEPS * N_ENVS}_vecnormalize.pkl")

    obs_rms = copy.deepcopy(train_vec_norm_env.obs_rms)
    ret_rms = copy.deepcopy(train_vec_norm_env.ret_rms)
    play_test_game(model, opponents, obs_rms, ret_rms)


def env_step_test(TEST_ROUNDS):
    env = BombermanGymEnv(CLASSIC_ENV_ARGS, opponents=[], layer_config=LAYER_CONFIG)

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
    env = BombermanGymEnv(CLASSIC_ENV_ARGS, opponents=opponents, layer_config=LAYER_CONFIG)
    return env


def main(N_ENVS, TOTAL_EPOCHS, N_DEMONSTRATION_EPISODES, opponents, PPO_PARAMS):
    freeze_support()
    env = get_env(N_ENVS, opponents)
    model = get_model(env, PPO_PARAMS)

    training_start = f"{datetime.now():%Y%m%d-%H%M%S}"

    demo_env = get_demo_environment(opponents)

    dataset = None

    for epoch in range(TOTAL_EPOCHS // 16 + 1):
        # PPO improvement
        run_epoch(
            PPO_PARAMS["n_steps"],
            N_ENVS,
            epoch,
            model,
            opponents,
            training_start,
        )

        """
        avg_reward = model.logger.name_to_value.get("rollout/ep_rew_mean", 0)
        if avg_reward < .1:
            dagger_step_cutoff = model.logger.name_to_value.get("rollout/ep_length_mean", 400) * 1.2

            # Collect states visited by the improved policy
            new_data = dagger_collect(
                demo_env,
                model,
                N_DEMONSTRATION_EPISODES,
                dagger_step_cutoff
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
                n_epochs=4,
            )
        """


if __name__ == "__main__":
    PPO_PARAMS = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.7,
        "target_kl": 0.02,
    }

    N_DEMONSTRATION_EPISODES = 50

    N_ENVS = 32
    TOTAL_EPOCHS = 1 + 50_000_000 // (PPO_PARAMS["n_steps"] * N_ENVS)
    opponents = []

    main(N_ENVS, TOTAL_EPOCHS, N_DEMONSTRATION_EPISODES, opponents, PPO_PARAMS)



