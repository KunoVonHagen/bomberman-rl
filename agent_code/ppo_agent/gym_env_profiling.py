import time
import pathlib
import numpy as np

from agent_code.ppo_agent.gym_environment import BombermanGymEnv
from environment import WorldArgs
from agent_code.my_agent.callbacks import setup as my_agent_setup, act as my_agent_act

CLASSIC_ENV_ARGS = WorldArgs(
    scenario="classic",
    seed=None,
    silence_errors=True,
    no_gui=True,
    make_video=False,
    save_replay=False,
    save_stats=False,
    turn_based=False,
    update_interval=0.1,
    log_dir=str(pathlib.Path(__file__).parent / "logs"),
    match_name=None,
    fps=60,
    replay=False,
    continue_without_training=False,
)


def benchmark_environment(n_envs=8, total_steps=50000, opponents=None):
    """
    Profiling for batched environments is best measured in total environment steps,
    rather than episodes, because games finish asynchronously.
    """
    if opponents is None:
        opponents = [(my_agent_setup, my_agent_act)] * 3

    env = BombermanGymEnv(
        CLASSIC_ENV_ARGS,
        opponents=opponents,
        n_envs=n_envs,
        auto_reset=True
    )

    print(f"Profiling batched environment with {n_envs} parallel games for {total_steps} steps...")
    print(f"(Max possible individual game steps: {n_envs * total_steps})\n")

    total_completed_games = 0
    start_time = time.perf_counter()

    obs, _ = env.reset()
    for step in range(total_steps):
        actions = env.action_space.sample()
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        finished_this_tick = np.sum(terminateds | truncateds)
        total_completed_games += finished_this_tick

    elapsed_time = time.perf_counter() - start_time
    env.close()

    total_individual_game_steps = n_envs * total_steps
    steps_per_second = total_individual_game_steps / elapsed_time if elapsed_time > 0 else float("inf")

    print("\n========== Batched Benchmark Results ==========")
    print(f"Parallel Environments (n_envs):  {n_envs}")
    print(f"Environment Steps Executed:      {total_steps:,}")
    print(f"Total Individual Game Steps:     {total_individual_game_steps:,}")
    print(f"Games Completed:                 {total_completed_games:,}")
    print(f"Total Wall-Clock Time:           {elapsed_time:.3f} s")
    print(f"Throughput (Steps/Second):       {steps_per_second:,.2f}")

    if total_completed_games > 0:
        print(f"Avg Steps per Completed Game:   {total_individual_game_steps / total_completed_games:.2f}")
    else:
        print("Warning: No games completed during the benchmark window.")


if __name__ == "__main__":
    benchmark_environment(
        n_envs=16,
        total_steps=4096,
        opponents=[(my_agent_setup, my_agent_act)] * 0
    )