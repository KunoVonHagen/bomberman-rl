import time
import pathlib

from gym_environment import BombermanGymEnv
from environment import WorldArgs
from agent_code.my_agent.callbacks import setup as my_agent_setup, act as my_agent_act
from agent_code.random_agent.callbacks import setup as random_agent_setup, act as random_agent_act


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


def benchmark_environment(num_episodes=100, opponents=None):
    if opponents is None:
        opponents = []

    env = BombermanGymEnv(
        CLASSIC_ENV_ARGS,
        opponents=opponents,
        layer_config=["base", "timer_channels"]#, "forecast", "danger_summary", "mobility"]
    )

    total_steps = 0
    total_time = 0.0

    print(f"Running {num_episodes} episodes...\n")

    for episode in range(num_episodes):
        obs, _ = env.reset()

        done = False
        steps = 0

        start = time.perf_counter()

        while not done:
            action = env.action_space.sample()  # Random agent
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            steps += 1

        elapsed = time.perf_counter() - start

        #fps = steps / elapsed if elapsed > 0 else float("inf")
        #print(f"Episode {episode + 1:3d}: {steps:4d} steps | {fps:8.2f} steps/s")

        total_steps += steps
        total_time += elapsed

    env.close()


    print("\n========== Benchmark ==========")
    print(f"Episodes:      {num_episodes}")
    print(f"Total steps:   {total_steps}")
    print(f"Total time:    {total_time:.3f} s")
    print(f"Average speed: {total_steps / total_time:.2f} steps/s")
    print(f"Average steps: {total_steps / num_episodes:.2f}")


if __name__ == "__main__":
    benchmark_environment(
        num_episodes=5000,
        opponents=[(my_agent_setup, my_agent_act)]*0
    )