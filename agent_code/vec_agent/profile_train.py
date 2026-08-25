"""
Run this on your training machine (not in a sandbox) to see where the
fps is actually going:

    python profile_train.py

It times, separately:
  1. sim.step() alone                      (physics ceiling, no torch)
  2. + encode_obs + SelfPlayOpponent        (rollout collection cost)
  3. PPO's model.train() gradient step      (backward-pass cost)

Compare (2) against benchmark.py's "full pipeline" number for the same
num_envs to see how much the self-play opponent specifically costs vs.
the RandomSafeBot baseline it replaced. Compare (2) vs (3) to see whether
rollout collection or gradient updates dominate total wall-clock.
"""
import time
import numpy as np
import torch

from sim import Settings
from gym_env import encode_obs
from bots import SelfPlayOpponent
from train_example import SmallGridCNN, SB3Compatible
from stable_baselines3 import PPO

N_ENVS = 512
N_STEPS = 100


def main():
    print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    opponent = SelfPlayOpponent(pool_size=5)
    venv = SB3Compatible(N_ENVS, Settings(scenario="classic"), opponent=opponent)
    model = PPO(
        "CnnPolicy", venv, n_steps=128, batch_size=4096, verbose=0,
        policy_kwargs=dict(features_extractor_class=SmallGridCNN,
                            features_extractor_kwargs=dict(features_dim=256),
                            normalize_images=False),
    )
    for _ in range(5):
        opponent.add_snapshot(model.policy)  # simulate a "warmed up" pool

    sim = venv.sim
    slots = np.tile(np.array([1, 2, 3]), (N_ENVS, 1))
    rng = np.random.default_rng(0)

    # 1. physics only
    t0 = time.time()
    for _ in range(N_STEPS):
        sim.step(rng.integers(0, 6, size=(N_ENVS, 4)))
    dt = time.time() - t0
    print(f"1. physics only:            {N_ENVS*N_STEPS/dt:>12,.0f} steps/s  ({dt/N_STEPS*1000:.2f} ms/step)")

    # 2. + encode_obs + self-play opponent (the actual rollout-collection cost)
    t0 = time.time()
    for _ in range(N_STEPS):
        actions = np.zeros((N_ENVS, 4), dtype=np.int64)
        actions[:, 0] = rng.integers(0, 6, size=N_ENVS)
        actions[:, 1:4] = opponent(sim, slots)
        events, done = sim.step(actions)
        _ = encode_obs(sim, np.zeros(N_ENVS, dtype=np.int64))
        sim.auto_reset_done(done)
    dt = time.time() - t0
    print(f"2. + obs + selfplay opp:    {N_ENVS*N_STEPS/dt:>12,.0f} steps/s  ({dt/N_STEPS*1000:.2f} ms/step)")

    # 3. PPO gradient update cost (collect one real rollout, then time .train())
    model.learn(total_timesteps=N_ENVS * 128)  # fills the rollout buffer once
    t0 = time.time()
    model.train()
    dt = time.time() - t0
    rollout_size = N_ENVS * 128
    print(f"3. model.train() on {rollout_size} samples: {dt:.2f}s  "
          f"({rollout_size/dt:,.0f} samples/s equivalent)")


if __name__ == "__main__":
    main()