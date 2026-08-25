"""
python benchmark.py

Reports two numbers per batch size:
  - "physics only": sim.step() alone (no obs encoding, no opponent policy,
    no resets) -- the ceiling for this NumPy implementation.
  - "full pipeline": step + observation encoding + opponent policy + auto
    reset, i.e. what you'd actually see driving PPO/etc through
    gym_env.BombermanVecEnv.
"""
import time
import numpy as np

from sim import VecBomberman, Settings
from gym_env import encode_obs
from bots import RandomSafeBot


def bench_physics_only(n_envs, n_steps=200, seed=0):
    sim = VecBomberman(n_envs, Settings(scenario="classic"), seed=seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for _ in range(n_steps):
        sim.step(rng.integers(0, 6, size=(n_envs, 4)))
    return n_envs * n_steps / (time.time() - t0)


def bench_full_pipeline(n_envs, n_steps=200, seed=0):
    sim = VecBomberman(n_envs, Settings(scenario="classic"), seed=seed)
    bot = RandomSafeBot()
    rng = np.random.default_rng(seed)
    slots = np.tile(np.array([1, 2, 3]), (n_envs, 1))
    t0 = time.time()
    for _ in range(n_steps):
        actions = np.zeros((n_envs, 4), dtype=np.int64)
        actions[:, 0] = rng.integers(0, 6, size=n_envs)
        actions[:, 1:4] = bot(sim, slots)
        events, done = sim.step(actions)
        _ = encode_obs(sim, np.zeros(n_envs, dtype=np.int64))
        sim.auto_reset_done(done)
    return n_envs * n_steps / (time.time() - t0)


if __name__ == "__main__":
    print(f"{'n_envs':>8} | {'physics only (steps/s)':>24} | {'full pipeline (steps/s)':>24}")
    for n_envs in [1, 64, 512, 2048, 8192]:
        p = bench_physics_only(n_envs)
        f = bench_full_pipeline(n_envs)
        print(f"{n_envs:>8} | {p:>24,.0f} | {f:>24,.0f}")
