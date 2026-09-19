from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import pathlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from multiprocessing import freeze_support

import numpy as np

from ..dqn_agent.config import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ..dqn_agent.gym_environment import ACTIONS, BombermanGymEnv, WorldArgs
from ..dqn_agent.replay_buffer import read_replay_file
from .model import MODEL_FILE, build_model, masked_greedy, masked_max, save_model

MANIFEST_FILE = "run_manifest.json"
LOG_FILE = "log.jsonl"
DEFAULT_OPPONENTS = ["agent_code.rule_based_agent.callbacks"] * 3


@dataclass
class FQIConfig:
    model: str = "forest"
    run_name: str | None = None
    runs_dir: str = "runs"
    n_envs: int = 32
    n_shards: int = 0
    epochs: int = 30
    steps_per_epoch: int = 400
    buffer_size: int = 300_000
    fit_samples: int = 0
    fitted_iterations: int = 2
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_epochs: int = 20
    ridge: float = 1e-2
    n_estimators: int = 50
    max_depth: int = 12
    min_samples_leaf: int = 20
    max_features: int = 10
    opponents: list[str] = field(default_factory=lambda: list(DEFAULT_OPPONENTS))
    scenario: str = "classic"
    env_version: int = 4
    eval_every: int = 5
    seed: int = 0
    replay: str | None = None

    def epsilon(self, epoch: int) -> float:
        if self.eps_epochs <= 0:
            return self.eps_end
        frac = min(1.0, epoch / self.eps_epochs)
        return self.eps_start + frac * (self.eps_end - self.eps_start)


class TransitionStore:
    def __init__(self, capacity: int, n_features: int, n_actions: int):
        self.capacity = int(capacity)
        self.features = np.zeros((self.capacity, n_features), dtype=np.float32)
        self.next_features = np.zeros((self.capacity, n_features), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones((self.capacity, n_actions), dtype=bool)
        self.pos = 0
        self.size = 0

    def add(self, features, actions, rewards, dones, next_features, next_masks) -> None:
        n = len(actions)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.features[idx] = features
        self.next_features[idx] = next_features
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.next_masks[idx] = next_masks
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def batch(self, n_samples: int, rng: np.random.Generator) -> tuple:
        idx = np.arange(self.size)
        if 0 < n_samples < self.size:
            idx = rng.choice(self.size, size=n_samples, replace=False)
        return (self.features[idx], self.actions[idx], self.rewards[idx], self.dones[idx],
                self.next_features[idx], self.next_masks[idx])


class SingleShardVecEnv:
    def __init__(self, env: BombermanGymEnv):
        self.env = env
        self.num_envs = env.n_envs
        self.single_observation_space = env.single_observation_space
        self._masks = None

    def reset(self):
        obs, _infos, masks = self.env.reset()
        self._masks = masks
        return obs

    def step(self, actions):
        obs, rewards, terminated, truncated, infos, masks = self.env.step(actions)
        self._masks = masks
        return obs, rewards, terminated | truncated, infos

    def action_masks(self):
        return self._masks

    def close(self):
        self.env.close()


def resolve_opponents(module_paths: list[str]) -> list:
    pairs = []
    for path in module_paths:
        module = importlib.import_module(path)
        pairs.append((module.setup, module.act))
    return pairs


def build_world_args(cfg: FQIConfig, log_dir: str) -> WorldArgs:
    env_cfg = DQN_DEFAULT_CONFIG.env
    return WorldArgs(
        scenario=cfg.scenario, seed=None, silence_errors=True, no_gui=True, make_video=False,
        save_replay=False, save_stats=False, turn_based=False, update_interval=env_cfg.update_interval,
        log_dir=log_dir, match_name=None, fps=env_cfg.fps, replay=None,
        continue_without_training=env_cfg.continue_without_training,
    )


def resolve_n_shards(n_envs: int, n_shards: int) -> int:
    if n_shards > 0:
        return n_shards
    wanted = max(1, min(8, os.cpu_count() or 1, n_envs))
    while n_envs % wanted:
        wanted -= 1
    return wanted


def make_env(cfg: FQIConfig, n_envs: int, n_shards: int, log_dir: str):
    opponents = resolve_opponents(cfg.opponents)
    if n_shards > 1:
        from ..dqn_agent.train import ShardedNativeBatchedVecEnv

        dqn_cfg = copy.deepcopy(DQN_DEFAULT_CONFIG)
        dqn_cfg.n_envs, dqn_cfg.n_shards = n_envs, n_shards
        dqn_cfg.env.env_version, dqn_cfg.env.scenario = cfg.env_version, cfg.scenario
        return ShardedNativeBatchedVecEnv(dqn_cfg, opponents, log_dir, n_shards=n_shards)
    env = BombermanGymEnv(
        build_world_args(cfg, log_dir), opponents=opponents, layer_config=DQN_DEFAULT_CONFIG.env.layer_config,
        env_version=cfg.env_version, n_envs=n_envs, reward_config=DQN_DEFAULT_CONFIG.rewards,
    )
    return SingleShardVecEnv(env)


def sample_masked(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = masks.astype(np.float64)
    weights[weights.sum(axis=1) == 0] = 1.0
    cumulative = np.cumsum(weights, axis=1)
    draws = rng.random(masks.shape[0]) * cumulative[:, -1]
    return np.minimum((draws[:, None] >= cumulative).sum(axis=1), masks.shape[1] - 1)


class EpisodeStats:
    def __init__(self, n_envs: int):
        self.rewards = np.zeros(n_envs)
        self.lengths = np.zeros(n_envs, dtype=np.int64)
        self.finished: list[dict] = []

    def update(self, rewards, dones, infos) -> None:
        self.rewards += rewards
        self.lengths += 1
        for i in np.flatnonzero(dones):
            info = infos[i]
            self.finished.append(dict(score=float(info["score"]), alive=float(info["alive"]),
                                      reward=float(self.rewards[i]), length=int(self.lengths[i]),
                                      won=float(info["score"] > max(info["opponent_scores"], default=-1))))
            self.rewards[i] = 0.0
            self.lengths[i] = 0

    def summary(self) -> dict:
        if not self.finished:
            return {}
        keys = ("score", "alive", "won", "reward", "length")
        out = {k: float(np.mean([f[k] for f in self.finished])) for k in keys}
        out["episodes"] = len(self.finished)
        return out


def q_values(model, features: np.ndarray, fitted: bool) -> np.ndarray:
    if not fitted:
        return np.zeros((features.shape[0], len(ACTIONS)), dtype=np.float32)
    return model.predict(features)


def collect(venv, model, fitted: bool, store: TransitionStore, n_steps: int, eps: float,
            rng: np.random.Generator, state: dict, stats: EpisodeStats) -> None:
    obs, masks = state["obs"], state["masks"]
    for _ in range(n_steps):
        q = q_values(model, obs["features"], fitted) + rng.random((obs["features"].shape[0], len(ACTIONS))) * 1e-6
        greedy = masked_greedy(q, masks)
        explore = rng.random(len(greedy)) < eps
        actions = np.where(explore, sample_masked(masks, rng), greedy)
        next_obs, rewards, dones, infos = venv.step(actions)
        next_masks = venv.action_masks()
        store.add(obs["features"], actions, rewards, dones, next_obs["features"], next_masks)
        stats.update(rewards, dones, infos)
        obs, masks = next_obs, next_masks
    state["obs"], state["masks"] = obs, masks


def fit(model, fitted: bool, store: TransitionStore, cfg: FQIConfig, rng: np.random.Generator, seed: int) -> dict:
    x, a, r, d, x2, m2 = store.batch(cfg.fit_samples, rng)
    td_error = None
    for _ in range(max(1, cfg.fitted_iterations)):
        v2 = masked_max(model.predict(x2), m2) if fitted else np.zeros(len(r), dtype=np.float32)
        y = r + cfg.gamma * (1.0 - d) * v2
        if fitted and td_error is None:
            td_error = float(np.mean(np.abs(model.predict(x)[np.arange(len(a)), a] - y)))
        model.fit(x, a, y, seed=seed)
        fitted = True
    q = model.predict(x)
    return dict(samples=int(len(a)), td_error=td_error, mean_q=float(q.mean()),
                mean_target=float(y.mean()), action_counts=np.bincount(a, minlength=len(ACTIONS)).tolist())


def evaluate(venv, model, fitted: bool) -> dict:
    obs = venv.reset()
    masks = venv.action_masks()
    stats = EpisodeStats(venv.num_envs)
    done_once = np.zeros(venv.num_envs, dtype=bool)
    while not done_once.all():
        actions = masked_greedy(q_values(model, obs["features"], fitted), masks)
        obs, rewards, dones, infos = venv.step(actions)
        masks = venv.action_masks()
        fresh = dones & ~done_once
        stats.update(rewards, fresh, infos)
        done_once |= dones
    return stats.summary()


def load_replay(path: str, n_features: int, store: TransitionStore) -> int:
    data = read_replay_file(path, fields=("features", "actions", "rewards", "dones", "action_masks"))
    n = int(data["n_rows"])
    features = np.asarray(data["features"][:n], dtype=np.float32)
    if features.shape[-1] != n_features:
        raise ValueError(f"replay features have width {features.shape[-1]}, the env produces {n_features}")
    actions, rewards, dones, masks = data["actions"][:n], data["rewards"][:n], data["dones"][:n], data["action_masks"][:n]
    for row in range(n - 1):
        store.add(features[row], actions[row], rewards[row], dones[row], features[row + 1], masks[row + 1])
    return max(0, n - 1) * features.shape[1]


def write_manifest(run_dir: pathlib.Path, cfg: FQIConfig, n_features: int) -> None:
    manifest = dict(
        config=asdict(cfg),
        env=dict(env_version=cfg.env_version, layer_config=list(DQN_DEFAULT_CONFIG.env.layer_config),
                 scenario=cfg.scenario),
        n_features=n_features,
        n_actions=len(ACTIONS),
        created=datetime.now().isoformat(timespec="seconds"),
    )
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def run(cfg: FQIConfig) -> pathlib.Path:
    freeze_support()
    rng = np.random.default_rng(cfg.seed)
    run_name = cfg.run_name or f"fqi_{cfg.model}_{datetime.now():%Y%m%d-%H%M%S}"
    cfg.run_name = run_name
    run_dir = pathlib.Path(cfg.runs_dir) / run_name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    cfg.n_shards = resolve_n_shards(cfg.n_envs, cfg.n_shards)
    venv = make_env(cfg, cfg.n_envs, cfg.n_shards, str(run_dir / "logs"))
    eval_env = make_env(cfg, cfg.n_envs, cfg.n_shards, str(run_dir / "logs")) if cfg.eval_every > 0 else None
    n_features = int(venv.single_observation_space["features"].shape[0])
    model = build_model(cfg.model, n_features, len(ACTIONS), ridge=cfg.ridge, n_estimators=cfg.n_estimators,
                        max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf, max_features=cfg.max_features)
    store = TransitionStore(cfg.buffer_size, n_features, len(ACTIONS))
    write_manifest(run_dir, cfg, n_features)
    print(f"run {run_name}: model={cfg.model} features={n_features} envs={cfg.n_envs}x{cfg.n_shards} "
          f"opponents={cfg.opponents}")

    fitted = False
    if cfg.replay:
        n = load_replay(cfg.replay, n_features, store)
        print(f"loaded {n} transitions from {cfg.replay}")

    state = {"obs": venv.reset(), "masks": venv.action_masks()}
    log = open(run_dir / LOG_FILE, "a", encoding="utf-8")
    for epoch in range(cfg.epochs):
        eps = cfg.epsilon(epoch)
        stats = EpisodeStats(venv.num_envs)
        t0 = time.time()
        collect(venv, model, fitted, store, cfg.steps_per_epoch, eps, rng, state, stats)
        t_collect = time.time() - t0

        t0 = time.time()
        fit_info = fit(model, fitted, store, cfg, rng, seed=cfg.seed + epoch)
        fitted = True
        t_fit = time.time() - t0
        save_model(model, run_dir / MODEL_FILE)

        record = dict(epoch=epoch, epsilon=round(eps, 4), transitions=store.size, collect_s=round(t_collect, 1),
                      fit_s=round(t_fit, 1), train=stats.summary(), fit=fit_info)
        if eval_env is not None and (epoch + 1) % cfg.eval_every == 0:
            t0 = time.time()
            record["eval"] = evaluate(eval_env, model, fitted)
            record["eval_s"] = round(time.time() - t0, 1)
        log.write(json.dumps(record) + "\n")
        log.flush()

        train, ev = record["train"], record.get("eval", {})
        td = fit_info["td_error"]
        print(f"epoch {epoch:3d} eps {eps:.2f} ts {store.size:7d} | collect {t_collect:5.1f}s fit {t_fit:5.1f}s "
              f"td {td if td is None else round(td, 3)} q {fit_info['mean_q']:6.2f} | "
              f"train score {train.get('score', float('nan')):5.2f} alive {train.get('alive', float('nan')):.2f} "
              f"reward {train.get('reward', float('nan')):6.2f} (n={train.get('episodes', 0)})"
              + (f" | eval score {ev['score']:5.2f} alive {ev['alive']:.2f} win {ev['won']:.2f} "
                 f"len {ev['length']:5.1f} (n={ev['episodes']})" if ev else ""), flush=True)
    log.close()
    venv.close()
    if eval_env is not None:
        eval_env.close()
    return run_dir


def parse_args(argv=None) -> FQIConfig:
    defaults = FQIConfig()
    p = argparse.ArgumentParser(description="Fitted Q-iteration on the hand-crafted features with a classical regressor")
    p.add_argument("--model", choices=("linear", "quadratic", "forest"), default=defaults.model)
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default=defaults.runs_dir)
    p.add_argument("--n-envs", type=int, default=defaults.n_envs)
    p.add_argument("--n-shards", type=int, default=defaults.n_shards,
                   help="env worker processes for collection and evaluation (0 = min(8, cores) dividing n_envs)")
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--steps-per-epoch", type=int, default=defaults.steps_per_epoch)
    p.add_argument("--buffer-size", type=int, default=defaults.buffer_size)
    p.add_argument("--fit-samples", type=int, default=defaults.fit_samples, help="0 fits on the whole store")
    p.add_argument("--fitted-iterations", type=int, default=defaults.fitted_iterations)
    p.add_argument("--gamma", type=float, default=defaults.gamma)
    p.add_argument("--eps-start", type=float, default=defaults.eps_start)
    p.add_argument("--eps-end", type=float, default=defaults.eps_end)
    p.add_argument("--eps-epochs", type=int, default=defaults.eps_epochs)
    p.add_argument("--ridge", type=float, default=defaults.ridge)
    p.add_argument("--n-estimators", type=int, default=defaults.n_estimators)
    p.add_argument("--max-depth", type=int, default=defaults.max_depth)
    p.add_argument("--min-samples-leaf", type=int, default=defaults.min_samples_leaf)
    p.add_argument("--max-features", type=int, default=defaults.max_features)
    p.add_argument("--opponents", nargs="+", default=defaults.opponents)
    p.add_argument("--scenario", default=defaults.scenario)
    p.add_argument("--env-version", type=int, default=defaults.env_version)
    p.add_argument("--eval-every", type=int, default=defaults.eval_every, help="0 disables the greedy evaluation")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--replay", default=None, help="replay_buffer.bin (or legacy .npz) of a DQN run to warm-start the store")
    args = p.parse_args(argv)
    return FQIConfig(**vars(args))


if __name__ == "__main__":
    run(parse_args())
