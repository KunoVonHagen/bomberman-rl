from __future__ import annotations

from dataclasses import dataclass, field, asdict, ClassVar
from typing import Optional
import json

from agent_code.ppo_agent.config import (
    EnvConfig,
    RewardConfig,
    SelfPlayConfig,
    OpponentArrangement,
    TrainingConfig as _PPOTrainingConfig,
    _coerce_value,
)


@dataclass
class DQNConfig:
    """Hyperparameters passed directly to stable_baselines3.DQN."""
    learning_rate: float = 1e-4
    buffer_size: int = 500_000
    learning_starts: int = 50_000
    batch_size: int = 256
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0


@dataclass
class TrainingConfig:
    run_name: Optional[str] = None
    runs_dir: str = "runs"
    n_envs: int = 32
    n_shards: int = 1
    total_timesteps: int = 50_000_000

    device: str = "auto"
    save_every_timesteps: int = 1_048_576
    eval_every_save: bool = True

    resume_from: Optional[str] = None
    resume_checkpoint: Optional[str] = None

    dqn: DQNConfig = field(default_factory=DQNConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    RESUMABLE_FIELDS: ClassVar[set] = {
        "total_timesteps",
        "save_every_timesteps",
        "eval_every_save",
        "device",
        "n_envs",
        "n_shards",
        "env.scenario_mix",
        "dqn.learning_rate",
        "dqn.exploration_fraction",
        "dqn.exploration_initial_eps",
        "dqn.exploration_final_eps",
        "dqn.target_update_interval",
        "dqn.train_freq",
        "dqn.gradient_steps",
        "dqn.tau",
        "dqn.gamma",
        "dqn.max_grad_norm",
        "self_play.enabled",
        "self_play.static_opponents",
        "self_play.arrangements",
        "self_play.allow_repeat_static_opponents",
        "self_play.shuffle_opponent_order",
        "self_play.resample_every_n_rollouts",
        "self_play.pool_size",
        "self_play.add_checkpoint_every_epochs",
        "self_play.sample_strategy",
        "self_play.latest_bias",
        "rewards.waited",
        "rewards.invalid_action",
        "rewards.bomb_dropped",
        "rewards.bomb_exploded",
        "rewards.crate_destroyed",
        "rewards.coin_found",
        "rewards.coin_collected",
        "rewards.killed_opponent",
        "rewards.killed_self",
        "rewards.got_killed",
        "rewards.opponent_eliminated",
        "rewards.survived_round",
        "rewards.coin_shaping_coef",
        "rewards.crate_shaping_coef",
        "rewards.danger_penalty_coef",
        "rewards.escape_bonus_coef",
        "rewards.trap_shaping_coef",
    }

    def apply_overrides(self, overrides: dict, *, restrict_to: set | None = None) -> list:
        """Apply a dict of dotted-key overrides, returning (key, old, new) triples."""
        applied = []
        for dotted_key, raw_value in overrides.items():
            if restrict_to is not None and dotted_key not in restrict_to:
                raise ValueError(
                    f"'{dotted_key}' can't be changed on --resume. "
                    f"Fields allowed on resume: {', '.join(sorted(restrict_to))}"
                )
            parts = dotted_key.split(".")
            obj = self
            for p in parts[:-1]:
                if not hasattr(obj, p):
                    raise ValueError(f"Unknown config path '{dotted_key}' (no '{p}')")
                obj = getattr(obj, p)
            leaf = parts[-1]
            if not hasattr(obj, leaf):
                raise ValueError(f"Unknown config path '{dotted_key}'")
            current = getattr(obj, leaf)
            new_value = _coerce_value(current, raw_value)
            setattr(obj, leaf, new_value)
            applied.append((dotted_key, current, new_value))
        return applied

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        d = dict(d)
        d["dqn"] = DQNConfig(**d.get("dqn", {}))
        d["env"] = EnvConfig(**_PPOTrainingConfig._migrate_env_dict(d.get("env", {})))
        d["self_play"] = SelfPlayConfig(**_PPOTrainingConfig._migrate_self_play_dict(d.get("self_play", {})))
        d["rewards"] = RewardConfig(**d.get("rewards", {}))
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


DEFAULT_CONFIG = TrainingConfig(
    run_name=None,
    n_envs=256,
    n_shards=32,
    total_timesteps=1_000_000_000_000,
    save_every_timesteps=256 * 1024 * 4,
    dqn=DQNConfig(
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=100_000,
        batch_size=512,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=20_000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
    ),
    env=EnvConfig(),
    self_play=SelfPlayConfig(
        enabled=True,
        static_opponents=[
            "agent_code.my_agent.callbacks",
            "agent_code.simple_agent.callbacks",
            "agent_code.rule_based_agent.callbacks",
            "agent_code.coin_collector_agent.callbacks",
        ],
        arrangements=[
            OpponentArrangement(n_static=0, n_self_play=3, weight=1.0),
        ],
        allow_repeat_static_opponents=True,
        shuffle_opponent_order=True,
        resample_every_n_rollouts=1,
        pool_size=16,
        add_checkpoint_every_epochs=1,
        sample_strategy="latest_biased",
        latest_bias=0.1,
    ),
)