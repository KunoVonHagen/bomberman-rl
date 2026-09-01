"""
config.py
-----------------------------------------------------------------------------
Central place for ALL training hyperparameters and settings. Everything the
training script needs to build the model, the environment, and to control
checkpointing / self-play lives here as a plain dataclass. Edit
`DEFAULT_CONFIG` at the bottom (or build your own `TrainingConfig`) instead
of touching train.py.

The whole config is dumped into each run's `run_manifest.json` so a run is
always self-describing and reproducible.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal, ClassVar
import json


def _coerce_value(current, raw):
    """Turn a raw CLI string into the same type as the field it's replacing.
    Only used by TrainingConfig.apply_overrides -- if `raw` isn't a string
    (already the right type), it's returned unchanged."""
    if not isinstance(raw, str):
        return raw
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))  # float() first so "1e6" style works for int fields too
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [x for x in raw.split(",") if x]
    if current is None:
        # Optional[...] field currently unset (e.g. clip_range_vf). Try float,
        # allow an explicit "none"/"null" to keep it unset, else pass through as str.
        if raw.strip().lower() in ("none", "null", ""):
            return None
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw  # str / Literal fields


@dataclass
class PPOConfig:
    """Passed straight through to sb3_contrib.MaskablePPO."""
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.7
    target_kl: float = 0.02


@dataclass
class EnvConfig:
    """Everything needed to build a WorldArgs + BombermanGymEnv."""
    scenario: str = "classic"
    seed: Optional[int] = None
    silence_errors: bool = True
    no_gui: bool = True
    make_video: bool = False
    save_replay: bool = False
    save_stats: bool = True
    turn_based: bool = False
    update_interval: float = 0.1
    match_name: Optional[str] = None
    fps: int = 60
    replay: Optional[str] = None
    continue_without_training: bool = False

    layer_config: List[str] = field(default_factory=lambda: [
        "base",
        "timer_channels",
        "forecast",
        "self_distance",
        "opponent_distance",
        "crate_potential",
        "danger_summary",
        "mobility",
        "crate_distance",
        "coin_distance",
    ])


@dataclass
class SelfPlayConfig:
    """Controls whether/how the agent trains against its own past checkpoints."""
    enabled: bool = False
    static_opponents: List[str] = field(default_factory=list)
    n_static_opponents: int = 1
    n_self_play_opponents: int = 1
    pool_size: int = 8
    add_checkpoint_every_epochs: int = 1
    sample_strategy: Literal["uniform", "latest_biased"] = "latest_biased"
    latest_bias: float = 0.5


@dataclass
class TrainingConfig:
    run_name: Optional[str] = None
    runs_dir: str = "runs"
    n_envs: int = 32
    n_shards: int = 1
    total_timesteps: int = 50_000_000

    device: str = "auto"
    n_demonstration_episodes: int = 50
    save_every_timesteps: int = 1_048_576
    eval_every_save: bool = True

    resume_from: Optional[str] = None
    resume_checkpoint: Optional[str] = None

    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    RESUMABLE_FIELDS: ClassVar[set] = {
        "total_timesteps",
        "save_every_timesteps",
        "eval_every_save",
        "device",
        "n_envs",
        "n_shards",
        "n_demonstration_episodes",
        "ppo.learning_rate",
        "ppo.n_steps",
        "ppo.batch_size",
        "ppo.n_epochs",
        "ppo.gamma",
        "ppo.gae_lambda",
        "ppo.clip_range",
        "ppo.clip_range_vf",
        "ppo.ent_coef",
        "ppo.vf_coef",
        "ppo.target_kl",
        "self_play.enabled",
        "self_play.static_opponents",
        "self_play.n_static_opponents",
        "self_play.n_self_play_opponents",
        "self_play.pool_size",
        "self_play.add_checkpoint_every_epochs",
        "self_play.sample_strategy",
        "self_play.latest_bias",
    }

    def apply_overrides(self, overrides: dict, *, restrict_to: set | None = None) -> list:
        """Apply {"dotted.path": raw_value} overrides in place. raw_value may be a
        CLI string (coerced to match the existing field's type) or an already-typed
        value. If `restrict_to` is given, any path not in it raises ValueError --
        pass TrainingConfig.RESUMABLE_FIELDS here when resuming a run so a typo or
        an over-eager override can't quietly change the model's architecture out
        from under its own checkpoint.

        Returns a list of (path, old_value, new_value) for logging.
        """
        applied = []
        for dotted_key, raw_value in overrides.items():
            if restrict_to is not None and dotted_key not in restrict_to:
                raise ValueError(
                    f"'{dotted_key}' can't be changed on --resume (it would change the "
                    f"model's architecture or environment, desyncing it from the loaded "
                    f"checkpoint). Fields allowed on resume: {', '.join(sorted(restrict_to))}"
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
        d["ppo"] = PPOConfig(**d.get("ppo", {}))
        d["env"] = EnvConfig(**d.get("env", {}))
        d["self_play"] = SelfPlayConfig(**d.get("self_play", {}))
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
    total_timesteps=1_000_000_000,
    save_every_timesteps=128 * 1024 * 8,
    ppo=PPOConfig(
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=8192,
        n_epochs=7,
        gamma=0.99,
        gae_lambda=0.97,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.025,
        vf_coef=0.7,
        target_kl=0.02,
    ),
    env=EnvConfig(),
    self_play=SelfPlayConfig(
        enabled=True,
        static_opponents=[
            #"agent_code.rule_based_agent.callbacks",
            #"agent_code.coin_collector_agent.callbacks",
            #"agent_code.random_agent.callbacks",
            #"agent_code.peaceful_agent.callbacks",
            #"agent_code.my_agent.callbacks",
            "agent_code.simple_agent.callbacks",
        ],
        n_static_opponents=1,
        n_self_play_opponents=2,
        pool_size=32,
        add_checkpoint_every_epochs=1,
        sample_strategy="latest_biased",
        latest_bias=0.1,
    ),
)