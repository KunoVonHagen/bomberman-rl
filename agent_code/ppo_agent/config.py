from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass, fields as dataclass_fields
from typing import Optional, List, Literal, ClassVar
import json


def _coerce_dataclass_list(cls, raw: str) -> list:
    """
    Coerce a raw string into a list of dataclass instances of type `cls`.
    The raw string can be a JSON array of objects, or a semicolon-separated list of comma-separated values corresponding to the dataclass fields.
    """
    trimmed = raw.strip()
    if trimmed.startswith("[") or trimmed.startswith("{"):
        parsed = json.loads(trimmed)
        return [cls(**item) if isinstance(item, dict) else item for item in parsed]

    field_specs = dataclass_fields(cls)
    items = []
    for chunk in trimmed.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) > len(field_specs):
            raise ValueError(
                f"Too many values in '{chunk}' for {cls.__name__} "
                f"(expected at most {len(field_specs)}: "
                f"{', '.join(f.name for f in field_specs)})"
            )
        kwargs = {}
        for f, val in zip(field_specs, parts):
            kwargs[f.name] = _coerce_value(f.default, val)
        items.append(cls(**kwargs))
    return items


def _coerce_value(current, raw):
    """
    Coerce a raw string value into the type of `current`. Handles bool, int, float, list, and None.
    """
    if not isinstance(raw, str):
        return raw
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        if current and is_dataclass(current[0]):
            return _coerce_dataclass_list(type(current[0]), raw)
        return [x for x in raw.split(",") if x]
    if current is None:
        if raw.strip().lower() in ("none", "null", ""):
            return None
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


@dataclass
class PPOConfig:
    """
    Hyperparameters for the PPO algorithm. These are passed directly to the
    Stable Baselines3 MaskablePPO constructor. See:
    https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    """
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
    """
    Configuration for the Bomberman environment.
    These are passed to the BombermanGymEnv constructor.
    """
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


@dataclass(frozen=True)
class OpponentArrangement:
    """
    Represents a single opponent lineup configuration for self-play.
    The agent will face `n_static` static opponents and `n_self_play` self-play opponents in each game.
    The `weight` determines how often this arrangement is sampled relative to others.
    """
    n_static: int = 1
    n_self_play: int = 2
    weight: float = 1.0


@dataclass
class SelfPlayConfig:
    """
    Configuration for self-play training.
    This allows the agent to train against a mix of static opponents and its own previous checkpoints.
    """
    enabled: bool = False
    static_opponents: List[str] = field(default_factory=list)

    arrangements: List[OpponentArrangement] = field(
        default_factory=lambda: [OpponentArrangement(n_static=1, n_self_play=2, weight=1.0)]
    )
    allow_repeat_static_opponents: bool = True
    shuffle_opponent_order: bool = True
    resample_every_n_rollouts: int = 1

    pool_size: int = 8
    add_checkpoint_every_epochs: int = 1
    sample_strategy: Literal["uniform", "latest_biased"] = "latest_biased"
    latest_bias: float = 0.5

    def __post_init__(self):
        totals = {a.n_static + a.n_self_play for a in self.arrangements}
        if len(totals) > 1:
            raise ValueError(
                "All self_play.arrangements must add up to the same total opponent "
                f"count (the game's player count is fixed) -- got totals {sorted(totals)}. "
                "Vary the static/self-play *mix* between arrangements, not the total."
            )


@dataclass
class TrainingConfig:
    run_name: Optional[str] = None
    runs_dir: str = "../ppo_agent/runs"
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
        "self_play.arrangements",
        "self_play.allow_repeat_static_opponents",
        "self_play.shuffle_opponent_order",
        "self_play.resample_every_n_rollouts",
        "self_play.pool_size",
        "self_play.add_checkpoint_every_epochs",
        "self_play.sample_strategy",
        "self_play.latest_bias",
    }

    def apply_overrides(self, overrides: dict, *, restrict_to: set | None = None) -> list:
        """
        Apply a dictionary of dotted-key overrides to this config object.
        If `restrict_to` is provided, only keys in that set are allowed to be overridden.
        Returns a list of (dotted_key, old_value, new_value) for each applied override.
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
        d["self_play"] = SelfPlayConfig(**cls._migrate_self_play_dict(d.get("self_play", {})))
        return cls(**d)

    @staticmethod
    def _migrate_self_play_dict(sp: dict) -> dict:
        """
        Migrate old self_play config dicts that used 'n_static_opponents' and 'n_self_play_opponents' to the new 'arrangements' format.
        """
        sp = dict(sp)
        n_static = sp.pop("n_static_opponents", None)
        n_self_play = sp.pop("n_self_play_opponents", None)
        if "arrangements" not in sp and (n_static is not None or n_self_play is not None):
            sp["arrangements"] = [{
                "n_static": n_static if n_static is not None else 1,
                "n_self_play": n_self_play if n_self_play is not None else 2,
                "weight": 1.0,
            }]
        return sp

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


DEFAULT_CONFIG = TrainingConfig(
    run_name=None,
    n_envs=32,
    n_shards=8,
    total_timesteps=1_000_000_000,
    save_every_timesteps=128 * 1024 * 8,
    ppo=PPOConfig(
        learning_rate=2e-4,
        n_steps=256,
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
            "agent_code.rule_based_agent.callbacks",
            "agent_code.coin_collector_agent.callbacks",
            "agent_code.simple_agent.callbacks",
            "agent_code.peaceful_agent.callbacks",
        ],
        arrangements=[
            OpponentArrangement(n_static=0, n_self_play=3, weight=3.0),
            OpponentArrangement(n_static=1, n_self_play=2, weight=3.0),
            OpponentArrangement(n_static=2, n_self_play=1, weight=2.0),
            OpponentArrangement(n_static=3, n_self_play=0, weight=1.0),
        ],
        allow_repeat_static_opponents=True,
        shuffle_opponent_order=True,
        resample_every_n_rollouts=1,
        pool_size=64,
        add_checkpoint_every_epochs=1,
        sample_strategy="latest_biased",
        latest_bias=0.1,
    ),
)