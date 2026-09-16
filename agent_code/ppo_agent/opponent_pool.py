from __future__ import annotations

import importlib
import pathlib
import random
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from stable_baselines3.common.save_util import json_to_data

from .config import EnvConfig, SelfPlayConfig, OpponentArrangement, ScenarioArrangement
from .checkpoint_manager import CheckpointManager
from .model import InferenceOptimizer
from .grouped_forward import (EXTRACTOR_KEYS, EXTRACTOR_PREFIX, grouped_extractor_forward, grouped_linear,
                              model_indices, observation_tensors, stack_state_dicts)
from .gym_environment import BombermanGymEnv, ACTION_INDICES
from environment import WorldArgs

OpponentPair = Tuple[Callable, Callable]

_MODEL_CACHE: Dict[str, "MaskablePPO"] = {}


def load_inference_model(model_path: str) -> "MaskablePPO":
    with zipfile.ZipFile(model_path) as archive:
        data = json_to_data(archive.read("data").decode(), custom_objects={"rollout_buffer_class": MaskableDictRolloutBuffer})
    policy_kwargs = dict(data.get("policy_kwargs") or {})
    policy_kwargs["optimizer_class"] = InferenceOptimizer
    return MaskablePPO.load(
        model_path,
        device="cpu",
        custom_objects={
            "n_envs": 1,
            "n_steps": 1,
            "rollout_buffer_class": MaskableDictRolloutBuffer,
            "policy_kwargs": policy_kwargs,
        },
    )
_OBS_ENV_CACHE: Dict[Tuple[Any, int, int], "BombermanGymEnv"] = {}

_BATCH_CAPACITIES = (1, 2, 4, 8, 16, 32, 64)

POLICY_KEYS = [
    "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
    "action_net.weight", "action_net.bias",
]
GROUPED_KEYS = [EXTRACTOR_PREFIX + key for key in EXTRACTOR_KEYS] + POLICY_KEYS
_GROUPED_STACK: Optional[tuple] = None
GROUPED_MIN_MODELS = 4


def _grouped_state_dict(model: "MaskablePPO") -> dict:
    return model.policy.state_dict()


def _grouped_stack():
    global _GROUPED_STACK
    paths = tuple(sorted(_MODEL_CACHE))
    if _GROUPED_STACK is None or _GROUPED_STACK[0] != paths:
        stacked = stack_state_dicts([_grouped_state_dict(_MODEL_CACHE[p]) for p in paths], GROUPED_KEYS)
        _GROUPED_STACK = (paths, {p: i for i, p in enumerate(paths)}, stacked)
    return _GROUPED_STACK[1], _GROUPED_STACK[2]


def _predict_per_model(owners, obs: dict, masks: np.ndarray) -> List[str]:
    names: List[Optional[str]] = [None] * len(owners)
    groups: Dict[str, Tuple[Any, list]] = {}
    for j, owner in enumerate(owners):
        groups.setdefault(owner.model_path, (owner, []))[1].append(j)
    for owner, members in groups.values():
        idx = np.asarray(members, dtype=np.int64)
        batch = {"grid_tensor": obs["grid_tensor"][idx], "features": obs["features"][idx]}
        for j, name in zip(members, owner.predict_batch(batch, masks[idx])):
            names[j] = name
    return names


def _batch_capacity(k: int) -> int:
    for capacity in _BATCH_CAPACITIES:
        if capacity >= k:
            return capacity
    return _BATCH_CAPACITIES[-1]


class _CheckpointOpponent:
    """
    Wraps a MaskablePPO checkpoint as an opponent for the BombermanGymEnv.
    Lazily loads the model and environment on first use, and caches them for subsequent calls.
    """

    def __init__(self, model_path: str, env_cfg: EnvConfig):
        self.model_path = model_path
        self.env_cfg = env_cfg
        self._model = None
        self._action_names = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_action_names"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_model(self):
        if self._model is None:
            self._model = _MODEL_CACHE.get(self.model_path)
            if self._model is None:
                self._model = load_inference_model(self.model_path)
                _MODEL_CACHE[self.model_path] = self._model
        elif self.model_path not in _MODEL_CACHE:
            _MODEL_CACHE[self.model_path] = self._model
        if self._action_names is None:
            self._action_names = {v: k for k, v in ACTION_INDICES.items() if k is not None}

    def _obs_env_with(self, capacity: int) -> BombermanGymEnv:
        layers = self.env_cfg.layer_config
        key = (None if layers is None else tuple(layers), int(self.env_cfg.env_version), capacity)
        env = _OBS_ENV_CACHE.get(key)
        if env is None:
            world_args = WorldArgs(
                scenario=self.env_cfg.scenario,
                seed=None,
                silence_errors=True,
                no_gui=True,
                make_video=False,
                save_replay=False,
                save_stats=False,
                turn_based=self.env_cfg.turn_based,
                update_interval=self.env_cfg.update_interval,
                log_dir=None,
                match_name=None,
                fps=self.env_cfg.fps,
                replay=False,
                continue_without_training=self.env_cfg.continue_without_training,
            )
            env = BombermanGymEnv(
                world_args,
                opponents=[((lambda handle: None), (lambda handle, state: "WAIT"))] * 3,
                layer_config=self.env_cfg.layer_config,
                env_version=self.env_cfg.env_version,
                n_envs=capacity,
            )
            env.reset()
            _OBS_ENV_CACHE[key] = env
        return env

    @property
    def batch_key(self) -> str:
        return self.model_path

    def setup(self, agent):
        self._ensure_model()

    def act(self, agent, game_state):
        return self.batch_act([agent], [game_state])[0]

    def predict_batch(self, obs: dict, masks: np.ndarray) -> List[str]:
        self._ensure_model()
        policy = self._model.policy
        if policy.training:
            policy.set_training_mode(False)
        obs_t, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            indices = policy._predict(obs_t, deterministic=False, action_masks=masks)
        return [self._action_names[int(i)] for i in indices.cpu().numpy().reshape(-1)]

    @staticmethod
    def predict_grouped(owners, obs: dict, masks: np.ndarray) -> List[str]:
        paths = [owner.model_path for owner in owners]
        n_models = len(set(paths))
        if n_models == 1:
            return owners[0].predict_batch(obs, masks)
        if n_models < GROUPED_MIN_MODELS:
            return _predict_per_model(owners, obs, masks)
        for owner in owners:
            owner._ensure_model()
        index, stacked = _grouped_stack()
        if stacked is None:
            return _predict_per_model(owners, obs, masks)
        grid, features = observation_tensors(obs)
        idx = model_indices(paths, index)
        with torch.no_grad():
            latent = grouped_extractor_forward(stacked, idx, grid, features)
            hidden = torch.tanh(grouped_linear(
                latent, stacked["mlp_extractor.policy_net.0.weight"], stacked["mlp_extractor.policy_net.0.bias"], idx))
            hidden = torch.tanh(grouped_linear(
                hidden, stacked["mlp_extractor.policy_net.2.weight"], stacked["mlp_extractor.policy_net.2.bias"], idx))
            logits = grouped_linear(hidden, stacked["action_net.weight"], stacked["action_net.bias"], idx)
            mask_t = torch.as_tensor(np.asarray(masks, dtype=bool)).reshape(logits.shape)
            logits = torch.where(mask_t, logits, torch.tensor(-1e8, dtype=logits.dtype))
            actions = torch.distributions.Categorical(logits=logits).sample()
        names = owners[0]._action_names
        return [names[int(a)] for a in actions.numpy()]

    def batch_act(self, agents, game_states) -> List[str]:
        names: List[str] = []
        limit = _BATCH_CAPACITIES[-1]
        for start in range(0, len(game_states), limit):
            chunk = game_states[start:start + limit]
            env = self._obs_env_with(_batch_capacity(len(chunk)))
            obs, masks = env.observations_from_game_states(chunk)
            names.extend(self.predict_batch(obs, masks))
        return names

    def as_pair(self) -> OpponentPair:
        return (self.setup, self.act)


def _make_checkpoint_opponent(model_path: str, env_cfg: EnvConfig) -> OpponentPair:
    return _CheckpointOpponent(model_path, env_cfg).as_pair()


class OpponentPool:
    def __init__(self, ckman: CheckpointManager, cfg: SelfPlayConfig):
        self.ckman = ckman
        self.cfg = cfg
        self.env_cfg: EnvConfig = ckman.config.env
        self._checkpoints: List[pathlib.Path] = []
        self._opponent_cache: Dict[str, OpponentPair] = {}
        self._last_descriptions: List[str] = []

        if cfg.enabled:
            self._checkpoints = list(ckman.list_checkpoints())[-cfg.pool_size:]

    @property
    def num_checkpoints(self) -> int:
        """Number of self-play checkpoints currently in this pool's window."""
        return len(self._checkpoints)

    def maybe_add_checkpoint(self, checkpoint_dir: Optional[pathlib.Path], _timesteps: int) -> None:
        """Call this after every saved checkpoint; it decides whether to add
        it to the self-play pool based on `add_checkpoint_every_epochs`."""
        if not self.cfg.enabled or checkpoint_dir is None:
            return

        every = max(1, self.cfg.add_checkpoint_every_epochs)
        saved_count = len(self.ckman.list_checkpoints())
        if saved_count % every == 0:
            self._checkpoints.append(checkpoint_dir)
            self._checkpoints = self._checkpoints[-self.cfg.pool_size:]
            self._prune_opponent_cache()

    def _prune_opponent_cache(self) -> None:
        """Drop cached opponent pairs for checkpoints that fell out of the
        active pool."""
        active_keys = {str(p.resolve()) for p in self._checkpoints}
        for key in list(self._opponent_cache):
            if key not in active_keys:
                del self._opponent_cache[key]

    def _sample_checkpoint(self) -> Optional[pathlib.Path]:
        if not self._checkpoints:
            return None
        if self.cfg.sample_strategy == "latest_biased" and random.random() < self.cfg.latest_bias:
            return self._checkpoints[-1]
        return random.choice(self._checkpoints)

    def _checkpoint_opponent(self, checkpoint_dir: pathlib.Path) -> OpponentPair:
        key = str(checkpoint_dir.resolve())
        if key not in self._opponent_cache:
            model_path = str((checkpoint_dir / "model.zip").resolve())
            self._opponent_cache[key] = _make_checkpoint_opponent(model_path, self.env_cfg)
        return self._opponent_cache[key]

    @staticmethod
    def _resolve_static(module_path: str) -> OpponentPair:
        module = importlib.import_module(module_path)
        return (module.setup, module.act)

    def _choose_arrangement(self) -> OpponentArrangement:
        """
        Randomly choose one of the configured opponent arrangements, weighted by
        their `weight` attribute. If all weights are <= 0, treat them as equal
        """
        arrangements = self.cfg.arrangements
        if len(arrangements) == 1:
            return arrangements[0]
        weights = [max(0.0, a.weight) for a in arrangements]
        if sum(weights) <= 0:
            weights = [1.0] * len(arrangements)
        return random.choices(arrangements, weights=weights, k=1)[0]

    def _choose_scenario_arrangement(self) -> ScenarioArrangement:
        """
        Randomly choose one entry from env_cfg.scenario_mix, weighted by
        `weight` -- same convention as _choose_arrangement for opponents.
        """
        mix = self.env_cfg.scenario_mix
        if len(mix) == 1:
            return mix[0]
        weights = [max(0.0, a.weight) for a in mix]
        if sum(weights) <= 0:
            weights = [1.0] * len(mix)
        return random.choices(mix, weights=weights, k=1)[0]

    def current_scenario(self) -> str:
        """
        Draw one scenario from env_cfg.scenario_mix (e.g. to mix
        "coin-heaven" into training for generalization). Falls back to
        env_cfg.scenario if no mix is set.

        Called once per individual game via OpponentSampler.sample_one() (see
        below) rather than once per PPO rollout, so the scenario mix is
        rerolled every game instead of persisting across a whole rollout.
        """
        mix = self.env_cfg.scenario_mix
        if not mix:
            return self.env_cfg.scenario
        return self._choose_scenario_arrangement().scenario

    def _sample_static_opponents(self, k: int) -> List[str]:
        if k <= 0:
            return []
        pool = self.cfg.static_opponents
        if not pool:
            raise ValueError(
                f"OpponentPool needs to fill {k} opponent slot(s) with static agents "
                "(self-play is disabled or its checkpoint pool is still empty) but "
                "self_play.static_opponents is empty. Add at least one static "
                "opponent module path to config.self_play.static_opponents."
            )
        if self.cfg.allow_repeat_static_opponents or k > len(pool):
            return [random.choice(pool) for _ in range(k)]
        return random.sample(pool, k=k)

    def current_opponents(self) -> List[OpponentPair]:
        """
        Return a list of (setup, act) callables for the opponents to use in the next match.
        The list is shuffled if `shuffle_opponent_order` is True.

        Called once per individual game via OpponentSampler.sample_one() (see
        below) rather than once per PPO rollout, so the opponent lineup is
        rerolled every game instead of persisting across a whole rollout.
        """
        arrangement = self._choose_arrangement()
        total_needed = arrangement.n_static + arrangement.n_self_play

        opponents: List[OpponentPair] = []
        descriptions: List[str] = []

        n_self_play_requested = arrangement.n_self_play if self.cfg.enabled else 0
        self_play_filled = 0
        for _ in range(n_self_play_requested):
            ckpt = self._sample_checkpoint()
            if ckpt is None:
                break
            opponents.append(self._checkpoint_opponent(ckpt))
            descriptions.append(f"checkpoint:{ckpt.parent.parent.name}/{ckpt.name}")
            self_play_filled += 1

        n_static_needed = total_needed - self_play_filled
        for path in self._sample_static_opponents(n_static_needed):
            opponents.append(self._resolve_static(path))
            descriptions.append(path)

        if self.cfg.shuffle_opponent_order and opponents:
            paired = list(zip(opponents, descriptions))
            random.shuffle(paired)
            opponents = [p[0] for p in paired]
            descriptions = [p[1] for p in paired]

        label = f"arrangement=(static:{n_static_needed},self_play:{self_play_filled})"
        self._last_descriptions = [label, *descriptions]
        return opponents

    def last_opponent_descriptions(self) -> List[str]:
        """
        Return a list of human-readable descriptions of the opponents used in the last call to `current_opponents()`.
        """
        return list(self._last_descriptions)

    @staticmethod
    def _normalized_weights(weights: List[float]) -> List[float]:
        clamped = [max(0.0, w) for w in weights]
        total = sum(clamped)
        if total <= 0:
            clamped = [1.0] * len(weights)
            total = float(len(weights))
        return [w / total for w in clamped]

    def arrangement_distribution(self) -> List[Tuple[str, float]]:
        """
        The probability of each configured opponent arrangement being drawn,
        matching _choose_arrangement's weighting exactly.
        """
        arrangements = self.cfg.arrangements
        probs = self._normalized_weights([a.weight for a in arrangements])
        return [
            (f"(static:{a.n_static},self_play:{a.n_self_play})", p)
            for a, p in zip(arrangements, probs)
        ]

    def scenario_distribution(self) -> List[Tuple[str, float]]:
        """
        The probability of each configured scenario being drawn, matching
        _choose_scenario_arrangement's weighting exactly.
        """
        mix = self.env_cfg.scenario_mix
        if not mix:
            return [(self.env_cfg.scenario, 1.0)]
        probs = self._normalized_weights([a.weight for a in mix])
        return [(a.scenario, p) for a, p in zip(mix, probs)]


class OpponentSampler:
    """
    A callable that samples opponents and scenarios for a single game.
    """

    def __init__(self, pool: "OpponentPool"):
        self.pool = pool

    def sample_one(self) -> Tuple[List[OpponentPair], str, List[str]]:
        """Draw one game's worth of opponents + scenario from the wrapped pool."""
        opponents = self.pool.current_opponents()
        descriptions = self.pool.last_opponent_descriptions()
        scenario = self.pool.current_scenario()
        return opponents, scenario, descriptions

    def preload(self, in_use=()) -> None:
        keep = set()
        for checkpoint in list(self.pool._checkpoints):
            _setup_fn, act_fn = self.pool._checkpoint_opponent(checkpoint)
            owner = getattr(act_fn, "__self__", None)
            if owner is not None and hasattr(owner, "_ensure_model"):
                owner._ensure_model()
                keep.add(owner.model_path)
        for owner in in_use:
            path = getattr(owner, "model_path", None)
            if path is not None:
                keep.add(path)
        for path in list(_MODEL_CACHE):
            if path not in keep:
                del _MODEL_CACHE[path]
