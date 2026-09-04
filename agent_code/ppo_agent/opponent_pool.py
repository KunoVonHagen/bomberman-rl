from __future__ import annotations

import importlib
import pathlib
import random
import tempfile
from typing import Callable, Dict, List, Optional, Tuple
from sb3_contrib import MaskablePPO

from .config import EnvConfig, SelfPlayConfig, OpponentArrangement
from .checkpoint_manager import CheckpointManager
from agent_code.ppo_agent.gym_environment import BombermanGymEnv, ACTION_INDICES
from environment import WorldArgs

OpponentPair = Tuple[Callable, Callable]


class _CheckpointOpponent:
    """
    Wraps a MaskablePPO checkpoint as an opponent for the BombermanGymEnv.
    Lazily loads the model and environment on first use, and caches them for subsequent calls.
    """

    def __init__(self, model_path: str, env_cfg: EnvConfig):
        self.model_path = model_path
        self.env_cfg = env_cfg
        self._model = None
        self._obs_env = None
        self._action_names = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_obs_env"] = None
        state["_action_names"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_ready(self):
        if self._model is None:
            self._model = MaskablePPO.load(
                self.model_path,
                custom_objects={"n_envs": 1, "n_steps": 1},
            )

        if self._obs_env is None:
            log_dir = tempfile.mkdtemp(prefix="checkpoint_opponent_")
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
                log_dir=log_dir,
                match_name=None,
                fps=self.env_cfg.fps,
                replay=False,
                continue_without_training=self.env_cfg.continue_without_training,
            )
            self._obs_env = BombermanGymEnv(world_args, opponents=[((lambda handle: None), (lambda handle, state: "WAIT"))] * 3, layer_config=self.env_cfg.layer_config)
            self._action_names = {v: k for k, v in ACTION_INDICES.items() if k is not None}

    def setup(self, agent):
        self._ensure_ready()
        self._obs_env.reset()

    def act(self, agent, game_state):
        self._ensure_ready()
        obs_env = self._obs_env
        obs = obs_env.observation_from_game_state(game_state)
        action_masks = obs_env.action_masks()

        action_idx, _ = self._model.predict(obs, deterministic=False, action_masks=action_masks)
        return self._action_names[int(action_idx)]

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

        totals = {a.n_static + a.n_self_play for a in cfg.arrangements}
        if len(totals) > 1:
            raise ValueError(
                "All self_play.arrangements must add up to the same total opponent "
                f"count -- got totals {sorted(totals)}."
            )

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