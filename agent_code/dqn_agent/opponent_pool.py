from __future__ import annotations

import importlib
import pathlib
import random
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

from environment import WorldArgs

from agent_code.dqn_agent.config import EnvConfig, SelfPlayConfig, OpponentArrangement, ScenarioArrangement
from agent_code.dqn_agent.checkpoint_manager import CheckpointManager
from agent_code.dqn_agent.gym_environment import BombermanGymEnv, ACTION_INDICES
from agent_code.dqn_agent.model import MaskableDQN

OpponentPair = Tuple[Callable, Callable]

_MODEL_CACHE: Dict[str, "MaskableDQN"] = {}
_OBS_ENV_CACHE: Dict[str, "BombermanGymEnv"] = {}


class _CheckpointOpponent:
    """Wrap a checkpointed DQN model as an in-game opponent."""

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
        """Load the model and observation environment lazily."""
        if self._model is None:
            self._model = _MODEL_CACHE.get(self.model_path)
            if self._model is None:
                self._model = MaskableDQN.load(self.model_path, device="cpu")
                _MODEL_CACHE[self.model_path] = self._model

        if self._obs_env is None:
            self._obs_env = _OBS_ENV_CACHE.get(self.model_path)
        if self._obs_env is not None:
            self._action_names = {v: k for k, v in ACTION_INDICES.items() if k is not None}
            return
        if self._obs_env is None:
            log_dir = tempfile.mkdtemp(prefix="dqn_checkpoint_opponent_")
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
            self._obs_env = BombermanGymEnv(
                world_args,
                opponents=[((lambda handle: None), (lambda handle, state: "WAIT"))] * 3,
                layer_config=self.env_cfg.layer_config,
                env_version=self.env_cfg.env_version,
            )
            _OBS_ENV_CACHE[self.model_path] = self._obs_env
            self._action_names = {v: k for k, v in ACTION_INDICES.items() if k is not None}

    def setup(self, agent):
        """Prepare the opponent environment before the match starts."""
        self._ensure_ready()
        self._obs_env.reset()

    def act(self, agent, game_state):
        """Pick the next action for the checkpointed model."""
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
    """Manage static and checkpoint-based opponents for self-play."""

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
        """Return the number of active checkpoint opponents."""
        return len(self._checkpoints)

    def maybe_add_checkpoint(self, checkpoint_dir: Optional[pathlib.Path], _timesteps: int) -> None:
        """Add a checkpoint to the pool when the cadence allows it."""
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
        """Sample one checkpoint from the active pool."""
        if not self._checkpoints:
            return None
        if self.cfg.sample_strategy == "latest_biased" and random.random() < self.cfg.latest_bias:
            return self._checkpoints[-1]
        return random.choice(self._checkpoints)

    def _checkpoint_opponent(self, checkpoint_dir: pathlib.Path) -> OpponentPair:
        """Return a cached checkpointed opponent."""
        key = str(checkpoint_dir.resolve())
        if key not in self._opponent_cache:
            model_path = str((checkpoint_dir / "model.zip").resolve())
            self._opponent_cache[key] = _make_checkpoint_opponent(model_path, self.env_cfg)
        return self._opponent_cache[key]

    @staticmethod
    def _resolve_static(module_path: str) -> OpponentPair:
        """Import a static agent module and expose its entry points."""
        module = importlib.import_module(module_path)
        return (module.setup, module.act)

    def _choose_arrangement(self) -> OpponentArrangement:
        """Randomly choose a configured opponent arrangement by weight."""
        arrangements = self.cfg.arrangements
        if len(arrangements) == 1:
            return arrangements[0]
        weights = [max(0.0, a.weight) for a in arrangements]
        if sum(weights) <= 0:
            weights = [1.0] * len(arrangements)
        return random.choices(arrangements, weights=weights, k=1)[0]

    def _choose_scenario_arrangement(self) -> ScenarioArrangement:
        """Randomly choose a scenario mix entry by weight."""
        mix = self.env_cfg.scenario_mix
        if len(mix) == 1:
            return mix[0]
        weights = [max(0.0, a.weight) for a in mix]
        if sum(weights) <= 0:
            weights = [1.0] * len(mix)
        return random.choices(mix, weights=weights, k=1)[0]

    def current_scenario(self) -> str:
        """Draw one scenario from the configured scenario mix."""
        mix = self.env_cfg.scenario_mix
        if not mix:
            return self.env_cfg.scenario
        return self._choose_scenario_arrangement().scenario

    def _sample_static_opponents(self, k: int) -> List[str]:
        """Sample static opponent module paths."""
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
        """Return the opponent callables for the next round."""
        arrangement = self._choose_arrangement()
        total_needed = arrangement.n_static + arrangement.n_self_play

        opponents: List[OpponentPair] = []
        descriptions: List[str] = []

        n_self_play_requested = arrangement.n_self_play if self.cfg.enabled else 0
        self_play_filled = 0
        for _ in range(n_self_play_requested):
            checkpoint = self._sample_checkpoint()
            if checkpoint is None:
                break
            opponents.append(self._checkpoint_opponent(checkpoint))
            descriptions.append(f"checkpoint:{checkpoint.parent.parent.name}/{checkpoint.name}")
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
        """Return the last chosen opponent descriptions."""
        return list(self._last_descriptions)

    @staticmethod
    def _normalized_weights(weights: List[float]) -> List[float]:
        """Normalize weights while handling negatives and zero totals."""
        clamped = [max(0.0, w) for w in weights]
        total = sum(clamped)
        if total <= 0:
            clamped = [1.0] * len(weights)
            total = float(len(weights))
        return [w / total for w in clamped]

    def arrangement_distribution(self) -> List[Tuple[str, float]]:
        """Return the current arrangement sampling weights."""
        arrangements = self.cfg.arrangements
        probs = self._normalized_weights([a.weight for a in arrangements])
        return [
            (f"(static:{a.n_static},self_play:{a.n_self_play})", p)
            for a, p in zip(arrangements, probs)
        ]

    def scenario_distribution(self) -> List[Tuple[str, float]]:
        """Return the current scenario sampling weights."""
        mix = self.env_cfg.scenario_mix
        if not mix:
            return [(self.env_cfg.scenario, 1.0)]
        probs = self._normalized_weights([a.weight for a in mix])
        return [(a.scenario, p) for a, p in zip(mix, probs)]


class OpponentSampler:
    """Sample a full opponent set and scenario for one match."""

    def __init__(self, pool: "OpponentPool"):
        self.pool = pool

    def sample_one(self) -> Tuple[List[OpponentPair], str, List[str]]:
        """Sample one match configuration from the pool."""
        opponents = self.pool.current_opponents()
        descriptions = self.pool.last_opponent_descriptions()
        scenario = self.pool.current_scenario()
        return opponents, scenario, descriptions