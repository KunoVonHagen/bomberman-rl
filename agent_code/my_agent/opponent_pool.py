from __future__ import annotations

import importlib
import pathlib
import random
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

from config import EnvConfig, SelfPlayConfig
from checkpoint_manager import CheckpointManager

OpponentPair = Tuple[Callable, Callable]


class _CheckpointOpponent:
    """
    Wraps a saved checkpoint (model + VecNormalize) as an opponent for self-play.
    The model and VecNormalize are loaded lazily on first use, and the Gym environment is also created lazily.
    This allows the opponent to be pickled and sent to subprocesses without loading the model or environment until needed.
    """

    def __init__(self, model_path: str, vecnorm_path: str, env_cfg: EnvConfig):
        self.model_path = model_path
        self.vecnorm_path = vecnorm_path
        self.env_cfg = env_cfg
        self._model = None
        self._vecnorm = None
        self._obs_env = None
        self._action_names = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_vecnorm"] = None
        state["_obs_env"] = None
        state["_action_names"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_ready(self):
        if self._model is None:
            from sb3_contrib import MaskablePPO
            self._model = MaskablePPO.load(
                self.model_path,
                custom_objects={"n_envs": 1, "n_steps": 1},
            )

        if self._vecnorm is None:
            import pickle
            with open(self.vecnorm_path, "rb") as f:
                self._vecnorm = pickle.load(f)

        if self._obs_env is None:
            from agent_code.my_agent.gym_environment import BombermanGymEnv, ACTION_INDICES
            from environment import WorldArgs

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
            self._obs_env = BombermanGymEnv(world_args, opponents=[], layer_config=self.env_cfg.layer_config)
            self._action_names = {v: k for k, v in ACTION_INDICES.items()}

    def setup(self, agent):
        self._ensure_ready()
        self._obs_env.reset()

    def act(self, agent, game_state):
        self._ensure_ready()
        obs_env = self._obs_env
        obs = obs_env.observation_from_game_state(game_state)
        obs = self._vecnorm.normalize_obs(obs)
        action_masks = obs_env.action_masks()

        action_idx, _ = self._model.predict(obs, deterministic=True, action_masks=action_masks)
        return self._action_names[int(action_idx)]

    def as_pair(self) -> OpponentPair:
        return (self.setup, self.act)


def _make_checkpoint_opponent(model_path: str, vecnorm_path: str, env_cfg: EnvConfig) -> OpponentPair:
    return _CheckpointOpponent(model_path, vecnorm_path, env_cfg).as_pair()


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
            vecnorm_path = str((checkpoint_dir / "vecnormalize.pkl").resolve())
            self._opponent_cache[key] = _make_checkpoint_opponent(model_path, vecnorm_path, self.env_cfg)
        return self._opponent_cache[key]

    @staticmethod
    def _resolve_static(module_path: str) -> OpponentPair:
        module = importlib.import_module(module_path)
        return (module.setup, module.act)

    def current_opponents(self) -> List[OpponentPair]:
        """Returns the list of (setup_fn, act_fn) pairs to pass as
        `opponents=` into WorldArgs/BombermanGymEnv for the next rollout."""
        opponents: List[OpponentPair] = []
        descriptions: List[str] = []

        for path in self.cfg.static_opponents:
            opponents.append(self._resolve_static(path))
            descriptions.append(path)

        if self.cfg.enabled:
            for _ in range(self.cfg.n_self_play_opponents):
                ckpt = self._sample_checkpoint()
                if ckpt is not None:
                    opponents.append(self._checkpoint_opponent(ckpt))
                    descriptions.append(f"checkpoint:{ckpt.parent.parent.name}/{ckpt.name}")

        self._last_descriptions = descriptions
        return opponents

    def last_opponent_descriptions(self) -> List[str]:
        """Plain-string description of the opponents returned by the most
        recent current_opponents() call — safe to put in JSON metadata,
        unlike the (setup, act) callables themselves."""
        return list(self._last_descriptions)