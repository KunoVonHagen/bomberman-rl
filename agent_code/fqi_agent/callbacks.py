from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np

import settings as s
from ..dqn_agent.gym_environment import ACTION_INDICES, BombermanGymEnv, WorldArgs
from .model import MODEL_FILE, load_model, masked_greedy

RUN: str = "fqi_forest"

AGENT_DIR = pathlib.Path(__file__).resolve().parent
ENV_PREFIX = "FQI_AGENT"
MANIFEST_FILE = "run_manifest.json"
_ACTION_NAMES = {v: k for k, v in ACTION_INDICES.items() if k is not None}


def _setting(name: str, default):
    return os.environ.get(f"{ENV_PREFIX}_{name}", default)


def _resolve_run_dir(run: str) -> pathlib.Path:
    candidates = [
        AGENT_DIR / "runs" / run,
        AGENT_DIR / run,
        pathlib.Path(run),
        AGENT_DIR.parent.parent / "runs" / run,
    ]
    for run_dir in candidates:
        if (run_dir / MANIFEST_FILE).exists() and (run_dir / MODEL_FILE).exists():
            return run_dir
    raise FileNotFoundError(
        f"fqi_agent: no run '{run}' with {MANIFEST_FILE} and {MODEL_FILE} in any of "
        f"{[str(c) for c in candidates]} -- set RUN at the top of callbacks.py"
    )


def _get_obs_env(manifest: dict) -> BombermanGymEnv:
    env = manifest["env"]
    world_args = WorldArgs(
        scenario=env.get("scenario", "classic"), seed=None, silence_errors=True, no_gui=True, make_video=False,
        save_replay=False, save_stats=False, turn_based=False, update_interval=0.1, log_dir=None,
        match_name=None, fps=60, replay=False, continue_without_training=False,
    )
    dummy_opponents = [((lambda handle: None), (lambda handle, state: "WAIT"))] * 3
    return BombermanGymEnv(
        world_args, opponents=dummy_opponents, layer_config=env["layer_config"], env_version=env["env_version"],
    )


def _make_warmup_state() -> dict:
    W, H = s.COLS, s.ROWS
    field = np.zeros((W, H), dtype=np.int8)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(W):
        for y in range(H):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    for x, y in ((3, 1), (1, 3), (7, 8), (8, 7), (9, 9)):
        field[x, y] = 1

    explosion_map = np.zeros((W, H))
    for x, y in ((5, 5), (5, 6), (5, 7)):
        explosion_map[x, y] = 1

    return {
        "round": 0,
        "step": 1,
        "field": field,
        "self": ("warmup", 0, False, (1, 1)),
        "others": [("warmup_opponent", 0, True, (W - 2, H - 2))],
        "bombs": [((1, 1), 3), ((W - 2, 1), 1)],
        "coins": [(1, 5), (11, 11)],
        "explosion_map": explosion_map,
        "user_input": None,
    }


def _choose(self, game_state: dict) -> str:
    obs_env = self._fqi_obs_env
    features = obs_env.observation_from_game_state(game_state)["features"]
    masks = obs_env.action_masks()[0]
    q = self._fqi_model.predict(features[None])[0]
    return _ACTION_NAMES[int(masked_greedy(q, masks))]


def _warm_up(self) -> None:
    time_start = time.time()
    state = _make_warmup_state()
    _choose(self, state)
    state["bombs"] = []
    state["explosion_map"] = np.zeros_like(state["explosion_map"])
    _choose(self, state)
    self.logger.info(f"fqi_agent.setup: warm-up finished in {time.time() - time_start:.2f}s")


def _fail(self, message: str) -> None:
    text = f"{message}\n{traceback.format_exc()}"
    self.logger.error(text)
    print(text, file=sys.stderr)


def setup(self):
    self._fqi_model = None
    self._fqi_obs_env = None

    try:
        run_dir = _resolve_run_dir(_setting("RUN", RUN))
        manifest = json.loads((run_dir / MANIFEST_FILE).read_text())
        model = load_model(run_dir / MODEL_FILE)
        obs_env = _get_obs_env(manifest)
        n_features = int(obs_env.single_observation_space["features"].shape[0])
        if n_features != model.n_features:
            raise ValueError(f"model expects {model.n_features} features, env version "
                             f"{manifest['env']['env_version']} produces {n_features}")
        self._fqi_obs_env = obs_env
        self._fqi_model = model
        self.logger.info(f"fqi_agent.setup: loaded {model.kind} model from {run_dir}")
    except Exception:
        _fail(self, "fqi_agent.setup: could not load the model, every action will be WAIT")
        return

    try:
        _warm_up(self)
    except Exception:
        _fail(self, "fqi_agent.setup: warm-up failed, the first steps may exceed the time limit")


def act(self, game_state: dict) -> str:
    if self._fqi_model is None:
        return "WAIT"

    time_start = time.time()
    try:
        chosen_action = _choose(self, game_state)
    except Exception:
        _fail(self, f"fqi_agent.act: failed at step {game_state.get('step')}, returning WAIT")
        return "WAIT"

    self.logger.debug(
        f"fqi_agent.act: step={game_state['step']}, round={game_state['round']}, "
        f"action={chosen_action}, time={time.time() - time_start:.3f}s"
    )
    return chosen_action
