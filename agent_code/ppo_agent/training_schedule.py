from __future__ import annotations

import json
import pathlib

STATIC_OPPONENT_POOL = [
    "agent_code.my_agent.callbacks",
    "agent_code.simple_agent.callbacks",
    "agent_code.rule_based_agent.callbacks",
    "agent_code.coin_collector_agent.callbacks",
]

DEFAULT_SCHEDULE = [
    {
        "at_timesteps": 0,
        "overrides": {
            "self_play.static_opponents": STATIC_OPPONENT_POOL,
            "self_play.arrangements": "0,0,1.5;1,0,2.0;0,1,2.0;2,0,1.5;1,1,1.5;0,2,1.0;2,1,1.0;1,2,0.75;0,3,0.5",
            "env.scenario_mix": "classic,0.85;coin-heaven,0.15",
            "ppo.learning_rate": 1e-5,
            "ppo.n_steps": 1024,
            "ppo.batch_size": 4096,
            "rewards.coin_collected": 1.0,
            "rewards.killed_opponent": 5.0,
            "rewards.trapping_bomb": 0.5,
            "rewards.waited": 0.0,
            "rewards.invalid_action": -0.05,
            "rewards.crate_destroyed": 0.3,
            "rewards.killed_self": -1.0,
            "rewards.got_killed": -5.0,
            "rewards.coin_shaping_coef": 0.05,
            "rewards.crate_shaping_coef": 0.02,
            "rewards.danger_penalty_coef": 0.05,
            "rewards.escape_bonus_coef": 0.05,
            "rewards.trap_shaping_coef": 0.03,
            "save_every_timesteps": 256 * 1024 * 4,
        },
    },
    {
        "at_timesteps": 15_000_000,
        "overrides": {
            "self_play.arrangements": "1,0,1.0;0,1,1.0;2,0,1.0;1,1,1.5;0,2,1.5;2,1,2.0;1,2,1.5;0,3,1.0;3,0,0.5",
            "env.scenario_mix": "classic,0.90;coin-heaven,0.10",
            "ppo.learning_rate": 7e-6,
        },
    },
    {
        "at_timesteps": 30_000_000,
        "gate": {"min_score": 2.0, "max_suicide_rate": 0.35, "max_delay_timesteps": 10_000_000},
        "overrides": {
            "rewards.trapping_bomb": 0.25,
            "rewards.waited": 0.0,
            "rewards.invalid_action": -0.025,
            "rewards.crate_destroyed": 0.15,
            "rewards.killed_self": -0.5,
            "rewards.got_killed": -2.5,
            "rewards.coin_shaping_coef": 0.025,
            "rewards.crate_shaping_coef": 0.01,
            "rewards.danger_penalty_coef": 0.025,
            "rewards.escape_bonus_coef": 0.025,
            "rewards.trap_shaping_coef": 0.015,
            "self_play.arrangements": "1,1,1.0;0,2,1.0;2,1,2.5;1,2,2.5;0,3,1.5;3,0,1.0",
        },
    },
    {
        "at_timesteps": 40_000_000,
        "overrides": {
            "env.scenario_mix": "classic,1.0",
            "self_play.arrangements": "2,1,2.0;1,2,2.5;0,3,2.0;3,0,1.0",
            "ppo.learning_rate": 4e-6,
        },
    },
    {
        "at_timesteps": 50_000_000,
        "overrides": {
            "rewards.trapping_bomb": 0.1,
            "rewards.waited": 0.0,
            "rewards.invalid_action": -0.01,
            "rewards.crate_destroyed": 0.06,
            "rewards.killed_self": -0.2,
            "rewards.got_killed": -1.0,
            "rewards.coin_shaping_coef": 0.01,
            "rewards.crate_shaping_coef": 0.0,
            "rewards.danger_penalty_coef": 0.01,
            "rewards.escape_bonus_coef": 0.01,
            "rewards.trap_shaping_coef": 0.006,
        },
    },
    {
        "at_timesteps": 60_000_000,
        "overrides": {
            "self_play.arrangements": "0,3,3.0;1,2,3.0;2,1,1.5;3,0,2.0",
            "ppo.learning_rate": 1e-6,
        },
    },
    {
        "at_timesteps": 70_000_000,
        "overrides": {
            "rewards.trapping_bomb": 0.0,
            "rewards.waited": 0.0,
            "rewards.invalid_action": 0.0,
            "rewards.crate_destroyed": 0.0,
            "rewards.killed_self": 0.0,
            "rewards.got_killed": 0.0,
            "rewards.coin_shaping_coef": 0.0,
            "rewards.crate_shaping_coef": 0.0,
            "rewards.danger_penalty_coef": 0.0,
            "rewards.escape_bonus_coef": 0.0,
            "rewards.trap_shaping_coef": 0.0,
        },
    },
    {
        "at_timesteps": 80_000_000,
        "overrides": {
            "self_play.arrangements": "0,3,1.0;1,2,2.0;2,1,2.0;3,0,2.0",
            "total_timesteps": 100_000_000,
            "ppo.learning_rate": 3e-7,
        },
    },
]


def load_schedule(path: str | None) -> list[dict]:
    """Load a schedule (a list of {"at_timesteps": absolute_step_count, "overrides": {...}}
    stages) from a JSON file, or return the built-in DEFAULT_SCHEDULE if path is None."""
    if path is None:
        return DEFAULT_SCHEDULE
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, list) or not all(
        isinstance(s, dict) and "at_timesteps" in s and "overrides" in s for s in data
    ):
        raise ValueError(
            f"schedule file {path} must be a JSON list of "
            f'{{"at_timesteps": <absolute count of total_timesteps>, "overrides": {{...}}}} '
            f"objects"
        )
    return sorted(data, key=lambda s: s["at_timesteps"])


def stage_index_for_timesteps(schedule: list[dict], timesteps_done: int) -> int:
    """
    Given a schedule and a count of timesteps done, return the index of the
    last stage whose "at_timesteps" threshold is less than or equal to
    timesteps_done. Returns -1 if no stages have been reached yet.
    """
    target_idx = -1
    for i, stage in enumerate(schedule):
        if timesteps_done >= stage["at_timesteps"]:
            target_idx = i
    return target_idx