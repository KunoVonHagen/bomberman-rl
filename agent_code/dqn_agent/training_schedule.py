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
            "total_timesteps": 150_000_000,
            "self_play.static_opponents": STATIC_OPPONENT_POOL,
            "self_play.arrangements": "0,0,1.0;0,1,1.5;1,0,1.0;0,2,1.5;1,1,1.5;0,3,2.0;1,2,1.5;2,1,1.0",
            "env.scenario_mix": "classic,0.85;coin-heaven,0.15",
            "dqn.learning_rate": 3e-5,
            "dqn.exploration_final_eps": 0.05,
            "dqn.softmax_beta_final": 20.0,
            "rewards.bomb_dropped": 0.01,
            "rewards.coin_found": 0.02,
            "rewards.coin_collected": 1.0,
            "rewards.killed_opponent": 5.0,
            "rewards.killed_self": -1.0,
            "rewards.got_killed": -5.0,
            "rewards.rival_killed_opponent": 0.0,
            "rewards.trapping_bomb": 0.5,
            "rewards.waited": 0.0,
            "rewards.invalid_action": -0.05,
            "rewards.crate_destroyed": 0.0,
            "rewards.crate_potential_coef": 0.05,
            "rewards.coin_shaping_coef": 0.05,
            "rewards.crate_shaping_coef": 0.02,
            "rewards.danger_penalty_coef": 0.05,
            "rewards.escape_bonus_coef": 0.05,
            "rewards.trap_shaping_coef": 0.03,
            "save_every_timesteps": 256 * 1024 * 4,
        },
    },
    {
        "at_timesteps": 20_000_000,
        "overrides": {
            "rewards.bomb_dropped": 0.0,
            "rewards.coin_found": 0.01,
            "dqn.exploration_final_eps": 0.04,
            "dqn.softmax_beta_final": 25.0,
        },
    },
    {
        "at_timesteps": 40_000_000,
        "overrides": {
            "rewards.coin_found": 0.0,
            "self_play.arrangements": "0,0,0.75;0,1,1.25;1,0,0.5;0,2,1.75;1,1,1.25;0,3,3.0;1,2,2.25;2,1,1.5",
            "env.scenario_mix": "classic,0.90;coin-heaven,0.10",
        },
    },
    {
        "at_timesteps": 55_000_000,
        "gate": {"min_score": 2.0, "max_suicide_rate": 0.35, "max_delay_timesteps": 10_000_000},
        "overrides": {
            "rewards.trapping_bomb": 0.25,
            "dqn.learning_rate": 2e-5,
            "rewards.waited": 0.0,
            "rewards.invalid_action": -0.03,
            "rewards.crate_destroyed": 0.0,
            "rewards.killed_self": -0.6,
            "rewards.got_killed": -3.0,
            "rewards.coin_shaping_coef": 0.03,
            "rewards.crate_shaping_coef": 0.012,
            "rewards.danger_penalty_coef": 0.03,
            "rewards.escape_bonus_coef": 0.03,
            "rewards.trap_shaping_coef": 0.018,
        },
    },
    {
        "at_timesteps": 70_000_000,
        "overrides": {
            "self_play.arrangements": "0,0,0.5;0,1,0.75;0,2,1.5;1,2,1.25;2,1,2.25;3,0,2.0",
            "dqn.exploration_final_eps": 0.03,
            "dqn.softmax_beta_final": 33.0,
        },
    },
    {
        "at_timesteps": 85_000_000,
        "overrides": {
            "rewards.trapping_bomb": 0.1,
            "dqn.learning_rate": 1e-5,
            "rewards.waited": 0.0,
            "rewards.invalid_action": -0.015,
            "rewards.crate_destroyed": 0.0,
            "rewards.killed_self": -0.35,
            "rewards.got_killed": -1.75,
            "rewards.coin_shaping_coef": 0.015,
            "rewards.crate_shaping_coef": 0.005,
            "rewards.danger_penalty_coef": 0.015,
            "rewards.escape_bonus_coef": 0.015,
            "rewards.trap_shaping_coef": 0.009,
        },
    },
    {
        "at_timesteps": 100_000_000,
        "overrides": {
            "self_play.arrangements": "0,0,0.4;0,2,1.0;2,1,1.5;3,0,2.0",
            "dqn.exploration_final_eps": 0.02,
            "dqn.softmax_beta_final": 50.0,
        },
    },
    {
        "at_timesteps": 115_000_000,
        "overrides": {
            "rewards.trapping_bomb": 0.0,
            "dqn.learning_rate": 6e-6,
            "rewards.waited": 0.0,
            "rewards.invalid_action": -0.005,
            "rewards.crate_destroyed": 0.0,
            "rewards.killed_self": -0.15,
            "rewards.got_killed": -0.75,
            "rewards.coin_shaping_coef": 0.005,
            "rewards.crate_shaping_coef": 0.0,
            "rewards.danger_penalty_coef": 0.005,
            "rewards.escape_bonus_coef": 0.005,
            "rewards.trap_shaping_coef": 0.003,
        },
    },
    {
        "at_timesteps": 130_000_000,
        "overrides": {
            "self_play.arrangements": "0,1,0.4;0,2,0.6;1,2,1.0;2,1,1.0;3,0,2.0",
            "env.scenario_mix": "classic,0.95;coin-heaven,0.05",
            "dqn.exploration_final_eps": 0.01,
            "dqn.softmax_beta_final": 100.0,
        },
    },
    {
        "at_timesteps": 142_000_000,
        "overrides": {
            "dqn.learning_rate": 3e-6,
            "dqn.exploration_final_eps": 0.005,
            "dqn.softmax_beta_final": 200.0,
            "rewards.waited": 0.0,
            "rewards.invalid_action": 0.0,
            "rewards.crate_destroyed": 0.0,
            "rewards.coin_shaping_coef": 0.0,
            "rewards.danger_penalty_coef": 0.0,
            "rewards.escape_bonus_coef": 0.0,
            "rewards.trap_shaping_coef": 0.0,
            "rewards.killed_self": -0.05,
            "rewards.got_killed": -0.25,
        },
    },
]


def load_schedule(path: str | None) -> list[dict]:
    """Load a schedule from a JSON file or return the built-in default."""
    if path is None:
        return DEFAULT_SCHEDULE
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, list) or not all(
        isinstance(stage, dict) and "at_timesteps" in stage and "overrides" in stage for stage in data
    ):
        raise ValueError(
            f"schedule file {path} must be a JSON list of "
            f'{{"at_timesteps": <absolute count of total_timesteps>, "overrides": {{...}}}} '
            f"objects"
        )
    return sorted(data, key=lambda stage: stage["at_timesteps"])


def stage_index_for_timesteps(schedule: list[dict], timesteps_done: int) -> int:
    """Return the last schedule stage reached by the run."""
    target_idx = -1
    for i, stage in enumerate(schedule):
        if timesteps_done >= stage["at_timesteps"]:
            target_idx = i
    return target_idx