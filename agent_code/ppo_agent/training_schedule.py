from __future__ import annotations

import argparse
import json
import pathlib
import time

STATIC_OPPONENT_POOL = [
    "agent_code.my_agent.callbacks",
    "agent_code.simple_agent.callbacks",
    "agent_code.rule_based_agent.callbacks",
    "agent_code.coin_collector_agent.callbacks",
]

DEFAULT_SCHEDULE = [
    {
        "at": 0.0,
        "overrides": {
            "self_play.static_opponents": STATIC_OPPONENT_POOL,
            "self_play.arrangements": "0,3,1.0",
            "ppo.learning_rate": 5e-4,
            "ppo.n_steps": 8192,
            "ppo.batch_size": 8192,
            "rewards.coin_collected": 1.0,
            "rewards.killed_opponent": 5.0,
            "rewards.waited": -0.01,
            "rewards.invalid_action": -0.2,
            "rewards.crate_destroyed": 0.3,
            "rewards.killed_self": -1.0,
            "rewards.got_killed": -5.0,
            "rewards.coin_shaping_coef": 0.05,
            "rewards.crate_shaping_coef": 0.02,
            "rewards.danger_penalty_coef": 0.05,
            "rewards.escape_bonus_coef": 0.05,
            "rewards.trap_shaping_coef": 0.10,
            "save_every_timesteps": 256 * 8192 * 8,
        },
    },
    {
        "at": 0.30,
        "overrides": {
            "ppo.learning_rate": 1e-5,
            "ppo.n_steps": 1024,
            "ppo.batch_size": 2048,
            "self_play.arrangements": "0,3,3.0;1,2,2.0;2,1,1.0",
            "save_every_timesteps": 256 * 1024 * 8,
        },
    },
    {
        "at": 0.45,
        "overrides": {
            "rewards.waited": -0.005,
            "rewards.invalid_action": -0.1,
            "rewards.crate_destroyed": 0.15,
            "rewards.killed_self": -0.5,
            "rewards.got_killed": -2.5,
            "rewards.coin_shaping_coef": 0.025,
            "rewards.crate_shaping_coef": 0.01,
            "rewards.danger_penalty_coef": 0.025,
            "rewards.escape_bonus_coef": 0.025,
            "rewards.trap_shaping_coef": 0.05,
            "self_play.arrangements": "0,3,1.0;1,2,2.0;2,1,2.0;3,0,1.0",
        },
    },
    {
        "at": 0.60,
        "overrides": {
            "self_play.arrangements": "1,2,1.0;2,1,2.0;3,0,2.0",
        },
    },
    {
        "at": 0.65,
        "overrides": {
            "rewards.waited": -0.002,
            "rewards.invalid_action": -0.04,
            "rewards.crate_destroyed": 0.06,
            "rewards.killed_self": -0.2,
            "rewards.got_killed": -1.0,
            "rewards.coin_shaping_coef": 0.01,
            "rewards.crate_shaping_coef": 0.0,
            "rewards.danger_penalty_coef": 0.01,
            "rewards.escape_bonus_coef": 0.01,
            "rewards.trap_shaping_coef": 0.02,
        },
    },
    {
        "at": 0.75,
        "overrides": {
            "self_play.arrangements": "2,1,1.0;3,0,3.0",
        },
    },
    {
        "at": 0.85,
        "overrides": {
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
        "at": 0.90,
        "overrides": {
            "self_play.arrangements": "3,0,1.0",
        },
    },
]


def load_schedule(path: str | None) -> list[dict]:
    if path is None:
        return DEFAULT_SCHEDULE
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, list) or not all(
        isinstance(s, dict) and "at" in s and "overrides" in s for s in data
    ):
        raise ValueError(
            f"schedule file {path} must be a JSON list of "
            f'{{"at": <fraction 0-1>, "overrides": {{...}}}} objects'
        )
    return sorted(data, key=lambda s: s["at"])


def wait_for_run_dir(run_dir: pathlib.Path, timeout_seconds: float, poll_seconds: float = 5.0) -> bool:
    """train.py's CheckpointManager creates run_dir at process start; give it
    a little while to show up rather than assuming it already exists."""
    waited = 0.0
    while not run_dir.exists():
        if waited >= timeout_seconds:
            return False
        time.sleep(poll_seconds)
        waited += poll_seconds
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", required=True,
                   help="Run directory to write live_overrides.json into (must match the "
                        "run train.py is writing to).")
    p.add_argument("--total-seconds", type=float, required=True,
                   help="Wall-clock budget the schedule's 'at' fractions are relative to -- "
                        "typically your SLURM --time, converted to seconds.")
    p.add_argument("--check-interval", type=float, default=300.0,
                   help="Seconds between checks for whether the next stage should activate "
                        "(default: 300 = 5 minutes; live_overrides.json is only rewritten "
                        "when the active stage actually changes).")
    p.add_argument("--schedule-file", type=str, default=None,
                   help="Optional JSON file describing the schedule (see module docstring). "
                        "Defaults to the built-in curriculum above (self-play/static "
                        "opponent mix, PPO learning-rate/n_steps/batch_size, and "
                        "reward-shaping anneal).")
    p.add_argument("--wait-timeout", type=float, default=1800.0,
                   help="Seconds to wait for --run-dir to be created by train.py before "
                        "giving up (default: 1800 = 30 minutes).")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    schedule = load_schedule(args.schedule_file)
    live_path = run_dir / "live_overrides.json"

    print(f"[training-schedule] waiting for {run_dir} to be created by train.py...", flush=True)
    if not wait_for_run_dir(run_dir, args.wait_timeout):
        print(f"[training-schedule] gave up waiting for {run_dir} after {args.wait_timeout:.0f}s "
              f"-- exiting without writing anything.", flush=True)
        return
    print(f"[training-schedule] {run_dir} found -- watching {len(schedule)} stage(s) "
          f"over a {args.total_seconds:.0f}s budget, checking every {args.check_interval:.0f}s.",
          flush=True)

    start = time.time()
    applied_idx = -1
    while True:
        elapsed = time.time() - start
        fraction = elapsed / args.total_seconds if args.total_seconds > 0 else 1.0

        target_idx = applied_idx
        for i, stage in enumerate(schedule):
            if fraction >= stage["at"]:
                target_idx = i

        if target_idx != applied_idx:
            stage = schedule[target_idx]
            live_path.write_text(json.dumps(stage["overrides"], indent=2))
            print(f"[training-schedule] t={elapsed:.0f}s ({fraction:.1%} of budget): "
                  f"activating stage {target_idx} (at>={stage['at']}) -> {live_path}", flush=True)
            for k, v in stage["overrides"].items():
                print(f"    {k} = {v}", flush=True)
            applied_idx = target_idx

        if fraction >= 1.0 and applied_idx == len(schedule) - 1:
            print("[training-schedule] final stage reached and time budget elapsed -- done.", flush=True)
            break

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()