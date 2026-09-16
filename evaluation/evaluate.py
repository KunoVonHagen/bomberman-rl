from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import hashlib
import importlib
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import platform
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import events as e
import settings as s
from environment import BombeRLeWorld, WorldArgs

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

EVAL_DIR = pathlib.Path(__file__).resolve().parent
STATS_DIR = EVAL_DIR / "stats"
LOG_DIR = EVAL_DIR / "logs"

AGENT = "<agent>"

MATCHUPS: Dict[str, Dict[str, Any]] = {
    "task1-coin-heaven-solo": {
        "scenario": "coin-heaven",
        "agents": [AGENT],
        "description": "Task 1: collect revealed coins on a board without crates or opponents",
    },
    "task2-classic-solo": {
        "scenario": "classic",
        "agents": [AGENT],
        "description": "Task 2: find and collect the hidden coins alone (classic crate density)",
    },
    "task2-loot-crate-solo": {
        "scenario": "loot-crate",
        "agents": [AGENT],
        "description": "Task 2 variant: 50 hidden coins, rewards efficient crate clearing",
    },
    "task3-hunt": {
        "scenario": "classic",
        "agents": [AGENT, "peaceful_agent", "coin_collector_agent"],
        "description": "Task 3: hunt the peaceful and the coin collector agent",
    },
    "task4-rule-based-x1": {
        "scenario": "classic",
        "agents": [AGENT, "rule_based_agent"],
        "description": "Task 4: duel against one rule-based agent",
    },
    "task4-rule-based-x3": {
        "scenario": "classic",
        "agents": [AGENT, "rule_based_agent", "rule_based_agent", "rule_based_agent"],
        "description": "Task 4: the `--my-agent` setting, three rule-based agents",
    },
    "grader-random-x3": {
        "scenario": "classic",
        "agents": [AGENT, "random_agent", "random_agent", "random_agent"],
        "description": "The submission test: one game against three random agents",
    },
    "heuristics-mix": {
        "scenario": "classic",
        "agents": [AGENT, "simple_agent", "my_agent", "rule_based_agent"],
        "description": "Mixed lineup with the team's heuristic agents and the rule-based agent",
    },
}
SUITE = [
    "task1-coin-heaven-solo",
    "task2-classic-solo",
    "task3-hunt",
    "task4-rule-based-x1",
    "task4-rule-based-x3",
    "grader-random-x3",
    "heuristics-mix",
]

EVENT_STATS = ("coins", "kills", "suicides", "crates", "bombs", "moves", "invalid")


def resolve_matchup(name: str, agent: str) -> Dict[str, Any]:
    if name not in MATCHUPS:
        raise KeyError(f"unknown matchup '{name}'; choose from {', '.join(MATCHUPS)}")
    spec = dict(MATCHUPS[name])
    spec["agents"] = [agent if a == AGENT else a for a in spec["agents"]]
    spec["name"] = name
    return spec


def _round_seed(seed: int, round_index: int) -> int:
    return int(np.random.SeedSequence([seed, round_index]).generate_state(1)[0])


AGENT_LOG_LEVEL = logging.WARNING
MAX_POOL_WORKERS = 61 if sys.platform == "win32" else 512
_APPLIED_ENV_KEYS: set = set()


def default_workers() -> int:
    return max(1, min(MAX_POOL_WORKERS, (os.cpu_count() or 2) - 1))


def _agent_env_snapshot() -> Dict[str, str]:
    return dict(os.environ)


def _apply_agent_env(snapshot: Dict[str, str]) -> None:
    global _APPLIED_ENV_KEYS
    for key in _APPLIED_ENV_KEYS - set(snapshot):
        os.environ.pop(key, None)
    os.environ.update(snapshot)
    _APPLIED_ENV_KEYS = set(snapshot)


def _make_world(agents: List[str], scenario: str, seed: Optional[int], log_dir: pathlib.Path,
                silence_errors: bool) -> BombeRLeWorld:
    log_dir.mkdir(parents=True, exist_ok=True)
    s.LOG_AGENT_CODE = AGENT_LOG_LEVEL
    s.LOG_AGENT_WRAPPER = AGENT_LOG_LEVEL
    args = WorldArgs(
        no_gui=True,
        fps=0,
        turn_based=False,
        update_interval=0,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir=str(log_dir),
        save_stats=False,
        match_name=None,
        seed=seed,
        silence_errors=silence_errors,
        scenario=scenario,
    )
    lineup = [(name, False) for name in agents]
    if "logging" in inspect.signature(BombeRLeWorld.__init__).parameters:
        return BombeRLeWorld(args, lineup, logging=False)
    return BombeRLeWorld(args, lineup)


def _board_id(world: BombeRLeWorld) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(world.arena).tobytes())
    digest.update(repr(sorted((c.x, c.y) for c in world.coins)).encode())
    digest.update(repr([(a.x, a.y) for a in world.agents]).encode())
    return digest.hexdigest()[:8]


def _close_world_loggers(world: BombeRLeWorld) -> None:
    names = ["BombeRLeWorld"]
    for a in world.agents:
        names += [f"{a.name}_code", f"{a.name}_wrapper"]
    for name in names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def play_rounds(agents: List[str], scenario: str, n_rounds: int, seed: Optional[int],
                log_dir: pathlib.Path, silence_errors: bool = False, round_offset: int = 0,
                progress=None) -> List[Dict[str, Any]]:
    world = _make_world(agents, scenario, seed, log_dir, silence_errors)
    try:
        records = []
        for local_index in range(n_rounds):
            round_index = round_offset + local_index
            if seed is not None:
                round_seed = _round_seed(seed, round_index)
                world.rng = np.random.default_rng(round_seed)
                np.random.seed(round_seed)
                random.seed(round_seed)
            else:
                round_seed = None
            world.new_round()
            board_id = _board_id(world)

            think_time_seen = {a: 0.0 for a in world.agents}
            steps_seen = {a: 0 for a in world.agents}
            think_time_max = {a: 0.0 for a in world.agents}
            slow_calls = {a: 0 for a in world.agents}
            forced_waits = {a: 0 for a in world.agents}
            skipped_steps = {a: 0 for a in world.agents}
            errors = {a: 0 for a in world.agents}
            died_at = {a: None for a in world.agents}
            killed_self = {a: False for a in world.agents}

            while world.running:
                active_before = list(world.active_agents)
                budget_before = {a: a.available_think_time for a in active_before}
                world.do_step()
                for a in active_before:
                    if a.statistics["steps"] > steps_seen[a]:
                        dt = a.statistics["time"] - think_time_seen[a]
                        think_time_seen[a] = a.statistics["time"]
                        steps_seen[a] = a.statistics["steps"]
                        if math.isinf(dt):
                            errors[a] += 1
                        else:
                            think_time_max[a] = max(think_time_max[a], dt)
                        if dt > s.TIMEOUT:
                            slow_calls[a] += 1
                        if dt > budget_before[a]:
                            forced_waits[a] += 1
                    else:
                        skipped_steps[a] += 1
                    if died_at[a] is None and a.dead:
                        died_at[a] = world.step
                        killed_self[a] = e.KILLED_SELF in a.events

            agent_records = []
            for a in world.agents:
                higher = sum(1 for b in world.agents if b is not a and b.score > a.score)
                equal = sum(1 for b in world.agents if b is not a and b.score == a.score)
                record = {
                    "name": a.name,
                    "code_name": a.code_name,
                    "score": a.score,
                    "rank": 1 + higher,
                    "win": len(world.agents) > 1 and higher == 0 and equal == 0,
                    "tie": len(world.agents) > 1 and higher == 0 and equal > 0,
                    "alive": not a.dead,
                    "died_at": died_at[a],
                    "killed_self": killed_self[a],
                    "steps_survived": world.step if died_at[a] is None else died_at[a],
                    "steps_acted": steps_seen[a],
                    "think_time_total": think_time_seen[a] if not math.isinf(think_time_seen[a]) else None,
                    "think_time_max": think_time_max[a],
                    "slow_calls": slow_calls[a],
                    "forced_waits": forced_waits[a],
                    "skipped_steps": skipped_steps[a],
                    "errors": errors[a],
                }
                for key in EVENT_STATS:
                    record[key] = a.statistics[key]
                agent_records.append(record)

            records.append({"round": round_index, "seed": round_seed, "board_id": board_id,
                            "length": world.step, "agents": agent_records})
            if progress is not None:
                progress()
        world.end()
        return records
    finally:
        _close_world_loggers(world)


def _pool_init(agents: List[str], agent_log_level: int) -> None:
    global AGENT_LOG_LEVEL
    AGENT_LOG_LEVEL = agent_log_level
    import torch
    torch.set_num_threads(1)
    for name in set(agents):
        try:
            importlib.import_module(f"agent_code.{name}.callbacks")
        except Exception:
            pass


def _pool_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    _apply_agent_env(task["env"])
    records = play_rounds(task["agents"], task["scenario"], task["n_rounds"], task["seed"],
                          pathlib.Path(task["log_dir"]), task["silence_errors"], round_offset=task["round_offset"])
    for r in records:
        r["worker"] = task["worker_id"]
    return records


_POOL: Optional[concurrent.futures.ProcessPoolExecutor] = None
_POOL_SIZE = 0


def worker_pool(workers: int, agents: List[str]) -> concurrent.futures.ProcessPoolExecutor:
    global _POOL, _POOL_SIZE
    workers = max(1, min(int(workers), MAX_POOL_WORKERS))
    if _POOL is None or _POOL_SIZE != workers:
        shutdown_pool()
        _POOL = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_pool_init, initargs=(list(agents), AGENT_LOG_LEVEL),
        )
        _POOL_SIZE = workers
    return _POOL


def shutdown_pool() -> None:
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
        _POOL = None
        _POOL_SIZE = 0


atexit.register(shutdown_pool)


def _prepare_agent_log_dirs(agents: List[str]) -> None:
    seen: List[str] = []
    for code_name in agents:
        names = {code_name}
        if agents.count(code_name) > 1:
            names.add(f"{code_name}_{seen.count(code_name)}")
        seen.append(code_name)
        for name in names:
            (REPO_ROOT / "agent_code" / name / "logs").mkdir(parents=True, exist_ok=True)


def play_parallel(agents: List[str], scenario: str, n_rounds: int, seed: Optional[int],
                  workers: int, log_dir: pathlib.Path, silence_errors: bool) -> List[Dict[str, Any]]:
    workers = max(1, min(workers, n_rounds))
    base, remainder = divmod(n_rounds, workers)
    chunks = [base + (1 if i < remainder else 0) for i in range(workers)]
    offsets = [sum(chunks[:i]) for i in range(workers)]

    if workers == 1:
        bar = tqdm(total=n_rounds, desc="rounds")
        records = play_rounds(agents, scenario, n_rounds, seed, log_dir / "worker_0", silence_errors,
                              progress=lambda: bar.update(1))
        bar.close()
        for r in records:
            r["worker"] = 0
        return records

    _prepare_agent_log_dirs(agents)
    env_snapshot = _agent_env_snapshot()
    tasks = [
        dict(worker_id=worker_id, agents=list(agents), scenario=scenario, n_rounds=n, seed=seed, round_offset=offset,
             log_dir=str(log_dir / f"worker_{worker_id}"), silence_errors=silence_errors, env=env_snapshot)
        for worker_id, (n, offset) in enumerate(zip(chunks, offsets))
    ]
    pool = worker_pool(workers, agents)
    futures = {pool.submit(_pool_task, task): task["worker_id"] for task in tasks}
    collected: Dict[int, List[Dict[str, Any]]] = {}
    with tqdm(total=n_rounds, desc="rounds") as bar:
        try:
            for future in concurrent.futures.as_completed(futures):
                worker_id = futures[future]
                collected[worker_id] = future.result()
                bar.update(len(collected[worker_id]))
        except concurrent.futures.process.BrokenProcessPool as exc:
            shutdown_pool()
            raise RuntimeError(f"an evaluation worker died: {exc}") from exc

    records = [r for worker_id in sorted(collected) for r in collected[worker_id]]
    if len(records) != n_rounds:
        raise RuntimeError(f"expected {n_rounds} round records, got {len(records)}")
    return records


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _std(values: List[float]) -> Optional[float]:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _sem(values: List[float]) -> Optional[float]:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None


def _rate(flags: List[bool]) -> Optional[float]:
    return _mean([float(f) for f in flags]) if flags else None


def _rate_sem(flags: List[bool]) -> Optional[float]:
    if not flags:
        return None
    p = sum(flags) / len(flags)
    return math.sqrt(p * (1 - p) / len(flags))


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        for a in r["agents"]:
            by_agent.setdefault(a["name"], []).append(a)

    n_players = len(records[0]["agents"]) if records else 0
    summary = {}
    for name, rows in by_agent.items():
        scores = [row["score"] for row in rows]
        acted = sum(row["steps_acted"] for row in rows)
        timed = [row["think_time_total"] for row in rows if row["think_time_total"] is not None]
        metrics = {
            "code_name": rows[0]["code_name"],
            "n_rounds": len(rows),
            "score_mean": _mean(scores),
            "score_std": _std(scores),
            "score_sem": _sem(scores),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "win_rate": _rate([row["win"] for row in rows]) if n_players > 1 else None,
            "win_rate_sem": _rate_sem([row["win"] for row in rows]) if n_players > 1 else None,
            "tie_rate": _rate([row["tie"] for row in rows]) if n_players > 1 else None,
            "rank_mean": _mean([row["rank"] for row in rows]) if n_players > 1 else None,
            "survival_rate": _rate([row["alive"] for row in rows]),
            "survival_rate_sem": _rate_sem([row["alive"] for row in rows]),
            "suicide_rate": _rate([row["killed_self"] for row in rows]),
            "killed_by_opponent_rate": _rate([(not row["alive"]) and not row["killed_self"] for row in rows]),
            "steps_survived_mean": _mean([row["steps_survived"] for row in rows]),
            "round_length_mean": _mean([r["length"] for r in records]),
            "think_time_mean_ms": 1000 * sum(timed) / acted if acted and timed else None,
            "think_time_max_ms": 1000 * max(row["think_time_max"] for row in rows) if rows else None,
            "slow_calls_per_round": _mean([row["slow_calls"] for row in rows]),
            "forced_waits_per_round": _mean([row["forced_waits"] for row in rows]),
            "skipped_steps_per_round": _mean([row["skipped_steps"] for row in rows]),
            "steps_lost_per_round": _mean([row["forced_waits"] + row["skipped_steps"] for row in rows]),
            "error_rounds": sum(1 for row in rows if row["errors"]),
        }
        for key in EVENT_STATS:
            metrics[f"{key}_mean"] = _mean([row[key] for row in rows])
        summary[name] = metrics
    return summary


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _describe_agent(code_name: str) -> Dict[str, Any]:
    folder = REPO_ROOT / "agent_code" / code_name
    info: Dict[str, Any] = {}
    callbacks = folder / "callbacks.py"
    if callbacks.exists():
        source = callbacks.read_text(encoding="utf-8", errors="replace")
        for key in ("RUN", "CHECKPOINT", "DETERMINISTIC"):
            match = re.search(rf"^{key}\s*(?::[^=]+)?=\s*(.+?)\s*$", source, re.MULTILINE)
            if match:
                info[key] = match.group(1)
    models = sorted(p for p in folder.rglob("*") if p.suffix in (".zip", ".pt", ".pth", ".pkl") and p.is_file())
    if models:
        info["model_files"] = {}
        for p in models[:20]:
            digest = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            info["model_files"][str(p.relative_to(folder)).replace("\\", "/")] = {
                "bytes": p.stat().st_size,
                "sha256": digest.hexdigest()[:16],
            }
    return info


def run_matchup(spec: Dict[str, Any], n_rounds: int, seed: Optional[int], workers: int,
                silence_errors: bool, tag: Optional[str] = None, note: Optional[str] = None,
                overwrite: bool = False) -> pathlib.Path:
    agents, scenario = spec["agents"], spec["scenario"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = tag or f"{agents[0]}__{spec.get('name', 'custom')}__{timestamp}"
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / f"{tag}.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; choose another --tag or pass --overwrite")
    workers = max(1, min(workers, n_rounds))
    log_dir = LOG_DIR / tag

    print(f"[{tag}] {scenario}: {' vs '.join(agents)} -- {n_rounds} rounds, seed {seed}, {workers} worker(s)")
    records = play_parallel(agents, scenario, n_rounds, seed, workers, log_dir, silence_errors)

    result = {
        "meta": {
            "created": timestamp,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "workers": workers,
            "note": note,
            "agents": {name: _describe_agent(name) for name in dict.fromkeys(agents)},
        },
        "matchup": {
            "name": spec.get("name", "custom"),
            "description": spec.get("description", ""),
            "scenario": scenario,
            "agents": agents,
            "n_rounds": n_rounds,
            "seed": seed,
        },
        "settings": {
            "MAX_STEPS": s.MAX_STEPS,
            "TIMEOUT": s.TIMEOUT,
            "BOMB_TIMER": s.BOMB_TIMER,
            "BOMB_POWER": s.BOMB_POWER,
            "EXPLOSION_TIMER": s.EXPLOSION_TIMER,
            "REWARD_COIN": s.REWARD_COIN,
            "REWARD_KILL": s.REWARD_KILL,
            "scenario": s.SCENARIOS[scenario],
        },
        "summary": summarize_records(records),
        "rounds": records,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    from evaluation.summarize import format_table
    result["_file"] = out_path.name
    print(format_table([result], candidate_only=False))
    print(f"written {out_path.relative_to(REPO_ROOT)}")
    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--n-rounds", type=int, default=100)
        p.add_argument("--seed", type=int, default=0,
                       help="run seed; every round is seeded from it and its index (-1: unseeded)")
        p.add_argument("--workers", type=int, default=default_workers(),
                       help="processes to split the rounds over (default: cores - 1)")
        p.add_argument("--agent-log-level", default="WARNING", type=str.upper,
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="log level of the agents' code/wrapper loggers during evaluation (default WARNING)")
        p.add_argument("--silence-errors", action="store_true",
                       help="keep playing when an agent throws (default: abort, as the tournament would)")
        p.add_argument("--note", help="free text stored in the result file, e.g. which checkpoint was evaluated")
        p.add_argument("--overwrite", action="store_true", help="replace an existing result file of the same tag")

    p_run = sub.add_parser("run", help="play one matchup")
    group = p_run.add_mutually_exclusive_group(required=True)
    group.add_argument("--matchup", choices=sorted(MATCHUPS), help="predefined lineup (needs --agent)")
    group.add_argument("--agents", nargs="+", help="explicit lineup; the first agent is the candidate")
    p_run.add_argument("--agent", help="agent under evaluation for --matchup")
    p_run.add_argument("--scenario", default="classic", choices=sorted(s.SCENARIOS),
                       help="scenario for --agents lineups")
    p_run.add_argument("--tag", help="result file name (default: <agent>__<matchup>__<timestamp>)")
    add_common(p_run)

    p_suite = sub.add_parser("suite", help="play the whole task ladder for one agent")
    p_suite.add_argument("--agent", required=True)
    p_suite.add_argument("--matchups", nargs="+", default=SUITE, choices=sorted(MATCHUPS))
    p_suite.add_argument("--tag-prefix", help="result files are named <prefix>__<matchup> instead of "
                                              "<agent>__<matchup>__<timestamp>")
    add_common(p_suite)

    sub.add_parser("list", help="show the predefined matchups")

    args = parser.parse_args(argv)
    if args.command == "list":
        for name, spec in MATCHUPS.items():
            print(f"{name:24s} {spec['scenario']:12s} {' vs '.join(spec['agents']):70s} {spec['description']}")
        return

    seed = None if args.seed < 0 else args.seed
    global AGENT_LOG_LEVEL
    AGENT_LOG_LEVEL = logging.getLevelName(args.agent_log_level)
    if args.command == "run":
        if args.matchup:
            if not args.agent:
                parser.error("--matchup requires --agent")
            spec = resolve_matchup(args.matchup, args.agent)
        else:
            if args.agent:
                parser.error("--agent is only used with --matchup; with --agents the first agent is the candidate")
            spec = {"name": "custom", "scenario": args.scenario, "agents": args.agents,
                    "description": "explicit lineup"}
        run_matchup(spec, args.n_rounds, seed, args.workers, args.silence_errors, args.tag, args.note,
                    args.overwrite)
    elif args.command == "suite":
        paths = []
        for name in args.matchups:
            tag = f"{args.tag_prefix}__{name}" if args.tag_prefix else None
            paths.append(run_matchup(resolve_matchup(name, args.agent), args.n_rounds, seed, args.workers,
                                     args.silence_errors, tag, args.note, args.overwrite))
        from evaluation.summarize import format_table, load_results
        print("\n== suite summary (candidate agent only) ==")
        print(format_table(load_results(paths), candidate_only=True))


if __name__ == "__main__":
    mp.freeze_support()
    main()
