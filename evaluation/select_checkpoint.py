from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import (
    LOG_DIR, MATCHUPS, STATS_DIR, _git_commit, default_workers, play_parallel, resolve_matchup, shutdown_pool,
    summarize_records,
)

VALIDATION_MATCHUPS = ["task2-classic-solo", "task3-hunt", "task4-rule-based-x3"]
CRITERIA = {"score": ("score_mean", "score_sem"), "win": ("win_rate", "win_rate_sem"),
            "survival": ("survival_rate", "survival_rate_sem")}
BEST_FILE = "best.txt"


def relative_path(path: pathlib.Path) -> str:
    path = pathlib.Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def find_run_dir(agent: str, run: str) -> pathlib.Path:
    agent_dir = REPO_ROOT / "agent_code" / agent
    candidates = [agent_dir / "runs" / run, agent_dir / run, pathlib.Path(run), REPO_ROOT / "runs" / run]
    for run_dir in candidates:
        if (run_dir / "checkpoints").is_dir():
            return run_dir
    raise FileNotFoundError(f"no run '{run}' with a checkpoints folder in {[str(c) for c in candidates]}")


def list_checkpoints(run_dir: pathlib.Path) -> List[pathlib.Path]:
    found = [p for p in (run_dir / "checkpoints").glob("checkpoint_*") if (p / "model.zip").exists()]
    return sorted(found, key=lambda p: int(p.name.split("_")[-1]))


def select_candidates(checkpoints: List[pathlib.Path], spec: List[str]) -> List[pathlib.Path]:
    if len(spec) == 1 and spec[0] == "all":
        return checkpoints
    if len(spec) == 1 and spec[0].startswith("last:"):
        return checkpoints[-int(spec[0][5:]):]
    if len(spec) == 1 and spec[0].startswith("every:"):
        step = int(spec[0][6:])
        chosen = checkpoints[::-1][::step][::-1]
        return chosen
    by_name = {p.name: p for p in checkpoints}
    missing = [name for name in spec if name not in by_name]
    if missing:
        raise FileNotFoundError(f"unknown checkpoints {missing}; available: {[p.name for p in checkpoints]}")
    return [by_name[name] for name in spec]


def read_metadata(checkpoint: pathlib.Path) -> Dict[str, Any]:
    path = checkpoint / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def timesteps_of(checkpoint: pathlib.Path) -> int:
    return int(checkpoint.name.split("_")[-1])


def _fmt(value: Optional[float], digits: int = 2, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100 * value:.0f}%" if percent else f"{value:.{digits}f}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def metadata_scores(meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    suite = meta.get("eval_suite") or {}
    out: Dict[str, Optional[float]] = {}
    for case, value in suite.items():
        name = case.split(".")[1] if case.startswith("agent_code.") else case
        out[name] = float(value["score"]) if isinstance(value, dict) and value.get("score") is not None else None
    return out


def rank_by_metadata(candidates: List[pathlib.Path]) -> Dict[str, Any]:
    rows, cases = [], []
    for ckpt in candidates:
        meta = read_metadata(ckpt)
        scores = metadata_scores(meta)
        for case in scores:
            if case not in cases:
                cases.append(case)
        valid = [v for v in scores.values() if v is not None]
        rows.append(dict(checkpoint=ckpt.name, timesteps=timesteps_of(ckpt), train_reward=meta.get("ep_rew_mean"),
                         cases=scores, overall=float(np.mean(valid)) if valid else None))
    scored = [r for r in rows if r["overall"] is not None]
    best = max(scored, key=lambda r: (r["overall"], r["timesteps"])) if scored else None
    return dict(cases=cases, rows=rows, best=best["checkpoint"] if best else None)


def print_metadata_table(result: Dict[str, Any]) -> None:
    headers = ["checkpoint", "steps", "train reward", *result["cases"], "overall"]
    rows = []
    for r in result["rows"]:
        mark = " *" if r["checkpoint"] == result["best"] else ""
        rows.append([r["checkpoint"] + mark, str(r["timesteps"]), _fmt(r["train_reward"]),
                     *[_fmt(r["cases"].get(c)) for c in result["cases"]], _fmt(r["overall"])])
    print(_table(headers, rows))


def play_run(agent: str, prefix: str, run: str, checkpoint: Optional[str], label: str, matchups: List[str],
             n_rounds: int, seed: int, workers: int, silence_errors: bool, log_dir: pathlib.Path,
             ensemble: Optional[List[str]] = None) -> Dict[str, Any]:
    os.environ[f"{prefix}_RUN"] = run
    if checkpoint is None:
        os.environ.pop(f"{prefix}_CHECKPOINT", None)
    else:
        os.environ[f"{prefix}_CHECKPOINT"] = checkpoint
    if ensemble:
        os.environ[f"{prefix}_ENSEMBLE"] = ",".join(ensemble)
    else:
        os.environ.pop(f"{prefix}_ENSEMBLE", None)
    out: Dict[str, Any] = {}
    for name in matchups:
        spec = resolve_matchup(name, agent)
        print(f"[{label}] {name}: {' vs '.join(spec['agents'])} -- {n_rounds} rounds, seed {seed}")
        records = play_parallel(spec["agents"], spec["scenario"], n_rounds, seed, workers,
                                log_dir / label / name, silence_errors)
        records.sort(key=lambda r: r["round"])
        summary = summarize_records(records)[agent]
        rounds = [next(a for a in r["agents"] if a["name"] == agent) for r in records]
        out[name] = dict(summary=summary, score=[float(a["score"]) for a in rounds],
                         win=[float(a["win"]) for a in rounds], alive=[float(a["alive"]) for a in rounds])
    return out


def play_candidate(agent: str, prefix: str, run: str, checkpoint: pathlib.Path, matchups: List[str],
                   n_rounds: int, seed: int, workers: int, silence_errors: bool, log_dir: pathlib.Path) -> Dict[str, Any]:
    return play_run(agent, prefix, run, checkpoint.name, checkpoint.name, matchups, n_rounds, seed, workers,
                    silence_errors, log_dir)


def score_run(results: Dict[str, Any], criterion: str) -> float:
    metric, _ = CRITERIA[criterion]
    return float(np.mean([results[m]["summary"][metric] for m in results]))


def rank_by_play(results: Dict[str, Dict[str, Any]], matchups: List[str], criterion: str) -> Dict[str, Any]:
    key = {"score": "score", "win": "win", "survival": "alive"}[criterion]
    per_round = {name: np.mean([np.asarray(res[m][key]) for m in matchups], axis=0) for name, res in results.items()}
    overall = {name: float(v.mean()) for name, v in per_round.items()}
    best = max(overall, key=lambda name: (overall[name], not name.startswith("ensemble("), name))
    rows = []
    for name, res in results.items():
        diff = per_round[name] - per_round[best]
        sem = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else None
        rows.append(dict(checkpoint=name, overall=overall[name], delta_vs_best=float(diff.mean()), delta_sem=sem,
                         matchups={m: res[m]["summary"] for m in matchups}))
    return dict(criterion=criterion, best=best, rows=rows)


def print_play_table(ranking: Dict[str, Any], matchups: List[str], metadata: Dict[str, Dict[str, Any]]) -> None:
    metric, sem_key = CRITERIA[ranking["criterion"]]
    percent = metric != "score_mean"
    headers = ["checkpoint", "steps", "train reward", *matchups, "overall", "win", "surv", "delta vs best"]
    rows = []
    for r in ranking["rows"]:
        mark = " *" if r["checkpoint"] == ranking["best"] else ""
        cells = []
        for m in matchups:
            s = r["matchups"][m]
            cells.append(_fmt(s[metric], percent=percent) + (f"+-{_fmt(s[sem_key], percent=percent)}" if s.get(sem_key) is not None else ""))
        win = np.mean([r["matchups"][m]["win_rate"] for m in matchups if r["matchups"][m]["win_rate"] is not None] or [np.nan])
        surv = np.mean([r["matchups"][m]["survival_rate"] for m in matchups])
        delta = _fmt(r["delta_vs_best"], percent=percent) + (f"+-{_fmt(r['delta_sem'], percent=percent)}" if r["delta_sem"] is not None else "")
        meta = metadata.get(r["checkpoint"], {})
        rows.append([r["checkpoint"] + mark, str(meta.get("timesteps", "")), _fmt(meta.get("ep_rew_mean")), *cells,
                     _fmt(r["overall"], percent=percent), _fmt(float(win), percent=True), _fmt(float(surv), percent=True), delta])
    print(_table(headers, rows))


def write_best(run_dir: pathlib.Path, checkpoint: str) -> pathlib.Path:
    pointer = run_dir / "checkpoints" / BEST_FILE
    pointer.write_text(checkpoint)
    return pointer


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Pick a run's checkpoint by validation score instead of 'latest'")
    p.add_argument("--agent", required=True, help="agent folder, e.g. ppo_agent")
    p.add_argument("--run", required=True, help="run name (or path) whose checkpoints are compared")
    p.add_argument("--checkpoints", nargs="+", default=["last:10"],
                   help="all | last:N | every:K | explicit checkpoint names (default last:10)")
    p.add_argument("--metadata", action="store_true",
                   help="rank by the evaluation stored in each checkpoint's metadata.json instead of playing")
    p.add_argument("--matchups", nargs="+", default=VALIDATION_MATCHUPS, choices=sorted(MATCHUPS))
    p.add_argument("--criterion", choices=sorted(CRITERIA), default="score")
    p.add_argument("--n-rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="same seed = same boards for every checkpoint")
    p.add_argument("--workers", type=int, default=default_workers(), help="evaluation processes (default: cores - 1)")
    p.add_argument("--silence-errors", action="store_true")
    p.add_argument("--env-prefix", help="environment-variable prefix the agent's callbacks read (default AGENT upper-cased)")
    p.add_argument("--ensemble", action="store_true",
                   help="also play the ensemble that averages all candidates (needs play mode)")
    p.add_argument("--mc-dropout", type=int, default=0,
                   help="dropout samples averaged per prediction for every candidate (dqn_agent with dropout > 0)")
    p.add_argument("--write-best", action="store_true", help="write the winner to <run>/checkpoints/best.txt")
    p.add_argument("--tag", help="result file name (default select__<agent>__<run>__<timestamp>)")
    args = p.parse_args(argv)

    run_dir = find_run_dir(args.agent, args.run)
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints with a model.zip in {run_dir / 'checkpoints'}")
    candidates = select_candidates(checkpoints, args.checkpoints)
    metadata = {c.name: read_metadata(c) for c in candidates}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"select__{args.agent}__{run_dir.name}__{timestamp}"
    print(f"[{tag}] {args.agent} run {run_dir.name}: {len(candidates)} of {len(checkpoints)} checkpoints")

    if args.metadata:
        ranking = rank_by_metadata(candidates)
        print_metadata_table(ranking)
        best = ranking["best"]
    else:
        prefix = args.env_prefix or args.agent.upper()
        if args.mc_dropout > 0:
            os.environ[f"{prefix}_MC_DROPOUT_SAMPLES"] = str(args.mc_dropout)
        results = {}
        for ckpt in candidates:
            results[ckpt.name] = play_candidate(args.agent, prefix, str(run_dir), ckpt,
                                                args.matchups, args.n_rounds, args.seed, args.workers,
                                                args.silence_errors, LOG_DIR / tag)
        if args.ensemble and len(candidates) > 1:
            label = f"ensemble({len(candidates)})"
            results[label] = play_run(args.agent, prefix, str(run_dir), None, label, args.matchups, args.n_rounds,
                                      args.seed, args.workers, args.silence_errors, LOG_DIR / tag,
                                      ensemble=[c.name for c in candidates])
        for name in ("RUN", "CHECKPOINT", "ENSEMBLE", "MC_DROPOUT_SAMPLES"):
            os.environ.pop(f"{prefix}_{name}", None)
        ranking = rank_by_play(results, args.matchups, args.criterion)
        print_play_table(ranking, args.matchups, metadata)
        best = ranking["best"]

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / f"{tag}.json"
    out_path.write_text(json.dumps(dict(
        meta=dict(created=timestamp, git_commit=_git_commit(), agent=args.agent,
                  run=relative_path(run_dir), mode="metadata" if args.metadata else "play", matchups=args.matchups,
                  n_rounds=args.n_rounds, seed=args.seed, criterion=args.criterion),
        candidates={c.name: dict(timesteps=timesteps_of(c), ep_rew_mean=metadata[c.name].get("ep_rew_mean"),
                                 eval_suite=metadata[c.name].get("eval_suite")) for c in candidates},
        ranking=ranking,
    ), indent=2, default=float))
    print(f"written {out_path.relative_to(REPO_ROOT)}")

    if best is None:
        print("no candidate has a validation score")
        return
    print(f"best checkpoint by {args.criterion}: {best}")
    if args.write_best and best.startswith("ensemble("):
        print(f"the ensemble wins; set ENSEMBLE = {[c.name for c in candidates]} in {args.agent}/callbacks.py instead of best.txt")
    elif args.write_best:
        pointer = write_best(run_dir, best)
        print(f"wrote {pointer} -- set CHECKPOINT = \"best\" in {args.agent}/callbacks.py to use it")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_pool()
