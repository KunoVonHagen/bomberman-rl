from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import LOG_DIR, STATS_DIR, _git_commit
from evaluation.select_checkpoint import CRITERIA, list_checkpoints, play_run, relative_path, score_run

SPACES_DIR = pathlib.Path(__file__).resolve().parent / "search_spaces"
STRATEGIES = ("grid", "random", "gp")


class GaussianProcess:
    def __init__(self, bandwidth: float = 1.0, noise: float = 0.1):
        self.bandwidth = float(bandwidth)
        self.noise = float(noise)
        self.x = np.zeros((0, 0))
        self.y = np.zeros(0)
        self.y_mean = 0.0
        self.y_std = 1.0
        self._alpha = np.zeros(0)
        self._inverse = np.zeros((0, 0))

    def kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-d2 / (2.0 * self.bandwidth ** 2))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.x = np.asarray(x, dtype=np.float64).reshape(len(y), -1)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if len(y) > 1 and y.std() > 0 else 1.0
        self.y = (y - self.y_mean) / self.y_std
        gram = self.kernel(self.x, self.x) + self.noise * np.eye(len(y))
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        self._inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
        self._alpha = self._inverse @ self.y
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.x.shape[1])
        g = self.kernel(self.x, x)
        mean = g.T @ self._alpha
        variance = 1.0 + self.noise - np.einsum("ij,ik,kj->j", g, self._inverse, g)
        std = np.sqrt(np.maximum(variance, 0.0))
        return mean * self.y_std + self.y_mean, std * self.y_std


class SearchSpace:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.name: str = spec["name"]
        self.agent: str = spec["agent"]
        self.prefix: str = spec.get("env_prefix", self.agent.upper())
        self.train: List[str] = list(spec["train"])
        self.style: str = spec.get("override_style", "set")
        self.run_name_key: str = spec.get("run_name_key", "run_name" if self.style == "set" else "--run-name")
        self.fixed: Dict[str, Any] = dict(spec.get("fixed", {}))
        self.parameters: List[Dict[str, Any]] = list(spec["parameters"])
        self.validation: Dict[str, Any] = dict(spec.get("validation", {}))
        self.checkpoint: Optional[str] = spec.get("checkpoint", "latest")
        for p in self.parameters:
            if not p.get("values"):
                raise ValueError(f"parameter {p.get('name')} needs a non-empty candidate list 'values'")

    @property
    def names(self) -> List[str]:
        return [p["name"] for p in self.parameters]

    def grid(self) -> List[Dict[str, Any]]:
        return [dict(zip(self.names, combo)) for combo in itertools.product(*(p["values"] for p in self.parameters))]

    def encode(self, points: Sequence[Dict[str, Any]]) -> np.ndarray:
        columns = []
        for p in self.parameters:
            values = p["values"]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                raw = np.array([float(pt[p["name"]]) for pt in points])
                scale = np.array([float(v) for v in values])
                if p.get("log", False):
                    raw, scale = np.log10(raw), np.log10(scale)
            else:
                index = {json.dumps(v): i for i, v in enumerate(values)}
                raw = np.array([index[json.dumps(pt[p["name"]])] for pt in points], dtype=np.float64)
                scale = np.arange(len(values), dtype=np.float64)
            spread = scale.std() if len(values) > 1 and scale.std() > 0 else 1.0
            columns.append((raw - scale.mean()) / spread)
        return np.stack(columns, axis=1)

    def command(self, params: Dict[str, Any], run_name: str) -> List[str]:
        default = (lambda n: n) if self.style == "set" else (lambda n: "--" + n.replace("_", "-"))
        args = {p["name"]: p.get("arg", default(p["name"])) for p in self.parameters}
        overrides = {**self.fixed, **{args[k]: v for k, v in params.items()}}
        cmd = list(self.train)
        if cmd and cmd[0] == "python":
            cmd[0] = sys.executable
        if self.style == "set":
            cmd += ["--set", f"{self.run_name_key}={run_name}"] + [f"{k}={_cli_value(v)}" for k, v in overrides.items()]
        else:
            cmd += [self.run_name_key, run_name]
            for k, v in overrides.items():
                cmd += [k, _cli_value(v)]
        return cmd


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def trial_key(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def train_trial(space: SearchSpace, params: Dict[str, Any], run_name: str, log_path: pathlib.Path) -> Dict[str, Any]:
    cmd = space.command(params, run_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    return dict(command=["python"] + cmd[1:], returncode=result.returncode, train_s=round(time.time() - t0, 1),
                log=relative_path(log_path))


def find_run_dir(agent: str, run_name: str) -> pathlib.Path:
    candidates = [REPO_ROOT / "runs" / run_name, REPO_ROOT / "agent_code" / agent / "runs" / run_name]
    for run_dir in candidates:
        if run_dir.is_dir():
            return run_dir
    raise FileNotFoundError(f"training did not produce a run folder for {run_name} in {[str(c) for c in candidates]}")


def validate_trial(space: SearchSpace, run_name: str, label: str, log_dir: pathlib.Path) -> Dict[str, Any]:
    v = space.validation
    run_dir = find_run_dir(space.agent, run_name)
    checkpoint = None
    if (run_dir / "checkpoints").is_dir():
        checkpoint = space.checkpoint
        if checkpoint == "latest" and list_checkpoints(run_dir):
            checkpoint = list_checkpoints(run_dir)[-1].name
    results = play_run(space.agent, space.prefix, str(run_dir), checkpoint, label, v.get("matchups", ["task4-rule-based-x3"]),
                       int(v.get("n_rounds", 20)), int(v.get("seed", 0)), int(v.get("workers", 1)),
                       bool(v.get("silence_errors", True)), log_dir)
    criterion = v.get("criterion", "score")
    return dict(run_dir=relative_path(run_dir), checkpoint=checkpoint, criterion=criterion,
                score=score_run(results, criterion),
                matchups={m: r["summary"] for m, r in results.items()})


def choose_random(candidates: List[Dict[str, Any]], n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    return [candidates[i] for i in sorted(idx)]


def choose_by_gp(space: SearchSpace, done: List[Dict[str, Any]], candidates: List[Dict[str, Any]],
                 bandwidth: float, noise: float, kappa: float) -> tuple[Dict[str, Any], Dict[str, Any]]:
    gp = GaussianProcess(bandwidth, noise).fit(space.encode([t["params"] for t in done]), [t["score"] for t in done])
    mean, std = gp.predict(space.encode(candidates))
    acquisition = mean + kappa * std
    best = int(np.argmax(acquisition))
    return candidates[best], dict(mean=float(mean[best]), std=float(std[best]), acquisition=float(acquisition[best]))


def surrogate_table(space: SearchSpace, done: List[Dict[str, Any]], bandwidth: float, noise: float) -> List[Dict[str, Any]]:
    if len(done) < 2:
        return []
    grid = space.grid()
    gp = GaussianProcess(bandwidth, noise).fit(space.encode([t["params"] for t in done]), [t["score"] for t in done])
    mean, std = gp.predict(space.encode(grid))
    tried = {trial_key(t["params"]): t["score"] for t in done}
    rows = [dict(params=g, mean=float(m), std=float(s), observed=tried.get(trial_key(g))) for g, m, s in zip(grid, mean, std)]
    rows.sort(key=lambda r: -r["mean"])
    return rows


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_trials(space: SearchSpace, trials: List[Dict[str, Any]]) -> None:
    headers = ["trial", *space.names, "score", "train s", "run"]
    rows = []
    for t in sorted(trials, key=lambda t: -(t["score"] if t["score"] is not None else -math.inf)):
        rows.append([str(t["id"]), *[str(t["params"][n]) for n in space.names], _fmt(t["score"]),
                     str(t.get("train_s", "")), t["run_name"]])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")


def load_space(path: str) -> SearchSpace:
    p = pathlib.Path(path)
    if not p.exists() and (SPACES_DIR / f"{path}.json").exists():
        p = SPACES_DIR / f"{path}.json"
    return SearchSpace(json.loads(p.read_text(encoding="utf-8")))


def run_search(space: SearchSpace, strategy: str, max_trials: Optional[int], initial: Optional[int], parallel: int,
               seed: int, bandwidth: float, noise: float, kappa: float, delete_runs: bool, tag: str) -> Dict[str, Any]:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / f"{tag}.json"
    state = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else dict(trials=[])
    trials: List[Dict[str, Any]] = state["trials"]
    tried = {trial_key(t["params"]) for t in trials}
    grid = space.grid()
    rng = np.random.default_rng(seed)
    budget = len(grid) if max_trials is None else min(max_trials, len(grid))
    log_dir = LOG_DIR / tag
    print(f"[{tag}] {space.agent}: {len(grid)} grid points, {len(trials)} done, strategy {strategy}, budget {budget}")

    def save() -> None:
        done = [t for t in trials if t["score"] is not None]
        best = max(done, key=lambda t: t["score"]) if done else None
        state.update(dict(
            meta=dict(created=state.get("meta", {}).get("created", datetime.now().strftime("%Y%m%d-%H%M%S")),
                      updated=datetime.now().strftime("%Y%m%d-%H%M%S"), git_commit=_git_commit(),
                      strategy=strategy, seed=seed, bandwidth=bandwidth, noise=noise, kappa=kappa),
            space=space.spec, trials=trials, best=best,
            surrogate=surrogate_table(space, done, bandwidth, noise) if strategy == "gp" else [],
        ))
        out_path.write_text(json.dumps(state, indent=2, default=float), encoding="utf-8")

    def run_one(params: Dict[str, Any], note: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        trial_id = len(trials) + 1
        run_name = f"tune_{space.name}_{trial_id:03d}"
        trial = dict(id=trial_id, params=params, run_name=run_name, score=None, proposal=note)
        trials.append(trial)
        tried.add(trial_key(params))
        return trial

    def finish(trial: Dict[str, Any], training: Dict[str, Any]) -> None:
        trial.update(training)
        if training["returncode"] != 0:
            print(f"trial {trial['id']} failed (exit {training['returncode']}), see {training['log']}")
            save()
            return
        validation = validate_trial(space, trial["run_name"], trial["run_name"], log_dir)
        trial.update(validation)
        print(f"trial {trial['id']}: {trial['params']} -> {validation['criterion']} {validation['score']:.3f} "
              f"(train {training['train_s']}s)")
        if delete_runs:
            shutil.rmtree(REPO_ROOT / validation["run_dir"], ignore_errors=True)
        save()

    remaining = [g for g in grid if trial_key(g) not in tried]
    if strategy in ("grid", "random"):
        chosen = remaining if strategy == "grid" else choose_random(remaining, budget - len(trials), rng)
        chosen = chosen[:max(0, budget - len(trials))]
        pending = [run_one(p) for p in chosen]
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futures = {pool.submit(train_trial, space, t["params"], t["run_name"], log_dir / f"{t['run_name']}.log"): t
                       for t in pending}
            for future in as_completed(futures):
                finish(futures[future], future.result())
    else:
        n_initial = initial if initial is not None else max(3, len(space.names) + 1)
        while len(trials) < budget and remaining:
            done = [t for t in trials if t["score"] is not None]
            if len(done) < n_initial:
                params, note = choose_random(remaining, 1, rng)[0], None
            else:
                params, note = choose_by_gp(space, done, remaining, bandwidth, noise, kappa)
                print(f"surrogate proposes {params} (mean {note['mean']:.3f}, std {note['std']:.3f})")
            trial = run_one(params, note)
            finish(trial, train_trial(space, params, trial["run_name"], log_dir / f"{trial['run_name']}.log"))
            remaining = [g for g in grid if trial_key(g) not in tried]

    save()
    print_trials(space, trials)
    if state.get("best"):
        print(f"best: {state['best']['params']} with {state['best'].get('criterion', 'score')} {state['best']['score']:.3f}")
    if strategy == "gp" and state.get("surrogate"):
        top = state["surrogate"][0]
        print(f"surrogate optimum: {top['params']} predicted {top['mean']:.3f} +- {top['std']:.3f}"
              + (" (observed)" if top["observed"] is not None else " (untried)"))
    print(f"written {out_path.relative_to(REPO_ROOT)}")
    return state


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Hyperparameter search on short training runs scored on the validation matchups")
    p.add_argument("--space", required=True, help="search-space JSON (path or name in evaluation/search_spaces)")
    p.add_argument("--strategy", choices=STRATEGIES, default="grid")
    p.add_argument("--max-trials", type=int, default=None, help="stop after this many trials (default: the whole grid)")
    p.add_argument("--initial", type=int, default=None, help="gp: random trials before the surrogate takes over")
    p.add_argument("--parallel", type=int, default=1, help="grid/random: training runs started concurrently")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bandwidth", type=float, default=1.0, help="gp: kernel bandwidth in standardised parameter units")
    p.add_argument("--noise", type=float, default=0.1, help="gp: noise variance added to the kernel matrix")
    p.add_argument("--kappa", type=float, default=1.0, help="gp: weight of the error bar when picking the next candidate")
    p.add_argument("--delete-runs", action="store_true", help="remove each trial's run folder after validation")
    p.add_argument("--tag", help="result file name (default tune__<space name>); an existing file is resumed")
    args = p.parse_args(argv)

    space = load_space(args.space)
    run_search(space, args.strategy, args.max_trials, args.initial, args.parallel, args.seed, args.bandwidth,
               args.noise, args.kappa, args.delete_runs, args.tag or f"tune__{space.name}")


if __name__ == "__main__":
    main()
