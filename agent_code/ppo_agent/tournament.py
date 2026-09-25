from __future__ import annotations

import argparse
import copy
import csv
import fnmatch
import json
import math
import multiprocessing as mp
import os
import pathlib
import random
import re
import time
import traceback
import warnings
from collections import OrderedDict, defaultdict
from multiprocessing import freeze_support

import numpy as np
import torch
from sb3_contrib.common.maskable.utils import get_action_masks

from . import callbacks as agent
from .opponent_pool import OpponentPool
from .train import make_test_env

N_PLAYERS = 4

warnings.filterwarnings("ignore", message="Could not deserialize object", category=UserWarning)

class Participant:
    def __init__(self, pid: str, run_dir: pathlib.Path, ckpt_dir: pathlib.Path, tta: int, deterministic: bool, seed: int):
        self.pid = pid
        self.cfg = agent._load_config(run_dir)
        self.policy = agent._load_model(ckpt_dir, self.cfg).policy
        self.policy.requires_grad_(False)
        self.tta = max(1, min(agent.N_SYMMETRIES, tta))
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self._obs_env = None
        self.errors = 0

    @property
    def obs_env(self):
        if self._obs_env is None:
            self._obs_env = agent._get_dummy_env(self.cfg)
        return self._obs_env

    def decide(self, obs: dict, masks) -> int:
        """obs: single (un-batched) observation as returned by observation_from_game_state, masks: (1, A).
        Mirrors callbacks._choose for a single model."""
        masks = np.asarray(masks)
        if masks.ndim == 1:
            masks = masks[None]
        mask_row = masks[0].astype(bool)
        if self.tta <= 1:
            action, _ = self.policy.predict(obs, deterministic=self.deterministic, action_masks=masks)
            return int(np.asarray(action).reshape(-1)[0])
        batch, tta_masks = agent._symmetry_batch(obs, mask_row, self.tta)
        with torch.no_grad():
            obs_t, _ = self.policy.obs_to_tensor(batch)
            probs = self.policy.get_distribution(obs_t, action_masks=tta_masks).distribution.probs.cpu().numpy()
        mean = np.where(mask_row, agent._average_symmetries(probs), 0.0)
        if self.deterministic:
            return int(np.argmax(mean))
        return int(self.rng.choice(len(mean), p=mean / mean.sum()))

    def as_opponent(self):
        names = agent._ACTION_NAMES

        def setup(handle):
            return None

        def act(handle, game_state):
            try:
                env = self.obs_env
                obs = env.observation_from_game_state(game_state)
                return names[self.decide(obs, env.action_masks())]
            except Exception:
                self.errors += 1
                if self.errors <= 3:
                    print(f"[{self.pid}] act failed, returning WAIT\n{traceback.format_exc()}", flush=True)
                return "WAIT"

        return setup, act


def run_game(learner: Participant, opponents: list, scenario: str, seed: int, logs_dir: str) -> dict:
    """One game: `learner` sits in the env's learner seat, `opponents` (3 (setup, act) pairs) in the other seats."""
    cfg = copy.deepcopy(learner.cfg)
    cfg.env.scenario = scenario
    cfg.env.seed = int(seed)
    cfg.env.save_stats = False
    cfg.env.make_video = False
    cfg.env.no_gui = True
    cfg.env.replay = None
    env = make_test_env(cfg, opponents, logs_dir, None)
    try:
        obs = env.reset()
        done = False
        while not done:
            masks = get_action_masks(env)
            single = {k: v[0] for k, v in obs.items()}
            action = learner.decide(single, masks)
            obs, _, dones, info = env.step(np.array([action]))
            done = bool(dones[0])
    finally:
        env.close()
    final = info[0]
    return dict(score=float(final["score"]), opp_scores=[float(x) for x in final.get("opponent_scores", [])],
                alive=float(final["alive"]), steps=float(final["step"]))


_W: dict = {}


def _worker_init(paths: dict, opts: dict) -> None:
    torch.set_num_threads(1)
    _W.update(paths=paths, opts=opts, cache=OrderedDict())


def _get_participant(pid: str) -> Participant:
    cache = _W["cache"]
    if pid in cache:
        cache.move_to_end(pid)
        return cache[pid]
    run_dir, ckpt_dir = _W["paths"][pid]
    o = _W["opts"]
    cache[pid] = Participant(pid, pathlib.Path(run_dir), pathlib.Path(ckpt_dir), o["tta"], o["deterministic"], o["seed"])
    while len(cache) > o["cache_size"]:
        cache.popitem(last=False)
    return cache[pid]


def play_match(job: dict) -> dict:
    try:
        o = _W["opts"]
        rng = random.Random(job["seed"])
        learner = _get_participant(job["learner"])
        seats = [p for p in job["candidates"] if p != job["learner"]] + list(job["fillers"])
        rng.shuffle(seats)
        pairs = [OpponentPool._resolve_static(s[len("filler:"):]) if s.startswith("filler:")
                 else _get_participant(s).as_opponent() for s in seats]
        result = run_game(learner, pairs, o["scenario"], job["seed"], o["logs_dir"])
        scores = {job["learner"]: result["score"]}
        if o["mode"] == "all":
            if len(result["opp_scores"]) != len(seats):
                raise RuntimeError(f"expected {len(seats)} opponent scores, got {result['opp_scores']}")
            for seat, sc in zip(seats, result["opp_scores"]):
                if not seat.startswith("filler:"):
                    scores[seat] = sc
        return dict(id=job["id"], stage=job["stage"], learner=job["learner"], seats=seats, scores=scores,
                    alive=result["alive"], steps=result["steps"])
    except Exception:
        return dict(id=job["id"], error=traceback.format_exc())


def _steps(path: pathlib.Path) -> int:
    m = re.search(r"(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def _stored_score(ckpt: pathlib.Path):
    try:
        meta = json.loads((ckpt / "metadata.json").read_text())
        return meta["tournament_eval"]["overall"]["score"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _walk_runs(root: pathlib.Path):
    """Find every run directory (one containing run_manifest.json) below `root`, at any depth."""
    runs, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        d = pathlib.Path(dirpath)
        if "run_manifest.json" in filenames:
            runs.append(d)
            dirnames[:] = []
        else:
            if "checkpoints" in dirnames:
                skipped.append(d)
            dirnames[:] = [n for n in dirnames if n not in ("checkpoints", "replays", "logs", "tensorboard")]
    return sorted(runs), skipped


def discover(args) -> dict:
    """pid ('<run path relative to --runs-dir>/checkpoint_xxx') -> (run_dir, checkpoint_dir)"""
    root = pathlib.Path(args.runs_dir)
    found = {}
    if args.candidates:
        for line in pathlib.Path(args.candidates).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            run, _, ckpt = line.rpartition("/")
            found[line] = (root / run, root / run / "checkpoints" / ckpt)
    else:
        runs, skipped = _walk_runs(root)
        for d in skipped:
            print(f"WARNING: {d} has a checkpoints/ folder but no run_manifest.json -> skipped", flush=True)
        print(f"{'run':<50} {'found':>6} {'used':>5}", flush=True)
        for run_dir in runs:
            name = run_dir.relative_to(root).as_posix() if run_dir != root else root.name
            if args.runs and not any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(run_dir.name, pat) for pat in args.runs):
                continue
            ckpts = sorted((c for c in (run_dir / "checkpoints").glob("checkpoint_*") if (c / "model.zip").exists()),
                           key=_steps)
            n_found = len(ckpts)
            ckpts = [c for c in ckpts if _steps(c) >= args.min_steps]
            ckpts = ckpts[::-1][::max(1, args.every)][::-1]
            if args.prefilter:
                ranked = sorted(ckpts, key=lambda c: (_stored_score(c) is not None, _stored_score(c) or 0.0), reverse=True)
                ckpts = sorted(ranked[:args.prefilter], key=_steps)
            if args.last:
                ckpts = ckpts[-args.last:]
            print(f"{name:<50} {n_found:>6} {len(ckpts):>5}", flush=True)
            for c in ckpts:
                found[f"{name}/{c.name}"] = (run_dir, c)
    if not found:
        raise SystemExit(f"no candidates found under '{root}'")
    missing = [p for p, (_, c) in found.items() if not (c / "model.zip").exists()]
    if missing:
        raise FileNotFoundError(f"no model.zip for: {missing[:5]}")
    return found


def build_schedule(active: list, stage: int, rounds: int, filler_prob: float, fillers: list, mode: str, seed: int) -> list:
    """
    Deterministic for (active, stage, seed) so a resumed run regenerates identical match ids.
    Candidates are shuffled into groups of 4 (`rounds` times). For every game each non-learner seat of the group is
    independently swapped for a random filler bot with probability `filler_prob` (the learner is always a candidate).
    """
    k = N_PLAYERS
    if len(active) < k:
        raise ValueError(f"need at least {k} candidates, have {len(active)}")
    rng = random.Random(f"{seed}-{stage}")
    matches = []
    for _ in range(rounds):
        order = list(active)
        rng.shuffle(order)
        groups = [order[i:i + k] for i in range(0, len(order), k)]
        if len(groups[-1]) < k:
            groups[-1] += rng.sample([p for p in active if p not in groups[-1]], k - len(groups[-1]))
        for group in groups:
            learners = group if mode == "learner" else [rng.choice(group)]
            for learner in learners:
                kept, filler_seats = [learner], []
                for other in (p for p in group if p != learner):
                    if fillers and rng.random() < filler_prob:
                        filler_seats.append(f"filler:{rng.choice(fillers)}")
                    else:
                        kept.append(other)
                matches.append(dict(id=f"s{stage}-{len(matches):06d}", stage=stage, candidates=kept,
                                    fillers=filler_seats, learner=learner, seed=rng.randrange(2 ** 31)))
    return matches


def standings(records: list) -> list:
    pts = defaultdict(list)
    for r in records:
        for pid, sc in r["scores"].items():
            pts[pid].append(sc)
    rows = []
    for pid, xs in pts.items():
        n = len(xs)
        sd = float(np.std(xs, ddof=1)) if n > 1 else float("nan")
        rows.append(dict(pid=pid, games=n, mean=float(np.mean(xs)), se=sd / math.sqrt(n) if n > 1 else float("nan"),
                         total=float(np.sum(xs))))
    rows.sort(key=lambda r: r["mean"], reverse=True)
    p_best = bootstrap_p_best({pid: xs for pid, xs in pts.items()})
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["p_best"] = p_best[r["pid"]]
    return rows


def bootstrap_p_best(pts: dict, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(0)
    ids = list(pts)
    means = np.empty((n_boot, len(ids)))
    for j, pid in enumerate(ids):
        x = np.asarray(pts[pid], dtype=float)
        means[:, j] = rng.choice(x, size=(n_boot, len(x))).mean(axis=1)
    winners = np.bincount(means.argmax(axis=1), minlength=len(ids)) / n_boot
    return {pid: float(w) for pid, w in zip(ids, winners)}


def n_keep(n_active: int, spec: float, minimum: int) -> int:
    n = math.ceil(spec * n_active) if spec <= 1 else int(spec)
    return max(min(minimum, n_active), min(n, n_active))


def write_csv(path: pathlib.Path, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "pid", "games", "mean", "se", "total", "p_best"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})


def print_top(rows: list, n: int = 15) -> None:
    print(f"  {'#':>3} {'checkpoint':<58} {'games':>5} {'mean pts':>9} {'±se':>6} {'P(best)':>8}", flush=True)
    for r in rows[:n]:
        print(f"  {r['rank']:>3} {r['pid']:<58} {r['games']:>5} {r['mean']:>9.2f} {r['se']:>6.2f} {r['p_best']:>8.2f}",
              flush=True)


def verify_order(participant: Participant, scenario: str, logs_dir: str, n: int = 6) -> bool:
    strong, dummy = "agent_code.coin_collector_agent.callbacks", "builtin:random"
    ok = True
    for pos in range(3):
        totals = np.zeros(3)
        for g in range(n):
            specs = [dummy] * 3
            specs[pos] = strong
            res = run_game(participant, [OpponentPool._resolve_static(s) for s in specs], scenario, 1000 + g, logs_dir)
            totals += np.asarray(res["opp_scores"], dtype=float)
        hit = int(np.argmax(totals)) == pos
        ok &= hit
        print(f"  coin collector in opponent slot {pos}: mean opponent_scores = {np.round(totals / n, 2).tolist()} "
              f"-> {'ok' if hit else 'MISMATCH'}", flush=True)
    return ok


def _parse_list(text: str, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="tournament_out")
    g = p.add_argument_group("candidates")
    g.add_argument("--runs", nargs="*", help="glob patterns of run names to include (default: all runs)")
    g.add_argument("--candidates", help="text file with one 'run/checkpoint_xxx' per line (overrides discovery)")
    g.add_argument("--every", type=int, default=1, help="use every n-th checkpoint of a run, counting back from the newest")
    g.add_argument("--last", type=int, default=0, help="only the newest n checkpoints per run (after --every)")
    g.add_argument("--min-steps", type=int, default=0, help="ignore checkpoints below this many timesteps")
    g.add_argument("--prefilter", type=int, default=0,
                   help="per run keep the n checkpoints with the best stored tournament_eval score (metadata.json)")
    g = p.add_argument_group("schedule")
    g.add_argument("--rounds", default="4,12,40",
                   help="games per candidate per stage, comma separated (one entry per stage)")
    g.add_argument("--keep", default="0.25,8",
                   help="survivors after each stage but the last: fraction (<=1) or count (>1)")
    g.add_argument("--mode", choices=["learner", "all"], default="all",
                   help="all: every game scores all candidates in it (checked automatically once, see --skip-order-check). "
                        "learner: only the score of the candidate in the env's controlled seat counts (needs no "
                        "assumptions, but ~4x more games for the same number of samples)")
    g.add_argument("--filler-prob", type=float, default=0.0,
                   help="probability that a (non-controlled) seat of a game is a random --fillers bot instead of a candidate")
    g.add_argument("--fillers", nargs="*", default=[], help="static opponent specs, e.g. agent_code.rule_based_agent.callbacks")
    g.add_argument("--skip-order-check", action="store_true", help="don't verify the opponent_scores order for --mode all")
    g.add_argument("--scenario", default="classic")
    g = p.add_argument_group("play")
    g.add_argument("--tta", type=int, default=agent.TTA_SYMMETRIES, help="symmetry TTA (default: value in callbacks.py)")
    g.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax (callbacks.py: DETERMINISTIC)")
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    g.add_argument("--cache-size", type=int, default=24, help="models kept in memory per worker")
    g.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart", action="store_true", help="delete previous results/state in --out and start over")
    p.add_argument("--dry-run", action="store_true", help="list candidates and planned number of games, then exit")
    p.add_argument("--verify-order", action="store_true", help="only check the opponent_scores ordering, then exit")
    return p.parse_args()


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rounds = _parse_list(args.rounds, int)
    keeps = _parse_list(args.keep, float)
    if len(keeps) != len(rounds) - 1:
        raise SystemExit(f"--keep needs {len(rounds) - 1} entries for {len(rounds)} stages")
    if args.filler_prob > 0 and not args.fillers:
        raise SystemExit("--filler-prob needs --fillers")
    k = N_PLAYERS

    candidates = discover(args)
    print(f"{len(candidates)} candidate checkpoints from {len({r for r, _ in candidates.values()})} runs", flush=True)
    (out / "candidates.txt").write_text("\n".join(sorted(candidates)) + "\n")
    paths = {pid: (str(r), str(c)) for pid, (r, c) in candidates.items()}
    opts = dict(tta=args.tta, deterministic=not args.stochastic, seed=args.seed, cache_size=args.cache_size,
                scenario=args.scenario, mode=args.mode, logs_dir=str(out / "logs"))

    if args.verify_order:
        torch.set_num_threads(1)
        pid = sorted(candidates)[0]
        r, c = candidates[pid]
        ok = verify_order(Participant(pid, r, c, 1, True, args.seed), args.scenario, opts["logs_dir"])
        print("opponent_scores order matches opponent order: " + ("YES, --mode all is safe" if ok else "NO, use --mode learner"))
        return

    if args.dry_run:
        n = len(candidates)
        for stage, rd in enumerate(rounds):
            games = rd * (math.ceil(n / k) * k if args.mode == "learner" else math.ceil(n / k))
            print(f"stage {stage}: {n} candidates x {rd} games each -> ~{games} games")
            if stage < len(keeps):
                n = n_keep(n, keeps[stage], N_PLAYERS)
        return

    fingerprint = dict(candidates=sorted(candidates), rounds=args.rounds, keep=args.keep, mode=args.mode,
                       filler_prob=args.filler_prob, fillers=args.fillers, scenario=args.scenario, tta=args.tta,
                       stochastic=args.stochastic, seed=args.seed)
    config_file = out / "config.json"
    if args.restart:
        for name in ("results.jsonl", "state.json", "config.json"):
            (out / name).unlink(missing_ok=True)
    if config_file.exists():
        old = json.loads(config_file.read_text())
        changed = sorted(k for k in fingerprint if old.get(k) != fingerprint[k])
        if changed:
            raise SystemExit(f"'{out}' holds an earlier tournament with different settings ({', '.join(changed)}). "
                             "Use a new --out, or --restart to discard it.")
    elif (out / "results.jsonl").exists() or (out / "state.json").exists():
        raise SystemExit(f"'{out}' holds results from an earlier tournament (no config.json). "
                         "Use a new --out, or --restart to discard it.")
    else:
        config_file.write_text(json.dumps(fingerprint))

    order_file = out / "order_check.json"
    if args.mode == "all" and not args.skip_order_check and not order_file.exists():
        print("checking that opponent_scores follow the opponent order (needed for --mode all) ...", flush=True)
        torch.set_num_threads(1)
        pid = sorted(candidates)[0]
        r, c = candidates[pid]
        if not verify_order(Participant(pid, r, c, 1, True, args.seed), args.scenario, opts["logs_dir"], n=8):
            raise SystemExit("opponent_scores do not follow the opponent order: rerun with --mode learner "
                             "(or --skip-order-check if you know the check is a false alarm)")
        order_file.write_text("ok")

    state_file = out / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else dict(stage=0, active=sorted(candidates))
    results_file = out / "results.jsonl"
    ctx = mp.get_context("spawn")

    for stage in range(state["stage"], len(rounds)):
        active = state["active"]
        schedule = build_schedule(active, stage, rounds[stage], args.filler_prob, args.fillers, args.mode, args.seed)
        done_ids, records = set(), []
        expected = {m["id"]: m["learner"] for m in schedule}
        if results_file.exists():
            for line in results_file.read_text().splitlines():
                rec = json.loads(line)
                if rec["stage"] == stage and expected.get(rec["id"]) == rec["learner"]:
                    done_ids.add(rec["id"])
                    records.append(rec)
        todo = [m for m in schedule if m["id"] not in done_ids]
        print(f"\n=== stage {stage}: {len(active)} candidates, {len(schedule)} games ({len(todo)} left) ===", flush=True)

        if todo:
            started, failed = time.time(), 0
            with ctx.Pool(min(args.workers, len(todo)), initializer=_worker_init, initargs=(paths, opts)) as pool, \
                    open(results_file, "a") as f:
                for i, rec in enumerate(pool.imap_unordered(play_match, todo, chunksize=1), 1):
                    if "error" in rec:
                        failed += 1
                        if failed <= 3:
                            print(f"game {rec['id']} failed:\n{rec['error']}", flush=True)
                        continue
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    records.append(rec)
                    if i % 25 == 0 or i == len(todo):
                        el = time.time() - started
                        print(f"  {i}/{len(todo)} games, {el / 60:.1f} min elapsed, ETA {(len(todo) - i) * el / i / 60:.1f} min",
                              flush=True)
            if failed:
                raise SystemExit(f"{failed} games failed (see the first tracebacks above). Fix the cause and re-run "
                                 "with the same --out: finished games are kept, failed ones are retried.")

        if not records:
            raise SystemExit(f"stage {stage} has no finished games")
        rows = standings(records)
        write_csv(out / f"standings_stage{stage}.csv", rows)
        print_top(rows)
        if stage < len(rounds) - 1:
            survivors = [r["pid"] for r in rows[:n_keep(len(rows), keeps[stage], N_PLAYERS)]]
            print(f"  -> keeping {len(survivors)} of {len(rows)}", flush=True)
            state = dict(stage=stage + 1, active=sorted(survivors))
        else:
            state = dict(stage=len(rounds), active=state["active"])
            write_csv(out / "final_ranking.csv", rows)
        state_file.write_text(json.dumps(state))

    final = out / "final_ranking.csv"
    if final.exists():
        best = next(csv.DictReader(open(final)))["pid"]
        run, _, ckpt = best.rpartition("/")
        print(f"\nBest checkpoint: {best}\n  callbacks.py -> RUN = \"{run}\"; CHECKPOINT = \"{ckpt}\"", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()