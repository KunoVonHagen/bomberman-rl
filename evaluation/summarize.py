from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COLUMNS = [
    ("tag", "tag"),
    ("matchup", "matchup"),
    ("agent", "agent"),
    ("n", "n"),
    ("score", "score mean +- sem"),
    ("win_rate", "win %"),
    ("survival_rate", "surv %"),
    ("coins_mean", "coins"),
    ("kills_mean", "kills"),
    ("suicides_mean", "suic"),
    ("invalid_mean", "inval"),
    ("steps_survived_mean", "steps"),
    ("steps_lost_per_round", "lost/rd"),
    ("think_time", "think ms mean/max"),
]


def load_results(paths: List[pathlib.Path]) -> List[Dict[str, Any]]:
    results = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        data["_file"] = pathlib.Path(path).name
        results.append(data)
    return results


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.0f}"


def _num(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def matchup_label(result: Dict[str, Any]) -> str:
    """The matchup name, or the lineup for explicit `run --agents` lineups."""
    name = result["matchup"]["name"]
    if name == "custom":
        return " vs ".join(result["matchup"]["agents"])
    return name


def rows_for(result: Dict[str, Any], candidate_only: bool) -> List[Dict[str, str]]:
    summary = result["summary"]
    rows = []
    for index, name in enumerate(summary):
        if candidate_only and index > 0:
            break
        m = summary[name]
        rows.append({
            "tag": result["_file"].removesuffix(".json") if index == 0 else "",
            "matchup": matchup_label(result) if index == 0 else "",
            "agent": name if index == 0 else f"  vs {name}",
            "n": str(m["n_rounds"]),
            "score": f"{_num(m['score_mean'])} +- {_num(m['score_sem'])}",
            "win_rate": _pct(m["win_rate"]),
            "survival_rate": _pct(m["survival_rate"]),
            "coins_mean": _num(m["coins_mean"]),
            "kills_mean": _num(m["kills_mean"]),
            "suicides_mean": _num(m["suicides_mean"]),
            "invalid_mean": _num(m["invalid_mean"], 1),
            "steps_survived_mean": _num(m["steps_survived_mean"], 0),
            "steps_lost_per_round": _num(m["steps_lost_per_round"]),
            "think_time": f"{_num(m['think_time_mean_ms'], 1)}/{_num(m['think_time_max_ms'], 0)}",
        })
    return rows


def format_table(results: List[Dict[str, Any]], candidate_only: bool = True) -> str:
    rows = [row for result in results for row in rows_for(result, candidate_only)]
    keys = [key for key, _ in COLUMNS]
    header = [title for _, title in COLUMNS]
    widths = [max(len(header[i]), *(len(row[k]) for row in rows)) for i, k in enumerate(keys)]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
        "|-" + "-|-".join("-" * w for w in widths) + "-|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[k].ljust(w) for k, w in zip(keys, widths)) + " |")
    return "\n".join(lines)


def write_csv(results: List[Dict[str, Any]], path: pathlib.Path, candidate_only: bool) -> None:
    """Write the raw per-agent metrics (all keys of the summary blocks) to CSV."""
    with open(path, "w", newline="") as f:
        writer = None
        for result in results:
            for index, (name, metrics) in enumerate(result["summary"].items()):
                if candidate_only and index > 0:
                    break
                row = {
                    "tag": result["_file"].removesuffix(".json"),
                    "matchup": matchup_label(result),
                    "scenario": result["matchup"]["scenario"],
                    "lineup": " vs ".join(result["matchup"]["agents"]),
                    "seed": result["matchup"]["seed"],
                    "git_commit": result["meta"].get("git_commit"),
                    "note": result["meta"].get("note"),
                    "agent": name,
                    **metrics,
                }
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="result files or glob patterns")
    parser.add_argument("--all-agents", action="store_true", help="also list the opponents of each lineup")
    parser.add_argument("--csv", type=pathlib.Path, help="additionally write the metrics as CSV")
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        if not matches and not pathlib.Path(pattern).exists():
            parser.error(f"no result file matches '{pattern}'")
        paths.extend(pathlib.Path(p) for p in (matches or [pattern]))
    results = load_results(paths)
    print(format_table(results, candidate_only=not args.all_agents))
    if args.csv:
        write_csv(results, args.csv, candidate_only=not args.all_agents)
        print(f"written {args.csv}")


if __name__ == "__main__":
    main()
