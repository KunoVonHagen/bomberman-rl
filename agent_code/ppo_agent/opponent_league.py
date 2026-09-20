from __future__ import annotations

from functools import partial
import dataclasses
import importlib
import random
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

OpponentPair = Tuple[Callable, Callable]

_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]
_KNOWN_MODS = ("eps", "bomb", "safe")

_SPEC_RE = re.compile(
    r"^(?P<path>[^@#\s]+)(?P<mods>(?:@[a-zA-Z_]+=[0-9.eE+-]+)*)(?:#(?P<tier>[A-Za-z0-9_-]+))?$"
)
_MOD_RE = re.compile(r"@([a-zA-Z_]+)=([0-9.eE+-]+)")

_BUILTIN_FACTORIES: Dict[str, Callable[[], OpponentPair]] = {}


@dataclasses.dataclass(frozen=True)
class StaticBot:
    """A parsed static-opponent spec. See module docstring for the grammar."""

    spec: str
    path: str
    eps: float = 0.0
    bomb: float = 0.0
    safe: float = 0.0
    tier: Optional[str] = None

    @property
    def is_builtin(self) -> bool:
        return self.path.startswith("builtin:")

    @property
    def is_noisy(self) -> bool:
        return bool(self.eps or self.bomb or self.safe)


def parse_static_spec(spec: str) -> StaticBot:
    """Parse a spec string into a :class:`StaticBot`. Raises ``ValueError`` on anything malformed."""
    raw = spec
    spec = spec.strip()
    match = _SPEC_RE.match(spec)
    if not match:
        raise ValueError(f"Malformed static-opponent spec: {raw!r}")

    path = match.group("path")
    tier = match.group("tier")
    mods = {name: float(value) for name, value in _MOD_RE.findall(match.group("mods") or "")}

    unknown = set(mods) - set(_KNOWN_MODS)
    if unknown:
        raise ValueError(
            f"Unknown modifier(s) {sorted(unknown)} in spec {raw!r}; known modifiers: {list(_KNOWN_MODS)}"
        )
    for name, value in mods.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Modifier @{name}= in spec {raw!r} must be in [0, 1], got {value}")

    if not path.startswith("builtin:") and "." not in path:
        raise ValueError(
            f"Static-opponent path {path!r} in spec {raw!r} must be a dotted module path "
            f"(e.g. 'agent_code.rule_based_agent.callbacks') or a 'builtin:*' name."
        )
    if path.startswith("builtin:") and path not in _BUILTIN_FACTORIES:
        raise ValueError(f"Unknown builtin opponent {path!r}; known builtins: {sorted(_BUILTIN_FACTORIES)}")

    return StaticBot(
        spec=raw, path=path,
        eps=mods.get("eps", 0.0), bomb=mods.get("bomb", 0.0), safe=mods.get("safe", 0.0),
        tier=tier,
    )


def spec_label(spec: str) -> str:
    """Short, log/metric-key-safe label for a spec, e.g. for TensorBoard tags or filenames."""
    bot = parse_static_spec(spec)
    if bot.path.startswith("builtin:"):
        short = bot.path.split(":", 1)[1]
    else:
        parts = bot.path.split(".")
        short = parts[-2] if len(parts) >= 2 and parts[-1] == "callbacks" else parts[-1]
    tag = [short]
    if bot.eps:
        tag.append(f"eps{bot.eps:g}")
    if bot.bomb:
        tag.append(f"bomb{bot.bomb:g}")
    if bot.safe:
        tag.append(f"safe{bot.safe:g}")
    if bot.tier:
        tag.append(bot.tier)
    return "_".join(tag)


def validate_static_specs(specs: Sequence[str], known_tiers: Iterable[str] = ()) -> None:
    """
    Check that each spec is well-formed, has a known tier (if any), and is unique within the list.
    """
    known_tiers = set(known_tiers)
    seen = set()
    for spec in specs:
        bot = parse_static_spec(spec)
        if bot.tier and known_tiers and bot.tier not in known_tiers:
            raise ValueError(
                f"Static opponent {spec!r} uses tier {bot.tier!r}, which has no matching entry in "
                f"self_play.static_tier_weights (known tiers: {sorted(known_tiers)}). Add one, or drop "
                "the '#tier' suffix."
            )
        if spec in seen:
            raise ValueError(f"Duplicate static-opponent spec: {spec!r}")
        seen.add(spec)


def _peaceful_pair() -> OpponentPair:
    return (lambda self: None), (lambda self, game_state: "WAIT")


def _random_pair() -> OpponentPair:
    return (lambda self: None), (lambda self, game_state: random.choice(_ACTIONS))


_BUILTIN_FACTORIES.update({
    "builtin:peaceful": _peaceful_pair,
    "builtin:random": _random_pair,
})


def _base_pair(path: str) -> OpponentPair:
    factory = _BUILTIN_FACTORIES.get(path)
    if factory is not None:
        return factory()
    module = importlib.import_module(path)
    return module.setup, module.act


def _noisy_act(act: Callable, bot: StaticBot, self, game_state):
    if bot.safe and random.random() < bot.safe:
        return "WAIT"
    action = act(self, game_state)
    if bot.eps and random.random() < bot.eps:
        action = random.choice(_ACTIONS)
    if bot.bomb and action == "BOMB" and random.random() < bot.bomb:
        action = "WAIT"
    return action

def _wrap_noise(setup: Callable, act: Callable, bot: StaticBot) -> OpponentPair:
    if not bot.is_noisy:
        return setup, act
    return setup, partial(_noisy_act, act, bot)


_RESOLVE_CACHE: Dict[str, OpponentPair] = {}


def resolve_static_spec(spec: str) -> OpponentPair:
    """Spec string -> ``(setup, act)``. Results are cached per distinct spec string."""
    cached = _RESOLVE_CACHE.get(spec)
    if cached is not None:
        return cached
    bot = parse_static_spec(spec)
    setup, act = _base_pair(bot.path)
    pair = _wrap_noise(setup, act, bot)
    _RESOLVE_CACHE[spec] = pair
    return pair


def _spec_weight(spec: str, tier_weights: Dict[str, float], priorities: Dict[str, float]) -> float:
    weight = 1.0
    if tier_weights:
        bot = parse_static_spec(spec)
        if bot.tier is not None:
            weight *= max(0.0, tier_weights.get(bot.tier, 1.0))
    if priorities:
        weight *= max(0.0, priorities.get(spec, 1.0))
    return weight


def sample_static_specs(
    pool: Sequence[str],
    k: int,
    *,
    tier_weights: Optional[Dict[str, float]] = None,
    priorities: Optional[Dict[str, float]] = None,
    allow_repeat: bool = True,
) -> List[str]:
    """
    Sample ``k`` static-opponent specs from ``pool``, optionally weighted by
    ``tier_weights`` and/or ``priorities``. If ``allow_repeat`` is False, the
    returned list will contain unique specs, and if ``k`` exceeds the pool size,
    the pool will be re-shuffled and drawn from again until ``k`` are chosen.
    """
    if k <= 0:
        return []
    pool = list(pool)
    if not pool:
        return []
    tier_weights = tier_weights or {}
    priorities = priorities or {}
    weights = [_spec_weight(spec, tier_weights, priorities) for spec in pool]
    if sum(weights) <= 0:
        weights = [1.0] * len(pool)

    if allow_repeat:
        return random.choices(pool, weights=weights, k=k)

    chosen: List[str] = []
    remaining_pool, remaining_weights = list(pool), list(weights)
    while len(chosen) < k:
        if not remaining_pool:
            remaining_pool, remaining_weights = list(pool), list(weights)
        total = sum(remaining_weights)
        if total <= 0:
            remaining_weights = [1.0] * len(remaining_pool)
            total = float(len(remaining_pool))
        pick = random.uniform(0, total)
        cursor = 0.0
        idx = len(remaining_pool) - 1
        for i, w in enumerate(remaining_weights):
            cursor += w
            if pick <= cursor:
                idx = i
                break
        chosen.append(remaining_pool.pop(idx))
        remaining_weights.pop(idx)
    return chosen


def ema(prev: Optional[float], new: float, alpha: float) -> float:
    """Exponential moving average. Seeds with ``new`` when there's no previous value."""
    if prev is None:
        return float(new)
    alpha = min(1.0, max(0.0, alpha))
    return alpha * float(new) + (1.0 - alpha) * float(prev)


def hardness_from_summary(summary: Optional[Dict[str, Optional[float]]]) -> Optional[float]:
    """
    Convert a per-spec tournament summary into a hardness score for PFSP sampling.
    """
    if not summary:
        return None
    win = summary.get("win")
    if win is not None:
        return float(max(0.0, min(1.0, 1.0 - win)))
    survived = summary.get("survived")
    if survived is not None:
        return float(max(0.0, min(1.0, 1.0 - survived)))
    return None


def priority_from_hardness(hardness: Optional[float], floor: float, power: float) -> float:
    """
    Convert a hardness score into a PFSP sampling priority.
    """
    if hardness is None:
        return 1.0
    floor = max(0.0, min(1.0, floor))
    weight = max(0.0, hardness) ** max(0.0, power)
    return floor + (1.0 - floor) * weight


def select_hall_of_fame(
    entries: Sequence[Tuple[int, "object"]],
    every_timesteps: int,
    max_count: int,
) -> List["object"]:
    """
    Thin a list of hall-of-fame entries (``(timesteps, path)``) to at most
    ``max_count`` entries, keeping the last checkpoint in each ``every_timesteps`` bucket and then sampling evenly across the remaining ones if needed.
    """
    if every_timesteps <= 0 or max_count <= 0 or not entries:
        return []
    ordered_entries = sorted(entries, key=lambda item: item[0])
    milestones: Dict[int, "object"] = {}
    for timesteps, path in ordered_entries:
        milestones[timesteps // every_timesteps] = path  # last checkpoint in each bucket wins
    ordered = [milestones[bucket] for bucket in sorted(milestones)]
    if len(ordered) <= max_count:
        return ordered
    if max_count == 1:
        return [ordered[-1]]
    idxs = sorted({round(i * (len(ordered) - 1) / (max_count - 1)) for i in range(max_count)})
    return [ordered[i] for i in idxs]


def thin_geometric(
    checkpoints: Sequence["object"],
    ratio: float = 1.6,
    min_gap: int = 1,
) -> List["object"]:
    """
    Thin a list of checkpoints geometrically, keeping the last checkpoint and then sampling earlier ones with a gap that grows by ``ratio`` each time, but never smaller than ``min_gap``.
    """
    n = len(checkpoints)
    if n <= 2:
        return list(checkpoints)
    kept_idx = [n - 1]
    gap = max(1, min_gap)
    idx = n - 1
    while idx - gap > 0:
        idx -= gap
        kept_idx.append(idx)
        gap = max(min_gap, round(gap * ratio))
    kept_idx.append(0)
    return [checkpoints[i] for i in sorted(set(kept_idx))]


DEFAULT_HELDOUT_POOL: List[str] = [
    "agent_code.rule_based_agent.callbacks#strong",
    "agent_code.rule_based_agent.callbacks@eps=0.2#medium",
    "agent_code.coin_collector_agent.callbacks#medium",
    "agent_code.coin_collector_agent.callbacks@bomb=0.5#weak",
    "agent_code.peaceful_agent.callbacks#weak",
    "agent_code.league_bots.hunter#medium",
    "agent_code.league_bots.hunter@eps=0.3#weak",
    "agent_code.league_bots.bomber#medium",
    "agent_code.league_bots.bomber@eps=0.3#weak",
    "builtin:random#weak",
]


def make_tournament_lineups(
    pool: Sequence[str],
    n_opponents: int = 3,
    n_lineups: int = 8,
    rng: Optional[random.Random] = None,
) -> List[List[str]]:
    """
    Make a list of ``n_lineups`` lineups, each containing ``n_opponents`` specs drawn from ``pool``.
    The same spec may appear in multiple lineups, but not more than once in the same lineup.
    If ``n_opponents`` exceeds the pool size, the pool will be re-shuffled and drawn from again until each lineup has enough opponents.
    """
    rng = rng or random
    pool = list(pool)
    if not pool or n_opponents <= 0 or n_lineups <= 0:
        return []
    needed = n_lineups * n_opponents
    bag: List[str] = []
    while len(bag) < needed:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        bag.extend(shuffled)
    bag = bag[:needed]
    return [bag[i * n_opponents:(i + 1) * n_opponents] for i in range(n_lineups)]


def merge_case_summaries(
    records: Sequence[Dict[str, object]],
    lineups: Sequence[Sequence[str]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Merge a list of per-case tournament records into a summary of overall and per-spec averages.
    """

    def _blank() -> Dict[str, list]:
        return {"score": [], "win": [], "survived": [], "length": [], "reward": []}

    def _finish(bucket: Dict[str, list]) -> Dict[str, Optional[float]]:
        finished = {key: (sum(values) / len(values) if values else None) for key, values in bucket.items()}
        finished["episodes"] = len(bucket["score"])
        return finished

    overall = _blank()
    per_spec: Dict[str, Dict[str, list]] = {}

    for record, lineup in zip(records, lineups):
        opponent_scores = [float(s) for s in (record.get("opponent_scores") or [])]
        won = float(float(record["score"]) > max(opponent_scores)) if opponent_scores else None

        for spec in lineup:
            bucket = per_spec.setdefault(spec, _blank())
            bucket["score"].append(float(record["score"]))
            bucket["survived"].append(float(record["survived"]))
            bucket["length"].append(float(record["length"]))
            bucket["reward"].append(float(record["reward"]))
            if won is not None:
                bucket["win"].append(won)

        overall["score"].append(float(record["score"]))
        overall["survived"].append(float(record["survived"]))
        overall["length"].append(float(record["length"]))
        overall["reward"].append(float(record["reward"]))
        if won is not None:
            overall["win"].append(won)

    result = {"overall": _finish(overall)}
    for spec, bucket in per_spec.items():
        result[spec] = _finish(bucket)
    return result