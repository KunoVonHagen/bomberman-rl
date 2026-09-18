#!/bin/bash
#SBATCH --job-name=dqn-tune
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gz300@uni-heidelberg.de

set -uo pipefail

mkdir -p logs

module purge
module load devel/python/3.11.7-gnu-14.2

source ~/envs/bomberman/bin/activate

cd "$HOME/bomberman-rl"

export PYTHONUNBUFFERED=1

SEED="${SEED:-0}"
PARALLEL="${PARALLEL:-1}"
DELETE_RUNS="${DELETE_RUNS:-1}"

# ---------------------------------------------------------------------------
# Expected compute budget
#
# Training speed: ~1M steps / 10 min
#
# Search 1: dqn_regularisation
#   5 weight_decay * 4 dropout = 20 trials
#   ~3h20m
#
# Search 2: dqn_default
#   5 learning_rate * 5 gamma * 4 target_update_interval
#     * 4 exploration_fraction
#   = 400 possible trials
#
#   Default budget: 100 trials
#   ~16h40m
#
# Search 3: dqn_shaping
#   5 coin * 5 crate * 5 danger * 5 trap
#   = 625 possible trials
#
#   Default budget: 100 trials
#   ~16h40m
#
# Total ~= 36h40m
#
# This leaves ~35h20m inside the 72h allocation for startup, validation,
# filesystem overhead, slower trials, and/or increasing the search budgets.
# ---------------------------------------------------------------------------

REG_MAX_TRIALS="${REG_MAX_TRIALS:-20}"
DEFAULT_MAX_TRIALS="${DEFAULT_MAX_TRIALS:-100}"
SHAPING_MAX_TRIALS="${SHAPING_MAX_TRIALS:-100}"

REG_TAG="${REG_TAG:-tune__dqn_regularisation}"
DEFAULT_TAG="${DEFAULT_TAG:-tune__dqn_default}"
SHAPING_TAG="${SHAPING_TAG:-tune__dqn_shaping}"

echo "============================================================"
echo " DQN hyperparameter tuning"
echo "============================================================"
echo "hostname: $(hostname)"
echo

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

echo "=== CPU ==="
echo "cpus visible to job: $(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')"
echo

echo "=== Search budget ==="
echo "regularisation : $REG_MAX_TRIALS trials"
echo "default        : $DEFAULT_MAX_TRIALS trials"
echo "shaping        : $SHAPING_MAX_TRIALS trials"
echo

run_search() {
    local SPACE="$1"
    local MAX_TRIALS="$2"
    local TAG="$3"

    echo
    echo "============================================================"
    echo " STARTING SEARCH: $SPACE"
    echo " max_trials=$MAX_TRIALS"
    echo " tag=$TAG"
    echo "============================================================"
    echo

    SEARCH_ARGS=(
        --space "$SPACE"
        --strategy grid
        --max-trials "$MAX_TRIALS"
        --parallel "$PARALLEL"
        --seed "$SEED"
        --tag "$TAG"
    )

    if [ "$DELETE_RUNS" = "1" ]; then
        SEARCH_ARGS+=(--delete-runs)
    fi

    START_TIME=$(date +%s)

    python3 -m evaluation.hyperparameter_search "${SEARCH_ARGS[@]}"
    RC=$?

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo
    echo "Search '$SPACE' exited with code $RC"
    echo "Elapsed: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"

    echo
    echo "=== Top results: $TAG ==="

    python3 - "$TAG" <<'EOF'
import json
import pathlib
import sys

path = pathlib.Path("evaluation/stats") / f"{sys.argv[1]}.json"

if not path.exists():
    print("No result file found.")
    sys.exit(0)

state = json.loads(path.read_text(encoding="utf-8"))

trials = state.get("trials", [])
done = sorted(
    (t for t in trials if t.get("score") is not None),
    key=lambda t: -t["score"],
)

print(f"{len(done)} scored trials of {len(trials)} started")

for t in done[:10]:
    print(
        f"  {t['score']:7.3f} "
        f"trial={t['id']:3d} "
        f"{t['params']}"
    )

if state.get("best"):
    best = state["best"]
    print()
    print(f"best score: {best['score']:.3f}")
    print(f"best params: {best['params']}")
EOF

    return "$RC"
}

# ---------------------------------------------------------------------------
# 1. DQN regularisation
#
# Search space:
#
#   weight_decay = [0, 1e-6, 1e-5, 1e-4, 1e-3]
#   dropout      = [0, 0.1, 0.2, 0.3]
#
# 5 * 4 = 20 trials
# ---------------------------------------------------------------------------

run_search \
    "dqn_regularisation" \
    "$REG_MAX_TRIALS" \
    "$REG_TAG"

REG_RC=$?

# ---------------------------------------------------------------------------
# 2. Main DQN hyperparameters
#
# Search space:
#
#   learning_rate          = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
#   gamma                  = [0.9, 0.95, 0.98, 0.99, 0.995]
#   target_update_interval = [1000, 3000, 10000, 30000]
#   exploration_fraction   = [0.05, 0.1, 0.2, 0.4]
#
# 5 * 5 * 4 * 4 = 400 possible combinations.
#
# Default to 100 trials to stay comfortably within the allocation.
# ---------------------------------------------------------------------------

run_search \
    "dqn_default" \
    "$DEFAULT_MAX_TRIALS" \
    "$DEFAULT_TAG"

DEFAULT_RC=$?

# ---------------------------------------------------------------------------
# 3. Reward shaping
#
# Search space:
#
#   coin_shaping_coef   = [0, 0.02, 0.05, 0.1, 0.2]
#   crate_shaping_coef  = [0, 0.01, 0.02, 0.05, 0.1]
#   danger_penalty_coef = [0, 0.02, 0.05, 0.1, 0.2]
#   trap_shaping_coef   = [0, 0.05, 0.1, 0.2, 0.4]
#
# 5 * 5 * 5 * 5 = 625 possible combinations.
#
# Default to 100 trials.
# ---------------------------------------------------------------------------

run_search \
    "dqn_shaping" \
    "$SHAPING_MAX_TRIALS" \
    "$SHAPING_TAG"

SHAPING_RC=$?

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo " DQN HYPERPARAMETER SEARCH COMPLETE"
echo "============================================================"
echo
echo "regularisation return code : $REG_RC"
echo "default return code        : $DEFAULT_RC"
echo "shaping return code        : $SHAPING_RC"
echo

for TAG in \
    "$REG_TAG" \
    "$DEFAULT_TAG" \
    "$SHAPING_TAG"
do
    echo "============================================================"
    echo "$TAG"
    echo "============================================================"

    python3 - "$TAG" <<'EOF'
import json
import pathlib
import sys

path = pathlib.Path("evaluation/stats") / f"{sys.argv[1]}.json"

if not path.exists():
    print("missing")
    sys.exit(0)

state = json.loads(path.read_text(encoding="utf-8"))

done = [
    t for t in state.get("trials", [])
    if t.get("score") is not None
]

print(f"scored: {len(done)} / {len(state.get('trials', []))}")

if state.get("best"):
    best = state["best"]
    print(f"best score : {best['score']:.3f}")
    print(f"best params: {best['params']}")

EOF

    echo
done

if [ "$REG_RC" -ne 0 ] ||
   [ "$DEFAULT_RC" -ne 0 ] ||
   [ "$SHAPING_RC" -ne 0 ]; then
    exit 1
fi

exit 0
