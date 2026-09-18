#!/bin/bash
#SBATCH --job-name=ppo-tune
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

cd $HOME/bomberman-rl


export PYTHONUNBUFFERED=1

SEED="${SEED:-0}"
PARALLEL="${PARALLEL:-1}"
DELETE_RUNS="${DELETE_RUNS:-1}"

# ---------------------------------------------------------------------------
# Expected compute budget
#
# Training speed: ~1M steps / 10 min
#
# Search 1:  ppo_regularisation
#   20 trials = ~3h20m
#
# Search 2:  ppo_default
#   192 trials = ~32h
#
# Search 3:  ppo_gae_epochs_vf
#   36 trials = ~6h
#
# Search 4:  ppo_batch_size
#   12 trials = ~2h
#
# Total ~= 43h20m
#
# This leaves ~4h40m for startup, validation, filesystem overhead,
# occasional slower trials, etc.
# ---------------------------------------------------------------------------

REG_MAX_TRIALS="${REG_MAX_TRIALS:-20}"
DEFAULT_MAX_TRIALS="${DEFAULT_MAX_TRIALS:-192}"
GAE_MAX_TRIALS="${GAE_MAX_TRIALS:-36}"
BATCH_MAX_TRIALS="${BATCH_MAX_TRIALS:-12}"

REG_TAG="${REG_TAG:-tune__ppo_regularisation}"
DEFAULT_TAG="${DEFAULT_TAG:-tune__ppo_default}"
GAE_TAG="${GAE_TAG:-tune__ppo_gae_epochs_vf}"
BATCH_TAG="${BATCH_TAG:-tune__ppo_batch_size}"

echo "============================================================"
echo " PPO hyperparameter tuning"
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
echo "gae/epochs/vf  : $GAE_MAX_TRIALS trials"
echo "batch size     : $BATCH_MAX_TRIALS trials"
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
# 1. Regularisation
# ---------------------------------------------------------------------------

run_search \
    "ppo_regularisation" \
    "$REG_MAX_TRIALS" \
    "$REG_TAG"

REG_RC=$?

# ---------------------------------------------------------------------------
# 2. Main PPO hyperparameters
# ---------------------------------------------------------------------------

run_search \
    "ppo_default" \
    "$DEFAULT_MAX_TRIALS" \
    "$DEFAULT_TAG"

DEFAULT_RC=$?

# ---------------------------------------------------------------------------
# 3. GAE / value-function optimisation
#
# Search space:
#
#   gae_lambda = [0.90, 0.95, 0.97, 0.99]
#   n_epochs   = [2, 4, 8]
#   vf_coef    = [0.3, 0.7, 1.0]
#
# 4 * 3 * 3 = 36 trials ~= 6 hours
# ---------------------------------------------------------------------------

run_search \
    "ppo_gae_epochs_vf" \
    "$GAE_MAX_TRIALS" \
    "$GAE_TAG"

GAE_RC=$?

# ---------------------------------------------------------------------------
# 4. Batch-size search
#
# Search space:
#
#   batch_size = [128, 256, 512]
#   gae_lambda = [0.95, 0.97, 0.99]
#   n_epochs   = [4, 8]
#
# 3 * 3 * 2 = 18 possible combinations.
#
# Default to 12 to stay safely inside the 48h allocation.
# ---------------------------------------------------------------------------

run_search \
    "ppo_batch_size" \
    "$BATCH_MAX_TRIALS" \
    "$BATCH_TAG"

BATCH_RC=$?

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo " PPO HYPERPARAMETER SEARCH COMPLETE"
echo "============================================================"
echo
echo "regularisation return code : $REG_RC"
echo "default return code        : $DEFAULT_RC"
echo "gae/epochs/vf return code  : $GAE_RC"
echo "batch-size return code     : $BATCH_RC"
echo

for TAG in \
    "$REG_TAG" \
    "$DEFAULT_TAG" \
    "$GAE_TAG" \
    "$BATCH_TAG"
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
   [ "$GAE_RC" -ne 0 ] ||
   [ "$BATCH_RC" -ne 0 ]; then
    exit 1
fi

exit 0