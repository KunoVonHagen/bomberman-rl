#!/bin/bash
#SBATCH --job-name=bomberman-sweep
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=00:30:00
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gz300@uni-heidelberg.de

set -uo pipefail
mkdir -p logs

module purge
module load devel/python/3.11.7-gnu-14.2

source ~/envs/bomberman/bin/activate

cd "$HOME/bomberman-rl"

echo "=== Node / GPU info ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

SWEEP_DIR="logs/sweep_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"
SUMMARY_CSV="$SWEEP_DIR/summary.csv"
echo "n_envs,n_shards,wall_seconds,avg_fps,fps_per_env,status" > "$SUMMARY_CSV"

# --- Combos to try: "n_envs:n_shards" -----------------------------------
# n_shards must divide n_envs and stay <= cpus-per-gpu (32). Edit freely.
COMBOS=(
    "64:8"
    "64:16"
    "64:32"
    "128:8"
    "128:16"
    "128:32"
    "256:16"
    "256:32"
)

for combo in "${COMBOS[@]}"; do
    n_envs="${combo%%:*}"
    n_shards="${combo##*:}"
    tag="envs${n_envs}_shards${n_shards}"
    combo_log="$SWEEP_DIR/${tag}.log"

    echo "=================================================================="
    echo "=== Running n_envs=${n_envs} n_shards=${n_shards}  (log: ${combo_log}) ==="
    echo "=================================================================="

    start_ts=$(date +%s)
    python -m agent_code.my_agent.train \
        --smoke-test --device cuda \
        --n-envs "$n_envs" --n-shards "$n_shards" \
        > "$combo_log" 2>&1
    rc=$?
    end_ts=$(date +%s)
    wall=$((end_ts - start_ts))

    if [ $rc -ne 0 ]; then
        echo "  -> FAILED (exit $rc) after ${wall}s, see ${combo_log}"
        echo "${n_envs},${n_shards},${wall},,,FAILED" >> "$SUMMARY_CSV"
        continue
    fi

    # Pull every "fps" value SB3's logger printed (one per rollout) and average.
    avg_fps=$(grep -oE '\bfps\b[^0-9]*[0-9]+' "$combo_log" \
        | grep -oE '[0-9]+$' \
        | awk '{sum+=$1; n++} END {if (n>0) printf "%.1f", sum/n; else print ""}')

    if [ -z "$avg_fps" ]; then
        echo "  -> ran ok in ${wall}s but no fps line found, see ${combo_log}"
        echo "${n_envs},${n_shards},${wall},,,NO_FPS_FOUND" >> "$SUMMARY_CSV"
        continue
    fi

    fps_per_env=$(awk -v f="$avg_fps" -v n="$n_envs" 'BEGIN{printf "%.2f", f/n}')
    echo "  -> wall=${wall}s avg_fps=${avg_fps} fps_per_env=${fps_per_env}"
    echo "${n_envs},${n_shards},${wall},${avg_fps},${fps_per_env},OK" >> "$SUMMARY_CSV"
done

echo
echo "=== Sweep finished. Per-combo logs + summary in: $SWEEP_DIR ==="
echo "=== Summary (sorted by avg_fps, best first) ==="
{
    head -n1 "$SUMMARY_CSV"
    tail -n +2 "$SUMMARY_CSV" | awk -F',' '$5!=""' | sort -t',' -k4 -n -r
} | column -s',' -t
