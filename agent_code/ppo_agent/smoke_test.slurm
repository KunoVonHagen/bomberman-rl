#!/bin/bash
#SBATCH --job-name=bomberman-smoketest
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoketest-%j.out
#SBATCH --error=logs/smoketest-%j.err

set -euo pipefail
mkdir -p logs

module purge
module load devel/python/3.11.7-gnu-14.2

source ~/envs/bomberman/bin/activate  # or module-load conda + conda activate

cd "$HOME/bomberman-rl"

echo "=== Node / GPU info ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

echo "=== Torch / CUDA sanity check ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
"
echo

# --- Background monitors: GPU + CPU utilization for the duration of the run ---
MONITOR_DIR="logs/monitor_${SLURM_JOB_ID}"
mkdir -p "$MONITOR_DIR"

# GPU utilization / memory every 2s
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 2 > "$MONITOR_DIR/gpu_util.csv" &
GPU_MON_PID=$!

# CPU / load every 2s (mpstat: per-core; needs sysstat module/package -- falls back to vmstat if missing)
if command -v mpstat &> /dev/null; then
    mpstat -P ALL 2 > "$MONITOR_DIR/cpu_util.txt" &
else
    vmstat 2 > "$MONITOR_DIR/cpu_util.txt" &
fi
CPU_MON_PID=$!

cleanup() {
    kill "$GPU_MON_PID" "$CPU_MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Running smoke test (tiny config: few envs, ~3 rollouts, checkpoint + eval game exercised) ==="
echo "--- pass 1: n_shards=1 (old single-process behavior, baseline) ---"
time python -m agent_code.my_agent.train --smoke-test --device cuda --n-shards 1
echo
echo "--- pass 2: n_shards=8 (sharded across subprocesses) ---"
time python -m agent_code.my_agent.train --smoke-test --device cuda --n-envs 8 --n-shards 8

echo
echo "=== Smoke test finished. Utilization logs written to: $MONITOR_DIR ==="
echo "Quick summary:"
echo "--- GPU utilization (%) samples ---"
awk -F',' 'NR>1 {gsub(/ %/,"",$2); print $2}' "$MONITOR_DIR/gpu_util.csv" | \
    awk '{sum+=$1; n++; if($1>max)max=$1} END {if(n>0) printf "avg=%.1f%% max=%d%% samples=%d\n", sum/n, max, n}'
