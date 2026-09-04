#!/bin/bash
#SBATCH --job-name=bomberman-ppo-short
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gz300@uni-heidelberg.de

module purge
module load devel/python/3.11.7-gnu-14.2

source ~/envs/bomberman/bin/activate

cd $HOME/bomberman-rl

RUN_NAME="${RUN_NAME:-bomberman_ppo_selfplay}"

TOTAL_SECONDS=$((72 * 3600))
LIVE_CHECK_INTERVAL=300

RUN_ROOT=$(python -c "from agent_code.ppo_agent.config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.runs_dir)")
RUN_DIR="$RUN_ROOT/$RUN_NAME"

echo "Run name: $RUN_NAME"
echo "Run dir:  $RUN_DIR"

if [ -d "$RUN_DIR" ]; then
    echo "Existing run dir found -- resuming."
    TRAIN_ARGS=(--resume "$RUN_NAME")
else
    echo "No existing run dir -- starting a new run."
    TRAIN_ARGS=(--set "run_name=$RUN_NAME")
fi

python -m agent_code.ppo_agent.training_schedule \
    --run-dir "$RUN_DIR" \
    --total-seconds "$TOTAL_SECONDS" \
    --check-interval "$LIVE_CHECK_INTERVAL" &
SCHEDULER_PID=$!

cleanup() {
    kill "$SCHEDULER_PID" 2>/dev/null
}
trap cleanup EXIT

srun python -m agent_code.ppo_agent.train "${TRAIN_ARGS[@]}" --device cuda

cleanup
