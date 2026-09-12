#!/bin/bash
#SBATCH --job-name=bomberman-dqn-short
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ha300@uni-heidelberg.de

module purge
module load devel/python/3.11.7-gnu-14.2

source ~/envs/bomberman/bin/activate

cd $HOME/bomberman-rl

RUN_NAME="${RUN_NAME:-bomberman_dqn_short_$(date +%Y%m%d_%H%M%S)}"

RUN_ROOT=$(python -c "from agent_code.dqn_agent.config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.runs_dir)")
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

srun python -m agent_code.dqn_agent.train "${TRAIN_ARGS[@]}" --device cuda
