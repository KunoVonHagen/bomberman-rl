#!/bin/bash
#SBATCH --job-name=bomberman-ppo
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gz300@uni-heidelberg.de

module purge
module load devel/python/3.11.7-gnu-14.2

source ~/envs/bomberman/bin/activate

cd $HOME/bomberman-rl

srun python -m agent_code.my_agent.train "$@"