#!/bin/bash
# Synthetic parameter recovery: fit Bayesian Loop to one synthetic user per
# array task, after slurm/synthetic_generate.sbatch has written the cohort.
#   sbatch --array=1-50 slurm/synthetic_recovery.sbatch
# All tasks write into experiments/synthetic_recovery_n50_<array job id>/.
#SBATCH --job-name=synthetic-recovery
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
export CUDA_VISIBLE_DEVICES=""
python run_experiments.py --synthetic --n 50 --gpus 0 \
  --shard "${SLURM_ARRAY_TASK_ID}/50" --run-id "${SLURM_ARRAY_JOB_ID}"
