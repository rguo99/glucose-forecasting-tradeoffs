#!/bin/bash
# One shard of the cohort for a GPU model, one user per GPU at a time.
# Usage: sbatch slurm/gpu_model_shard.sbatch <model> <k/N>
#   <model>: lstm | lstm_ensemble | transformer | transformer_ensemble | transformer_conformal
#SBATCH --job-name=gpu-model
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
MODEL=$1
SHARD=$2
cd "$SLURM_SUBMIT_DIR"
THREADS=1 source slurm/_env.sh
GPU_IDS=$(seq -s, 0 $(( ${SLURM_GPUS_ON_NODE:-2} - 1 )))
python run_cohort.py --model "$MODEL" --gpus "$GPU_IDS" --shard "$SHARD"
