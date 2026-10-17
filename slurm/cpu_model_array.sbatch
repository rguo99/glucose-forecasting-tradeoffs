#!/bin/bash
# One CPU model fit per array task (one user per task).
# Usage: sbatch slurm/cpu_model_array.sbatch <model> [prior_scale]
#   <model>: global | ar1_pure | ar1_rls | lstm_conformal
#   prior_scale (Bayesian Loop only): 0.30 (default), 0.10 or 0.50
# Submitted by slurm/submit_all.sh, which sets the array size and N_USERS to
# the number of analysed users. To submit by hand:
#   sbatch -p <partition> --array=1-<N> --export=ALL,N_USERS=<N> slurm/cpu_model_array.sbatch global
#SBATCH --job-name=cpu-model
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
MODEL=$1
PRIOR=${2:-0.30}
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
export CUDA_VISIBLE_DEVICES=""
python run_cohort.py --model "$MODEL" --gpus 0 --shard "${SLURM_ARRAY_TASK_ID}/${N_USERS:?set N_USERS to the number of analysed users}" --prior-scale "$PRIOR"
