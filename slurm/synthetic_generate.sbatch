#!/bin/bash
# Generate the synthetic cohort for parameter recovery (50 users, seed 0),
# including the per-user feature files. Run before slurm/synthetic_recovery.sbatch.
#SBATCH --job-name=synthetic-generate
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
export CUDA_VISIBLE_DEVICES=""
python synthetic/generate_synthetic_loop.py --n 50 --seed 0
