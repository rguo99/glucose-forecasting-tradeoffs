#!/bin/bash
# Evaluation, tables and figures (CPU only), after all model runs finish.
#SBATCH --job-name=paper-outputs
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
THREADS=1 source slurm/_env.sh
export CUDA_VISIBLE_DEVICES=""
export T1DEXI_RAW_DIR=${T1DEXI_RAW_DIR:?set T1DEXI_RAW_DIR (raw T1DEXI DM/LB/PR parquet)}
python make_paper_outputs.py
