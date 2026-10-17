#!/usr/bin/env bash
# Submit every job needed to reproduce the paper, chained with Slurm
# dependencies, so that the tables and figures are built automatically once all
# model fits have finished.
#
#   export T1DEXI_DATA_DIR=/path/to/t1dexi-preprocessing/output
#   export T1DEXI_RAW_DIR=/path/to/T1DEXI          # raw domains (DM, LB, PR parquet)
#   export RESULTS_DIR=/path/to/results             # optional; default ./output
#   CPU_PARTITION=normal GPU_PARTITION=gpu bash slurm/submit_all.sh
#
# Optional settings (environment variables):
#   CPU_PARTITION, GPU_PARTITION  partitions for CPU and GPU jobs (required)
#   CPU_CONSTRAINT, GPU_CONSTRAINT  node features passed as -C (default: none)
#   GPU_SHARDS                      GPU jobs per neural model (default 4)
#   MAX_PARALLEL                    concurrent array tasks per CPU model (default 100)
#   WITH_TIMING=1                   also run the MCMC wall-time experiment
#   DRY_RUN=1                       print the sbatch commands without submitting
#
# Run from the repository root, after activating the environment and editing
# slurm/_env.sh if needed. Logs go to ./logs/.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${T1DEXI_DATA_DIR:?set T1DEXI_DATA_DIR to the t1dexi-preprocessing output root}"
: "${T1DEXI_RAW_DIR:?set T1DEXI_RAW_DIR to the raw T1DEXI folder}"
: "${CPU_PARTITION:?set CPU_PARTITION}"
: "${GPU_PARTITION:?set GPU_PARTITION}"
GPU_SHARDS=${GPU_SHARDS:-4}
MAX_PARALLEL=${MAX_PARALLEL:-100}
mkdir -p logs

CPU_OPTS=(-p "$CPU_PARTITION" --parsable)
GPU_OPTS=(-p "$GPU_PARTITION" --parsable)
[ -n "${CPU_CONSTRAINT:-}" ] && CPU_OPTS+=(-C "$CPU_CONSTRAINT")
[ -n "${GPU_CONSTRAINT:-}" ] && GPU_OPTS+=(-C "$GPU_CONSTRAINT")

submit() {  # prints the job id (or a placeholder in dry-run mode)
  if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "DRYRUN"; else sbatch "$@"; fi
}

# Number of analysed users, from the cohort definition in eval_config.py
# (397 eligible pump users minus 2 with duplicate CGM timestamps = 395 for T1DEXI).
N_USERS=$(OMP_NUM_THREADS=1 python -c "
import config, eval_config
print(len(eval_config.load_eligible_users(config.data_dir(), config.filter_dir(), verbose=False)))")
echo "Analysed users: $N_USERS"

JOBS=()
ARRAY=(--array="1-${N_USERS}%${MAX_PARALLEL}" --export="ALL,N_USERS=${N_USERS}")

# CPU models: Bayesian Loop (main prior and two sensitivity priors), Pure AR1,
# AR1-RLS and LSTM-Conformal; one user per array task.
for PRIOR in 0.30 0.10 0.50; do
  JOBS+=("$(submit "${CPU_OPTS[@]}" "${ARRAY[@]}" -J "bayesian-loop-${PRIOR}" slurm/cpu_model_array.sbatch global "$PRIOR")")
done
for MODEL in ar1_pure ar1_rls lstm_conformal; do
  JOBS+=("$(submit "${CPU_OPTS[@]}" "${ARRAY[@]}" -J "$MODEL" slurm/cpu_model_array.sbatch "$MODEL")")
done

# GPU models, split into GPU_SHARDS jobs each.
for MODEL in lstm lstm_ensemble transformer transformer_ensemble transformer_conformal; do
  for K in $(seq 1 "$GPU_SHARDS"); do
    JOBS+=("$(submit "${GPU_OPTS[@]}" -J "$MODEL-$K" slurm/gpu_model_shard.sbatch "$MODEL" "$K/$GPU_SHARDS")")
  done
done

# Synthetic parameter recovery (CPU): generate the cohort, then one user per array task.
SYN_GEN=$(submit "${CPU_OPTS[@]}" -J synthetic-generate slurm/synthetic_generate.sbatch)
JOBS+=("$SYN_GEN")
JOBS+=("$(submit "${CPU_OPTS[@]}" --array=1-50 --dependency="afterok:${SYN_GEN}" -J synthetic-recovery slurm/synthetic_recovery.sbatch)")

# Optional: MCMC wall-time experiment (Table S6), on CPU (2 cores) and on one
# GPU. Timings depend on hardware.
if [ -n "${WITH_TIMING:-}" ]; then
  JOBS+=("$(submit "${CPU_OPTS[@]}" -c 2 -J mcmc-timing-cpu slurm/scalability_timing.sbatch cpu)")
  JOBS+=("$(submit "${GPU_OPTS[@]}" -G 1 -c 10 -J mcmc-timing-gpu slurm/scalability_timing.sbatch gpu)")
fi

# Tables and figures once every fit has finished successfully.
DEP=$(IFS=:; echo "${JOBS[*]}")
FINAL=$(submit "${CPU_OPTS[@]}" -J paper-outputs --dependency="afterok:${DEP}" slurm/make_paper_outputs.sbatch)

echo "Submitted ${#JOBS[@]} model jobs; tables and figures: job ${FINAL}"
echo "Outputs: \${RESULTS_DIR:-./output}/paper/{tables,figures}/"
