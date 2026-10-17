#!/bin/bash
# MCMC wall time vs number of training instances (Table S6): one user, five
# training-set sizes (100-500 instances), run one after another on the same
# device so the timings are comparable. Run once per device:
#   sbatch -p <cpu_partition> -c 2 slurm/scalability_timing.sbatch cpu
#   sbatch -p <gpu_partition> -G 1 -c 10 slurm/scalability_timing.sbatch gpu
# Results: $RESULTS_DIR/experiments/scalability_real_user37_<device>/
# (scalability_results.csv, scalability_fit.json), read by supplementary_tables.py.
#SBATCH --job-name=scalability-timing
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
DEVICE=${1:?usage: sbatch slurm/scalability_timing.sbatch cpu|gpu}
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
USER_ID=37
RESULTS=${RESULTS_DIR:-$PWD/output}
BATCH_DIR="$RESULTS/experiments/scalability_real_user${USER_ID}_${DEVICE}"
CUR="$RESULTS/scalability/user${USER_ID}_curation"
python scalability_timing.py --user-id "$USER_ID" --prep-only

if [ "$DEVICE" = cpu ]; then
  # Same configuration as the main-cohort fits: two CPU cores, no GPU.
  export CUDA_VISIBLE_DEVICES=""
  NUM_CPUS=2
  PIN=$(python -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:2])))")
  RUN=(taskset -c "$PIN")
else
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  NUM_CPUS=${SLURM_CPUS_PER_TASK:-1}
  RUN=()
fi

for N_TRAIN in 100 200 300 400 500; do
  ${RUN[@]+"${RUN[@]}"} python real_data_evaluation.py --model ar1 --use-carb-uncertainty --carb-uncertainty-type global \
    --inference-method mcmc --num-warmup 1000 --num-samples 1000 --sampling-mode chronological \
    --num-cpus "$NUM_CPUS" --data-path "$T1DEXI_DATA_DIR/processed_data/non_exercise_8h" \
    --curation-dir "$CUR/n${N_TRAIN}" --output-path "$BATCH_DIR/n${N_TRAIN}" \
    --all-users --target-user-id "$USER_ID"
done
python scalability_timing.py --user-id "$USER_ID" --results-only --batch-dir "$BATCH_DIR"
