# Shared setup sourced by the Slurm templates. Edit to match your system.
# Activate the environment from environment.yml, e.g.:
#   source /path/to/conda/bin/activate glucose-forecasting
export T1DEXI_DATA_DIR=${T1DEXI_DATA_DIR:?set T1DEXI_DATA_DIR to the t1dexi-preprocessing output root}
# export RESULTS_DIR=/path/to/results
# Cap BLAS/OpenMP thread pools to the allocation (they otherwise size
# themselves from the node's core count).
export OMP_NUM_THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
