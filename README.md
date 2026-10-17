# Decision-Relevant Evaluation Framework for Probabilistic Glucose Forecasting

This repository accompanies the paper:

> Guo R, Levine ME, Wu D, Zaharieva DP, Lal RA, Fox EB, Johari R.
> Clinical and Statistical Metrics Reveal Tradeoffs in Glucose Forecasting for
> Type 1 Diabetes Decision Support. 2026.

Code developed by Rachel Guo.

It reproduces the paper's experiments, tables and figures, fitting and evaluating
nine probabilistic glucose forecasting models on the T1DEXI dataset:

| Paper name | Code | Hardware used for the paper |
|---|---|---|
| Bayesian Loop | `models/ar1.py` + `models/loop_model.py` (`run_cohort.py --model global`) | CPU, 2 cores per user |
| Pure AR1 | `models/ar1_pure.py` (`--model ar1_pure`) | CPU, 2 cores per user |
| Adaptive AR1-RLS | `models/ar1_rls.py` (`--model ar1_rls`) | CPU, 2 cores per user |
| LSTM-MC | `models/lstm_mc.py` (`--model lstm`) | GPU |
| LSTM-Ensemble | `models/lstm_ensemble.py` (`--model lstm_ensemble`) | GPU |
| LSTM-Conformal | `models/lstm_conformal.py` (`--model lstm_conformal`) | CPU, 2 cores per user |
| Transformer-MC | `models/transformer_mc.py` (`--model transformer`) | GPU |
| Transformer-Ensemble | `models/transformer_ensemble.py` (`--model transformer_ensemble`) | GPU |
| Transformer-Conformal | `models/transformer_conformal.py` (`--model transformer_conformal`) | GPU |

Data preprocessing (raw T1DEXI files to curated 8-hour instances) lives in a
separate repository, [`t1dexi-preprocessing`](https://github.com/rguo99/t1dexi-preprocessing),
so it can be reused on its own. This repository takes the preprocessed output of
`t1dexi-preprocessing` as its input.

## Repository layout

| Files | Role |
|---|---|
| `config.py`, `eval_config.py` | Paths (from environment variables), the analysed cohort, and which model runs feed the evaluation |
| `run_cohort.py` | Fits one model for every user in the cohort (or one shard), one subprocess per user |
| `real_data_evaluation.py`, `InferenceRunner.py`, `run_experiments.py` | Per-user fitting entry point, the inference engine shared by all nine models, and the per-user job queue (plus the synthetic experiment) |
| `models/` | The nine models (table above); `models/cf_artifacts.py` saves and reloads trained neural weights |
| `utils/` | Model-input preparation, per-user result bundles, seeding |
| `synthetic/` | Synthetic-cohort generator for parameter recovery (seed 0, 50 users); writes the cohort and its ground-truth parameters under `$RESULTS_DIR/synthetic/` |
| `make_paper_outputs.py` | Runs every evaluation step below in order and collects the paper's tables and figures |
| `evaluate.py`, `paper_tables.py`, `extra_tables.py`, `supplementary_tables.py` | Metrics for all models, and the results and supplementary tables |
| `tier_table.py`, `scorecard_figure.py`, `qualitative_figure.py`, `figure_selection.py`, `figure_selection.json` | Tier heatmap and the instance-level figures (instances pinned in `figure_selection.json`) |
| `convergence.py`, `parameter_analysis.py`, `sensitivity.py`, `compute_parameter_recovery.py` | Bayesian Loop MCMC diagnostics, posterior parameters, prior sensitivity, synthetic parameter recovery |
| `heterogeneity_drivers.py`, `completeness_analysis.py`, `scalability_timing.py` | Per-user failure correlates, meal-reporting completeness, MCMC run time vs training-set size |
| `slurm/` | Slurm templates and the one-command driver `submit_all.sh` |

## 1. Data access

T1DEXI is available through [Vivli](https://vivli.org) (DOI 10.25934/PR00008428.1)
under a data use agreement. No T1DEXI data are included in this repository.
The only participant-level information is `figure_selection.json`, which
identifies the five example instances plotted in the paper's figures by T1DEXI
study ID and window start time, so that those figures can be regenerated
exactly. It contains no glucose, insulin, meal or other measurements.

1. Request T1DEXI from Vivli.
2. Run `t1dexi-preprocessing` at tag `v1.0.0` on the raw files (see its README).
3. Point this code at its output:

```bash
export T1DEXI_DATA_DIR=/path/to/preprocessing/output   # the folder used as T1DEXI_OUTPUT_DIR there
export T1DEXI_RAW_DIR=/path/to/T1DEXI                   # the same raw T1DEXI folder used there
export RESULTS_DIR=/path/to/results                     # optional, default ./output
```

Both variables point at folders `t1dexi-preprocessing` already uses, so no
files need copying or moving. `T1DEXI_RAW_DIR` is needed only for the
heterogeneity-driver tables, which read the `DM`, `LB` and `PR` parquet files
that the preprocessing pipeline's first step writes next to the raw `.xpt` files.

`config.py` documents the expected layout. All model runs, caches, tables and
figures are written under `$RESULTS_DIR`.

## 2. Installation

Python 3.11. The pinned versions are those used for the paper.

```bash
conda env create -f environment.yml     # GPU (CUDA 12) build
conda activate glucose-forecasting
# or, with pip:
pip install -r requirements.txt         # GPU (CUDA 12) build
pip install -r requirements-cpu.txt     # CPU-only build (no CUDA wheels)
```

**CPU vs GPU.** Bayesian Loop, Pure AR1, AR1-RLS and LSTM-Conformal were run on
CPU for the paper, and everything after model fitting (evaluation, tables,
figures) is CPU-only. The other five neural models were trained on NVIDIA
GPUs with CUDA 12. They also run on CPU, which is fine for a smoke test but slow
for the full cohort, and CPU results are not bit-identical to GPU results.

**Tested on** Linux (CentOS 7, glibc 2.17) with NVIDIA RTX 2080 Ti / L40S GPUs.
The pins (jax 0.7.0, torch 2.6.0, pandas 2.3.2, pyarrow 20.0.0) are the last
releases with wheels for glibc 2.17; newer versions should work on newer
systems but have not been verified.

**Threads.** Cap BLAS/OpenMP threads to the CPUs you allocate, e.g.
`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`; otherwise
they size their pools from the whole node.

## 3. Quickstart (two users, CPU, a few minutes per model)

```bash
export CUDA_VISIBLE_DEVICES=""
python run_cohort.py --model global   --only-uids 829,1012 --tag smoke
python run_cohort.py --model ar1_pure --only-uids 829,1012 --tag smoke
python run_cohort.py --model lstm     --only-uids 829,1012 --tag smoke
```

`--tag` writes to `$RESULTS_DIR/experiments/smoke_*`, which the evaluation
scripts ignore.

## 4. Full reproduction

### 4.1 Cohort

`eval_config.load_eligible_users` selects the analysed cohort from the
preprocessing output: pump users (88 multiple-daily-injection users excluded)
with at least 150 training and 10 test instances and no duplicate CGM
timestamps (2 users excluded). That gives the **395 users** reported in the
paper. The log of every run prints these counts.

### 4.2 Model runs

On a Slurm cluster, one command submits every fit and then builds the tables
and figures once all fits have finished (jobs are chained with
`--dependency=afterok`):

```bash
export T1DEXI_DATA_DIR=/path/to/t1dexi-preprocessing/output
export T1DEXI_RAW_DIR=/path/to/T1DEXI
export RESULTS_DIR=/path/to/results
CPU_PARTITION=<cpu_partition> GPU_PARTITION=<gpu_partition> bash slurm/submit_all.sh
```

It reads the cohort size from the data (395 users) and submits:

- Bayesian Loop at the main prior (0.30) and the two sensitivity priors
  (`--prior-scale 0.10` and `0.50`; the latter selects the loose prior, scale
  0.70, in `models/ar1.py`), Pure AR1, AR1-RLS and LSTM-Conformal as CPU array jobs, one user per task;
- the five GPU models in `GPU_SHARDS` (default 4) jobs each;
- the synthetic parameter-recovery run (generate 50 simulated users, then one user per array task);
- `make_paper_outputs.py`, which starts after everything above succeeds.

`DRY_RUN=1` prints the `sbatch` commands without submitting, `WITH_TIMING=1`
adds the MCMC wall-time experiment (Table S6), and `CPU_CONSTRAINT` /
`GPU_CONSTRAINT` select node types. The individual templates
(`slurm/cpu_model_array.sbatch`, `slurm/gpu_model_shard.sbatch`, ...) can also
be submitted by hand; their headers show how. A resubmitted or requeued job skips users that already have results
for that model and prior scale, so no user is fitted twice (`run_cohort.py --force`
refits them). Edit `slurm/_env.sh` to activate
your environment first.

Without Slurm, run the same steps in this order (each `run_cohort.py` call fits
the users one after another, so this takes days on one machine):

```bash
# CPU models (run on CPU for the paper; hide GPUs so they match it)
export CUDA_VISIBLE_DEVICES=""
for PRIOR in 0.30 0.10 0.50; do python run_cohort.py --model global --prior-scale $PRIOR --gpus 0; done
for MODEL in ar1_pure ar1_rls lstm_conformal; do python run_cohort.py --model $MODEL --gpus 0; done
python synthetic/generate_synthetic_loop.py --n 50 --seed 0
python run_experiments.py --synthetic --n 50 --gpus 0

# GPU models
unset CUDA_VISIBLE_DEVICES
for MODEL in lstm lstm_ensemble transformer transformer_ensemble transformer_conformal; do
  python run_cohort.py --model $MODEL --gpus 0      # GPU ids, e.g. 0,1
done

# Tables and figures (CPU only)
CUDA_VISIBLE_DEVICES="" python make_paper_outputs.py
```

Set the thread caps from section 2 first. The optional MCMC run-time table
(Table S6) is `slurm/scalability_timing.sbatch`, run once with `cpu` and once
with `gpu`.

Approximate cost for the full cohort, from the runs behind the paper:

| Step | Resources per task | Wall time |
|---|---|---|
| Bayesian Loop, per user (1000 warmup, 1000 samples, 4 chains) | 2 CPU cores, 12 GB | median 14 min, max 34 min |
| Pure AR1, per user | 2 CPU cores | 2-3 min |
| AR1-RLS, per user | 2 CPU cores | about 2 min |
| LSTM-Conformal, 1/10 of the cohort | 2 CPU cores | about 75 min |
| Each GPU model, 1/2 of the cohort | 2 GPUs, 8 CPU cores | 1.5-2.5 h |
| Evaluation and all tables/figures (`make_paper_outputs.py`) | 16 CPU cores, 64 GB | about 1.5 h |

Each model run saves the full posterior/predictive samples per user and, for
the neural models, the trained weights (`models/cf_artifacts.py`), so the
evaluation can be rerun without refitting.

### 4.3 Tables and figures

```bash
python make_paper_outputs.py      # or: sbatch slurm/make_paper_outputs.sbatch
```

This runs every analysis step in order (`STEPS` in the script) and collects
the outputs in `$RESULTS_DIR/paper/{figures,tables}/` under the file names the
manuscript uses:

| Paper item | File | Produced by |
|---|---|---|
| Figure 1 (schematic) | not generated by code | |
| Figure 2, single-instance scorecard | `fig_scorecard.pdf` | `scorecard_figure.py` |
| Figure 3, tier heatmap | `fig_tier_heatmap.pdf` | `tier_table.py` |
| Figure 4, hypoglycemia ROC | `fig_roc_hypo.pdf` | `evaluate.py` |
| Figure 5, metrics by horizon | `fig_horizons.pdf` | `evaluate.py` |
| Figure 6, metrics by instance type | `fig_stratified.pdf` | `evaluate.py` |
| Figure 7, per-user metric distributions | `fig_user_level_metric_distributions.pdf` | `evaluate.py` |
| Figure 8, representative instances | `fig_qualitative.pdf` | `qualitative_figure.py` |
| Table 1, main results (4 h) | `tab_main_results.tex` | `paper_tables.py` |
| 30-min results | `tab_main_results_30min.tex` | `extra_tables.py` |
| Instance percentiles (Figure 8) | `tab_instance_percentiles.tex` | `extra_tables.py` |
| Dataset summary | `tab_dataset.tex` | `evaluate.py` |
| MCMC convergence (tables, histograms) | `tab_convergence*.tex`, `convergence_histograms_*.pdf` | `convergence.py` |
| Posterior parameters | `parameter_table.tex`, `kde_parameter_distributions.pdf` | `parameter_analysis.py` |
| Prior sensitivity | `sensitivity_predictive.pdf`, `sensitivity_parameters.pdf` | `sensitivity.py` |
| Synthetic parameter recovery | `tab_param_recovery*.tex` | `compute_parameter_recovery.py` |
| Heterogeneity drivers (Bayesian Loop, all models) | `tab_heterogeneity_drivers.tex`, `tab_heterogeneity_drivers_all_models.tex` | `heterogeneity_drivers.py` |
| Heterogeneity driver directions | `tab_heterogeneity_driver_directions.tex` | `heterogeneity_drivers.py` |
| Completeness prevalence | `tab_completeness_prevalence.tex` | `completeness_analysis.py` + `supplementary_tables.py` |
| Completeness-stratified results | `tab_completeness_stratified.tex` | `completeness_analysis.py` + `paper_tables.py` |
| CGM composition of train/test forecast windows | `tab_glucose_composition.tex` | `supplementary_tables.py` |
| MCMC run time vs training-set size (Table S6) | `tab_scalability.tex` | `slurm/scalability_timing.sbatch` (cpu and gpu runs) + `supplementary_tables.py` |
| Instance-gap histogram | `fig_instance_gap_histogram.pdf` | `t1dexi-preprocessing`: `figures/instance_gap_histogram.py --metrics-df <metrics_df.pkl>` |

Inline tables of the manuscript text (metric definitions, curation rules) are
not data-derived. Counts in table captions (users, instances, severity bins)
are computed from the data when the tables are written. The run-time table
needs the optional timing runs (`WITH_TIMING=1` in `slurm/submit_all.sh`);
timings depend on the hardware.

The instances shown in Figures 2 and 8 are listed in `figure_selection.json`
(user id and window start time). `qualitative_figure.py --reselect` reruns the
automatic selection instead.

## 5. Reproducibility

- **Seeds.** Every model's random seed is derived from the user id and model
  name (`utils/seeding.py`), so reruns are deterministic on the same hardware
  and software. `BAYESIANUQ_SEED_BASE` shifts all seeds.
- **GPU determinism.** `run_cohort.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  for the per-user processes. GPU results are exact on the same GPU model and
  library versions, and statistically equivalent elsewhere.
- **Hardware dependence.** Floating-point results depend on the processor:
  rerunning this code on the same CPU or GPU model as the paper's runs
  reproduces every posterior and prediction sample bit for bit (checked for all
  nine models on a sample of users). A full rerun on whatever CPUs and GPUs were
  available gave identical Pure AR1 and AR1-RLS results, Bayesian Loop cohort
  means within 0.03, and neural-model means within about 0.2 mg/dL RMSE and
  1 percentage point for coverage, sensitivity and specificity; no conclusion
  changes. Quantities that threshold per-user values move more: the share of
  users with sensitivity or specificity below 0.10, and the per-model
  significance counts built on it in `tab_heterogeneity_drivers*.tex` and
  `tab_heterogeneity_driver_directions.tex`. In particular, LSTM-Ensemble's
  specificity and sensitivity failure rates are nearly equal (1.8% and 1.7% in
  the paper), so the metric its column is tested against can change on a refit.
  MCMC diagnostics (`tab_convergence*.tex`) and the MCMC run times (Table S6)
  also depend on the hardware.
- **Provenance.** Each saved result records the seed, library versions and
  device (`utils/seeding.env_provenance`).
- **Hardware behind the paper's results.** Bayesian Loop, Pure AR1, AR1-RLS
  and LSTM-Conformal ran on CPU (2 cores per user). The GPU models ran on
  whichever NVIDIA GPU was free, so the device varies by user:

  | Model | Users per GPU type |
  |---|---|
  | LSTM-MC | RTX 2080 Ti 132, L40S 132, V100-SXM2 131 |
  | LSTM-Ensemble | TITAN Xp 132, RTX 3090 132, RTX 2080 Ti 66, H100 65 |
  | Transformer-MC | TITAN Xp 198, L40S 197 |
  | Transformer-Ensemble | L40S 170, H100 135, RTX 3090 84, RTX 2080 Ti 5, TITAN Xp 1 |
  | Transformer-Conformal | H100 198, RTX 3090 167, RTX 2080 Ti 30 |

  The instances in the figures (`figure_selection.json`) were fitted on:

  | Figure instance (T1DEXI id) | LSTM-MC | LSTM-Ens | Transformer-MC | Transformer-Ens | Transformer-Conf |
  |---|---|---|---|---|---|
  | Figure 2 scorecard (829) | RTX 2080 Ti | TITAN Xp | L40S | RTX 3090 | RTX 3090 |
  | Figure 8 Meal (58) | V100-SXM2 | H100 | L40S | H100 | RTX 2080 Ti |
  | Figure 8 Carb (11) | V100-SXM2 | H100 | L40S | H100 | RTX 2080 Ti |
  | Figure 8 Correction (166) | V100-SXM2 | H100 | L40S | L40S | RTX 3090 |
  | Figure 8 Stable (707) | RTX 2080 Ti | TITAN Xp | L40S | L40S | RTX 3090 |

  Retraining on the same GPU type reproduces these forecasts exactly. On a
  different GPU type, a neural model's forecast for a single instance can change
  noticeably even though its averages do not: seeds fix the random draws, not the
  GPU's floating-point arithmetic (newer GPUs may also use TF32 in cuDNN), and
  small differences grow over training. For example, retraining LSTM-MC for user
  829 on an L40S changed its 4-hour RMSE on the Figure 2 instance from 32.4 to
  15.9 mg/dL, while that user's mean over all 150 test instances changed from
  21.6 to 21.9 mg/dL. Bayesian Loop and the AR1 baselines are not affected in
  this way.
- **Training protocols.** Transformer-MC uses early stopping disabled and
  validation-fold checkpoint selection (`run_cohort.py` sets
  `TRANSFORMER_PATIENCE_OVERRIDE` and `TRANSFORMER_USE_VAL_SELECTION_OVERRIDE`);
  the other neural models use patience 100 without validation selection.
- **Modeling simplification.** Bayesian Loop is driven by delivered basal
  insulin rather than net basal (delivered minus scheduled); see the Methods section on Bayesian Loop.

## 6. Citation

See `CITATION.cff`. If you use the preprocessing, please also cite the netIOB
work it builds on (see the `t1dexi-preprocessing` README).

Bayesian Loop's deterministic forecast (`models/loop_model.py`) is our own JAX
implementation of the Loop prediction algorithm. It draws on two sources:

- LoopKit Authors. *Loop Prediction Algorithm*. LoopDocs.
  <https://loopkit.github.io/loopdocs/operation/algorithm/prediction>
- Tidepool Project. *PyLoopKit*: a Python implementation of the Loop algorithm.
  <https://github.com/tidepool-org/PyLoopKit>

It simplifies and deviates from Loop in some ways so that it can be used as an
offline forecaster on logged data; the paper's Methods section on Bayesian Loop
describes how. Results for Bayesian Loop therefore describe this implementation,
not the Loop app or PyLoopKit.

## License

MIT (see `LICENSE`). PyLoopKit's BSD-2-Clause license is reproduced in
`THIRD_PARTY_NOTICES`.
