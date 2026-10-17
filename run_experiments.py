#!/usr/bin/env python3
"""
run_experiments.py
==================
Per-user job queue shared by the model runners, plus the synthetic
parameter-recovery experiment.

Each user is fitted by a separate real_data_evaluation.py subprocess; one
subprocess runs per listed GPU index, and the next user is started as soon
as one finishes. The real-data cohort runs are launched with run_cohort.py,
which imports the command builders and run_queue() from here.

Usage (synthetic parameter recovery, Bayesian Loop only):
  python synthetic/generate_synthetic_loop.py --n 50 --seed 0
  python run_experiments.py --synthetic --n 50 --gpus 0
  python run_experiments.py --synthetic --n 50 --gpus 0 --shard 3/50 --run-id r1   # one user
"""

import os
import glob
import time
import pickle
import argparse
import subprocess
from datetime import datetime

import config

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Real data MCMC settings
MCMC_WARMUP   = 1000
MCMC_SAMPLES  = 1000
NUM_CHAINS    = 4

# Synthetic MCMC settings — fewer needed since data is clean physics,
# no real-world noise/complexity, and we only need posterior means
SYN_MCMC_WARMUP  = 1000
SYN_MCMC_SAMPLES = 1000
SYN_NUM_CHAINS   = 4

# Size Ray to the CPUs Slurm actually gave us, rather than hardcoding 20.
# WHY: the `gpu` QOS caps a user at cpu=128 AND gres/gpu=16. At 20 CPUs per job
# you hit the CPU cap (QOSMaxCpuPerUserLimit) after ~6 jobs and never reach the
# 16-GPU ceiling, so jobs queue behind a limit that buys nothing -- Ray workers
# take num_cpus=2 each (see ray.remote(...) in real_data_evaluation.py), and a
# 2-GPU job is GPU-bound long before it is CPU-bound. Reading the allocation
# means `sbatch --cpus-per-task=10` now actually lowers the footprint instead of
# leaving Ray to claim 20 CPUs it cannot use.
NUM_CPUS      = int(os.environ.get('SLURM_CPUS_PER_TASK', 20))
SAMPLING_MODE = "chronological"

# RLS has no MCMC/SVI sampling loop and no GPU dependency, so it gets its
# own (much smaller) per-worker CPU budget -- letting more users run in
# parallel within the same total core allocation than the MCMC baselines need.
RLS_NUM_CPUS  = 4

PROJECT_DIR   = config.REPO_ROOT
SCRIPT        = "real_data_evaluation.py"
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.environ['EXPERIMENT_TIMESTAMP'] = RUN_TIMESTAMP

GPUS          = [0, 1, 2, 3, 4]
POLL_INTERVAL = 30

# ── SYNTHETIC PATHS ───────────────────────────────────────────────────────────
SYN_UID_START = 2001


# ── COMMAND BUILDERS ──────────────────────────────────────────────────────────

def global_cmd(user_id, data_path, curation_dir, batch_dir):
    """Real data command — full MCMC settings."""
    return (
        f"python {SCRIPT} "
        f"--model ar1 "
        f"--use-carb-uncertainty "
        f"--carb-uncertainty-type global "
        f"--inference-method mcmc "
        f"--num-warmup {MCMC_WARMUP} "
        f"--num-samples {MCMC_SAMPLES} "
        f"--num-chains {NUM_CHAINS} "
        f"--sampling-mode {SAMPLING_MODE} "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def global_cmd_synthetic(user_id, data_path, curation_dir, batch_dir):
    """Synthetic command — reduced MCMC settings for speed."""
    return (
        f"python {SCRIPT} "
        f"--model ar1 "
        f"--use-carb-uncertainty "
        f"--carb-uncertainty-type global "
        f"--inference-method mcmc "
        f"--num-warmup {SYN_MCMC_WARMUP} "
        f"--num-samples {SYN_MCMC_SAMPLES} "
        f"--num-chains {SYN_NUM_CHAINS} "
        f"--sampling-mode {SAMPLING_MODE} "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def lstm_cmd(user_id, data_path, curation_dir, batch_dir):
    return (
        f"python {SCRIPT} "
        f"--model lstm_mc "
        f"--inference-method svi "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def transformer_cmd(user_id, data_path, curation_dir, batch_dir):
    """
    Transformer-MC baseline — same 96-step input grid, MC-Dropout UQ, and
    per-user train/test split as lstm_cmd, but a higher-capacity attention
    architecture. (rules out LSTM
    capacity as a confound).
    """
    return (
        f"python {SCRIPT} "
        f"--model transformer_mc "
        f"--inference-method svi "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def lstm_ensemble_cmd(user_id, data_path, curation_dir, batch_dir):
    """
    LSTM deep-ensemble baseline (models/lstm_ensemble.py) — same 96-step
    input grid, architecture (hidden_dim=32), and per-user train/test split
    as lstm_cmd, but UQ comes from an ensemble of independently-trained
    members instead of MC-Dropout.
    (rule out MC-Dropout as a weak-UQ confound, separate from the
    capacity axis already covered by transformer_cmd).
    """
    return (
        f"python {SCRIPT} "
        f"--model lstm_ensemble "
        f"--inference-method svi "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def lstm_conformal_cmd(user_id, data_path, curation_dir, batch_dir):
    """
    LSTM split-conformal baseline (models/lstm_conformal.py) — same
    architecture as lstm_cmd; UQ comes from a split-conformal calibration
    fold carved out of the training period (never the test period, to
    avoid leaking test-period information and to keep the comparison fair
    against the other UQ methods, none of which see test data before
    forecasting)..
    """
    return (
        f"python {SCRIPT} "
        f"--model lstm_conformal "
        f"--inference-method svi "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def transformer_conformal_cmd(user_id, data_path, curation_dir, batch_dir):
    """
    Transformer split-conformal baseline (models/transformer_conformal.py) —
    same higher-capacity Transformer architecture as transformer_cmd; UQ
    comes from a split-conformal calibration fold, mirroring
    lstm_conformal_cmd's design exactly. Isolates whether the shared
    MC-Dropout failure mode seen in both LSTM_MC and Transformer_MC (fixed
    false-alarm-rate operating point) tracks the UQ method or the
    architecture, by holding architecture fixed and swapping only UQ method.
    """
    return (
        f"python {SCRIPT} "
        f"--model transformer_conformal "
        f"--inference-method svi "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def transformer_ensemble_cmd(user_id, data_path, curation_dir, batch_dir):
    """
    Transformer deep-ensemble baseline (models/transformer_ensemble.py) — the
    diagonal cell of the architecture x UQ-method 2x2. LSTM-Ensemble is the
    strongest non-Bayesian-Loop model in the current field, so Transformer +
    deep ensemble is the configuration most likely to beat Bayesian Loop at
    the <=2hr horizons where the paper's central clinical claim lives. Exists
    to test that claim adversarially rather than assume it.
    """
    return (
        f"python {SCRIPT} "
        f"--model transformer_ensemble "
        f"--inference-method svi "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def ar1_baseline_cmd(user_id, data_path, curation_dir, batch_dir):
    """
    AR1 pure baseline command — no carb/insulin uncertainty modelling.
    Uses MCMC with the same warmup/sample settings as Global for fair comparison.
    Model script lives in the same models/ folder as ar1 and lstm_mc.
    """
    return (
        f"python {SCRIPT} "
        f"--model ar1_pure "
        f"--inference-method mcmc "
        f"--num-warmup {MCMC_WARMUP} "
        f"--num-samples {MCMC_SAMPLES} "
        f"--num-chains {NUM_CHAINS} "
        f"--sampling-mode {SAMPLING_MODE} "
        f"--num-cpus {NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


def rls_ar1_baseline_cmd(user_id, data_path, curation_dir, batch_dir):
    """
    Adaptive AR(1)-via-RLS baseline command (models/ar1_rls.py).
    No MCMC/SVI sampling happens for this model — InferenceRunner short-
    circuits to a closed-form per-instance RLS fit — but we still pass
    --inference-method mcmc so real_data_evaluation.py takes the simple
    sequential per-user code path (no Ray SVI parallelism needed).
    """
    return (
        f"python {SCRIPT} "
        f"--model ar1_rls "
        f"--inference-method mcmc "
        f"--sampling-mode {SAMPLING_MODE} "
        f"--num-cpus {RLS_NUM_CPUS} "
        f"--data-path {data_path} "
        f"--curation-dir {curation_dir} "
        f"--output-path {batch_dir} "
        f"--all-users "
        f"--target-user-id {user_id}"
    )


# ── DYNAMIC GPU QUEUE ─────────────────────────────────────────────────────────

def run_queue(user_ids, cmd_fn, model_name, batch_dir, gpus,
              dry_run=False, synthetic=False, prior_scale=0.30):
    log_dir = os.path.join(batch_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    gpu_procs = {g: None for g in gpus}
    queue     = list(user_ids)
    completed = []
    failed    = []

    print(f"\n{'='*60}")
    print(f"[{model_name}] {len(queue)} users, {len(gpus)} GPUs")
    print(f"{'='*60}")

    while queue or any(v is not None for v in gpu_procs.values()):

        # Check finished jobs
        for gpu, slot in list(gpu_procs.items()):
            if slot is None:
                continue
            proc, uid, start_time = slot
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start_time
                if ret == 0:
                    completed.append(uid)
                    print(f"  GPU{gpu} done user {uid} ({elapsed/60:.1f}min)  "
                          f"[{len(completed)}/{len(completed)+len(failed)+len(queue)}]")
                else:
                    failed.append(uid)
                    print(f"  ERROR: GPU{gpu} FAILED user {uid} "
                          f"(exit={ret}, {elapsed/60:.1f}min)")
                gpu_procs[gpu] = None

        # Assign free GPUs
        for gpu, slot in list(gpu_procs.items()):
            if slot is not None or not queue:
                continue
            uid      = queue.pop(0)
            cmd      = cmd_fn(uid)
            log_path = os.path.join(
                log_dir, f"{model_name}_user{uid}_gpu{gpu}.log")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"]          = str(gpu)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["EXPERIMENT_TIMESTAMP"]          = RUN_TIMESTAMP
            env["PRIOR_SCALE"]                   = str(prior_scale)
            if synthetic:
                env["SYNTHETIC_DATA_MODE"] = "true"

            print(f"  GPU{gpu} <- user {uid}  ({len(queue)} remaining)")

            if dry_run:
                print(f"     [DRY RUN] {cmd}")
                gpu_procs[gpu] = None
                completed.append(uid)
            else:
                with open(log_path, 'w') as logf:
                    proc = subprocess.Popen(
                        cmd, shell=True, env=env,
                        stdout=logf, stderr=logf,
                        cwd=PROJECT_DIR
                    )
                gpu_procs[gpu] = (proc, uid, time.time())

        if any(v is not None for v in gpu_procs.values()) or queue:
            if not dry_run:
                time.sleep(POLL_INTERVAL)

    print(f"\n[{model_name}] Complete.")
    print(f"  Completed : {len(completed)}")
    if failed:
        print(f"  Failed    : {len(failed)} -> {failed}")
    return completed, failed


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--synthetic',   action='store_true', required=True,
                        help='Run synthetic parameter recovery (Bayesian Loop only). '
                             'Real-data runs use run_cohort.py.')
    parser.add_argument('--n',           type=int, default=50,
                        help='Synthetic cohort size (default: 50)')
    parser.add_argument('--gpus',        type=str, default='0',
                        help='Comma-separated GPU IDs, one concurrent user per ID (default: 0)')
    parser.add_argument('--shard',       type=str, default=None,
                        help='k/N: fit only the k-th of N slices of the synthetic users (1-indexed)')
    parser.add_argument('--run-id',      type=str, default=None,
                        help='Run directory suffix (default: timestamp); give every shard the same id')
    parser.add_argument('--dry-run',     action='store_true',
                        help='Print commands without launching')
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(',')]

    # One subprocess runs per GPU index concurrently; split the CPU allocation
    # between them. Floor of 2 because Ray workers request num_cpus=2.
    global NUM_CPUS
    _total_cpus = NUM_CPUS
    NUM_CPUS = max(2, _total_cpus // len(gpus))
    print(f"CPU budget : {_total_cpus} allocated / {len(gpus)} concurrent "
          f"subprocess(es) -> --num-cpus {NUM_CPUS} each")

    import eval_config
    n = args.n
    # Array tasks pass the same --run-id so they all write into one run directory.
    global RUN_TIMESTAMP
    if args.run_id:
        RUN_TIMESTAMP = args.run_id
    syn_data, syn_cur = eval_config.synthetic_paths(n)
    batch_dir = os.path.join(config.EXPERIMENTS_DIR, f"synthetic_recovery_n{n}_{RUN_TIMESTAMP}")
    user_ids  = list(range(SYN_UID_START, SYN_UID_START + n))
    if args.shard:
        k, nsh = (int(x) for x in args.shard.split("/"))
        assert 1 <= k <= nsh
        user_ids = user_ids[(n * (k - 1)) // nsh:(n * k) // nsh]
        print(f"SHARD {k}/{nsh}: users {user_ids}")

    if not os.path.exists(os.path.join(syn_data, "by_user", "cgm")) or not os.path.exists(syn_cur):
        print(f"[ERROR] Synthetic data not found: {syn_data}")
        print(f"  Run: python synthetic/generate_synthetic_loop.py --n {n} --seed 0")
        raise SystemExit(1)

    print(f"SYNTHETIC PARAMETER RECOVERY — N={n} users, Bayesian Loop only")
    print(f"Data      : {syn_data}")
    print(f"Curation  : {syn_cur}")
    print(f"Batch dir : {batch_dir}")
    print(f"GPUs      : {gpus}")
    print(f"MCMC      : warmup={SYN_MCMC_WARMUP} samples={SYN_MCMC_SAMPLES} chains={SYN_NUM_CHAINS}")

    os.makedirs(os.path.join(batch_dir, "logs"), exist_ok=True)
    cmd_fn = lambda uid: global_cmd_synthetic(uid, syn_data, syn_cur, batch_dir)  # noqa: E731
    _, failed = run_queue(user_ids, cmd_fn, "Synthetic_Global", batch_dir, gpus,
                          dry_run=args.dry_run, synthetic=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
