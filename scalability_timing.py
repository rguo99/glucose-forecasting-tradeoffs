#!/usr/bin/env python3
"""
scalability_timing.py
====================================
Real-data scalability experiment: run Global MCMC on a single real user
at N=100, 200, 300, 400, 500 TRAIN instances simultaneously across 5 GPUs.

Instances are selected with 50/50 meal/non-meal balance using the same
classification logic as curate_data_v5.py.

Train/test split:
  - TRAIN: N randomly selected instances (50/50 meal/non-meal) from
           the chronologically EARLIER 70% of all valid instances
  - TEST:  Fixed 150 instances randomly selected from the chronologically
           LATER 30% of all valid instances (same across all N values,
           so test set is identical for fair comparison)

This mirrors the approach in curate_data_v6.py and gives a proper
held-out test set for UQ evaluation via get_metrics.py.

Steps:
  1. Load and classify all instances for the target user
  2. Create curation pkls (train varies by N, test is fixed)
  3. Launch all 5 jobs simultaneously, one per GPU
  4. Collect runtimes + evaluate UQ metrics on test set
  5. Save scalability_results.csv + metrics summary

Usage:
  python scalability_timing.py
  python scalability_timing.py --user-id 42
  python scalability_timing.py --dry-run
  python scalability_timing.py --prep-only
  python scalability_timing.py --results-only --batch-dir output/experiments/scalability_real_user37_YYYY-MM-DD_HH-MM-SS
"""

import os
import time
import json
import glob
import pickle
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

# CONFIG
import config

PROJECT_DIR      = config.REPO_ROOT
SCRIPT           = "real_data_evaluation.py"
RUN_TIMESTAMP    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.environ['EXPERIMENT_TIMESTAMP'] = RUN_TIMESTAMP

INSTANCE_COUNTS  = [100, 200, 300, 400, 500]
GPUS             = [0, 1, 2, 3, 4]

MCMC_WARMUP      = 1000
MCMC_SAMPLES     = 1000
NUM_CPUS         = 10
SAMPLING_MODE    = "chronological"

DEFAULT_USER_ID  = "37"
DEFAULT_DATA_DIR = None   # defaults to config.data_dir()
RANDOM_SEED      = 42

TRAIN_FRAC       = 0.70   # earlier 70% to the train pool, later 30% to the test pool
TEST_SIZE        = 150    # fixed test size across all N values

# Classification thresholds — identical to curate_data_v5/v6
MIN_CGM_POINTS      = 48
UNEXPLAINED_RISE_TH = 40.0
MEAL_WINDOW_MIN     = -30
MEAL_WINDOW_MAX     =  60
MIN_CARBS           = 10.0
MIN_BOLUS           = 0.5


# LOAD AND CLASSIFY

def load_user_instances(user_id, data_dir):
    inst_path = os.path.join(data_dir, 'non_exercise.csv')
    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"Instance file not found: {inst_path}")
    df     = pd.read_csv(inst_path)
    id_col = next((c for c in ['user_id', 'id', 'patient_id'] if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"No user ID column in {inst_path}.")
    df[id_col] = df[id_col].astype(str).str.split('.').str[0]
    user_df    = df[df[id_col] == str(user_id)].copy()
    if user_df.empty:
        raise ValueError(f"User {user_id} not found.")
    user_df['time_series_start'] = pd.to_datetime(user_df['time_series_start'])
    user_df = user_df.sort_values('time_series_start').reset_index(drop=True)
    user_df['ex'] = user_df['ex'].astype(int)
    print(f"  Found {len(user_df):,} total instances for user {user_id}")
    return user_df


def load_user_features(user_id, data_dir):
    by_user_dir = os.path.join(data_dir, 'by_user')
    features    = {}
    for name in ['cgm', 'meals', 'bolus', 'basal']:
        fp = os.path.join(by_user_dir, name, f'{user_id}.parquet')
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            features[name] = {
                eid: grp.reset_index(drop=True)
                for eid, grp in df.groupby('ex')
            }
            print(f"  Loaded {name:8s}: {len(df):>7,} rows  "
                  f"across {len(features[name]):>5,} instances")
        else:
            features[name] = {}
    return features


def classify_instance(eid, features):
    cgm   = features.get('cgm',   {}).get(eid, pd.DataFrame())
    meals = features.get('meals', {}).get(eid, pd.DataFrame())
    bolus = features.get('bolus', {}).get(eid, pd.DataFrame())
    if cgm.empty or len(cgm) < MIN_CGM_POINTS:
        return 'BAD'
    time_col = next(
        (c for c in ['timestamp', 'datestamp', 'time'] if c in cgm.columns), None)
    if not time_col:
        return 'BAD'
    cgm           = cgm.copy()
    cgm[time_col] = pd.to_datetime(cgm[time_col], format='mixed')
    cgm           = cgm.sort_values(time_col)
    t0            = cgm.iloc[47][time_col]
    future_cgm    = cgm[cgm[time_col] >= t0]['cgm'].values
    if len(future_cgm) < 12:
        return 'BAD'
    rise        = np.max(future_cgm) - future_cgm[0]
    total_carbs = 0.0
    if not meals.empty:
        m_col = next(
            (c for c in ['timestamp', 'datestamp', 'time'] if c in meals.columns), None)
        if m_col:
            meals        = meals.copy()
            meals[m_col] = pd.to_datetime(meals[m_col])
            meals['rel'] = (meals[m_col] - t0).dt.total_seconds() / 60.0
            relevant     = meals[(meals['rel'] >= MEAL_WINDOW_MIN) &
                                  (meals['rel'] <= MEAL_WINDOW_MAX)]
            total_carbs  = (relevant['meal_dose'].sum()
                            if 'meal_dose' in relevant.columns else 0.0)
    if rise > UNEXPLAINED_RISE_TH and total_carbs < 5.0:
        return 'BAD'
    if total_carbs >= MIN_CARBS:
        return 'MEAL'
    total_bolus = 0.0
    if not bolus.empty:
        b_col = next(
            (c for c in ['timestamp', 'datestamp', 'time'] if c in bolus.columns), None)
        if b_col:
            bolus        = bolus.copy()
            bolus[b_col] = pd.to_datetime(bolus[b_col])
            bolus['rel'] = (bolus[b_col] - t0).dt.total_seconds() / 60.0
            relevant_b   = bolus[(bolus['rel'] >= MEAL_WINDOW_MIN) &
                                  (bolus['rel'] <= MEAL_WINDOW_MAX)]
            total_bolus  = (relevant_b['bolus'].sum()
                            if 'bolus' in relevant_b.columns else 0.0)
    if total_bolus >= MIN_BOLUS or (rise < 20.0 and np.std(future_cgm) < 20.0):
        return 'NON_MEAL'
    return 'BAD'


def build_label_cache(all_eids, features):
    cache = {}
    for i, eid in enumerate(all_eids):
        cache[eid] = classify_instance(eid, features)
        if (i + 1) % 500 == 0 or (i + 1) == len(all_eids):
            print(f"  Classified {i+1}/{len(all_eids)}...", flush=True)
    return cache


# SELECTION

def sample_balanced(pool_eids, label_cache, n, rng):
    """Random 50/50 meal/non-meal sample from pool. Graceful degradation."""
    meals     = [e for e in pool_eids if label_cache.get(e) == 'MEAL']
    non_meals = [e for e in pool_eids if label_cache.get(e) == 'NON_MEAL']
    n_meal    = min(n // 2,       len(meals))
    n_nonmeal = min(n - n_meal,   len(non_meals))
    selected  = []
    if n_meal > 0:
        selected += rng.choice(meals,     size=n_meal,    replace=False).tolist()
    if n_nonmeal > 0:
        selected += rng.choice(non_meals, size=n_nonmeal, replace=False).tolist()
    return sorted(selected)


def make_curation_pkls(user_id, label_cache, eid_to_time, curation_base):
    """
    Build one curation subdir per N.

    Train pool: chronologically earlier TRAIN_FRAC of valid instances
    Test pool:  chronologically later (1-TRAIN_FRAC) of valid instances

    Train varies by N (100, 200, ... 500).
    Test is FIXED at TEST_SIZE across all N — same held-out instances
    for fair comparison of UQ metrics across training set sizes.

    No filtered_user_instances.pkl is saved — this ensures filter_df
    in real_data_evaluation.py returns all instances unfiltered, matching
    the scalability experiment behavior that gave realistic ISF recovery.
    """
    counts = Counter(label_cache.values())
    print(f"\n  Label counts: MEAL={counts['MEAL']}  "
          f"NON_MEAL={counts['NON_MEAL']}  BAD={counts['BAD']}")

    # Chronological 70/30 split of all valid instances
    all_valid = sorted(
        [eid for eid, lbl in label_cache.items()
         if lbl != 'BAD' and eid in eid_to_time]
    )
    cut        = int(len(all_valid) * TRAIN_FRAC)
    train_pool = all_valid[:cut]
    test_pool  = all_valid[cut:]

    print(f"  Train pool: {len(train_pool)} instances (earlier {int(TRAIN_FRAC*100)}%)")
    print(f"  Test pool:  {len(test_pool)} instances (later {int((1-TRAIN_FRAC)*100)}%)")

    # Fixed test set — same across all N for fair comparison
    rng_test  = np.random.default_rng(RANDOM_SEED)
    test_ids  = sample_balanced(test_pool, label_cache, TEST_SIZE, rng_test)
    test_labels = Counter(label_cache[e] for e in test_ids)
    print(f"  Fixed test : {len(test_ids)} instances  "
          f"[MEAL={test_labels['MEAL']}  NON_MEAL={test_labels['NON_MEAL']}]")

    # Always regenerate
    if os.path.exists(curation_base):
        shutil.rmtree(curation_base)

    curation_dirs = []
    for n_train in INSTANCE_COUNTS:
        if counts['MEAL'] < n_train // 2 or \
                counts['NON_MEAL'] < n_train - n_train // 2:
            print(f"  N={n_train:>3}: SKIPPED — insufficient instances")
            continue

        # Train: random balanced from train pool, different seed per N
        rng_train  = np.random.default_rng(RANDOM_SEED + n_train)
        train_ids  = sample_balanced(train_pool, label_cache, n_train, rng_train)
        train_lbls = Counter(label_cache[e] for e in train_ids)

        # Verify no overlap between train and test
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            print(f"  N={n_train:>3}: WARNING — {len(overlap)} train/test overlaps removed")
            train_ids = [e for e in train_ids if e not in overlap]

        print(f"  N={n_train:>3}: train={len(train_ids)}  "
              f"[MEAL={train_lbls['MEAL']}  NON_MEAL={train_lbls['NON_MEAL']}]  "
              f"test={len(test_ids)}")

        cur_dir = os.path.join(curation_base, f"n{n_train}")
        os.makedirs(cur_dir, exist_ok=True)

        # NOTE: No filtered_user_instances.pkl — filter_df will return all
        # instances unfiltered, so inference sees the full instance pool.
        # train/test split is handled by InferenceRunner via these pkls.
        with open(os.path.join(cur_dir, 'train_instances.pkl'), 'wb') as f:
            pickle.dump(train_ids, f)
        with open(os.path.join(cur_dir, 'test_instances.pkl'), 'wb') as f:
            pickle.dump(test_ids, f)

        # filtered_user_instances.pkl contains ONLY train+test so InferenceRunner
        # can identify this user's instances — but we name it differently to avoid
        # filter_df picking it up at the hardcoded paths
        combined = {str(user_id): sorted(set(train_ids + test_ids))}
        with open(os.path.join(cur_dir, 'curation_instances.pkl'), 'wb') as f:
            pickle.dump(combined, f)

        curation_dirs.append((n_train, cur_dir))

    return curation_dirs, test_ids


# LAUNCH

def launch_all(user_id, data_dir, curation_dirs, batch_dir, dry_run=False):
    log_dir = os.path.join(PROJECT_DIR, batch_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = {}
    for (n_train, cur_dir), gpu in zip(curation_dirs, GPUS):
        job_out_dir = os.path.join(batch_dir, f"n{n_train}")
        cmd = (
            f"python {SCRIPT} "
            f"--model ar1 "
            f"--use-carb-uncertainty "
            f"--carb-uncertainty-type global "
            f"--inference-method mcmc "
            f"--num-warmup {MCMC_WARMUP} "
            f"--num-samples {MCMC_SAMPLES} "
            f"--sampling-mode {SAMPLING_MODE} "
            f"--num-cpus {NUM_CPUS} "
            f"--data-path {data_dir} "
            f"--curation-dir {cur_dir} "
            f"--output-path {job_out_dir} "
            f"--all-users "
            f"--target-user-id {user_id}"
        )
        log_path = os.path.join(log_dir, f"scale_n{n_train}_gpu{gpu}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]          = str(gpu)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["EXPERIMENT_TIMESTAMP"]          = RUN_TIMESTAMP

        print(f"  GPU{gpu} <- N={n_train} train  ->  {job_out_dir}")
        if dry_run:
            print(f"     [DRY RUN] {cmd}")
            procs[n_train] = None
        else:
            with open(log_path, 'w') as logf:
                proc = subprocess.Popen(
                    cmd, shell=True, env=env,
                    stdout=logf, stderr=logf,
                    cwd=PROJECT_DIR
                )
            procs[n_train] = (proc, time.time(), gpu)

    if dry_run:
        return {}

    print(f"\nWaiting for all {len(procs)} jobs to complete...")
    runtimes = {}
    while procs:
        for n, slot in list(procs.items()):
            proc, start_time, gpu = slot
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start_time
                status  = "" if ret == 0 else "FAILED"
                print(f"  {status} N={n} done (GPU{gpu}, "
                      f"{elapsed/60:.1f}min, exit={ret})")
                runtimes[n] = elapsed
                del procs[n]
        if procs:
            time.sleep(30)
    return runtimes


# COLLECT RESULTS AND EVALUATE METRICS

def collect_results(user_id, batch_dir):
    """
    Read sampling_duration_sec and UQ metrics from each job's result bundle.
    Returns list of dicts with runtime + key metrics per N.
    """
    results = []
    for n_train in INSTANCE_COUNTS:
        job_dir = os.path.join(PROJECT_DIR, batch_dir, f"n{n_train}")
        pattern = os.path.join(job_dir, "*", "results",
                               f"user_{user_id}_results.pkl")
        found = False
        for pkl_path in glob.glob(pattern):
            try:
                with open(pkl_path, 'rb') as f:
                    bundle = pickle.load(f)

                rt      = bundle.get('sampling_duration_sec')
                n_tr    = len(bundle.get('train_indices', []))
                n_te    = len(bundle.get('test_indices',  []))

                if rt is None:
                    continue

                row = {
                    'n_train':     n_tr   if n_tr > 0 else n_train,
                    'n_test':      n_te,
                    'runtime_sec': rt,
                    'runtime_min': rt / 60,
                }

                # Compute UQ metrics on test set
                all_preds = bundle.get('all_preds', bundle.get('all_preds_cpu', {}))
                obs_arr   = bundle.get('obs_glucose',
                            bundle.get('observed_glucose_array'))
                test_idx  = np.asarray(bundle.get('test_indices',
                            bundle.get('test_idx', [])))

                if all_preds and obs_arr is not None and len(test_idx) > 0:
                    samples = np.asarray(
                        all_preds.get('obs', all_preds.get('pred_mean')))
                    if samples.ndim == 2:
                        samples = samples[np.newaxis, :, :]

                    # Align samples and obs to test set
                    if samples.shape[1] == len(obs_arr):
                        test_samp = samples[:, test_idx, :]
                        test_obs  = obs_arr[test_idx]
                    elif samples.shape[1] == len(test_idx):
                        test_samp = samples
                        test_obs  = obs_arr[test_idx]
                    else:
                        test_samp = None

                    if test_samp is not None and len(test_obs) > 0:
                        mean_pred = np.mean(test_samp, axis=0)  # (n_test, T)
                        lo        = np.percentile(test_samp, 2.5,  axis=0)
                        hi        = np.percentile(test_samp, 97.5, axis=0)

                        T = min(48, test_obs.shape[1], mean_pred.shape[1])

                        # RMSE
                        row['RMSE'] = float(np.sqrt(np.mean(
                            (test_obs[:, :T] - mean_pred[:, :T]) ** 2)))

                        # Coverage (95% PI)
                        row['Coverage'] = float(np.mean(
                            (test_obs[:, :T] >= lo[:, :T]) &
                            (test_obs[:, :T] <= hi[:, :T])))

                        # PI Width
                        row['Width'] = float(np.mean(hi[:, :T] - lo[:, :T]))

                        # CRPS (mean across instances and time)
                        crps_vals = []
                        for i in range(min(test_obs.shape[0], 50)):  # cap at 50 for speed
                            for t in range(T):
                                s = test_samp[:, i, t]
                                crps_vals.append(
                                    float(np.mean(np.abs(s - test_obs[i, t]))
                                          - 0.5 * np.mean(np.abs(s[1:] - s[:-1]))))
                        row['CRPS'] = float(np.mean(crps_vals)) if crps_vals else np.nan

                        # ISF posterior mean (key diagnostic)
                        param_samples = bundle.get('param_samples', {})
                        map_params    = bundle.get('map_params', {})
                        for k in ['theta_ISF', 'ISF', 'isf']:
                            v = param_samples.get(k, map_params.get(k))
                            if v is not None:
                                row['ISF_mean']   = float(np.mean(v))
                                row['ISF_median'] = float(np.median(v))
                                row['ISF_std']    = float(np.std(v))
                                break
                        for k in ['theta_CR', 'CR', 'cr']:
                            v = param_samples.get(k, map_params.get(k))
                            if v is not None:
                                row['CR_mean'] = float(np.mean(v))
                                break

                results.append(row)
                found = True
                break

            except Exception as e:
                print(f"  Warning: could not load {pkl_path}: {e}")

        if not found:
            print(f"  Warning: no result bundle found for N={n_train}")

    return results


# REPORT

def report_scaling(user_id, results, out_dir):
    if not results:
        print("No results to report.")
        return

    df = pd.DataFrame(results).sort_values('n_train')
    df.to_csv(os.path.join(out_dir, "scalability_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(out_dir, 'scalability_results.csv')}")

    # Runtime table
    print(f"\n{'='*70}")
    print(f"SCALABILITY RESULTS — REAL DATA (user {user_id})")
    print(f"{'='*70}")

    has_metrics = 'RMSE' in df.columns
    header = f"{'N train':<10} {'N test':<8} {'Runtime (min)':<16} {'Sec/inst':<10}"
    if has_metrics:
        header += f" {'RMSE':<8} {'Coverage':<10} {'Width':<8} {'CRPS':<8}"
        if 'ISF_mean' in df.columns:
            header += f" {'ISF mean':<10} {'ISF std':<8}"
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        line = (f"{row['n_train']:<10.0f} "
                f"{row.get('n_test', 0):<8.0f} "
                f"{row['runtime_min']:<16.1f} "
                f"{row['runtime_sec']/row['n_train']:<10.2f}")
        if has_metrics:
            line += (f" {row.get('RMSE', float('nan')):<8.2f}"
                     f" {row.get('Coverage', float('nan')):<10.3f}"
                     f" {row.get('Width', float('nan')):<8.1f}"
                     f" {row.get('CRPS', float('nan')):<8.2f}")
            if 'ISF_mean' in df.columns:
                line += (f" {row.get('ISF_mean', float('nan')):<10.1f}"
                         f" {row.get('ISF_std', float('nan')):<8.1f}")
        print(line)

    # Linear fit on runtime
    if len(df) >= 3:
        x      = df['n_train'].values.astype(float)
        y      = df['runtime_sec'].values.astype(float)
        A      = np.vstack([x, np.ones(len(x))]).T
        m, b   = np.linalg.lstsq(A, y, rcond=None)[0]
        y_pred = m * x + b
        r2     = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        resid  = np.std(y - y_pred)

        print(f"\n  Runtime linear fit: {m:.1f}s × N_train + {b:.1f}s")
        print(f"  R² = {r2:.4f}   residual std = {resid:.1f}s")

        fit = {
            'user_id':                      user_id,
            'slope_sec_per_train_instance': m,
            'intercept_sec':                b,
            'r2':                           r2,
            'residual_std_sec':             resid,
            'n_points':                     len(df),
            'n_train_range':                [int(x.min()), int(x.max())],
        }
        with open(os.path.join(out_dir, "scalability_fit.json"), 'w') as f:
            json.dump(fit, f, indent=2)

    # UQ metrics trend
    if has_metrics and 'RMSE' in df.columns:
        print(f"\n  UQ metrics vs N_train (test set = fixed {TEST_SIZE} instances):")
        print(f"  Does more training data improve prediction quality?")
        rmse_vals = df['RMSE'].values
        if not np.all(np.isnan(rmse_vals)):
            trend = "improving" if rmse_vals[-1] < rmse_vals[0] else "flat/mixed"
            print(f"  RMSE trend: {rmse_vals[0]:.2f} -> {rmse_vals[-1]:.2f}  {trend}")
        if 'ISF_mean' in df.columns:
            print(f"  ISF estimates: {df['ISF_mean'].values}")
            print(f"  (Realistic ISF for T1D: 20-80 mg/dL/U)")

    print(f"{'='*70}")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--user-id',      type=str, default=DEFAULT_USER_ID)
    parser.add_argument('--data-dir',     type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--dry-run',      action='store_true')
    parser.add_argument('--prep-only',    action='store_true')
    parser.add_argument('--results-only', action='store_true')
    parser.add_argument('--batch-dir',    type=str, default=None)
    args = parser.parse_args()

    user_id       = args.user_id
    data_dir      = args.data_dir or config.data_dir()
    abs_data_dir  = os.path.join(PROJECT_DIR, data_dir)
    curation_base = os.path.join(config.RESULTS_DIR, "scalability",
                                 f"user{user_id}_curation")
    batch_dir     = args.batch_dir or os.path.join(
        config.EXPERIMENTS_DIR, f"scalability_real_user{user_id}_{RUN_TIMESTAMP}")
    abs_batch_dir = os.path.join(PROJECT_DIR, batch_dir)
    os.makedirs(abs_batch_dir, exist_ok=True)

    if args.results_only:
        if args.batch_dir is None:
            parser.error("--results-only requires --batch-dir")
        results = collect_results(user_id, batch_dir)
        report_scaling(user_id, results, abs_batch_dir)
        return

    print(f"\n{'='*60}")
    print(f"REAL-DATA SCALABILITY EXPERIMENT")
    print(f"  User ID         : {user_id}")
    print(f"  Data dir        : {data_dir}")
    print(f"  Instance counts : {INSTANCE_COUNTS}  (N = train instances)")
    print(f"  Train pool      : earlier {int(TRAIN_FRAC*100)}% of valid instances")
    print(f"  Test set        : fixed {TEST_SIZE} from later {int((1-TRAIN_FRAC)*100)}%")
    print(f"  Random seed     : {RANDOM_SEED}")
    if not args.prep_only:
        print(f"  Batch dir       : {batch_dir}")
    print(f"{'='*60}\n")

    print("Step 1: Loading instance list...")
    user_df = load_user_instances(user_id, abs_data_dir)

    print("\nStep 2: Loading per-user feature parquets...")
    features = load_user_features(user_id, abs_data_dir)

    print(f"\nStep 3: Classifying {len(user_df)} instances...")
    label_cache = build_label_cache(user_df['ex'].tolist(), features)
    eid_to_time = dict(zip(user_df['ex'], user_df['time_series_start']))

    print(f"\nStep 4: Creating curation pkls (always fresh)...")
    curation_dirs, test_ids = make_curation_pkls(
        user_id, label_cache, eid_to_time, curation_base)

    if not curation_dirs:
        print("[ERROR] No viable N values.")
        return

    if args.prep_only:
        print(f"\nCuration pkls written to {curation_base}")
        return

    if args.dry_run:
        print(f"\nStep 5: Launching jobs (DRY RUN)...")
        launch_all(user_id, data_dir, curation_dirs, batch_dir, dry_run=True)
        print("\n[DRY RUN complete]")
        return

    print(f"\nStep 5: Launching {len(curation_dirs)} parallel jobs...")
    wall_runtimes = launch_all(user_id, data_dir, curation_dirs, batch_dir)

    print(f"\nStep 6: Collecting results and computing UQ metrics...")
    results = collect_results(user_id, batch_dir)

    if not results and wall_runtimes:
        print("  Bundle collection failed — falling back to wall-clock runtimes")
        results = [{'n_train': n, 'runtime_sec': rt, 'runtime_min': rt / 60}
                   for n, rt in sorted(wall_runtimes.items())]

    report_scaling(user_id, results, abs_batch_dir)


if __name__ == "__main__":
    main()