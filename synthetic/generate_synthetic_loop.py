#!/usr/bin/env python3
"""
generate_synthetic_loop.py
==========================
Expanded synthetic cohort for parameter recovery experiments.

Generates N_USERS synthetic users by randomly sampling physiological
parameters from clinically realistic ranges (fixed seed for reproducibility).

Parameter ranges:
  ISF         : 15 – 80  mg/dL per unit  (covers resistant → sensitive)
  CR          : 4  – 20  g per unit
  sigma_obs   : 5  – 20  mg/dL           (quiet → noisy/degraded sensor)
  sigma_carb  : 0.05 – 0.30              (mild → severe carb misreporting)

Usage:
  python synthetic/generate_synthetic_loop.py --n 30
  python synthetic/generate_synthetic_loop.py --n 30 --seed 42

  The paper's synthetic-recovery experiment uses --n 50 --seed 0.

Interleaving order: MEAL, SNACK, FASTING, BOLUS (repeating round-robin)
This ensures carb-rich instances appear early chronologically so the
inference runner's chronological train/test split sees balanced instance
types in both halves.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_config  # noqa: E402
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib
from itertools import zip_longest
matplotlib.use('Agg')

# ── CONFIG ────────────────────────────────────────────────────────────────────
TIMESTEP_MINS    = 5
HISTORY_MINS     = 4 * 60
FUTURE_MINS      = 4 * 60
WINDOW_MINS      = HISTORY_MINS + FUTURE_MINS
HISTORY_STEPS    = HISTORY_MINS // TIMESTEP_MINS   # 48
FUTURE_STEPS     = FUTURE_MINS  // TIMESTEP_MINS   # 48
WINDOW_STEPS     = WINDOW_MINS  // TIMESTEP_MINS   # 96

# Fixed insulin kinetics (same as original — not varied)
I_PEAK           = 75
I_DUR            = 360
C_ABS            = 120   # minutes — fixed 2h absorption

# Instance counts per user (same as original)
N_MEAL_WINDOWS    = 100
N_SNACK_WINDOWS   = 60
N_BOLUS_WINDOWS   = 80
N_FASTING_WINDOWS = 60

MEAL_CARB_RANGE   = (30, 90)
MEAL_START_RANGE  = (80, 130)
SNACK_CARB_RANGE  = (10, 30)
SNACK_START_RANGE = (80, 130)
BOLUS_DOSE_RANGE  = (1.0, 3.0)
BOLUS_START_RANGE = (200, 300)

TRAIN_FRACTION    = 0.50
MIN_GLUCOSE       = 70.0
BASE_SEED         = 42   # for instance-level RNG per user

# ── PHYSIOLOGICAL PARAMETER RANGES (sampled uniformly) ───────────────────────
ISF_RANGE         = (15.0,  80.0)   # mg/dL per unit
CR_RANGE          = (4.0,   20.0)   # g per unit
SIGMA_OBS_RANGE   = (5.0,   20.0)   # mg/dL sensor noise
SIGMA_CARB_RANGE  = (0.05,  0.30)   # log-scale carb error std

# User IDs start at 2001 to avoid collision with original N=5 cohort (1001-1005)
UID_START         = 2001


# ── PHYSICS ───────────────────────────────────────────────────────────────────

def insulin_action_fraction(t, peak_time=I_PEAK, duration=I_DUR):
    t = np.clip(t, 0.0, None)
    t_peak    = np.clip(t, 0.0, peak_time)
    peak_val  = 0.9 * (t_peak / peak_time) ** 2
    decay_p   = np.clip((t - peak_time) / (duration - peak_time), 0.0, 1.0)
    decay_val = 0.9 * np.exp(-2.5 * decay_p)
    return np.where(t <= peak_time, peak_val,
           np.where(t <= duration,  decay_val, 0.0))


def carb_absorption_fraction(t, c_abs=C_ABS):
    return np.clip(t / (1.5 * c_abs), 0.0, 1.0)


def get_norm_factor():
    dt_array = np.arange(0, 600, TIMESTEP_MINS)
    return insulin_action_fraction(dt_array).sum()


NORM = get_norm_factor()   # pre-compute once


def compute_min_net_delta(bolus_dose, true_carb_amount, ISF, CR):
    min_delta = 0.0
    for step in range(1, FUTURE_STEPS + 1):
        dt = step * TIMESTEP_MINS
        ins_effect = 0.0
        if bolus_dose > 0:
            dt_arr     = np.arange(TIMESTEP_MINS, dt + TIMESTEP_MINS, TIMESTEP_MINS)
            rates      = insulin_action_fraction(dt_arr)
            ins_effect = -ISF * bolus_dose * rates.sum() / NORM
        carb_effect = 0.0
        if true_carb_amount > 0:
            carb_effect = (ISF / CR) * true_carb_amount * carb_absorption_fraction(dt)
        min_delta = min(min_delta, ins_effect + carb_effect)
    return min_delta


def simulate_window(start_glucose, bolus_dose, true_carb_amount, ISF, CR, sigma_obs, rng):
    cgm_clean = np.full(WINDOW_STEPS, start_glucose)
    for step in range(HISTORY_STEPS + 1, WINDOW_STEPS):
        dt = (step - HISTORY_STEPS) * TIMESTEP_MINS
        ins_effect = 0.0
        if bolus_dose > 0:
            dt_arr     = np.arange(TIMESTEP_MINS, dt + TIMESTEP_MINS, TIMESTEP_MINS)
            rates      = insulin_action_fraction(dt_arr)
            ins_effect = -ISF * bolus_dose * rates.sum() / NORM
        carb_effect = 0.0
        if true_carb_amount > 0:
            carb_effect = (ISF / CR) * true_carb_amount * carb_absorption_fraction(dt)
        cgm_clean[step] = start_glucose + ins_effect + carb_effect
    cgm_noisy = cgm_clean + rng.normal(0, sigma_obs, WINDOW_STEPS)
    return cgm_noisy, cgm_clean


# ── COHORT GENERATION ─────────────────────────────────────────────────────────

def sample_cohort_params(n_users, param_seed=0):
    """
    Randomly sample physiological parameters for n_users.
    Uses a separate seed from instance-level RNG so cohort is reproducible
    regardless of n_users.
    """
    rng = np.random.default_rng(param_seed)

    users = {}
    for i in range(n_users):
        uid = UID_START + i
        isf        = float(rng.uniform(*ISF_RANGE))
        cr         = float(rng.uniform(*CR_RANGE))
        sigma_obs  = float(rng.uniform(*SIGMA_OBS_RANGE))
        sigma_carb = float(rng.uniform(*SIGMA_CARB_RANGE))

        # Derive a rough TDD-style label for description
        tdd_approx = 1800 / isf
        if tdd_approx < 30:
            desc = "Insulin Sensitive"
        elif tdd_approx > 60:
            desc = "Insulin Resistant"
        else:
            desc = "Average"

        users[uid] = {
            'desc':          desc,
            'ISF':           isf,
            'CR':            cr,
            'sigma_obs':     sigma_obs,
            'carb_error_std': sigma_carb,
            'i_peak':        I_PEAK,
            'i_dur':         I_DUR,
            'c_abs':         C_ABS,
        }
    return users


def write_by_user(out_base):
    """Split cgm/basal/bolus/meals into by_user/<feature>/<uid>.parquet."""
    for name in ['cgm', 'basal', 'bolus', 'meals']:
        pq = os.path.join(out_base, f"{name}.parquet")
        src = pq if os.path.exists(pq) else os.path.join(out_base, f"{name}.csv")
        df = pd.read_parquet(src) if src.endswith('.parquet') else pd.read_csv(src)
        out = os.path.join(out_base, 'by_user', name)
        os.makedirs(out, exist_ok=True)
        for uid, g in df.groupby('user_id'):
            g.to_parquet(os.path.join(out, f"{int(uid)}.parquet"), index=False)
        print(f"  by_user/{name}: {df['user_id'].nunique()} users from {os.path.basename(src)}")


def generate_cohort(n_users=30, param_seed=0, out_base=None):
    print("=" * 70)
    print(f"SYNTHETIC COHORT GENERATOR v2  (N={n_users} users)")
    print(f"  ISF range       : {ISF_RANGE} mg/dL/U")
    print(f"  CR range        : {CR_RANGE} g/U")
    print(f"  sigma_obs range : {SIGMA_OBS_RANGE} mg/dL")
    print(f"  sigma_carb range: {SIGMA_CARB_RANGE} (log-scale)")
    print(f"  Param seed      : {param_seed}")
    print(f"  Interleave order: MEAL, SNACK, FASTING, BOLUS (round-robin)")
    print("=" * 70)

    default_data, curation_dir = eval_config.synthetic_paths(n_users)
    if out_base is None:
        out_base = default_data

    os.makedirs(out_base,    exist_ok=True)
    os.makedirs(curation_dir, exist_ok=True)

    cohort = sample_cohort_params(n_users, param_seed=param_seed)

    # Print sampled parameters
    print(f"\n{'UID':<6} {'Desc':<20} {'ISF':>6} {'CR':>6} "
          f"{'σ_obs':>7} {'σ_carb':>8}")
    print("-" * 58)
    for uid, p in cohort.items():
        print(f"{uid:<6} {p['desc']:<20} {p['ISF']:>6.1f} {p['CR']:>6.1f} "
              f"{p['sigma_obs']:>7.2f} {p['carb_error_std']:>8.3f}")

    # Save ground truth
    truth_dict = {
        str(uid): {
            'ISF':           p['ISF'],
            'CR':            p['CR'],
            'NOISE':         p['sigma_obs'],    # sigma_obs
            'carb_error_std': p['carb_error_std'],
            'desc':          p['desc'],
        }
        for uid, p in cohort.items()
    }
    with open(os.path.join(out_base, "true_parameters.json"), "w") as f:
        json.dump(truth_dict, f, indent=4)
    print(f"\nGround truth saved -> {out_base}/true_parameters.json")

    # ── Generate instances ────────────────────────────────────────────────────
    all_cgm_rows   = []
    all_bolus_rows = []
    all_meal_rows  = []
    all_basal_rows = []

    filtered_user_instances = {}
    train_instances = []
    test_instances  = []

    global_ex_id = 1
    anchor       = pd.Timestamp("2026-03-01 00:00:00")

    for uid, params in cohort.items():
        u_str = str(uid)
        ISF   = params['ISF']
        CR    = params['CR']
        sobs  = params['sigma_obs']
        scarb = params['carb_error_std']

        filtered_user_instances[u_str] = []
        rng = np.random.default_rng(BASE_SEED + uid)

        instances_this_user = []
        window_counter      = 0

        def make_instance(event_type, bolus_dose, true_carb, logged_carb, start_glucose):
            nonlocal global_ex_id, window_counter
            eid           = global_ex_id
            global_ex_id += 1
            window_counter += 1

            w_start    = anchor + pd.Timedelta(hours=12 * (window_counter - 1))
            times      = [w_start + pd.Timedelta(minutes=i * TIMESTEP_MINS)
                          for i in range(WINDOW_STEPS)]
            event_time = times[HISTORY_STEPS]

            cgm_noisy, _ = simulate_window(
                start_glucose, bolus_dose, true_carb, ISF, CR, sobs, rng)

            for ts, g in zip(times, cgm_noisy):
                all_cgm_rows.append(
                    {'user_id': u_str, 'timestamp': ts, 'cgm': g, 'ex': eid})
            for ts in times:
                all_basal_rows.append(
                    {'user_id': u_str, 'timestamp': ts, 'basal_rate': 1.0, 'ex': eid})
            if bolus_dose > 0:
                all_bolus_rows.append(
                    {'user_id': u_str, 'timestamp': event_time, 'bolus': bolus_dose, 'ex': eid})
            if logged_carb > 0:
                all_meal_rows.append(
                    {'user_id': u_str, 'timestamp': event_time, 'carbs': logged_carb, 'ex': eid})

            filtered_user_instances[u_str].append(eid)
            return eid

        # ── Build parameter specs for each type first (preserves RNG sequence) ──

        meal_specs = []
        for _ in range(N_MEAL_WINDOWS):
            logged_carb   = rng.uniform(*MEAL_CARB_RANGE)
            bolus_dose    = logged_carb / CR
            err_mult      = np.clip(1.0 + rng.normal(0, scarb), 0.4, 2.5)
            true_carb     = logged_carb * err_mult
            min_delta     = compute_min_net_delta(bolus_dose, true_carb, ISF, CR)
            start_glucose = max(rng.uniform(*MEAL_START_RANGE), MIN_GLUCOSE - min_delta)
            meal_specs.append(('MEAL', bolus_dose, true_carb, logged_carb, start_glucose))

        snack_specs = []
        for _ in range(N_SNACK_WINDOWS):
            logged_carb   = rng.uniform(*SNACK_CARB_RANGE)
            err_mult      = np.clip(1.0 + rng.normal(0, scarb), 0.4, 2.5)
            true_carb     = logged_carb * err_mult
            start_glucose = rng.uniform(*SNACK_START_RANGE)
            snack_specs.append(('UNBOLUSED_SNACK', 0.0, true_carb, logged_carb, start_glucose))

        fasting_specs = []
        for _ in range(N_FASTING_WINDOWS):
            start_glucose = rng.uniform(100, 140)
            fasting_specs.append(('FASTING', 0.0, 0.0, 0.0, start_glucose))

        bolus_specs = []
        for _ in range(N_BOLUS_WINDOWS):
            start_glucose = rng.uniform(*BOLUS_START_RANGE)
            max_safe      = (start_glucose - MIN_GLUCOSE) / ISF
            bolus_dose    = float(np.clip(rng.uniform(*BOLUS_DOSE_RANGE), 0.5, max_safe))
            bolus_specs.append(('PURE_BOLUS', bolus_dose, 0.0, 0.0, start_glucose))

        # ── Interleave round-robin: M, S, F, B, M, S, F, B, ... ─────────────
        # Meal and snack come first so carb-rich instances appear early
        # chronologically — ensures the inference runner's chronological
        # train/test split sees balanced instance types in both halves.
        interleaved_specs = []
        for group in zip_longest(meal_specs, snack_specs, fasting_specs, bolus_specs):
            for spec in group:
                if spec is not None:
                    interleaved_specs.append(spec)

        # ── Call make_instance in interleaved order ───────────────────────────
        for spec in interleaved_specs:
            event_type, bolus_dose, true_carb, logged_carb, start_glucose = spec
            instances_this_user.append(
                make_instance(event_type, bolus_dose, true_carb, logged_carb, start_glucose))

        # ── Train / test split — purely chronological, balance guaranteed ─────
        n_train = int(len(instances_this_user) * TRAIN_FRACTION)
        for i, eid in enumerate(instances_this_user):
            (train_instances if i < n_train else test_instances).append(eid)

        # ── Verify balance ────────────────────────────────────────────────────
        # After interleaving M,S,F,B the positions are:
        #   meal    at indices 0, 4, 8, ...
        #   snack   at indices 1, 5, 9, ...
        #   fasting at indices 2, 6, 10, ...
        #   bolus   at indices 3, 7, 11, ...
        train_set    = set(instances_this_user[:n_train])
        meal_eids    = set(instances_this_user[i] for i in range(0, len(instances_this_user), 4))
        snack_eids   = set(instances_this_user[i] for i in range(1, len(instances_this_user), 4))
        fasting_eids = set(instances_this_user[i] for i in range(2, len(instances_this_user), 4))
        bolus_eids   = set(instances_this_user[i] for i in range(3, len(instances_this_user), 4))
        n_meal   = len(meal_eids    & train_set)
        n_snack  = len(snack_eids   & train_set)
        n_fast   = len(fasting_eids & train_set)
        n_bolus  = len(bolus_eids   & train_set)
        print(f"    Train balance: {n_meal} meal | {n_snack} snack | {n_fast} fasting | {n_bolus} bolus")

        n = len(instances_this_user)
        print(f"  User {uid} ({params['desc']:20s}): {n} instances | "
              f"ISF={ISF:.1f} CR={CR:.1f} σ_obs={sobs:.1f} σ_carb={scarb:.3f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(all_cgm_rows).to_csv(
        os.path.join(out_base, "cgm.csv"),   index=False)
    pd.DataFrame(all_basal_rows).to_csv(
        os.path.join(out_base, "basal.csv"), index=False)
    pd.DataFrame(all_bolus_rows).to_csv(
        os.path.join(out_base, "bolus.csv"), index=False)
    pd.DataFrame(all_meal_rows).to_csv(
        os.path.join(out_base, "meals.csv"), index=False)

    # Also save parquet versions for compatibility with feature extraction pipeline
    pd.DataFrame(all_cgm_rows).to_parquet(
        os.path.join(out_base, "glucose.parquet"), index=False)
    pd.DataFrame(all_basal_rows).to_parquet(
        os.path.join(out_base, "basal.parquet"),   index=False)
    pd.DataFrame(all_bolus_rows).to_parquet(
        os.path.join(out_base, "bolus.parquet"),   index=False)
    pd.DataFrame(all_meal_rows).to_parquet(
        os.path.join(out_base, "meals.parquet"),   index=False)

    # Per-user feature files (by_user/<feature>/<uid>.parquet), the layout
    # real_data_evaluation.py reads. Built from the files just written
    # (cgm.csv, and the .parquet files for the other features).
    write_by_user(out_base)

    # ── Save curation pkls ────────────────────────────────────────────────────
    with open(os.path.join(curation_dir, "filtered_user_instances.pkl"), "wb") as f:
        pickle.dump(filtered_user_instances, f)
    with open(os.path.join(curation_dir, "train_instances.pkl"), "wb") as f:
        pickle.dump(train_instances, f)
    with open(os.path.join(curation_dir, "test_instances.pkl"), "wb") as f:
        pickle.dump(test_instances, f)

    # ── Save cohort summary ───────────────────────────────────────────────────
    summary_rows = [
        {
            'uid':           uid,
            'desc':          p['desc'],
            'ISF':           round(p['ISF'], 2),
            'CR':            round(p['CR'], 2),
            'sigma_obs':     round(p['sigma_obs'], 2),
            'sigma_carb':    round(p['carb_error_std'], 3),
            'n_instances':   len(filtered_user_instances[str(uid)]),
        }
        for uid, p in cohort.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_base, "cohort_summary.csv"), index=False)

    print(f"\n{'='*70}")
    print(f"COHORT STATISTICS")
    print(f"{'='*70}")
    print(f"  N users         : {n_users}")
    print(f"  ISF  mean±std   : {summary_df['ISF'].mean():.1f} ± {summary_df['ISF'].std():.1f}")
    print(f"  CR   mean±std   : {summary_df['CR'].mean():.1f} ± {summary_df['CR'].std():.1f}")
    print(f"  σ_obs mean±std  : {summary_df['sigma_obs'].mean():.1f} ± {summary_df['sigma_obs'].std():.1f}")
    print(f"  σ_carb mean±std : {summary_df['sigma_carb'].mean():.3f} ± {summary_df['sigma_carb'].std():.3f}")
    print(f"  Total instances : {sum(len(v) for v in filtered_user_instances.values())}")
    print(f"  Train instances : {len(train_instances)}")
    print(f"  Test instances  : {len(test_instances)}")
    print(f"\n  Output dir      : {out_base}")
    print(f"  Curation dir    : {curation_dir}")
    print(f"{'='*70}")
    print(f"\nDone. Run inference with:")
    print(f"   BATCH_LIST = ['synthetic_cohort_n{n_users}']")
    print(f"   DATA_DIR   = '{out_base}'")
    print(f"   FILTER_DIR = '{curation_dir}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate expanded synthetic cohort for parameter recovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python synthetic/generate_synthetic_loop.py --n 30
  python synthetic/generate_synthetic_loop.py --n 25 --seed 123
        """
    )
    parser.add_argument('--n',    type=int, default=30,
                        help='Number of synthetic users (default: 30)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for parameter sampling (default: 0)')
    parser.add_argument('--out',  type=str, default=None,
                        help='Output base directory (default: auto from N)')
    args = parser.parse_args()

    generate_cohort(n_users=args.n, param_seed=args.seed, out_base=args.out)


if __name__ == "__main__":
    main()