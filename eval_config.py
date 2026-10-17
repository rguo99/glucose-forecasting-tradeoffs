#!/usr/bin/env python3
"""
eval_config.py
==============
Single source of truth for the evaluation cohort and for which model-run
directories feed the evaluation scripts.

Usage:
    from eval_config import get_config
    cfg = get_config(prior_scale=0.30)

    cfg['BATCH_LIST']    # absolute paths of model-run directories
    cfg['TARGET_USERS']  # analysed cohort (list of int user ids)
    cfg['DATA_DIR']      # instance table + per-user features
    cfg['FILTER_DIR']    # curated train/test instance lists
    cfg['OUT_DIR']       # where tables/figures are written

Model-run directories
---------------------
run_cohort.py writes one directory per (model, prior scale, shard):
    <EXPERIMENTS_DIR>/batch_<model>[_scale<tag>][_shard<k>of<N>]_<timestamp>/
Every non-scale ``batch_*`` directory is used for the main analysis (prior
scale 0.30). For the prior-sensitivity analyses (0.10, 0.50), the Bayesian
Loop directories of the main run are replaced by the matching
``_scale<tag>_`` directories; all other models are prior-independent.
A (model, user) pair found in more than one directory is an error
(see evaluate.find_bundle_paths), so results are never chosen by sort order.
"""

import os
import glob as _glob
import pickle
from collections import defaultdict

import config

MAIN_PRIOR_SCALE = 0.30

# Eligibility thresholds for the analysed cohort.
MIN_TRAIN, MIN_TEST = 150, 10



def scale_tag(prior_scale):
    return str(prior_scale).replace('.', 'p')


def out_dir_for(prior_scale=MAIN_PRIOR_SCALE):
    name = 'main' if prior_scale == MAIN_PRIOR_SCALE else f'main_scale{scale_tag(prior_scale)}'
    return os.path.join(config.MAIN_RESULTS_DIR, name)


def _is_scale_dir(path):
    return '_scale' in os.path.basename(path)


def _is_bayesian_loop_dir(path):
    return '_global_' in os.path.basename(path) or os.path.basename(path).startswith('batch_global')


def resolve_batches(prior_scale=MAIN_PRIOR_SCALE):
    """Model-run directories for the given prior scale (absolute paths, sorted)."""
    all_dirs = sorted(d for d in _glob.glob(os.path.join(config.EXPERIMENTS_DIR, 'batch_*'))
                      if os.path.isdir(d))
    main = [d for d in all_dirs if not _is_scale_dir(d)]
    if prior_scale == MAIN_PRIOR_SCALE:
        out = main
    else:
        tag = f'_scale{scale_tag(prior_scale)}_'
        scale_dirs = [d for d in all_dirs if tag in os.path.basename(d)]
        if not scale_dirs:
            raise FileNotFoundError(
                f'No Bayesian Loop runs for prior scale {prior_scale} '
                f'(expected batch_*{tag}* under {config.EXPERIMENTS_DIR}).')
        out = [d for d in main if not _is_bayesian_loop_dir(d)] + scale_dirs
    print(f'  prior={prior_scale}: {len(out)} model-run dirs under {config.EXPERIMENTS_DIR}')
    return out


def find_duplicate_cgm_users(data_dir, users):
    """Users whose per-user CGM file has repeated timestamps."""
    import pandas as pd
    dup = []
    for uid in users:
        p = os.path.join(data_dir, 'by_user', 'cgm', f'{uid}.parquet')
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        ts_col = next((c for c in ['datestamp', 'timestamp', 'LBDTC'] if c in df.columns), None)
        if ts_col is None:
            continue
        key = ['ex', ts_col] if 'ex' in df.columns else [ts_col]
        if df.duplicated(subset=key).any():
            dup.append(uid)
    return dup


def load_eligible_users(data_dir, filter_dir, exclude_duplicate_cgm=True, verbose=True):
    """Eligible pump users: >= MIN_TRAIN train and >= MIN_TEST test instances,
    excluding multiple-daily-injection (MDI) users and, optionally, users with
    duplicate CGM timestamps. Returns a sorted list of str user ids."""
    import pandas as pd

    with open(os.path.join(filter_dir, 'train_instances.pkl'), 'rb') as f:
        train_ids = set(pickle.load(f))
    with open(os.path.join(filter_dir, 'test_instances.pkl'), 'rb') as f:
        test_ids = set(pickle.load(f))

    non_ex = pd.read_csv(os.path.join(data_dir, 'non_exercise.csv'), usecols=['ex', 'id'])
    ex_to_user = non_ex.set_index('ex')['id'].apply(lambda x: str(int(float(x)))).to_dict()

    train_by_user = defaultdict(list)
    test_by_user = defaultdict(list)
    for eid in train_ids:
        uid = ex_to_user.get(eid)
        if uid:
            train_by_user[uid].append(eid)
    for eid in test_ids:
        uid = ex_to_user.get(eid)
        if uid:
            test_by_user[uid].append(eid)

    mdi_users = set()
    device_path = config.device_parquet()
    if os.path.exists(device_path):
        dev_df = pd.read_parquet(device_path)
        id_col = next((c for c in ['USUBJID', 'user_id', 'id'] if c in dev_df.columns), None)
        if id_col and 'DXTRT' in dev_df.columns:
            mdi_users = set(
                dev_df[dev_df['DXTRT'].str.upper().str.contains('MULTIPLE DAILY', na=False)]
                [id_col].astype(str).unique())
    else:
        raise FileNotFoundError(f'Device table not found: {device_path}')

    eligible = sorted([uid for uid in train_by_user
                       if len(train_by_user[uid]) >= MIN_TRAIN
                       and len(test_by_user.get(uid, [])) >= MIN_TEST
                       and uid not in mdi_users], key=int)
    users = eligible
    dup = []
    if exclude_duplicate_cgm:
        dup = find_duplicate_cgm_users(data_dir, eligible)
        users = [u for u in eligible if u not in set(dup)]
    if verbose:
        print(f'  Cohort: {len(eligible)} eligible pump users '
              f'({len(mdi_users)} MDI users excluded), '
              f'{len(dup)} excluded for duplicate CGM timestamps {dup}, '
              f'{len(users)} analysed')
    return users


# SYNTHETIC (parameter recovery)
SYN_N = 50
SYN_UID_START = 2001


def synthetic_paths(n=SYN_N):
    base = os.path.join(config.RESULTS_DIR, 'synthetic')
    return (os.path.join(base, 'processed_data', f'synthetic_cohort_n{n}'),
            os.path.join(base, 'data_filtering', f'synthetic_cohort_n{n}'))


def _synthetic_config(n):
    data_dir, filter_dir = synthetic_paths(n)
    runs = sorted(_glob.glob(os.path.join(config.EXPERIMENTS_DIR, f'synthetic_recovery_n{n}_*')))
    if len(runs) > 1:
        raise RuntimeError(f'More than one synthetic_recovery_n{n}_* run found; keep exactly one: {runs}')
    out_dir = os.path.join(config.MAIN_RESULTS_DIR, f'synthetic_n{n}')
    os.makedirs(out_dir, exist_ok=True)
    return {
        'BATCH_LIST':    runs,
        'TARGET_USERS':  list(range(SYN_UID_START, SYN_UID_START + n)),
        'DATA_DIR':      data_dir,
        'FILTER_DIR':    filter_dir,
        'OUT_DIR':       out_dir,
        'INST_CSV_NAME': 'basal.csv',
    }


def get_config(prior_scale=MAIN_PRIOR_SCALE, synthetic=False, n=SYN_N, resolve_batches_now=True):
    """Config dict for the real-data cohort (or the synthetic cohort)."""
    if synthetic:
        return _synthetic_config(n)
    data_dir, filter_dir = config.data_dir(), config.filter_dir()
    cfg = {
        'BATCH_LIST':    resolve_batches(prior_scale) if resolve_batches_now else [],
        'TARGET_USERS':  [int(u) for u in load_eligible_users(data_dir, filter_dir)],
        'DATA_DIR':      data_dir,
        'FILTER_DIR':    filter_dir,
        'OUT_DIR':       out_dir_for(prior_scale),
        'INST_CSV_NAME': 'non_exercise.csv',
        'PRIOR_SCALE':   prior_scale,
    }
    os.makedirs(cfg['OUT_DIR'], exist_ok=True)
    return cfg
