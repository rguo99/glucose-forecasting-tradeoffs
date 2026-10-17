#!/usr/bin/env python3
"""evaluate.py
===========
Main evaluation: per-instance, per-horizon metrics for all nine models, the
main and supplementary results tables, and the evaluation figures.

For every analysed user, the saved posterior/predictive samples of each model
(<run dir>/<model folder>/results/user_<uid>_results.pkl) are scored at
horizons of 30 min to 4 h:
  - statistical/UQ metrics: RMSE, CRPS (all-pairs energy form), 95% PI
    coverage and width, interval score;
  - directional metrics: net trend accuracy, trajectory correlation, mean
    directional bias;
  - clinical metrics: hypo/hyperglycemia sensitivity and specificity, alarm
    operating points (false-alarm rate, PPV), Clarke error grid zones.
Metric prefixes in metrics_df.pkl: G_ Bayesian Loop, L_ LSTM-MC, A_ Pure AR1,
R_ AR1-RLS, T_ Transformer-MC, E_ LSTM-Ensemble, C_ LSTM-Conformal,
F_ Transformer-Conformal, N_ Transformer-Ensemble.

Usage:
    python evaluate.py --prior 0.30      # main analysis
    python evaluate.py --prior 0.10      # prior sensitivity (Bayesian Loop prior scale)

Outputs (eval_config.out_dir_for(prior)): metrics_df.pkl, tables.tex,
tables/*.tex, fig_horizons, instance_type_comparison (fig_stratified),
fig_user_level_metric_distributions, fig_roc_hypo, plot_data/.
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   15,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.titlesize': 16,
    'lines.linewidth':  2,
    'axes.linewidth':   1.2,
})

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr, wilcoxon
from tqdm import tqdm

import config

warnings.filterwarnings('ignore')

if 'numpy.core' in sys.modules and 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = sys.modules['numpy.core']
    import numpy.core.numeric, numpy.core.multiarray
    sys.modules['numpy._core.numeric']    = sys.modules['numpy.core.numeric']
    sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print('WARNING: fastdtw not installed — DTW metrics skipped')

# ── MODEL IDENTIFIERS ─────────────────────────────────────────────────────────
MODEL_SAFE   = 'Global'
MODEL_BASE   = 'LSTM_MC'
MODEL_AR1    = 'AR1Pure'
MODEL_AR1RLS = 'AR1RLS'  # Adaptive AR(1) via RLS with forgetting factor
MODEL_TRANSFORMER = 'TransformerMC'  # Architecture-capacity comparator
MODEL_LSTM_ENSEMBLE  = 'LSTMEnsemble'
MODEL_LSTM_CONFORMAL = 'LSTMConformal'
MODEL_TRANSFORMER_CONFORMAL = 'TransformerConformal'  # UQ-method comparator, Transformer backbone — isolates whether the LSTM_MC/Transformer_MC shared MC-Dropout failure mode is UQ-method- or architecture-driven
MODEL_TRANSFORMER_ENSEMBLE  = 'TransformerEnsemble'  # completes the architecture x UQ-method 2x2; strongest-expected non-Bayesian-Loop config, so it's the adversarial test of the <=2hr clinical claim

LABEL_SAFE   = 'Bayesian Loop'
LABEL_BASE   = 'LSTM-MC'
LABEL_AR1    = 'Pure AR1'
LABEL_AR1RLS = 'Adaptive AR1-RLS'
LABEL_TRANSFORMER = 'Transformer-MC'
LABEL_LSTM_ENSEMBLE  = 'LSTM-Ensemble'
LABEL_LSTM_CONFORMAL = 'LSTM-Conformal'
LABEL_TRANSFORMER_CONFORMAL = 'Transformer-Conformal'
LABEL_TRANSFORMER_ENSEMBLE  = 'Transformer-Ensemble'

# NOTE: MODEL_AR1RLS/MODEL_TRANSFORMER/MODEL_LSTM_ENSEMBLE/MODEL_LSTM_CONFORMAL/
# MODEL_TRANSFORMER_CONFORMAL are intentionally excluded from ALL_MODELS so
# existing figures/tables (many of which assume exactly the original three
# models) are unaffected. Pass models=[..., *EXTRA_MODELS keys] explicitly to
# include them.
ALL_MODELS   = [MODEL_SAFE, MODEL_BASE, MODEL_AR1]
ALL_LABELS   = {MODEL_SAFE: LABEL_SAFE, MODEL_BASE: LABEL_BASE, MODEL_AR1: LABEL_AR1,
                MODEL_AR1RLS: LABEL_AR1RLS, MODEL_TRANSFORMER: LABEL_TRANSFORMER,
                MODEL_LSTM_ENSEMBLE: LABEL_LSTM_ENSEMBLE,
                MODEL_LSTM_CONFORMAL: LABEL_LSTM_CONFORMAL,
                MODEL_TRANSFORMER_CONFORMAL: LABEL_TRANSFORMER_CONFORMAL,
                MODEL_TRANSFORMER_ENSEMBLE: LABEL_TRANSFORMER_ENSEMBLE}
ALL_PREFIXES = {MODEL_SAFE: 'G', MODEL_BASE: 'L', MODEL_AR1: 'A', MODEL_AR1RLS: 'R',
                MODEL_TRANSFORMER: 'T', MODEL_LSTM_ENSEMBLE: 'E',
                MODEL_LSTM_CONFORMAL: 'C', MODEL_TRANSFORMER_CONFORMAL: 'F',
                MODEL_TRANSFORMER_ENSEMBLE: 'N'}

# Models beyond the three core models (Bayesian Loop, LSTM-MC, Pure AR1). They
# are included wherever their result bundles are present (get_active_models /
# _extra_models_present).
EXTRA_MODELS = [(MODEL_AR1RLS, 'R'), (MODEL_TRANSFORMER, 'T'),
                (MODEL_LSTM_ENSEMBLE, 'E'), (MODEL_LSTM_CONFORMAL, 'C'),
                (MODEL_TRANSFORMER_CONFORMAL, 'F'),
                (MODEL_TRANSFORMER_ENSEMBLE, 'N')]


# ── DISPLAY ORDER for every table and figure ─────────────────────────────────
# Bayesian Loop, LSTM family (MC -> Ensemble -> Conformal), Transformer family
# (same order), then the two classical AR baselines. This is a taxonomy order,
# not a performance ranking: the UQ-method effect reads down a family block and
# the architecture effect reads across at matched offsets (positions 2<->5,
# 3<->6, 4<->7). Ensemble precedes Conformal by increasing interval width.
DISPLAY_ORDER = ['G', 'L', 'E', 'C', 'T', 'N', 'F', 'A', 'R']


def ordered_prefixes(available=None):
    """Return DISPLAY_ORDER filtered to the prefixes actually available.

    available: iterable of prefixes to keep, or None for "all known models".
    Any prefix not in DISPLAY_ORDER (e.g. a future baseline someone forgot to
    add) is appended at the end rather than silently dropped.
    """
    if available is None:
        available = ['G', 'L', 'A'] + [pfx for _, pfx in EXTRA_MODELS]
    avail = list(available)
    ordered = [p for p in DISPLAY_ORDER if p in avail]
    ordered += [p for p in avail if p not in DISPLAY_ORDER]
    return ordered


HORIZONS       = [6, 12, 18, 24, 36, 48]
HORIZON_LABELS = ['30 min', '60 min', '90 min', '120 min', '180 min', '240 min']
FULL_HORIZON   = 48

# Clinical contexts (build_or_load_instance_type_map): MEAL >= _MEAL_CARBS g, CARB_INSTANCE
# >= _MIN_CARBS g, CORRECTION >= _MIN_BOLUS U with < _MIN_CARBS g, otherwise STABLE. The totals are
# over all carbohydrate and bolus entries attached to the instance, i.e. its full 8-hour window
# (4-hour history + 4-hour forecast), as in the paper. _MEAL_WIN_* (minutes from the forecast
# origin) is used only when the meal/bolus tables carry no instance-id column.
_MIN_CARBS    = 10.0
_MEAL_CARBS   = 40.0
_MIN_BOLUS    = 0.5
_MEAL_WIN_MIN = -30
_MEAL_WIN_MAX =  60

INSTANCE_TYPES = ['MEAL', 'CARB_INSTANCE', 'CORRECTION', 'STABLE']
INSTANCE_LABELS = {
    'MEAL':          r'Meal ($\geq$40g carbs)',
    'CARB_INSTANCE': r'Carb Instance (10--39g carbs)',
    'CORRECTION':    r'Correction ($\geq$0.5U bolus)',
    'STABLE':        r'Stable',
}

_IMAP_CACHE_FNAME = f'instance_type_map_meal{int(_MEAL_CARBS)}g.pkl'


def imap_cache_path(inst_csv):
    """Cache file for the instance-type map. Instance ids are only stable for
    a given build of the preprocessed data, so the cache name is tied to the
    instance table it was computed from."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    st = os.stat(inst_csv)
    fp = hashlib.sha256(f'{os.path.realpath(inst_csv)}|{st.st_size}|{int(st.st_mtime)}'.encode()).hexdigest()[:12]
    return os.path.join(config.CACHE_DIR, f'{fp}_{_IMAP_CACHE_FNAME}')

MODEL_COLORS  = {MODEL_SAFE: '#2196F3', MODEL_BASE: '#FF5722', MODEL_AR1: '#4CAF50', MODEL_AR1RLS: '#9C27B0',
                 MODEL_TRANSFORMER: '#FF9800', MODEL_LSTM_ENSEMBLE: '#009688',
                 MODEL_LSTM_CONFORMAL: '#795548', MODEL_TRANSFORMER_CONFORMAL: '#3F51B5',
                 MODEL_TRANSFORMER_ENSEMBLE: '#C2185B'}  # distinct from LSTM-Ensemble's teal
MODEL_MARKERS = {MODEL_SAFE: 'o',       MODEL_BASE: 's',       MODEL_AR1: '^',       MODEL_AR1RLS: 'D',
                 MODEL_TRANSFORMER: 'P', MODEL_LSTM_ENSEMBLE: 'X',
                 MODEL_LSTM_CONFORMAL: '*', MODEL_TRANSFORMER_CONFORMAL: 'h',
                 MODEL_TRANSFORMER_ENSEMBLE: '<'}
# Per-model linestyles, distinct from every other model's even when lines
# overlap and colors are hard to tell apart (e.g. fig_roc_hypo with up to 9
# models overlaid) -- used alongside MODEL_COLORS, not instead of it.
MODEL_LINESTYLES = {MODEL_SAFE: '-',       MODEL_BASE: '--',      MODEL_AR1: ':',       MODEL_AR1RLS: '-.',
                     MODEL_TRANSFORMER: (0, (3, 1, 1, 1)),
                     MODEL_LSTM_ENSEMBLE: (0, (5, 1)),
                     MODEL_LSTM_CONFORMAL: (0, (1, 1)),
                     MODEL_TRANSFORMER_CONFORMAL: (0, (3, 5, 1, 5)),
                     MODEL_TRANSFORMER_ENSEMBLE: (0, (5, 2, 1, 2, 1, 2))}


# ============================================================================
# DATA LOADING
# ============================================================================

def label_folder(folder):
    n = folder.lower()
    if 'transformer_conformal' in n: return MODEL_TRANSFORMER_CONFORMAL
    if 'transformer_ensemble' in n: return MODEL_TRANSFORMER_ENSEMBLE
    if 'transformer_mc'  in n: return MODEL_TRANSFORMER
    if 'lstm_conformal'  in n: return MODEL_LSTM_CONFORMAL
    if 'lstm_ensemble'   in n: return MODEL_LSTM_ENSEMBLE
    if 'lstm_mc'         in n: return MODEL_BASE
    if 'ar1_rls'         in n: return MODEL_AR1RLS
    if 'ar1_pure'        in n: return MODEL_AR1
    if 'iid_noise_fixed' in n or 'iid_fixed' in n: return 'IIDFixed'
    if 'iid_noise'       in n: return 'IID'
    if 'ar1'             in n:
        return MODEL_SAFE if 'carb_uncert' in n else 'AR1'
    return None


def load_bundle(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    all_preds   = d.get('all_preds', d.get('all_preds_cpu', {}))
    raw_samples = np.asarray(all_preds.get('obs', all_preds.get('pred_mean')))
    if raw_samples.ndim == 2:
        raw_samples = raw_samples[np.newaxis, :, :]
    raw_obs  = np.asarray(d.get('obs_glucose', d.get('observed_glucose_array')))
    uid      = d.get('user_id')
    test_idx = np.asarray(d.get('test_indices', d.get('test_idx', [])))
    raw_ids  = np.asarray(d.get('instance_ids', []))

    if raw_samples.shape[1] == len(raw_obs):
        samples  = raw_samples[:, test_idx, :]
        obs      = raw_obs[test_idx]
        test_ids = raw_ids[test_idx] if len(raw_ids) > 0 else np.arange(len(test_idx))
    elif raw_samples.shape[1] == len(test_idx):
        samples  = raw_samples
        obs      = raw_obs[test_idx]
        test_ids = raw_ids[test_idx] if len(raw_ids) == len(raw_obs) else raw_ids
    else:
        m_ = min(len(test_idx), raw_samples.shape[1])
        samples  = raw_samples[:, :m_, :]
        obs      = raw_obs[test_idx[:m_]]
        test_ids = raw_ids[test_idx[:m_]] if len(raw_ids) > 0 else np.arange(m_)

    train_ids = np.asarray(d.get('train_indices', d.get('train_ids', [])))
    if len(train_ids) > 0 and len(raw_ids) > 0 and train_ids.max() < len(raw_ids):
        train_ids = raw_ids[train_ids]

    return {
        'obs': obs, 'samples': samples, 'user_id': uid,
        'test_idx': test_idx, 'test_ids': test_ids,
        'train_ids': train_ids,
        'map_params': d.get('map_params', {}),
        'param_samples': d.get('param_samples', {}),
    }


def load_all_bundles(batch_dirs, target_users, models=None):
    if models is None:
        models = ALL_MODELS
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_data = {m: {} for m in models}

    def _load_one(label, uid, path):
        try:
            return label, uid, load_bundle(path)
        except Exception:
            return label, uid, None

    tasks = []
    for batch_dir in batch_dirs:
        if not os.path.exists(batch_dir):
            print(f'  WARNING: Missing: {batch_dir}')
            continue
        for folder in sorted(glob.glob(os.path.join(batch_dir, '20*'))):
            if not os.path.isdir(folder): continue
            label = label_folder(folder)
            if label not in all_data: continue
            for uid in target_users:
                p = os.path.join(folder, 'results', f'user_{uid}_results.pkl')
                if os.path.exists(p):
                    tasks.append((label, uid, p))

    print(f'  Loading {len(tasks)} bundles in parallel...')
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_load_one, *t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks),
                           desc='  Loading bundles'):
            label, uid, bundle = future.result()
            if bundle is not None:
                all_data[label][uid] = bundle
    return all_data


def find_bundle_paths(batch_dirs, target_users, models=None):
    """Like load_all_bundles, but returns file paths instead of loaded
    bundles. Deserializing every model's full posterior-sample array for
    every user upfront (what load_all_bundles does) means the ENTIRE
    cohort's data sits resident in the main process for the whole run —
    for 8 models x 395 users that's the dominant memory cost (it's what
    OOM'd this pipeline at 48GB). Callers should instead defer the actual
    load_bundle() call to wherever/whenever a single user's data is
    actually needed (typically inside a Pool worker, one user at a time),
    so peak memory scales with worker count, not cohort size x model count.
    Cheap: just filesystem existence checks, no pickle deserialization."""
    if models is None:
        models = ALL_MODELS
    all_paths = {m: {} for m in models}
    for batch_dir in batch_dirs:
        if not os.path.exists(batch_dir):
            print(f'  WARNING: Missing: {batch_dir}')
            continue
        for folder in sorted(glob.glob(os.path.join(batch_dir, '20*'))):
            if not os.path.isdir(folder): continue
            label = label_folder(folder)
            if label not in all_paths: continue
            for uid in target_users:
                p = os.path.join(folder, 'results', f'user_{uid}_results.pkl')
                if os.path.exists(p):
                    prev = all_paths[label].get(uid)
                    if prev is not None and prev != p:
                        raise RuntimeError(
                            f'{label} user {uid} has results in more than one run '
                            f'directory:\n  {prev}\n  {p}\nKeep exactly one.')
                    all_paths[label][uid] = p
    return all_paths


def get_completed_users(batch_dirs, models=None):
    if models is None:
        models = ALL_MODELS
    completed = {m: set() for m in models}
    for batch_dir in batch_dirs:
        if not os.path.exists(batch_dir):
            continue
        for folder in sorted(glob.glob(os.path.join(batch_dir, '20*'))):
            if not os.path.isdir(folder): continue
            label = label_folder(folder)
            if label not in completed: continue
            results_dir = os.path.join(folder, 'results')
            if not os.path.isdir(results_dir): continue
            for pkl in glob.glob(os.path.join(results_dir, 'user_*_results.pkl')):
                fname = os.path.basename(pkl)
                uid   = fname.replace('user_', '').replace('_results.pkl', '')
                completed[label].add(uid)
    for m in models:
        print(f'  Completed on disk — {m}: {len(completed[m])} users')
    return completed


def _nan_bundle(ref_bundle):
    obs  = ref_bundle['obs']
    samp = ref_bundle['samples']
    return {
        'obs':      obs,
        'samples':  np.full_like(samp, np.nan, dtype=float),
        'test_ids': ref_bundle['test_ids'],
    }


# ============================================================================
# INSTANCE TYPE MAP
# ============================================================================

def build_or_load_instance_type_map(inst_df, meals_df, bolus_df, cache_path):
    """Instance id -> clinical context, from the total carbohydrates and bolus insulin in each
    instance's 8-hour window (see the constants above). Cached per preprocessing build."""
    if os.path.exists(cache_path):
        print('  Loading cached instance type map...')
        with open(cache_path, 'rb') as f:
            imap = pickle.load(f)
        print(f'  Loaded {len(imap):,} cached classifications')
        return imap

    print(f'  Classifying {len(inst_df):,} instances...')

    def _clean(v):
        s = str(v)
        return s[:-2] if s.endswith('.0') else s

    inst_uid = next((c for c in ['user_id','id','USUBJID'] if c in inst_df.columns), None)
    inst_id  = next((c for c in ['instance_id','ex','id']  if c in inst_df.columns), None)
    m_uid    = next((c for c in ['id','user_id','USUBJID'] if c in meals_df.columns), None)
    m_date   = next((c for c in ['datestamp','timestamp']  if c in meals_df.columns), None)
    carb_col = next((c for c in ['carbs','meal_dose']      if c in meals_df.columns), None)
    b_uid    = next((c for c in ['id','user_id']           if c in bolus_df.columns), None)
    b_date   = next((c for c in ['timestamp','datestamp']  if c in bolus_df.columns), None)
    dose_col = next((c for c in ['bolus','dose']           if c in bolus_df.columns), None)

    inst_df = inst_df.copy()
    inst_df['_uid'] = inst_df[inst_uid].apply(_clean)
    inst_df['_eid'] = inst_df[inst_id].apply(_clean)
    inst_df['_t0']  = (pd.to_datetime(inst_df['time_series_start']).dt.tz_localize(None)
                       + pd.Timedelta(hours=4))
    inst_df['_ws']  = inst_df['_t0'] + pd.Timedelta(minutes=_MEAL_WIN_MIN)
    inst_df['_we']  = inst_df['_t0'] + pd.Timedelta(minutes=_MEAL_WIN_MAX)

    meals_df = meals_df.copy()
    meals_df['_uid'] = meals_df[m_uid].apply(_clean)
    meal_ex_col = next((c for c in ['ex','instance_id'] if c in meals_df.columns), None)

    if meal_ex_col is not None:
        meals_df['_eid'] = meals_df[meal_ex_col].apply(_clean)
        carbs_per = meals_df.groupby('_eid')[carb_col].sum()
    else:
        meals_df[m_date] = pd.to_datetime(meals_df[m_date]).dt.tz_localize(None)
        has_time = ((meals_df[m_date].dt.hour != 0).any() or
                    (meals_df[m_date].dt.minute != 0).any())
        if has_time:
            merged = inst_df[['_eid','_uid','_ws','_we']].merge(
                meals_df[['_uid', m_date, carb_col]], on='_uid', how='left')
            merged = merged[(merged[m_date] >= merged['_ws']) &
                            (merged[m_date] <= merged['_we'])]
        else:
            meals_df['_date']   = meals_df[m_date].dt.date
            inst_df['_ws_date'] = inst_df['_ws'].dt.date
            inst_df['_we_date'] = inst_df['_we'].dt.date
            merged = inst_df[['_eid','_uid','_ws_date','_we_date']].merge(
                meals_df[['_uid','_date', carb_col]], on='_uid', how='left')
            merged = merged[(merged['_date'] >= merged['_ws_date']) &
                            (merged['_date'] <= merged['_we_date'])]
        carbs_per = merged.groupby('_eid')[carb_col].sum()

    bolus_per = pd.Series(dtype=float)
    if all([b_date, dose_col]) and not bolus_df.empty:
        bolus_df = bolus_df.copy()
        bolus_ex_col = next((c for c in ['ex','instance_id'] if c in bolus_df.columns), None)
        if bolus_ex_col is not None:
            bolus_df['_eid'] = bolus_df[bolus_ex_col].apply(_clean)
            bolus_per = bolus_df.groupby('_eid')[dose_col].sum()
        else:
            bolus_df['_uid'] = bolus_df[b_uid].apply(_clean)
            bolus_df[b_date] = pd.to_datetime(bolus_df[b_date]).dt.tz_localize(None)
            bmerged = inst_df[['_eid','_uid','_ws','_we']].merge(
                bolus_df[['_uid', b_date, dose_col]], on='_uid', how='left')
            bmerged = bmerged[(bmerged[b_date] >= bmerged['_ws']) &
                              (bmerged[b_date] <= bmerged['_we'])]
            bolus_per = bmerged.groupby('_eid')[dose_col].sum()

    imap   = {}
    counts = {'MEAL': 0, 'CARB_INSTANCE': 0, 'CORRECTION': 0, 'STABLE': 0}
    for eid in tqdm(list(inst_df['_eid']), desc='  Classifying', leave=False):
        carbs = carbs_per.get(eid, 0.0)
        bolus = bolus_per.get(eid, 0.0)
        if carbs >= _MEAL_CARBS:
            imap[eid] = 'MEAL';          counts['MEAL'] += 1
        elif carbs >= _MIN_CARBS:
            imap[eid] = 'CARB_INSTANCE'; counts['CARB_INSTANCE'] += 1
        elif bolus >= _MIN_BOLUS:
            imap[eid] = 'CORRECTION';    counts['CORRECTION'] += 1
        else:
            imap[eid] = 'STABLE';        counts['STABLE'] += 1

    print(f'  Classification counts: {counts}')
    with open(cache_path, 'wb') as f:
        pickle.dump(imap, f)
    return imap


# ============================================================================
# METRICS
# ============================================================================

def interval_score(obs, lo, hi, alpha=0.05):
    width  = hi - lo
    pen_lo = (2/alpha) * np.maximum(lo - obs, 0)
    pen_hi = (2/alpha) * np.maximum(obs - hi, 0)
    return float(np.mean(width + pen_lo + pen_hi))


def clarke_zones(ref, pred):
    """Clarke Error Grid zones A, D, E (Clarke et al., Diabetes Care 1987), mg/dL.
    Standard precedence A, E, C, then D (C checked so D never overlaps it).
    Arrays broadcast. Returns boolean masks (zA, zD, zE)."""
    zA = (np.abs(pred - ref) <= 0.2 * ref) | ((ref <= 70) & (pred <= 70))
    zE = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    zC = (((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) |
          ((ref >= 130) & (ref <= 180) & (pred <= 1.4 * ref - 182)))
    zD_raw = (((ref >= 240) & (pred >= 70) & (pred <= 180)) |
              ((ref <= 175 / 3) & (pred >= 70) & (pred <= 180)) |
              ((ref >= 175 / 3) & (ref <= 70) & (pred >= 1.2 * ref)))
    zD = zD_raw & ~zA & ~zE & ~zC
    return zA, zD, zE


def compute_instance_metrics(obs_i, samples_i, horizon_steps):
    """
    Compute per-instance metrics at a given horizon.
    Returns dict with model-prefix-agnostic keys (caller adds prefix).

    directional metrics
    -----------------------
    P_Net_Trend (float in [0,1] or NaN):
        Fraction of posterior samples where sign(s[k,T-1] - s[k,0])
        equals sign(o[T-1] - o[0]).  Pure sign agreement — no ±threshold.
        NaN when o[T-1] == o[0] (flat observed; sign undefined).

    P_Mean_Dir_Bias (float, mg/dL, can be negative):
        Mean over posterior samples of (s[k,T-1] - o[T-1]).
        Measures systematic endpoint bias across ALL instances.
        Negative = model over-predicts glucose drop (Bayesian Loop at 4h).
        Not bounded; belongs in main results table, NOT tier heatmap.

    P_Traj_Corr (float in [-1,1] or NaN):
        Pearson correlation between the posterior-mean trajectory and the
        observed trajectory over the full forecast window. Magnitude/shape-
        aware companion to P_Net_Trend (which only checks endpoint sign).
        NaN if either trajectory is flat (zero variance).
    """
    nan_keys = [
        'RMSE', 'IS', 'CRPS', 'Coverage', 'Width',
        'CEG_A', 'CEG_D', 'CEG_E',
        'DTW',
        # directional
        'P_Net_Trend',
        'P_Mean_Dir_Bias',
        'P_Traj_Corr',
        # posterior safety
        'P_CEG_A', 'P_CEG_D', 'P_CEG_E',
        'P_CEG_A_samples', 'P_CEG_D_samples', 'P_CEG_E_samples',
        'P_Hypo_Prob', 'P_Hyper_Prob',
        'P_Hypo_Sens', 'P_Hypo_Spec',
        'P_Hyper_Sens', 'P_Hyper_Spec',
        'P_Hypo_Prob_60min', 'P_Hyper_Prob_60min',
        'P_Hypo_Sens_60min', 'P_Hypo_Spec_60min',
        'P_Hyper_Sens_60min', 'P_Hyper_Spec_60min',
    ]
    if samples_i.shape[0] == 0 or np.all(np.isnan(samples_i)):
        return {k: np.nan for k in nan_keys}

    T      = min(horizon_steps, obs_i.shape[0], samples_i.shape[1])
    o      = obs_i[:T]
    s      = samples_i[:, :T]
    mean_t = np.mean(s, axis=0)
    lo, hi = np.percentile(s, [2.5, 97.5], axis=0)

    out = {}

    # ── Standard point / interval metrics ────────────────────────────────────
    out['RMSE']     = float(np.sqrt(np.mean((o - mean_t) ** 2)))
    out['Coverage'] = float(np.mean((o >= lo) & (o <= hi)))
    out['Width']    = float(np.mean(hi - lo))
    out['IS']       = interval_score(o, lo, hi)

    # CRPS in energy form, with the spread term E|X-X'| averaged over all sample
    # pairs, via the sorted-sample identity
    # mean_{i,j}|x_i-x_j| = (2/n^2) * sum_i (2i-n-1) x_(i).
    n_s = s.shape[0]
    w = (2.0 * np.arange(1, n_s + 1) - n_s - 1.0)
    crps_vals = []
    for t in range(T):
        samp_t = s[:, t]
        spread = 2.0 * float(np.dot(w, np.sort(samp_t))) / (n_s * n_s)
        crps_vals.append(float(np.mean(np.abs(samp_t - o[t])) - 0.5 * spread))
    out['CRPS'] = float(np.mean(crps_vals))

    # ── CEG (posterior mean) ──────────────────────────────────────────────────
    # Standard Clarke et al. (1987) zones (clarke_zones).
    zA, zD, zE = clarke_zones(o, mean_t)
    out['CEG_A'] = float(np.mean(zA) * 100)
    out['CEG_D'] = float(np.mean(zD) * 100)
    out['CEG_E'] = float(np.mean(zE) * 100)

    # ── DTW ───────────────────────────────────────────────────────────────────
    if HAS_DTW:
        try:
            out['DTW'] = float(fastdtw(
                o.reshape(-1, 1), mean_t.reshape(-1, 1), dist=euclidean)[0])
        except Exception:
            out['DTW'] = np.nan
    else:
        out['DTW'] = np.nan

    # =========================================================================
    # POSTERIOR (P_) METRICS
    # =========================================================================

    # ── P_Net_Trend — pure sign agreement, no threshold ─────────────
    # Sidesteps the ±5 vs ±20 mg/dL threshold controversy entirely.
    # NaN when observed net change is exactly 0 (sign undefined — rare).
    obs_net = o[-1] - o[0]
    if obs_net == 0.0:
        out['P_Net_Trend']   = np.nan
        out['Obs_Net_Sign']  = 0.0
    else:
        obs_sign     = np.sign(obs_net)           # +1 or -1
        sample_nets  = s[:, -1] - s[:, 0]         # (n_samp,)
        sample_signs = np.sign(sample_nets)        # +1, 0, or -1
        # samples with net==0 count as disagreement (conservative, rare)
        out['P_Net_Trend']  = float(np.mean(sample_signs == obs_sign))
        # Exposed for meal/bolus-conditional directional-accuracy analysis
        # (compute_meal_response_accuracy) — ground truth, identical across
        # models for the same instance; downstream code references the
        # G-prefixed copy, same convention as the hypo/hyper truth sentinels.
        out['Obs_Net_Sign'] = float(obs_sign)

    # ── P_Mean_Dir_Bias — Mean Directional Bias (mg/dL) ──────────────────
    # Unconditional over ALL instances.  Negative → systematic downward bias.
    # Captures what magnitude-sensitive slope metrics tried to measure, without
    # the phase-penalisation problem of first-difference correlation.
    out['P_Mean_Dir_Bias'] = float(np.mean(s[:, -1] - o[-1]))

    # ── P_Traj_Corr — Trajectory Correlation ──────────────────────────────────
    # Pearson correlation between the posterior-mean trajectory and the
    # observed trajectory over the full forecast window. Unlike P_Net_Trend
    # (endpoint sign only), this rewards matching the shape/magnitude of the
    # trajectory, not just its net direction. NaN if either trajectory is
    # flat (undefined correlation).
    if T < 2 or np.std(o) == 0 or np.std(mean_t) == 0:
        out['P_Traj_Corr'] = np.nan
    else:
        out['P_Traj_Corr'] = float(pearsonr(mean_t, o)[0])

    # ── P_CEG (full posterior) ────────────────────────────────────────────────
    # CEG zones are reported for the posterior-mean trajectory (P_CEG_*); the
    # per-sample zone fractions are kept as P_CEG_*_samples.
    zA_p, zD_p, zE_p = clarke_zones(o[np.newaxis, :], s)
    out['P_CEG_A_samples'] = float(np.mean(zA_p) * 100)
    out['P_CEG_D_samples'] = float(np.mean(zD_p) * 100)
    out['P_CEG_E_samples'] = float(np.mean(zE_p) * 100)
    out['P_CEG_A'] = out['CEG_A']
    out['P_CEG_D'] = out['CEG_D']
    out['P_CEG_E'] = out['CEG_E']

    # ── P_Hypo/Hyper probs and sens/spec ──────────────────────────────────────
    out['P_Hypo_Prob']  = float(np.mean(np.min(s, axis=1) < 70))
    out['P_Hyper_Prob'] = float(np.mean(np.max(s, axis=1) > 180))

    if np.any(o < 70):
        out['P_Hypo_Sens'] = out['P_Hypo_Prob'];  out['P_Hypo_Spec'] = np.nan
    else:
        out['P_Hypo_Sens'] = np.nan;              out['P_Hypo_Spec'] = 1.0 - out['P_Hypo_Prob']

    if np.any(o > 180):
        out['P_Hyper_Sens'] = out['P_Hyper_Prob']; out['P_Hyper_Spec'] = np.nan
    else:
        out['P_Hyper_Sens'] = np.nan;              out['P_Hyper_Spec'] = 1.0 - out['P_Hyper_Prob']

    steps_60min = min(12, T)
    out['P_Hypo_Prob_60min']  = float(np.mean(np.min(s[:, :steps_60min], axis=1) < 70))
    out['P_Hyper_Prob_60min'] = float(np.mean(np.max(s[:, :steps_60min], axis=1) > 180))

    if np.any(o[:steps_60min] < 70):
        out['P_Hypo_Sens_60min'] = out['P_Hypo_Prob_60min']
        out['P_Hypo_Spec_60min'] = np.nan
    else:
        out['P_Hypo_Sens_60min'] = np.nan
        out['P_Hypo_Spec_60min'] = 1.0 - out['P_Hypo_Prob_60min']

    if np.any(o[:steps_60min] > 180):
        out['P_Hyper_Sens_60min'] = out['P_Hyper_Prob_60min']
        out['P_Hyper_Spec_60min'] = np.nan
    else:
        out['P_Hyper_Sens_60min'] = np.nan
        out['P_Hyper_Spec_60min'] = 1.0 - out['P_Hyper_Prob_60min']

    return out


# ============================================================================
# POST-PROCESSING: compute_and_fill_auroc  (retained for supplement)
# ============================================================================

def _auroc_score(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask   = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[mask], labels[mask]
    if len(scores) < 10 or labels.sum() < 2 or (1 - labels).sum() < 2:
        return np.nan
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        thresholds = np.linspace(0, 1, 201)
        tprs, fprs = [], []
        for tau in thresholds:
            alarm = scores >= tau
            tp = np.sum(alarm  & (labels == 1)); fn = np.sum(~alarm & (labels == 1))
            fp = np.sum(alarm  & (labels == 0)); tn = np.sum(~alarm & (labels == 0))
            tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        order = np.argsort(fprs)
        return float(np.trapz(np.array(tprs)[order], np.array(fprs)[order]))


def compute_and_fill_auroc(df):
    """
    Compute Hypo/Hyper AUROC for supplement / ROC figure.
    NOT included in METRICS_MAIN (Table 2).
    """
    df = df.copy()
    auroc_specs = [
        ('Hypo_AUROC',        'P_Hypo_Prob',       'P_Hypo_Sens'),
        ('Hyper_AUROC',       'P_Hyper_Prob',      'P_Hyper_Sens'),
        ('Hypo_AUROC_60min',  'P_Hypo_Prob_60min', 'P_Hypo_Sens_60min'),
        ('Hyper_AUROC_60min', 'P_Hyper_Prob_60min','P_Hyper_Sens_60min'),
    ]
    for prefix in ordered_prefixes():
        for out_suffix, score_suffix, sentinel_suffix in auroc_specs:
            score_col    = f'{prefix}_{score_suffix}'
            sentinel_col = f'{prefix}_{sentinel_suffix}'
            out_col      = f'{prefix}_{out_suffix}'
            out_all_col  = f'{prefix}_{out_suffix}_ALL'
            if score_col not in df.columns or sentinel_col not in df.columns:
                continue
            df[out_col] = np.nan; df[out_all_col] = np.nan
            true_labels = df[sentinel_col].notna().astype(float)
            for (horizon, itype), grp_idx in df.groupby(['horizon','itype']).groups.items():
                df.loc[grp_idx, out_col] = _auroc_score(
                    df.loc[grp_idx, score_col].values,
                    true_labels.loc[grp_idx].values)
            for horizon, grp_idx in df.groupby('horizon').groups.items():
                df.loc[grp_idx, out_all_col] = _auroc_score(
                    df.loc[grp_idx, score_col].values,
                    true_labels.loc[grp_idx].values)

    print('\n  AUROC summary (ALL itype, supplement):')
    for prefix, label in ([('G', LABEL_SAFE), ('L', LABEL_BASE), ('A', LABEL_AR1)]
                          + [(pfx, ALL_LABELS[mk]) for mk, pfx in EXTRA_MODELS]):
        for suffix in ['Hypo_AUROC', 'Hyper_AUROC']:
            col = f'{prefix}_{suffix}_ALL'
            if col not in df.columns: continue
            per_h = df.groupby('horizon')[col].first()
            vals  = '  '.join(f'{h}: {v:.3f}' for h, v in per_h.items()
                              if not np.isnan(v))
            print(f'    {label:<14} {suffix}: {vals}')
    return df


# ============================================================================
# BINARY ALARM OPERATING-POINT ANALYSIS
# ============================================================================
#
# Hypo/hyper sensitivity and specificity are reported elsewhere as continuous
# posterior-probability-weighted expectations. This section converts the
# alarm score into a binary decision at two operating points and reports the
# resulting false alarm rate (1 - specificity):
#   (a) a fixed sensitivity target of 0.90, common across models (matches the
#       reference lines in Figure~\ref{fig:roc_hypo});
#   (b) each model's own Youden's-J-optimal threshold (argmax sens+spec-1),
#       i.e. its best achievable sensitivity/specificity tradeoff.
# Reported over ALL instances/contexts at the 4-hour horizon, for the 60-min
# and 4-hr safety windows.

def _operating_point_stats(scores, labels, target_sens=0.90, target_spec=0.90):
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=float)
    mask   = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[mask], labels[mask]
    out = {'thresh90': np.nan, 'spec90': np.nan, 'far90': np.nan, 'ppv90': np.nan,
           'threshJ': np.nan, 'sensJ': np.nan, 'specJ': np.nan, 'farJ': np.nan, 'ppvJ': np.nan,
           # Fixed-specificity operating point:
           # "if I insist on <=10% false alarms, how much sensitivity do I get".
           'threshSpec90': np.nan, 'sensAtSpec90': np.nan, 'ppvAtSpec90': np.nan}
    if len(scores) < 10 or labels.sum() < 2 or (1 - labels).sum() < 2:
        return out
    thresholds = np.linspace(0, 1, 1001)
    # Vectorized threshold sweep (was a 1001-iteration Python loop) -- this
    # function is now called per-user/per-context (up to ~3,200 times for a
    # full 8-model run), not just a handful of times for pooled-cohort
    # stats, so the sweep itself needed to stop being the bottleneck.
    # alarm[i, j] = would instance i alarm at threshold j.
    alarm = scores[:, None] >= thresholds[None, :]           # (n_instances, 1001)
    pos = labels == 1
    tp = alarm[pos].sum(axis=0).astype(float)
    fn = (~alarm[pos]).sum(axis=0).astype(float)
    tn = (~alarm[~pos]).sum(axis=0).astype(float)
    fp = alarm[~pos].sum(axis=0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        senss = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        specs = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        ppvs  = np.where(tp + fp > 0, tp / (tp + fp), np.nan)

    # Fixed sensitivity target: among thresholds achieving sens >= target,
    # take the highest (most conservative alarm rule still meeting the
    # target), which minimizes the false alarm rate at that sensitivity.
    ok = senss >= target_sens
    if ok.any():
        idx  = np.where(ok)[0]
        best = idx[np.argmax(thresholds[idx])]
        out['thresh90'] = float(thresholds[best])
        out['spec90']   = float(specs[best])
        out['far90']    = float(1 - specs[best])
        out['ppv90']    = float(ppvs[best])

    # Fixed specificity target: among thresholds achieving spec >= target,
    # take the LOWEST (most liberal alarm rule still meeting the target),
    # which maximizes the resulting sensitivity at that specificity.
    ok_spec = specs >= target_spec
    if ok_spec.any():
        idx_s     = np.where(ok_spec)[0]
        best_spec = idx_s[np.argmin(thresholds[idx_s])]
        out['threshSpec90'] = float(thresholds[best_spec])
        out['sensAtSpec90'] = float(senss[best_spec])
        out['ppvAtSpec90']  = float(ppvs[best_spec])

    # Youden's J: threshold maximizing sensitivity + specificity - 1.
    j     = senss + specs - 1
    jbest = int(np.argmax(j))
    out['threshJ'] = float(thresholds[jbest])
    out['sensJ']   = float(senss[jbest])
    out['specJ']   = float(specs[jbest])
    out['farJ']    = float(1 - specs[jbest])
    out['ppvJ']    = float(ppvs[jbest])
    return out


def compute_operating_points(df):
    """Returns {(prefix, name): stats_dict} for Hypo/Hyper x 60min/4hr,
    computed over ALL instances/contexts at the 4-hour horizon."""
    d4h = df[df['horizon'] == '240 min']
    specs = [
        ('Hypo_60min',  'P_Hypo_Prob_60min',  'P_Hypo_Sens_60min'),
        ('Hypo_4hr',    'P_Hypo_Prob',        'P_Hypo_Sens'),
        ('Hyper_60min', 'P_Hyper_Prob_60min', 'P_Hyper_Sens_60min'),
        ('Hyper_4hr',   'P_Hyper_Prob',       'P_Hyper_Sens'),
    ]
    results = {}
    for prefix in ordered_prefixes():
        for name, score_suffix, sentinel_suffix in specs:
            score_col    = f'{prefix}_{score_suffix}'
            sentinel_col = f'{prefix}_{sentinel_suffix}'
            if score_col not in d4h.columns or sentinel_col not in d4h.columns:
                continue
            labels = d4h[sentinel_col].notna().astype(float)
            results[(prefix, name)] = _operating_point_stats(
                d4h[score_col].values, labels.values)
    return results


def write_table_false_alarm(f, df, out_dir=None):
    """Two supplementary tables : false alarm rate at a
    fixed sensitivity=0.90 target, and at each model's Youden-optimal point.

    LAYOUT: models are ROWS, (context x metric) are columns. The original
    layout had it the other way round -- 4 context rows x 3 metrics x N
    models = 3N numeric columns -- which was already ~2000pt overfull at 7
    models (pdflatex truncated it after ~4 columns in the compiled PDF) and
    would have been 27 columns at 9. Transposed it is a fixed 12 numeric
    columns regardless of how many models exist, with one row per model, which
    also matches the models-as-rows convention already used by the tier table.
    """
    op = compute_operating_points(df)
    ordered = ordered_models(df)
    contexts = [('Hypo 60-min',  'Hypo_60min'),
                ('Hypo 4-hr',    'Hypo_4hr'),
                ('Hyper 60-min', 'Hyper_60min'),
                ('Hyper 4-hr',   'Hyper_4hr')]

    def _pct(v): return '—' if np.isnan(v) else f'{v*100:.1f}'
    def _tau(v): return '—' if np.isnan(v) else f'{v:.2f}'

    def _emit(handles, caption, label, sub_hdrs, keys, fmts):
        """sub_hdrs: 3 per-context column names; keys/fmts: matching op keys."""
        n_sub = len(sub_hdrs)
        ncols = len(contexts) * n_sub
        for h in handles:
            h.write(f'% TABLE: {label} \n')
            h.write(r'\begin{table*}[htbp]' + '\n' + r'\centering' + '\n'
                    + r'\footnotesize' + '\n')
            h.write(f'\\caption{{{caption}}}\n\\label{{{label}}}\n')
            h.write(f'\\begin{{tabular}}{{l{"r" * ncols}}}\n' + r'\toprule' + '\n')
            # context group header
            h.write('Model & ' + ' & '.join(
                f'\\multicolumn{{{n_sub}}}{{c}}{{{cl}}}' for cl, _ in contexts)
                + r' \\' + '\n')
            offs = [2 + i * n_sub for i in range(len(contexts))]
            h.write(''.join(f'\\cmidrule(lr){{{o}-{o + n_sub - 1}}}' for o in offs) + '\n')
            # metric sub-header
            h.write(' & ' + ' & '.join(sub_hdrs * len(contexts)) + r' \\' + '\n'
                    + r'\midrule' + '\n')
            for lbl, pfx in ordered:
                cells = []
                for _, key in contexts:
                    st = op.get((pfx, key), {})
                    cells += [fn(st.get(k, np.nan)) for k, fn in zip(keys, fmts)]
                h.write(f'  {lbl} & ' + ' & '.join(cells) + r' \\' + '\n')
            h.write(r'\bottomrule' + '\n' + r'\end{tabular}' + '\n'
                    + r'\end{table*}' + '\n\n')
            if h is not f: h.close()

    # ── Table A: fixed sensitivity = 0.90 ──────────────────────────────────
    caption90 = (
        r'Binary alarm operating-point analysis at a fixed sensitivity '
        r'target of $0.90$: the continuous posterior-probability alarm score is '
        r'converted to a binary decision at the threshold $\tau$ achieving '
        r'sensitivity $\geq 0.90$ (matching the reference line in '
        r'Figure~\ref{fig:roc_hypo}). Spec = specificity at $\tau$; '
        r'FAR = false alarm rate ($1-$Spec, \%), the fraction of true-negative '
        r'instances that would trigger a false alarm at this threshold. '
        r'A dash means the sensitivity target is unreachable at any threshold.'
    )
    handles = [f]
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
        handles.append(open(os.path.join(out_dir, 'tables', 'tab_false_alarm_fixed.tex'), 'w'))
    _emit(handles, caption90, 'tab:false_alarm_fixed',
          [r'$\tau$', r'Spec (\%)', r'FAR (\%)'],
          ['thresh90', 'spec90', 'far90'], [_tau, _pct, _pct])

    # ── Table B: per-model Youden's-J-optimal point ────────────────────────
    captionJ = (
        r"Binary alarm operating-point analysis at each model's own "
        r"Youden's-$J$-optimal threshold (maximizing sensitivity $+$ "
        r"specificity $-1$), i.e.\ each model's best achievable "
        r"sensitivity--specificity tradeoff rather than a common fixed target. "
        r'FAR = false alarm rate ($1-$specificity, \%).'
    )
    handles = [f]
    if out_dir:
        handles.append(open(os.path.join(out_dir, 'tables', 'tab_false_alarm_youden.tex'), 'w'))
    _emit(handles, captionJ, 'tab:false_alarm_youden',
          [r'Sens (\%)', r'Spec (\%)', r'FAR (\%)'],
          ['sensJ', 'specJ', 'farJ'], [_pct, _pct, _pct])


def compute_user_metrics(args):
    """extra_path_pairs: list of (prefix, path_or_None) — one entry per
    EXTRA_MODELS tuple. Unlike G/L/A (always computed, NaN-filled if
    missing), an extra model's columns are simply omitted when its path
    is None — preserves the original R/T-only-if-present behavior for any
    number of extra models.

    Bundles are loaded HERE, inside the worker process, one user at a time
    — not pre-loaded in the main process and pickled across the Pool IPC
    boundary. This is what keeps peak memory bounded by --workers instead
    of by cohort-size x model-count (see find_bundle_paths)."""
    uid, g_path, l_path, a_path, imap, horizons, extra_path_pairs = args
    bundle_g = load_bundle(g_path)
    obs_g  = bundle_g['obs'];  samp_g = bundle_g['samples']
    test_ids_g = bundle_g['test_ids']
    bundle_l = load_bundle(l_path) if l_path else _nan_bundle(bundle_g)
    obs_l  = bundle_l['obs'];  samp_l = bundle_l['samples']
    has_ar1 = a_path is not None
    bundle_a = load_bundle(a_path) if has_ar1 else None
    obs_a   = bundle_a['obs']     if has_ar1 else obs_g
    samp_a  = bundle_a['samples'] if has_ar1 else np.full_like(samp_g, np.nan, dtype=float)

    extra_active = [(pfx, load_bundle(p)) for pfx, p in extra_path_pairs if p is not None]
    n = min([obs_g.shape[0], obs_l.shape[0], obs_a.shape[0],
             samp_g.shape[1], samp_l.shape[1], samp_a.shape[1]]
            + [b['obs'].shape[0] for _, b in extra_active]
            + [b['samples'].shape[1] for _, b in extra_active])
    rows = []
    for i in range(n):
        eid = str(test_ids_g[i]) if i < len(test_ids_g) else str(i)
        if eid.endswith('.0'): eid = eid[:-2]
        itype = imap.get(eid, 'STABLE')
        for h_steps, h_label in zip(horizons, HORIZON_LABELS):
            mg = compute_instance_metrics(obs_g[i], samp_g[:, i], h_steps)
            ml = compute_instance_metrics(obs_l[i], samp_l[:, i], h_steps)
            ma = compute_instance_metrics(obs_a[i], samp_a[:, i], h_steps)
            row = {
                **{f'G_{k}': v for k, v in mg.items()},
                **{f'L_{k}': v for k, v in ml.items()},
                **{f'A_{k}': v for k, v in ma.items()},
                'eid': eid, 'itype': itype, 'horizon': h_label, 'uid': uid,
            }
            for prefix, bundle in extra_active:
                m = compute_instance_metrics(bundle['obs'][i], bundle['samples'][:, i], h_steps)
                row.update({f'{prefix}_{k}': v for k, v in m.items()})
            rows.append(row)
    return rows


# ============================================================================
# STATS HELPERS
# ============================================================================

def wilcoxon_pval(vals_a, vals_b):
    """Instance-level paired Wilcoxon (N=instances). Retained for supplementary
    sensitivity checks only — see wilcoxon_pval_peruser for the primary test.
    Instances are drawn from overlapping rolling windows per user (only a
    4-hour minimum gap against a 4-hour horizon), so adjacent instances share
    autocorrelated errors and violate the independence assumption behind
    Wilcoxon. This inflates N and anti-conservatively shrinks p-values."""
    a = np.array(vals_a, dtype=float); b = np.array(vals_b, dtype=float)
    if len(a) == 0 or len(b) == 0 or len(a) != len(b): return np.nan
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10: return np.nan
    try:
        _, p = wilcoxon(a[mask], b[mask]); return float(p)
    except Exception:
        return np.nan


def wilcoxon_pval_peruser(df, col_a, col_b, uid_col='uid'):
    """Paired Wilcoxon on per-user means (N=users). This is the primary test
    unit: users are independent, whereas instances within a user are not
    (see wilcoxon_pval docstring). Each user's per-instance values are
    averaged first, then the paired test runs on those N=395 aggregates."""
    sub = df[[uid_col, col_a, col_b]].dropna(subset=[col_a, col_b])
    if sub.empty: return np.nan
    agg = sub.groupby(uid_col)[[col_a, col_b]].mean()
    if len(agg) < 10: return np.nan
    try:
        _, p = wilcoxon(agg[col_a].values, agg[col_b].values); return float(p)
    except Exception:
        return np.nan


def pval_str(p):
    if np.isnan(p): return ''
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    return 'ns'


# ============================================================================
# LATEX HELPERS
# ============================================================================

def latex_header(caption, label, cols):
    col_spec = 'l' + 'r' * len(cols)
    return '\n'.join([
        r'\begin{table}[htbp]', r'\centering', r'\small',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        f'\\begin{{tabular}}{{{col_spec}}}', r'\toprule',
        ' & '.join(['Metric'] + cols) + r' \\', r'\midrule',
    ])


def latex_footer():
    return '\n'.join([r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    return f'{val:.{decimals}f}'


def write_metric_row_three(f, metric_label, g_vals, l_vals, a_vals,
                           lower_better=True, decimals=2, pct=False,
                           signed=False, extra_vals=None, pvals=None):
    """
    One table row for three models (+ optional extra baselines) + p-values.
    signed=True: P_Mean_Dir_Bias — no automatic bolding (diagnostic, not rankable).
    extra_vals: optional list of value-lists for EXTRA_MODELS (e.g. AR1RLS,
    TransformerMC), appended after Pure AR1. p-value for each is vs. g_vals,
    mirroring the existing G-vs-L / G-vs-A convention.
    pvals: optional precomputed p-values (e.g. per-user Wilcoxon via
    wilcoxon_pval_peruser) to use instead of the default instance-level test.
    """
    scale = 100 if pct else 1
    extra_vals = extra_vals or []
    all_vals  = [g_vals, l_vals, a_vals] + extra_vals
    all_means = [np.nanmean(v) if v else np.nan for v in all_vals]

    if signed:
        def _fmt_bold(m): return fmt(m * scale, decimals)
    else:
        best = min if lower_better else max
        finite = [v for v in all_means if not np.isnan(v)]
        best_mean = best(finite) if finite else np.nan
        def _fmt_bold(m):
            s = fmt(m * scale, decimals)
            return f'\\textbf{{{s}}}' if (not np.isnan(m) and m == best_mean) else s

    if pvals is None:
        pvals = [wilcoxon_pval(g_vals, v) for v in [l_vals, a_vals] + extra_vals]
    means_str = ' & '.join(_fmt_bold(m) for m in all_means)
    pvals_str = ' & '.join(pval_str(p) for p in pvals)
    f.write(f'  {metric_label} & {means_str} & {pvals_str} \\\\\n')


# ============================================================================
# TABLE 1 — DATASET STATISTICS
# ============================================================================

def write_table1_dataset(f, common_users, g_paths, imap):
    """g_paths: {uid: path} for the Global/Bayesian-Loop bundle — loaded one
    user at a time (not requiring the whole cohort pre-loaded)."""
    n_users = len(common_users)
    train_counts = {t: 0 for t in INSTANCE_TYPES}; train_counts['TOTAL'] = 0
    test_counts  = {t: 0 for t in INSTANCE_TYPES}; test_counts['TOTAL']  = 0
    train_per_user, test_per_user = [], []
    for uid in common_users:
        path = g_paths.get(uid)
        bundle = load_bundle(path) if path else None
        if not bundle: continue
        test_ids = bundle.get('test_ids', [])
        test_per_user.append(len(test_ids))
        for eid in test_ids:
            eid_str = str(eid).replace('.0', '')
            itype = imap.get(eid_str, 'STABLE')
            test_counts[itype] += 1; test_counts['TOTAL'] += 1
        train_ids = bundle.get('train_ids', [])
        train_per_user.append(len(train_ids))
        for eid in train_ids:
            eid_str = str(eid).replace('.0', '')
            itype = imap.get(eid_str, 'STABLE')
            train_counts[itype] += 1; train_counts['TOTAL'] += 1

    def _iqr(counts):
        if not counts or sum(counts) == 0: return '---'
        return (f'{np.median(counts):.0f} '
                f'[{np.percentile(counts,25):.0f}--{np.percentile(counts,75):.0f}]')

    total_inst = train_counts['TOTAL'] + test_counts['TOTAL']
    f.write('% TABLE 1: DATASET STATISTICS\n')
    f.write(r'\begin{table}[!tp]' + '\n\centering\n')
    f.write(r'\caption{Dataset statistics for the selected T1DEXI cohort.}' + '\n')
    f.write(r'\label{tab:dataset}' + '\n')
    f.write(r'\begin{adjustbox}{max width=\linewidth}' + '\n')
    f.write(r'\begin{tabular}{lrrr}' + '\n\\toprule\n')
    f.write(r'Characteristic & Train Set & Test Set & Total \\' + '\n\midrule\n')
    f.write(f'Participants & --- & --- & {n_users} \\\\\n')
    f.write(f'Total Instances & {train_counts["TOTAL"]:,} & {test_counts["TOTAL"]:,} & {total_inst:,} \\\\\n')
    f.write(f'Instances per user (Median [IQR]) & {_iqr(train_per_user)} & {_iqr(test_per_user)} & --- \\\\\n')
    f.write(r'\midrule' + '\n')
    for itype in INSTANCE_TYPES:
        label  = INSTANCE_LABELS[itype]
        tr_c   = train_counts[itype]; te_c = test_counts[itype]; tot_c = tr_c + te_c
        tr_pct = tr_c / train_counts['TOTAL'] * 100 if train_counts['TOTAL'] > 0 else 0
        te_pct = te_c / test_counts['TOTAL']  * 100 if test_counts['TOTAL']  > 0 else 0
        tot_pct= tot_c / total_inst * 100 if total_inst > 0 else 0
        f.write(f'\\quad {label} & {tr_c:,} ({tr_pct:.1f}\\%) & '
                f'{te_c:,} ({te_pct:.1f}\\%) & {tot_c:,} ({tot_pct:.1f}\\%) \\\\\n')
    f.write(r'\midrule' + '\n')
    f.write(r'Prediction horizon & \multicolumn{3}{r}{4 hours (48 steps $\times$ 5 min)} \\' + '\n')
    f.write(r'Training window & \multicolumn{3}{r}{4 hours (48 CGM readings)} \\' + '\n')
    f.write(r'Train/test split & \multicolumn{3}{r}{Chronological + 8h washout} \\' + '\n')
    f.write(r'\bottomrule' + '\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n\n')


# ============================================================================
# TABLE 2 — MAIN RESULTS 
# ============================================================================
#
# Row tuple: (label, col_suffix, lower_better, pct, decimals, signed)
# signed=True → no bolding (P_Mean_Dir_Bias is diagnostic, not rankable)
#
METRICS_MAIN = [
    # ── Statistical ──────────────────────────────────────────────────────────
    ('RMSE (mg/dL)',                    'RMSE',               True,  False, 1, False),
    ('CRPS (mg/dL)',                    'CRPS',               True,  False, 2, False),
    ('Coverage (95\\% PI)',             'Coverage',           False, True,  1, False),
    ('PI Width (mg/dL)',                'Width',              True,  False, 1, False),
    ('Interval Score',                  'IS',                 True,  False, 1, False),
    (r'\midrule', None, None, None, None, None),
    # ── Directional ──────────────────────────────────────────────────────────
    # Net Trend Accuracy: fraction of samples with correct endpoint sign.
    # No ±threshold. NaN when observed net change = 0.
    # Column name kept as P_Net_Trend for downstream script compatibility.
    ('Net Trend Accuracy (\\%)',        'P_Net_Trend',        False, True,  1, False),
    # Trajectory Correlation: Pearson r between predicted-mean and observed
    # trajectory over the full horizon. Magnitude/shape-aware companion to
    # Net Trend Accuracy (which is endpoint-sign-only).
    ('Trajectory Correlation (r)',      'P_Traj_Corr',        False, False, 2, False),
    # Mean Directional Bias: mean endpoint error in mg/dL over ALL instances.
    # Negative = systematic downward bias (over-predicted drop).
    # Not bolded — diagnostic, not a ranking metric.
    ('Mean Directional Bias (mg/dL)',   'P_Mean_Dir_Bias',    False, False, 1, True),
    (r'\midrule', None, None, None, None, None),
    # ── Hypoglycaemia safety ──────────────────────────────────────────────────
    ('Hypo Sensitivity (4-hr, \\%)',    'P_Hypo_Sens',        False, True,  1, False),
    ('Hypo Specificity (4-hr, \\%)',    'P_Hypo_Spec',        False, True,  1, False),
    ('Hypo Sensitivity (60-min, \\%)',  'P_Hypo_Sens_60min',  False, True,  1, False),
    ('Hypo Specificity (60-min, \\%)',  'P_Hypo_Spec_60min',  False, True,  1, False),
    (r'\midrule', None, None, None, None, None),
    # ── Hyperglycaemia safety ─────────────────────────────────────────────────
    ('Hyper Sensitivity (4-hr, \\%)',   'P_Hyper_Sens',       False, True,  1, False),
    ('Hyper Specificity (4-hr, \\%)',   'P_Hyper_Spec',       False, True,  1, False),
    ('Hyper Sensitivity (60-min, \\%)', 'P_Hyper_Sens_60min', False, True,  1, False),
    ('Hyper Specificity (60-min, \\%)', 'P_Hyper_Spec_60min', False, True,  1, False),
    (r'\midrule', None, None, None, None, None),
    # ── Clarke Error Grid ─────────────────────────────────────────────────────
    ('CEG Zone A (\\%)',                'P_CEG_A',            False, False, 1, False),
    ('CEG Zone D (\\%)',                'P_CEG_D',            True,  False, 2, False),
    ('CEG Zone E (\\%)',                'P_CEG_E',            True,  False, 2, False),
]


def _write_three_model_body(f, df, metrics, ordered):
    """ordered: [(label, prefix)] in DISPLAY_ORDER from ordered_models(df).
    ordered[0] is the p-value reference (Bayesian Loop/'G') and ordered[1] is
    LSTM-MC/'L' -- those two keep their paired dropna; every model from index
    2 on gets the independent dropna Pure AR1 always had.
    """
    prefixes = [pfx for _, pfx in ordered]
    ref, second = prefixes[0], prefixes[1]
    d = df[df['horizon'] == '240 min']
    for row in metrics:
        label, col, lower, pct, dec, signed = row
        if label.startswith(r'\midrule'):
            f.write(r'\midrule' + '\n'); continue
        paired = d[[f'{ref}_{col}', f'{second}_{col}']].dropna()
        vals = [paired[f'{ref}_{col}'].tolist(), paired[f'{second}_{col}'].tolist()]
        vals += [d[f'{pfx}_{col}'].dropna().tolist() for pfx in prefixes[2:]]
        # Primary significance test is per-user (N=users), not per-instance:
        # instances are autocorrelated (overlapping rolling windows), users
        # are independent. See wilcoxon_pval_peruser docstring.
        pvals = [wilcoxon_pval_peruser(d, f'{ref}_{col}', f'{pfx}_{col}')
                 for pfx in prefixes[1:]]
        write_metric_row_three(f, label, vals[0], vals[1],
                               vals[2] if len(vals) > 2 else [],
                               lower_better=lower, decimals=dec, pct=pct,
                               signed=signed, extra_vals=vals[3:], pvals=pvals)


def ordered_models(df):
    """[(label, prefix)] for every model with real data in df, in the paper's
    canonical DISPLAY_ORDER (see that constant for the full rationale).

    Position 0 is always Bayesian Loop ('G'), which every table's p-values are
    computed against, and position 1 is always LSTM-MC ('L'), preserving the
    G/L paired-dropna in _write_three_model_body. Every model from position 2
    on gets the same independent-dropna treatment Pure AR1 already had, so
    switching to this ordering is a pure reordering with no change to how any
    number is computed.
    """
    avail = [(ALL_LABELS[MODEL_SAFE], 'G'), (ALL_LABELS[MODEL_BASE], 'L'),
             (ALL_LABELS[MODEL_AR1], 'A')]
    avail += [(ALL_LABELS[mk], pfx) for mk, pfx in EXTRA_MODELS]
    have = [(lbl, pfx) for lbl, pfx in avail
            if f'{pfx}_RMSE' in df.columns and df[f'{pfx}_RMSE'].notna().any()]
    by_pfx = {pfx: (lbl, pfx) for lbl, pfx in have}
    out = [by_pfx[p] for p in DISPLAY_ORDER if p in by_pfx]
    out += [t for t in have if t[1] not in DISPLAY_ORDER]   # never silently drop
    return out


def _extra_models_present(df):
    """Which EXTRA_MODELS (AR1RLS, TransformerMC, LSTM-Ensemble, LSTM-Conformal,
    Transformer-Conformal) have real data in df."""
    present = []
    for model_key, pfx in EXTRA_MODELS:
        col = f'{pfx}_RMSE'
        if col in df.columns and df[col].notna().any():
            present.append((ALL_LABELS[model_key], pfx))
    return present


def get_active_models(df):
    """Ordered (model_key, prefix) list for every model with real (non-NaN)
    RMSE data in df — G, L, A, plus any EXTRA_MODELS present. Single source
    of truth for the plotting functions, replacing the old
    has_lstm/has_ar1/has_rls/has_transformer boolean-kwarg pattern (which
    doesn't scale past 2 extra models)."""
    candidates = [(MODEL_SAFE, 'G'), (MODEL_BASE, 'L'), (MODEL_AR1, 'A')] + EXTRA_MODELS
    have = [(mk, pfx) for mk, pfx in candidates
            if f'{pfx}_RMSE' in df.columns and df[f'{pfx}_RMSE'].notna().any()]
    # Return in the paper's canonical DISPLAY_ORDER so figure legends match
    # table column order. Safe to reorder here (unlike the positional table
    # writers): every plot labels each series from its own model_key, so
    # label/colour/marker travel with the data.
    by_pfx = {pfx: (mk, pfx) for mk, pfx in have}
    out = [by_pfx[p] for p in DISPLAY_ORDER if p in by_pfx]
    out += [t for t in have if t[1] not in DISPLAY_ORDER]
    return out


def _write_hypo_alarm_rows(f, op, ordered):
    """Binary alarm operating-point rows : False-Alarm
    Rate and PPV at the fixed sensitivity=0.90 threshold (matching the
    reference line in Fig.~roc_hypo), hypoglycaemia / 4-hour only — the
    acute-risk direction already emphasized elsewhere (horizon figure, main
    narrative). Full hypo+hyper x 60-min/4-hr coverage remains in the
    supplementary tab_false_alarm_fixed/tab_false_alarm_youden tables.
    No paired significance test: these are pooled-cohort operating-point
    statistics, not per-instance/per-user values, so p-value cells are left
    blank rather than fabricating a test.
    """
    prefixes = [pfx for _, pfx in ordered]

    def _vals(stat_key):
        return [[op.get((pfx, 'Hypo_4hr'), {}).get(stat_key, np.nan)] for pfx in prefixes]

    far = _vals('far90')
    write_metric_row_three(
        f, 'Hypo False-Alarm Rate (Sens=0.90, 4-hr, \\%)',
        far[0], far[1], far[2], lower_better=True, decimals=1, pct=True,
        extra_vals=far[3:], pvals=[np.nan] * (len(ordered) - 1))

    ppv = _vals('ppv90')
    write_metric_row_three(
        f, 'Hypo PPV (Sens=0.90, 4-hr, \\%)',
        ppv[0], ppv[1], ppv[2], lower_better=False, decimals=1, pct=True,
        extra_vals=ppv[3:], pvals=[np.nan] * (len(ordered) - 1))

    # Second operating point: sensitivity at the most liberal threshold with
    # specificity >= 0.90.
    sens_at_spec = _vals('sensAtSpec90')
    write_metric_row_three(
        f, 'Hypo Sensitivity (Spec=0.90, 4-hr, \\%)',
        sens_at_spec[0], sens_at_spec[1], sens_at_spec[2], lower_better=False,
        decimals=1, pct=True, extra_vals=sens_at_spec[3:],
        pvals=[np.nan] * (len(ordered) - 1))


def write_table2_main(f, df, out_dir=None):
    ordered      = ordered_models(df)
    all_labels   = [lbl for lbl, _ in ordered]
    model_names  = (', '.join(all_labels[:-1]) + ' and ' + all_labels[-1]
                    if len(all_labels) > 1 else all_labels[0])
    caption = (
        f'Aggregate performance: {model_names} '
        r'(4-hour horizon). Bold = best (not shown for Mean Directional Bias). '
        r'Net Trend Accuracy: fraction of posterior samples agreeing with observed '
        r'endpoint sign (no threshold). '
        r'Trajectory Correlation: Pearson $r$ between predicted-mean and observed '
        r'trajectory over the full horizon (magnitude/shape-aware companion to '
        r'Net Trend Accuracy). '
        r'Mean Directional Bias: mean endpoint error (mg/dL); negative = '
        r'over-prediction of glucose drop. '
        r'Hypo False-Alarm Rate / PPV / Sensitivity(Spec=0.90): the continuous '
        r'posterior-weighted Hypo Sensitivity/Specificity above converted to a '
        r'binary alarm decision at the two reference-line operating points in '
        r'Fig.~\ref{fig:roc_hypo} -- sensitivity $\geq 0.90$ (FAR $=1-$specificity, '
        r'PPV $=$ precision of a raised alarm) and specificity $\geq 0.90$ '
        r'(resulting sensitivity if false alarms are capped at 10\%). '
        r'Hypoglycaemia/4-hr only (see Supplement for the full hypo+hyper '
        r'$\times$ 60-min/4-hr breakdown); no significance test '
        r'(pooled-cohort operating point, not a per-user statistic). '
        + '; '.join(f'$p_{{{i+1}}}$: BLoop vs {lbl}'
                    for i, lbl in enumerate(all_labels[1:]))
        + r' (paired Wilcoxon on per-user means, $N=' + str(df['uid'].nunique()) + r'$ users; instances are '
        r'non-independent due to overlapping rolling windows, see Supplement). '
        r'$*p<0.05$, $**p<0.01$, $***p<0.001$.'
    )
    cols = all_labels + [f'$p_{i+1}$' for i in range(len(all_labels) - 1)]
    op = compute_operating_points(df)
    handles = [f]
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
        handles.append(open(os.path.join(out_dir, 'tables', 'tab_main_results.tex'), 'w'))
    for h in handles:
        h.write('% TABLE 2: MAIN RESULTS\n')
        h.write(latex_header(caption, 'tab:main_results', cols) + '\n')
        _write_three_model_body(h, df, METRICS_MAIN, ordered)
        h.write(r'\midrule' + '\n')
        _write_hypo_alarm_rows(h, op, ordered)
        h.write(latex_footer() + '\n\n')
        if h is not f: h.close()


# ============================================================================
# TABLE: AR1 BASELINE COMPARISON
# ============================================================================

def write_table_ar1_baseline(f, df, out_dir=None):
    # EXEMPT from DISPLAY_ORDER on purpose. This table's whole structure is
    # "improvement over the Pure AR1 baseline" -- Pure AR1 is the reference
    # every Delta and p-value is computed against, and it has three column
    # groups (means for all models, then Deltas and p-values for all models
    # EXCEPT the reference). Moving the reference to the far right while the
    # Delta columns silently omit it would be harder to read, not easier, so
    # this one keeps Bayesian Loop / LSTM-MC / Pure AR1 / extras, with the
    # reference adjacent to the models it anchors. Its caption states the
    # reference explicitly, so no reader is misled by the different order.
    extra_present = _extra_models_present(df)
    extra_labels  = [lbl for lbl, _ in extra_present]
    all_labels    = [LABEL_SAFE, LABEL_BASE, LABEL_AR1] + extra_labels
    caption = (
        f'Improvement over {LABEL_AR1} baseline (4-hour horizon). '
        r'$\Delta$ = model $-$ Pure AR1. '
        r'$p$: paired Wilcoxon vs Pure AR1, on per-user means ($N=' + str(df['uid'].nunique()) + r'$ users). '
        r'$*p<0.05$, $**p<0.01$, $***p<0.001$.'
    )
    cols = (all_labels
            + ['$\\Delta$ BLoop', '$\\Delta$ LSTM'] + [f'$\\Delta$ {lbl}' for lbl in extra_labels]
            + ['$p$ BLoop', '$p$ LSTM'] + [f'$p$ {lbl}' for lbl in extra_labels])
    d = df[df['horizon'] == '240 min']
    metrics = [
        ('RMSE (mg/dL)',                  'RMSE',             True,  False, 1),
        ('CRPS (mg/dL)',                  'CRPS',             True,  False, 2),
        ('Coverage (95\\% PI)',           'Coverage',         False, True,  1),
        ('PI Width (mg/dL)',              'Width',            True,  False, 1),
        ('Net Trend Accuracy (\\%)',      'P_Net_Trend', False, True,  1),
        ('Trajectory Correlation (r)',    'P_Traj_Corr',      False, False, 2),
        ('Mean Directional Bias (mg/dL)', 'P_Mean_Dir_Bias',  False, False, 1),
        ('Hypo Sensitivity (4-hr, \\%)', 'P_Hypo_Sens',      False, True,  1),
        ('CEG Zone D (\\%)',              'P_CEG_D',          True,  False, 2),
    ]
    handles = [f]
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
        handles.append(open(os.path.join(out_dir, 'tables', 'tab_ar1_baseline.tex'), 'w'))
    for h in handles:
        h.write('% TABLE: AR1 BASELINE COMPARISON\n')
        col_spec = 'l' + 'r' * len(cols)
        h.write(r'\begin{table}[htbp]\centering\small' + '\n')
        h.write(f'\\caption{{{caption}}}\n\\label{{tab:ar1_baseline}}\n')
        h.write(f'\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n')
        h.write(' & '.join(['Metric'] + cols) + r' \\' + '\n\midrule\n')
        for label, col, lower, pct, dec in metrics:
            scale  = 100 if pct else 1
            base_cols = [f'G_{col}', f'L_{col}', f'A_{col}'] + [f'{pfx}_{col}' for _, pfx in extra_present]
            paired = d[base_cols].dropna(subset=[f'G_{col}', f'L_{col}', f'A_{col}'])
            gv = paired[f'G_{col}'].values
            lv = paired[f'L_{col}'].values
            av = paired[f'A_{col}'].values
            # Each extra baseline paired against Pure AR1 independently — its
            # own NaN pattern need not match G/L/A's.
            extra_paired = [d[[f'{pfx}_{col}', f'A_{col}']].dropna() for _, pfx in extra_present]
            extra_v  = [ep[f'{pfx}_{col}'].values for ep, (_, pfx) in zip(extra_paired, extra_present)]
            extra_av = [ep[f'A_{col}'].values      for ep in extra_paired]
            gm = np.nanmean(gv); lm = np.nanmean(lv); am = np.nanmean(av)
            extra_m  = [np.nanmean(ev) if len(ev) else np.nan for ev in extra_v]
            extra_am = [np.nanmean(eav) if len(eav) else np.nan for eav in extra_av]
            dg = gm - am;        dl = lm - am
            extra_d = [em - eam for em, eam in zip(extra_m, extra_am)]
            pg = wilcoxon_pval_peruser(d, f'G_{col}', f'A_{col}')
            pl = wilcoxon_pval_peruser(d, f'L_{col}', f'A_{col}')
            extra_p = [wilcoxon_pval_peruser(d, f'{pfx}_{col}', f'A_{col}')
                       for _, pfx in extra_present]
            _f  = lambda v: fmt(v * scale, dec)
            _fd = lambda v: (f'{v*scale:+.{dec}f}' if not np.isnan(v) else '—')
            means_str = ' & '.join(_f(v) for v in [gm, lm, am] + extra_m)
            deltas_str = ' & '.join(_fd(v) for v in [dg, dl] + extra_d)
            pvals_str  = ' & '.join(pval_str(p) for p in [pg, pl] + extra_p)
            h.write(f'  {label} & {means_str} & {deltas_str} & {pvals_str} \\\\\n')
        h.write(r'\bottomrule\end{tabular}\end{table}' + '\n\n')
        if h is not f: h.close()


# ============================================================================
# TABLE 3 — MULTI-HORIZON
# ============================================================================

def write_table3_horizons(f, df):
    ordered       = ordered_models(df)
    all_labels    = [lbl for lbl, _ in ordered]
    disp_prefixes = [pfx for _, pfx in ordered]
    model_labels  = all_labels
    n_models = len(model_labels)
    model_names = ', '.join(model_labels)

    f.write('% TABLE 3: MULTI-HORIZON RESULTS\n')
    f.write(r'\begin{table}[htbp]\centering\small' + '\n')
    f.write(
        f'\\caption{{Accuracy at multiple horizons: {model_names}. '
        r'Net Trend Accuracy: fraction of posterior samples agreeing with observed '
        r'endpoint sign.}}' + '\n'
    )
    f.write(r'\label{tab:horizons}' + '\n')
    f.write('\\begin{tabular}{l' + 'r' * (n_models * 4) + '}' + '\n\\toprule\n')
    hdr_cols = ['30 min', '60 min', '90 min', '120 min']
    offsets  = [2 + i * n_models for i in range(4)]
    f.write(' & '.join([''] + [f'\\multicolumn{{{n_models}}}{{c}}{{{h}}}' for h in hdr_cols]) + r' \\' + '\n')
    f.write(''.join([f'\\cmidrule(lr){{{o}-{o+n_models-1}}}' for o in offsets]) + '\n')
    model_hdrs = (' & ' + ' & '.join(model_labels)) * 4
    f.write(f'Metric{model_hdrs} \\\\\n\midrule\n')
    for metric, col, lower, dec, pct in [
        ('RMSE',               'RMSE',             True,  1, False),
        ('Net Trend Acc (\\%)','P_Net_Trend',  False, 1, True),
        ('Interval Score',     'IS',               True,  1, False),
    ]:
        scale = 100 if pct else 1
        row_str = f'  {metric}'
        for h_label in HORIZON_LABELS[:4]:
            d  = df[df['horizon'] == h_label]
            # header (model_labels) and values both come from `ordered`
            means = [np.nanmean(d[f'{pfx}_{col}'].dropna()) * scale
                     for pfx in disp_prefixes]
            finite = [v for v in means if not np.isnan(v)]
            bv = (min if lower else max)(finite) if finite else np.nan
            _fb = lambda v, _bv=bv: (f'\\textbf{{{fmt(v, dec)}}}' if (not np.isnan(v) and v == _bv) else fmt(v, dec))
            row_str += ''.join(f' & {_fb(v)}' for v in means)
        f.write(row_str + r' \\' + '\n')
    f.write(r'\bottomrule\end{tabular}\end{table}' + '\n\n')


# ============================================================================
# TABLE 4 — STRATIFIED BY INSTANCE TYPE
# ============================================================================

def write_table4_stratified(f, df, out_dir=None):
    ordered       = ordered_models(df)
    all_labels    = [lbl for lbl, _ in ordered]
    disp_prefixes = [pfx for _, pfx in ordered]
    ref           = disp_prefixes[0]
    caption = (
        r'Performance stratified by clinical context (4-hour horizon). '
        + '; '.join(f'$p_{{{i+1}}}$: BLoop vs {lbl}'
                    for i, lbl in enumerate(all_labels[1:]))
        + r' (paired Wilcoxon on per-user means within each context)'
        + r'. Bold = best (not shown for Mean Directional Bias).'
    )
    d4h = df[df['horizon'] == '240 min']
    metric_specs = [
        # (label, col, lower, pct, dec, signed)
        ('RMSE (mg/dL)',                  'RMSE',              True,  False, 1, False),
        ('Net Trend Accuracy (\\%)',      'P_Net_Trend',  False, True,  1, False),
        ('Mean Directional Bias (mg/dL)', 'P_Mean_Dir_Bias',   False, False, 1, True),
        ('CEG Zone D (\\%)',              'P_CEG_D',           True,  False, 2, False),
        ('Hypo Sens, 60-min (\\%)',       'P_Hypo_Sens_60min', False, True,  1, False),
        ('Hypo Spec, 60-min (\\%)',       'P_Hypo_Spec_60min', False, True,  1, False),
        ('Hypo Sens, 4-hr (\\%)',         'P_Hypo_Sens',       False, True,  1, False),
        ('Hypo Spec, 4-hr (\\%)',         'P_Hypo_Spec',       False, True,  1, False),
    ]
    cols = all_labels + [f'$p_{i+1}$' for i in range(len(all_labels) - 1)]
    handles = [f]
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
        handles.append(open(os.path.join(out_dir, 'tables', 'tab_stratified.tex'), 'w'))
    for h in handles:
        h.write('% TABLE 4: STRATIFIED\n')
        h.write(r'\begin{table}[htbp]\centering\small' + '\n')
        h.write(f'\\caption{{{caption}}}\n\\label{{tab:stratified}}\n')
        h.write(f'\\begin{{tabular}}{{l{"r"*len(cols)}}}\n\\toprule\n')
        h.write('Context / Metric & ' + ' & '.join(cols) + r' \\' + '\n\midrule\n')
        first_type = True
        for itype in INSTANCE_TYPES:
            sub = d4h[d4h['itype'] == itype]
            if len(sub) == 0: continue
            if not first_type: h.write(r'\midrule' + '\n')
            first_type = False
            h.write(f'\\multicolumn{{{len(cols)+1}}}{{l}}'
                    f'{{\\textit{{{INSTANCE_LABELS[itype]} ($N={len(sub):,}$)}}}} \\\\\n')
            for label, col, lower, pct, dec, signed in metric_specs:
                # Values and p-values are built by iterating `ordered` (the
                # SAME display-ordered list `cols` is built from), so header
                # and value order cannot drift apart. `ref` is position 0
                # (Bayesian Loop), which every p-value is computed against.
                all_means = [np.nanmean(v) if len(v) else np.nan
                             for v in (sub[f'{pfx}_{col}'].dropna().tolist()
                                       for pfx in disp_prefixes)]
                all_p = [wilcoxon_pval_peruser(sub, f'{ref}_{col}', f'{pfx}_{col}')
                         for pfx in disp_prefixes[1:]]
                scale = 100 if pct else 1
                if signed:
                    _f = lambda v: fmt(v * scale, dec)
                else:
                    finite = [v for v in all_means if not np.isnan(v)]
                    bv = (min if lower else max)(finite) if finite else np.nan
                    def _f(v, _bv=bv):
                        s = fmt(v * scale, dec)
                        return f'\\textbf{{{s}}}' if (not np.isnan(v) and v == _bv) else s
                _p = lambda p: ('' if np.isnan(p) else
                                ('$***$' if p < 0.001 else '$**$' if p < 0.01
                                 else '$*$' if p < 0.05 else 'ns'))
                means_str = ' & '.join(_f(v) for v in all_means)
                pvals_str = ' & '.join(_p(p) for p in all_p)
                h.write(f'  \\quad {label} & {means_str} & {pvals_str} \\\\\n')
        h.write(r'\bottomrule\end{tabular}\end{table}' + '\n\n')
        if h is not f: h.close()


# ============================================================================
# TABLE — MEAL/BOLUS-CONDITIONAL DIRECTIONAL ACCURACY
# ============================================================================

def compute_meal_response_accuracy(df, itypes=('MEAL', 'CORRECTION')):
    """Meal/bolus-conditional directional accuracy. NOT a causal-effect
    estimate — this is an observational refinement of the itype
    stratification already in write_table4_stratified/plot_instance_type_
    comparison (which conditions Net Trend Accuracy on meal/bolus context),
    additionally conditioning on the *observed* response direction: among
    Meal/Correction instances where glucose was actually seen to rise (or
    fall), does the model's posterior directional call agree? Confounds
    (e.g. concurrent unlogged interventions) are not controlled for."""
    d4h = df[df['horizon'] == '240 min']
    prefixes = ordered_prefixes()
    if 'G_Obs_Net_Sign' not in d4h.columns:
        return {}  # pkl predates Obs_Net_Sign — caller prints a warning
    rows = {}
    for itype in itypes:
        for direction, sign in [('Rise', 1.0), ('Fall', -1.0)]:
            sub = d4h[(d4h['itype'] == itype) & (d4h['G_Obs_Net_Sign'] == sign)]
            rows[(itype, direction)] = {
                pfx: sub[f'{pfx}_P_Net_Trend'].dropna()
                for pfx in prefixes if f'{pfx}_P_Net_Trend' in sub.columns
            }
    return rows


def write_table_meal_response_accuracy(f, df, out_dir=None):
    """Supplementary table (not a causal claim — see
    compute_meal_response_accuracy docstring)."""
    ordered       = ordered_models(df)
    all_labels    = [lbl for lbl, _ in ordered]
    disp_prefixes = [pfx for _, pfx in ordered]
    # POSITIONAL: `cols` below is [LABEL_SAFE, LABEL_BASE, LABEL_AR1] +
    # extra_labels, and row values are emitted by iterating this list, so the
    # two must agree. See the note in _write_hypo_alarm_rows.
    prefixes = [pfx for _, pfx in ordered]
    caption = (
        r'Meal/bolus-triggered directional accuracy (4-hour horizon), '
        r'conditioned on the \emph{observed} glucose response direction: '
        r'among Meal ($\geq$40g carbs) or Correction ($\geq$0.5U bolus) '
        r'instances where glucose was observed to rise (or fall), does the '
        r'model correctly call that direction? This is an observational '
        r'stratification of Net Trend Accuracy (cf. Table~\ref{tab:stratified}), '
        r'not a causal-effect estimate — confounds and counterfactual '
        r'validity are not assessed. Bold = best.'
    )
    cols = all_labels
    rows = compute_meal_response_accuracy(df)
    if not rows:
        print('  WARNING: G_Obs_Net_Sign not found — pkl predates this metric; '
              'tab_meal_response_accuracy will be all em-dashes. '
              'Re-run evaluate.py to backfill it.')
    itype_label = {'MEAL': r'Meal ($\geq$40g)', 'CORRECTION': r'Correction ($\geq$0.5U)'}
    handles = [f]
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
        handles.append(open(os.path.join(out_dir, 'tables', 'tab_meal_response_accuracy.tex'), 'w'))
    for h in handles:
        h.write('% TABLE: MEAL/BOLUS-CONDITIONAL DIRECTIONAL ACCURACY (not causal)\n')
        h.write(latex_header(caption, 'tab:meal_response_accuracy', cols) + '\n')
        for itype in ('MEAL', 'CORRECTION'):
            for direction in ('Rise', 'Fall'):
                vals  = rows.get((itype, direction), {})
                means = [np.nanmean(vals[pfx]) if pfx in vals and len(vals[pfx]) else np.nan
                         for pfx in prefixes]
                finite = [v for v in means if not np.isnan(v)]
                best   = max(finite) if finite else np.nan
                cells  = [f'\\textbf{{{fmt(v*100,1)}}}' if (not np.isnan(v) and v == best)
                          else fmt(v*100, 1) for v in means]
                n = len(vals.get('G', []))
                h.write(f'  {itype_label[itype]}, Observed {direction} (N={n}) & '
                        + ' & '.join(cells) + r' \\' + '\n')
        h.write(latex_footer() + '\n\n')
        if h is not f: h.close()


# ============================================================================
# TABLE 5 — RMSE-DIRECTION PARADOX
# ============================================================================

def write_table5_paradox(f, df):
    RMSE_LOW = 35.0; NTA_HIGH = 0.60; NTA_LOW = 0.45
    d4h = df[df['horizon'] == '240 min'].copy()

    def classify(rmse, nta):
        low  = rmse <= RMSE_LOW
        good = (not np.isnan(nta)) and nta >= NTA_HIGH
        bad  = np.isnan(nta) or nta <= NTA_LOW
        if low  and good: return 'GOOD'
        if low  and bad:  return 'FLAT'
        if not low and good: return 'PENALISED'
        if not low and bad:  return 'BAD'
        return 'AMBIGUOUS'

    ordered       = ordered_models(df)
    all_labels    = [lbl for lbl, _ in ordered]
    disp_prefixes = [pfx for _, pfx in ordered]
    base_prefixes = disp_prefixes
    base_cols = [c for pfx in base_prefixes for c in (f'{pfx}_RMSE', f'{pfx}_P_Net_Trend')]
    paired = d4h[base_cols].dropna(
        subset=['G_RMSE','G_P_Net_Trend','L_RMSE','L_P_Net_Trend'])
    n = len(paired)
    if n == 0: f.write('% Paradox table: no paired data\n\n'); return

    cls_by_prefix = {
        pfx: [classify(r[f'{pfx}_RMSE'], r[f'{pfx}_P_Net_Trend']) for _, r in paired.iterrows()]
        for pfx in base_prefixes
    }

    cats = [
        ('GOOD',      'Low RMSE + Correct Direction'),
        ('FLAT',      'Low RMSE + Wrong Direction'),
        ('PENALISED', 'High RMSE + Correct Direction'),
        ('BAD',       'High RMSE + Wrong Direction'),
        ('AMBIGUOUS', 'Ambiguous'),
    ]
    model_labels = all_labels
    f.write('% TABLE 5: RMSE-DIRECTION PARADOX\n')
    f.write(r'\begin{table}[htbp]\centering' + '\n')
    f.write(f'\\caption{{RMSE--direction paradox ({" / ".join(model_labels)}). '
            r'Low RMSE: $\leq 35$\,mg/dL; correct direction: NTA $\geq 0.60$; '
            r'wrong direction: NTA $\leq 0.45$.}}\n')
    f.write(r'\label{tab:paradox}' + '\n')
    f.write('\\begin{tabular}{l' + 'r' * len(model_labels) + '}\\toprule' + '\n')
    f.write('Category & ' + ' & '.join(model_labels) + r' \\' + '\n\midrule\n')
    for key, label in cats:
        counts = [cls_by_prefix[pfx].count(key) for pfx in base_prefixes]
        _s = lambda c: f'{c:,} ({c/n*100:.1f}\\%)'
        f.write(f'  {label} & ' + ' & '.join(_s(c) for c in counts) + r' \\' + '\n')
    f.write(r'\bottomrule\end{tabular}\end{table}' + '\n\n')


# ============================================================================
# SUMMARY TXT
# ============================================================================

def write_summary_txt(f, df, n_users, counts):
    d4h = df[df['horizon'] == '240 min']
    extra_present = _extra_models_present(df)
    extra_counts_str = ''.join(
        f', {ALL_LABELS[mk]}={counts.get(mk, 0)}' for mk, _ in EXTRA_MODELS if mk in counts
    )
    f.write('='*70 + '\n')
    f.write(f'EVALUATION SUMMARY — {LABEL_SAFE}={counts[MODEL_SAFE]}, '
            f'{LABEL_BASE}={counts[MODEL_BASE]}, {LABEL_AR1}={counts[MODEL_AR1]}'
            f'{extra_counts_str} '
            f'/ {n_users} users\n')
    f.write('  Directional: P_Net_Trend (pure sign) | P_Mean_Dir_Bias (mg/dL)\n')
    f.write('='*70 + '\n\n')
    for label, prefix in [(LABEL_SAFE,'G'), (LABEL_BASE,'L'), (LABEL_AR1,'A')] + \
                         [(lbl, pfx) for lbl, pfx in extra_present]:
        f.write(f'{label}:\n')
        for col, lbl, pct in [
            ('RMSE',               'RMSE (mg/dL)',               False),
            ('IS',                 'Interval Score',              False),
            ('CRPS',               'CRPS',                       False),
            ('Coverage',           'Coverage (95% PI)',           True),
            ('P_Net_Trend',   'Net Trend Accuracy (sign,%)', True),
            ('P_Mean_Dir_Bias',    'Mean Directional Bias (mg/dL)',False),
            ('P_Hypo_Sens',        'Hypo Sensitivity (4-hr,%)',   True),
            ('P_Hypo_Spec',        'Hypo Specificity (4-hr,%)',   True),
            ('P_Hypo_Sens_60min',  'Hypo Sensitivity (60-min,%)', True),
            ('P_Hypo_Spec_60min',  'Hypo Specificity (60-min,%)', True),
            ('P_Hyper_Sens',       'Hyper Sensitivity (4-hr,%)',  True),
            ('P_Hyper_Spec',       'Hyper Specificity (4-hr,%)',  True),
            ('P_CEG_A',            'CEG Zone A (%)',              False),
            ('P_CEG_D',            'CEG Zone D (%)',              False),
        ]:
            vals = d4h[f'{prefix}_{col}'].dropna()
            if len(vals) == 0: continue
            m = vals.mean() * (100 if pct else 1)
            s = vals.std()  * (100 if pct else 1)
            f.write(f'  {lbl:<42} {m:.2f} ± {s:.2f}\n')
        f.write('\n')

    f.write('Head-to-head wins (4h horizon):\n')
    for col, label, lower in [
        ('RMSE',            'RMSE',             True),
        ('P_Net_Trend','Net Trend (sign)',  False),
        ('IS',              'Interval Score',    True),
    ]:
        g = d4h[f'G_{col}'].values; l = d4h[f'L_{col}'].values; a = d4h[f'A_{col}'].values
        mask_gl = ~(np.isnan(g) | np.isnan(l))
        mask_ga = ~(np.isnan(g) | np.isnan(a))
        if mask_gl.sum() == 0: continue
        gv, lv = g[mask_gl], l[mask_gl]
        g_wins = np.sum(gv < lv) if lower else np.sum(gv > lv)
        l_wins = np.sum(lv < gv) if lower else np.sum(lv > gv)
        p_gl   = wilcoxon_pval(gv.tolist(), lv.tolist())
        f.write(f'  {label:<22} {LABEL_SAFE} {g_wins}/{mask_gl.sum()} '
                f'({g_wins/mask_gl.sum()*100:.1f}%)  '
                f'{LABEL_BASE} {l_wins}/{mask_gl.sum()} '
                f'({l_wins/mask_gl.sum()*100:.1f}%)  p={p_gl:.3f}\n')
        if mask_ga.sum() > 0:
            gv2, av = g[mask_ga], a[mask_ga]
            g_wins2 = np.sum(gv2 < av) if lower else np.sum(gv2 > av)
            a_wins  = np.sum(av  < gv2) if lower else np.sum(av  > gv2)
            p_ga    = wilcoxon_pval(gv2.tolist(), av.tolist())
            f.write(f'  {"":22} {LABEL_SAFE} {g_wins2}/{mask_ga.sum()} '
                    f'({g_wins2/mask_ga.sum()*100:.1f}%) vs {LABEL_AR1}  '
                    f'{LABEL_AR1} {a_wins}/{mask_ga.sum()} '
                    f'({a_wins/mask_ga.sum()*100:.1f}%)  p={p_ga:.3f}\n')


# ============================================================================
# PLOTS
# ============================================================================

def compute_operating_points_by_horizon(df, target_sens=0.90, metric='Hypo'):
    """Per-horizon fixed-sensitivity operating point, for the horizon
    figure's False-Alarm-Rate panel. Unlike compute_operating_points
    (which fixes the 4-hr instance set and compares 60-min-vs-4-hr *sentinel
    windows*), this computes one operating point per horizon bucket using
    that horizon's own P_<metric>_Prob/P_<metric>_Sens columns — directly
    analogous to how the existing Sensitivity/Specificity horizon-figure
    panels already vary by horizon.

    metric: 'Hypo' or 'Hyper', selecting which pair of columns to read."""
    prefixes = ordered_prefixes()
    results = {}
    for h in HORIZON_LABELS:
        d_h = df[df['horizon'] == h]
        for prefix in prefixes:
            score_col    = f'{prefix}_P_{metric}_Prob'
            sentinel_col = f'{prefix}_P_{metric}_Sens'
            if score_col not in d_h.columns or sentinel_col not in d_h.columns:
                continue
            labels = d_h[sentinel_col].notna().astype(float)
            results[(prefix, h)] = _operating_point_stats(
                d_h[score_col].values, labels.values, target_sens=target_sens)
    return results


def compute_operating_points_by_group(df, group_col, horizon='240 min', metric='Hypo'):
    """Per-group (e.g. per-uid, per-itype) operating-point stats at a
    fixed horizon — for the per-user and instance-type figures' FAR@Sens=0.90
    / Sensitivity@Spec=0.90 panels. Small groups (e.g. a user with only one
    true-hypo test instance) gracefully return NaN via
    _operating_point_stats's own len/count guards rather than erroring —
    those groups just show up as missing in the resulting violin/bar, same
    as any other per-user metric with insufficient data.

    metric: 'Hypo' or 'Hyper', selecting which pair of columns to read."""
    prefixes = ordered_prefixes()
    d = df[df['horizon'] == horizon]
    results = {}
    for prefix in prefixes:
        score_col    = f'{prefix}_P_{metric}_Prob'
        sentinel_col = f'{prefix}_P_{metric}_Sens'
        if score_col not in d.columns or sentinel_col not in d.columns:
            continue
        labels_all = d[sentinel_col].notna().astype(float)
        for group_val, idx in d.groupby(group_col).groups.items():
            results[(prefix, group_val)] = _operating_point_stats(
                d.loc[idx, score_col].values, labels_all.loc[idx].values)
    return results


_HORIZON_OP_POINT_KEYS = {
    'FAR90_HYPO':  ('far90', 'Hypo'),  'SENS_AT_SPEC90_HYPO':  ('sensAtSpec90', 'Hypo'),
    'FAR90_HYPER': ('far90', 'Hyper'), 'SENS_AT_SPEC90_HYPER': ('sensAtSpec90', 'Hyper'),
}

# Row 1: RMSE, Coverage, PI Width, Interval Score
# Row 2: Net Trend Accuracy, Trajectory Correlation, Mean Directional Bias, CEG Zone D
# Row 3: Hypo Sensitivity, Hypo Specificity, Hypo FAR (Sens=0.90), Hypo Sensitivity (Spec=0.90)
# Row 4: Hyper Sensitivity, Hyper Specificity, Hyper FAR (Sens=0.90), Hyper Sensitivity (Spec=0.90)
# (key, ylabel, lower_is_better, df column suffix)
HORIZON_METRICS = [
    ('RMSE',      'RMSE (mg/dL)',                       True,  'RMSE'),
    ('Coverage',  'Coverage (95% PI)',                   False, 'Coverage'),
    ('Width',     'PI Width (mg/dL)',                    True,  'Width'),
    ('IS',        'Interval Score',                      True,  'IS'),
    ('NTA',       'Net Trend Accuracy',                  False, 'P_Net_Trend'),
    ('TrajCorr',  'Trajectory Correlation (r)',          False, 'P_Traj_Corr'),
    ('MDB',       'Mean Directional Bias (mg/dL)',       False, 'P_Mean_Dir_Bias'),
    ('CEGD',      'CEG Zone D (%)',                      True,  'P_CEG_D'),
    ('HypoSens',  'Hypo Sensitivity',                    False, 'P_Hypo_Sens'),
    ('HypoSpec',  'Hypo Specificity',                    False, 'P_Hypo_Spec'),
    ('HypoFAR',   'Hypo FAR (Sens=0.90)',                True,  'FAR90_HYPO'),
    ('HypoSAS',   'Hypo Sensitivity (Spec=0.90)',        False, 'SENS_AT_SPEC90_HYPO'),
    ('HyperSens', 'Hyper Sensitivity',                   False, 'P_Hyper_Sens'),
    ('HyperSpec', 'Hyper Specificity',                   False, 'P_Hyper_Spec'),
    ('HyperFAR',  'Hyper FAR (Sens=0.90)',                True,  'FAR90_HYPER'),
    ('HyperSAS',  'Hyper Sensitivity (Spec=0.90)',        False, 'SENS_AT_SPEC90_HYPER'),
]


def compute_horizon_plot_data(df, active=None):
    """Tidy long-form table of exactly what plot_horizon_comparison draws
    (mean/sem per metric x model x horizon). This -- not metrics_df.pkl --
    is the right thing to save once and reload for aesthetic-only replots
    (fonts, legend, layout): a few hundred rows instead of one row per
    test instance; also usable as the figures' source data. See save_plot_data."""
    active = active or get_active_models(df)
    op_cache = {}  # metric ('Hypo'/'Hyper') -> compute_operating_points_by_horizon(df, metric=...)
    rows = []
    for key, ylabel, lower, col in HORIZON_METRICS:
        for model_key, prefix in active:
            if col in _HORIZON_OP_POINT_KEYS:
                stat_key, op_metric = _HORIZON_OP_POINT_KEYS[col]
                if op_metric not in op_cache:
                    op_cache[op_metric] = compute_operating_points_by_horizon(df, metric=op_metric)
                op_by_h = op_cache[op_metric]
                for h in HORIZON_LABELS:
                    st = op_by_h.get((prefix, h))
                    if st is None or np.isnan(st[stat_key]): continue
                    rows.append((key, ylabel, lower, model_key, prefix, h, st[stat_key], np.nan))
                continue
            for h in HORIZON_LABELS:
                sub = df[df['horizon'] == h][f'{prefix}_{col}'].dropna()
                if len(sub) == 0: continue
                rows.append((key, ylabel, lower, model_key, prefix, h, sub.mean(), sub.sem()))
    return pd.DataFrame(rows, columns=['metric', 'ylabel', 'lower_better',
                                        'model_key', 'prefix', 'horizon', 'mean', 'sem'])


def _model_family(model_key):
    """Bucket a model key into its family for legend grouping. DISPLAY_ORDER
    already sorts G, then all LSTM-* variants, then all Transformer-*
    variants, then AR1-*, but matplotlib's ncol legend layout fills
    column-major -- so without this, family members land scattered across
    different rows/columns instead of clustered together."""
    if 'LSTM' in model_key: return 'LSTM'
    if 'Transformer' in model_key: return 'Transformer'
    if 'AR1' in model_key: return 'AR1'
    return 'Bayesian Loop'


def shared_family_legend(fig, handles, labels, model_keys, fontsize,
                         bbox_to_anchor_fn, **legend_kwargs):
    """One shared bottom legend, grouped into column-blocks by model family
    (Bayesian Loop / LSTM* / Transformer* / AR1*) so each family reads as
    a contiguous column rather than being scattered across matplotlib's
    default column-major fill order. Shorter families are padded with
    blank (invisible) entries so every column has the same row count.
    `bbox_to_anchor_fn(nrow)` returns the (x, y) anchor once nrow is known.
    Legend box style (boxed, light gray edge) is unified across figures --
    do not make some figures' legends frameless."""
    families, family_index = [], {}
    for h, l, mk in zip(handles, labels, model_keys):
        fam = _model_family(mk)
        if fam not in family_index:
            family_index[fam] = len(families)
            families.append([])
        families[family_index[fam]].append((h, l))

    ncol = len(families)
    nrow = max(len(f) for f in families)
    blank = Line2D([], [], color='none', label='')
    grid_handles, grid_labels = [], []
    for fam in families:
        for i in range(nrow):
            h, l = fam[i] if i < len(fam) else (blank, '')
            grid_handles.append(h); grid_labels.append(l)

    legend = fig.legend(
        grid_handles, grid_labels,
        loc='lower center',
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        handlelength=2.2,
        handletextpad=0.6,
        bbox_to_anchor=bbox_to_anchor_fn(nrow),
        **legend_kwargs,
    )
    return legend, nrow


def plot_horizon_comparison(df=None, out_dir=None, active=None, plot_data=None, meta=None):
    """
    16-panel horizon figure (4 rows x 4 cols).

    Pass either `df` (full per-instance metrics_df) to compute+plot in one
    shot, or a cached `plot_data`/`meta` (from compute_horizon_plot_data /
    load_plot_data) to redraw from the small cached table without touching
    the full df at all -- use this path for aesthetic-only changes.

    Row 3's last two panels are the two hypoglycemia operating
    points (the sensitivity=0.90/specificity=0.90 reference lines on
    fig_roc_hypo), binarized -- see _write_hypo_alarm_rows for the same
    pair in the main table. Row 4 is the hyperglycemia analog of Row 3.
    """
    if plot_data is None:
        active = active or get_active_models(df)
        plot_data = compute_horizon_plot_data(df, active)
        meta = {'n_users': int(df['uid'].nunique()),
                'n_inst': int(len(df[df['horizon'] == '240 min']))}
    else:
        active = active or [tuple(t) for t in
                            plot_data[['model_key', 'prefix']].drop_duplicates().values]

    fig, axes = plt.subplots(4, 4, figsize=(19, 17))
    axes = axes.flatten()

    for ax, (key, ylabel, lower, col) in zip(axes, HORIZON_METRICS):
        sub_metric = plot_data[plot_data['metric'] == key]
        for model_key, prefix in active:
            sub = sub_metric[sub_metric['prefix'] == prefix]
            if len(sub) == 0: continue
            sub = sub.set_index('horizon')
            h_labels = [h for h in HORIZON_LABELS if h in sub.index]
            if not h_labels: continue
            means = sub.loc[h_labels, 'mean'].values
            sems  = sub.loc[h_labels, 'sem'].values
            x = range(len(h_labels))
            ax.plot(x, means, marker=MODEL_MARKERS[model_key],
                    color=MODEL_COLORS[model_key], label=ALL_LABELS[model_key],
                    linewidth=2, markersize=6)
            if col not in _HORIZON_OP_POINT_KEYS:
                ax.fill_between(x, means - sems, means + sems, alpha=0.12,
                                color=MODEL_COLORS[model_key])

        if key == 'MDB':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

        ax.set_xticks(range(len(HORIZON_LABELS)))
        ax.set_xticklabels([h.replace(' min','') for h in HORIZON_LABELS], fontsize=13)
        ax.set_xlabel('Lead time (min)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(ylabel, fontsize=15, fontweight='bold')
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(True, alpha=0.3)
        if key not in ('MDB',):
            ax.text(0.98, 0.02, '↓ better' if lower else '↑ better',
                    transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
                    color='gray')

    # Shared legend (bottom of figure, once) instead of per-panel legends,
    # grouped into column-blocks by model family (see shared_family_legend).
    handles, labels = axes[0].get_legend_handles_labels()
    model_keys = [mk for mk, _ in active]
    _, legend_nrow = shared_family_legend(
        fig, handles, labels, model_keys, fontsize=15,
        bbox_to_anchor_fn=lambda nrow: (0.5, -0.004 - 0.014 * nrow),
    )

    plt.tight_layout(rect=[0, 0.02 * legend_nrow, 1, 1])
    for ext in ['.pdf', '.png']:
        plt.savefig(os.path.join(out_dir, f'fig_horizons{ext}'),
                    bbox_inches='tight', pad_inches=0.15,
                    dpi=200 if ext == '.pdf' else 150)
    plt.close()
    print('  fig_horizons.pdf')


_ITYPE_OP_POINT_KEYS = {
    'FAR90_HYPO':  ('far90', 'Hypo'),  'SENS_AT_SPEC90_HYPO':  ('sensAtSpec90', 'Hypo'),
    'FAR90_HYPER': ('far90', 'Hyper'), 'SENS_AT_SPEC90_HYPER': ('sensAtSpec90', 'Hyper'),
}

# Row 4 is the hyperglycemia analog of Row 3.
ITYPE_METRICS = [
    ('RMSE',            'RMSE (mg/dL)',                 True),
    ('Coverage',        'Coverage (95% PI)',            False),
    ('Width',           'PI Width (mg/dL)',             True),
    ('IS',              'Interval Score',               True),
    ('P_Net_Trend',     'Net Trend Accuracy',           False),
    ('P_Traj_Corr',     'Trajectory Correlation (r)',   False),
    ('P_Mean_Dir_Bias', 'Mean Directional Bias',        False),
    ('P_CEG_D',         'CEG Zone D (%)',                True),
    ('P_Hypo_Sens',     'Hypo Sensitivity',             False),
    ('P_Hypo_Spec',     'Hypo Specificity',             False),
    ('FAR90_HYPO',      'Hypo FAR (Sens=0.90)',          True),
    ('SENS_AT_SPEC90_HYPO', 'Hypo Sensitivity (Spec=0.90)', False),
    ('P_Hyper_Sens',     'Hyper Sensitivity',            False),
    ('P_Hyper_Spec',     'Hyper Specificity',            False),
    ('FAR90_HYPER',      'Hyper FAR (Sens=0.90)',         True),
    ('SENS_AT_SPEC90_HYPER', 'Hyper Sensitivity (Spec=0.90)', False),
]


def compute_itype_plot_data(df, active=None):
    """Tidy long-form table of exactly what plot_instance_type_comparison
    draws (mean/sem per metric x model x clinical-context). See
    compute_horizon_plot_data for why this -- not metrics_df.pkl -- is
    what should be cached/reloaded for aesthetic-only replots."""
    d4h = df[df['horizon'] == '240 min']
    itypes_present = [t for t in INSTANCE_TYPES if (d4h['itype'] == t).sum() > 0]
    active = active or get_active_models(df)
    op_cache = {}  # metric ('Hypo'/'Hyper') -> compute_operating_points_by_group(df, 'itype', metric=...)
    rows = []
    for col, ylabel, lower in ITYPE_METRICS:
        for model_key, prefix in active:
            if col in _ITYPE_OP_POINT_KEYS:
                stat_key, op_metric = _ITYPE_OP_POINT_KEYS[col]
                if op_metric not in op_cache:
                    op_cache[op_metric] = compute_operating_points_by_group(df, 'itype', metric=op_metric)
                op_by_itype = op_cache[op_metric]
                for it in itypes_present:
                    st = op_by_itype.get((prefix, it), {})
                    val = st.get(stat_key, np.nan)
                    if np.isnan(val): continue
                    rows.append((col, ylabel, lower, model_key, prefix, it, val, np.nan))
                continue
            for it in itypes_present:
                sub = d4h[d4h['itype'] == it][f'{prefix}_{col}'].dropna()
                if len(sub) == 0: continue
                rows.append((col, ylabel, lower, model_key, prefix, it, sub.mean(), sub.sem()))
    data = pd.DataFrame(rows, columns=['metric', 'ylabel', 'lower_better',
                                        'model_key', 'prefix', 'itype', 'mean', 'sem'])
    return data, itypes_present


def plot_instance_type_comparison(df=None, out_dir=None, active=None, plot_data=None, meta=None):
    """16-panel bar chart, same metric set/ordering as plot_horizon_comparison
    (see that function's docstring for the row rationale, including Row 4's
    hyper/hypo rationale), grouped by clinical context instead of horizon.

    Pass either `df` to compute+plot in one shot, or a cached `plot_data`/
    `meta` (from compute_itype_plot_data / load_plot_data) to redraw from
    the small cached table without touching the full df -- use this path
    for aesthetic-only changes."""
    short = {'MEAL':'Meal\n(≥40g)', 'CARB_INSTANCE':'Carb\n(10-39g)',
             'CORRECTION':'Correction', 'STABLE':'Stable'}
    if plot_data is None:
        active = active or get_active_models(df)
        plot_data, itypes_present = compute_itype_plot_data(df, active)
        meta = {'itypes_present': itypes_present}
    else:
        active = active or [tuple(t) for t in
                            plot_data[['model_key', 'prefix']].drop_duplicates().values]
        itypes_present = meta['itypes_present']

    n_models = len(active)
    # Leave more inter-group breathing room as model count grows — at 5
    # models a 0.85 cluster only leaves 0.15 between groups, and fixed-size
    # error-bar caps (capsize is in points, not data units) visually bridge
    # that gap even when the bars themselves don't touch.
    cluster_frac = 0.85 if n_models <= 3 else 0.70
    width    = cluster_frac / n_models
    offsets  = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
    capsize  = 4 if n_models <= 3 else 2.5
    fig, axes = plt.subplots(4, 4, figsize=(19, 17.5)); axes = axes.flatten()
    x = np.arange(len(itypes_present))
    legend_handles, legend_labels = None, None
    for ax, (col, ylabel, lower) in zip(axes, ITYPE_METRICS):
        sub_metric = plot_data[plot_data['metric'] == col]
        for (model_key, prefix), offset in zip(active, offsets):
            sub = sub_metric[sub_metric['prefix'] == prefix].set_index('itype').reindex(itypes_present)
            means = sub['mean'].values
            if col in _ITYPE_OP_POINT_KEYS:
                ax.bar(x + offset, means, width,
                       label=ALL_LABELS[model_key], color=MODEL_COLORS[model_key],
                       alpha=0.82)
                continue
            sems = sub['sem'].values
            ax.bar(x + offset, means, width, yerr=sems,
                   label=ALL_LABELS[model_key], color=MODEL_COLORS[model_key],
                   alpha=0.82, capsize=capsize, error_kw={'elinewidth': 1})
        if col == 'P_Mean_Dir_Bias':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels([short[t] for t in itypes_present], fontsize=13)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(ylabel, fontsize=15, fontweight='bold')
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    # One shared legend below the grid instead of per-axis legends — with
    # several models, per-axis legends were wide enough to overlap bars in
    # whichever panel matplotlib's 'best' placement chose, grouped into
    # column-blocks by model family (see shared_family_legend).
    model_keys = [mk for mk, _ in active]
    _, legend_nrow = shared_family_legend(
        fig, legend_handles, legend_labels, model_keys, fontsize=15,
        bbox_to_anchor_fn=lambda nrow: (0.5, -0.003 - 0.015 * nrow),
    )
    plt.tight_layout(rect=[0, 0.03 * legend_nrow, 1, 0.96])
    path = os.path.join(out_dir, 'instance_type_comparison.pdf')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=200); plt.close()
    print(f'  {path}')


_USER_OP_POINT_KEYS = {
    'FAR90_HYPO':  ('far90', 'Hypo'),  'SENS_AT_SPEC90_HYPO':  ('sensAtSpec90', 'Hypo'),
    'FAR90_HYPER': ('far90', 'Hyper'), 'SENS_AT_SPEC90_HYPER': ('sensAtSpec90', 'Hyper'),
}

# Row 4 is the hyperglycemia analog of Row 3.
USER_LEVEL_METRICS = [
    ('RMSE',             'RMSE (mg/dL)',                  True),
    ('Coverage',         'Coverage (95% PI)',             False),
    ('Width',            'PI Width (mg/dL)',              True),
    ('IS',               'Interval Score',                True),
    ('P_Net_Trend',      'Net Trend Accuracy',            False),
    ('P_Traj_Corr',      'Trajectory Correlation (r)',    False),
    ('P_Mean_Dir_Bias',  'Mean Directional Bias (mg/dL)', False),
    ('P_CEG_D',          'CEG Zone D (%)',                 True),
    ('P_Hypo_Sens',      'Hypo Sensitivity',              False),
    ('P_Hypo_Spec',      'Hypo Specificity',              False),
    ('FAR90_HYPO',       'Hypo FAR (Sens=0.90)',           True),
    ('SENS_AT_SPEC90_HYPO', 'Hypo Sensitivity (Spec=0.90)', False),
    ('P_Hyper_Sens',      'Hyper Sensitivity',             False),
    ('P_Hyper_Spec',      'Hyper Specificity',             False),
    ('FAR90_HYPER',       'Hyper FAR (Sens=0.90)',          True),
    ('SENS_AT_SPEC90_HYPER', 'Hyper Sensitivity (Spec=0.90)', False),
]


def compute_user_level_plot_data(df, active=None):
    """Tidy long-form table of exactly what plot_user_level_metrics draws
    (per-user value per metric x model -- violins need the distribution,
    not just mean/sem, but this is still tiny: ~metrics x models x users
    rows). See compute_horizon_plot_data for why this -- not
    metrics_df.pkl -- is what should be cached/reloaded for
    aesthetic-only replots."""
    d4h = df[df['horizon'] == '240 min']
    active = active or get_active_models(df)
    op_cache = {}  # metric ('Hypo'/'Hyper') -> compute_operating_points_by_group(df, 'uid', metric=...)
    rows = []
    for col, ylabel, lower in USER_LEVEL_METRICS:
        for model_key, prefix in active:
            if col in _USER_OP_POINT_KEYS:
                stat_key, op_metric = _USER_OP_POINT_KEYS[col]
                if op_metric not in op_cache:
                    op_cache[op_metric] = compute_operating_points_by_group(df, 'uid', metric=op_metric)
                op_by_user = op_cache[op_metric]
                per_user = pd.Series({
                    uid: op_by_user.get((prefix, uid), {}).get(stat_key, np.nan)
                    for uid in d4h['uid'].unique()
                }).dropna()
            else:
                per_user = d4h.groupby('uid')[f'{prefix}_{col}'].mean().dropna()
            for uid, val in per_user.items():
                rows.append((col, ylabel, lower, model_key, prefix, uid, val))
    data = pd.DataFrame(rows, columns=['metric', 'ylabel', 'lower_better',
                                        'model_key', 'prefix', 'uid', 'value'])
    return data


def plot_user_level_metrics(df=None, out_dir=None, active=None, plot_data=None, meta=None):
    """16-panel per-user violin grid, same metric set/ordering as
    plot_horizon_comparison (see that function's docstring for the row
    rationale, including Row 4's hyper/hypo rationale). Users with too few
    true-hypo/hyper test instances for _operating_point_stats to resolve a
    FAR90/SENS_AT_SPEC90 threshold simply drop out of those panels' violins
    (NaN, dropped below) rather than erroring — same graceful-degradation
    behavior as every other per-user metric here when a user lacks enough
    data.

    Pass either `df` to compute+plot in one shot, or a cached `plot_data`/
    `meta` (from compute_user_level_plot_data / load_plot_data) to redraw
    from the small cached table without touching the full df -- use this
    path for aesthetic-only changes."""
    if plot_data is None:
        active = active or get_active_models(df)
        plot_data = compute_user_level_plot_data(df, active)
        meta = {'n_users': int(df[df['horizon'] == '240 min']['uid'].nunique())}
    else:
        active = active or [tuple(t) for t in
                            plot_data[['model_key', 'prefix']].drop_duplicates().values]

    fig, axes = plt.subplots(4, 4, figsize=(20, 19.5)); axes = axes.flatten()
    for ax, (col, ylabel, lower) in zip(axes, USER_LEVEL_METRICS):
        sub_metric = plot_data[plot_data['metric'] == col]
        data, colors_list = [], []
        for model_key, prefix in active:
            vals = sub_metric[sub_metric['prefix'] == prefix]['value'].values
            data.append(vals)
            colors_list.append(MODEL_COLORS[model_key])
        vp = ax.violinplot(data, positions=range(len(data)),
                           showmedians=True, showextrema=True)
        for pc, color in zip(vp['bodies'], colors_list):
            pc.set_facecolor(color); pc.set_alpha(0.7)
        vp['cmedians'].set_color('black')
        if col == 'P_Mean_Dir_Bias':
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        # Model identity now comes from the shared bottom legend (color-coded)
        # instead of repeating rotated model-name x-tick labels in every one
        # of the 16 panels.
        ax.set_xticks([])
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'Per-user {ylabel}', fontsize=15, fontweight='bold')
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(True, alpha=0.3, axis='y')

    # Shared legend (bottom of figure, once), grouped into column-blocks by
    # model family (see shared_family_legend) -- built from Patch swatches
    # since violinplot bodies aren't themselves legend-labeled.
    handles = [Patch(facecolor=MODEL_COLORS[mk], alpha=0.7) for mk, _ in active]
    labels  = [ALL_LABELS[mk] for mk, _ in active]
    model_keys = [mk for mk, _ in active]
    _, legend_nrow = shared_family_legend(
        fig, handles, labels, model_keys, fontsize=15,
        bbox_to_anchor_fn=lambda nrow: (0.5, -0.004 - 0.014 * nrow),
    )
    plt.tight_layout(rect=[0, 0.02 * legend_nrow, 1, 1])
    path = os.path.join(out_dir, 'fig_user_level_metric_distributions.pdf')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.15, dpi=200); plt.close()
    print(f'  {path}')


# ============================================================================
# CACHED PLOT DATA — small tidy tables for aesthetic-only replots
# ============================================================================
# fig_horizons / fig_stratified / fig_user_level_metric_distributions are
# drawn from small aggregated tables (a few hundred KB) computed from the
# per-instance metrics_df.pkl. They are saved under plot_data/ as the figures'
# source data.

def extract_all_plot_data(df, active=None):
    active = active or get_active_models(df)
    horizon_data = compute_horizon_plot_data(df, active)
    itype_data, itypes_present = compute_itype_plot_data(df, active)
    user_data = compute_user_level_plot_data(df, active)
    meta = {
        'active':         [list(t) for t in active],
        'n_users':        int(df['uid'].nunique()),
        'n_inst':         int(len(df[df['horizon'] == '240 min'])),
        'itypes_present': itypes_present,
    }
    return horizon_data, itype_data, user_data, meta


def save_plot_data(out_dir, horizon_data, itype_data, user_data, meta):
    plot_dir = os.path.join(out_dir, 'plot_data')
    os.makedirs(plot_dir, exist_ok=True)
    horizon_data.to_csv(os.path.join(plot_dir, 'horizon.csv'), index=False)
    itype_data.to_csv(os.path.join(plot_dir, 'itype.csv'), index=False)
    user_data.to_csv(os.path.join(plot_dir, 'user_level.csv'), index=False)
    with open(os.path.join(plot_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  plot_data/ saved ({plot_dir})')


def plot_rmse_vs_nta(df, out_dir, active=None):
    """Per-user RMSE vs Net Trend Accuracy — RMSE-direction tradeoff figure."""
    d4h = df[df['horizon'] == '240 min']
    active = active or get_active_models(df)
    user_means = {pfx: d4h.groupby('uid')[[f'{pfx}_RMSE', f'{pfx}_P_Net_Trend']].mean()
                  for _, pfx in active}
    g_user = user_means['G']
    a_user = user_means.get('A')
    fig, ax = plt.subplots(figsize=(10, 7))
    # Bayesian Loop <-> Pure AR1 connector lines: a specific 2-model visual
    # device (not generalized to every active model — with 8 models this
    # would be unreadable spaghetti).
    if a_user is not None:
        common_ga = g_user.index.intersection(a_user.index)
        for uid in common_ga:
            gx = g_user.loc[uid,'G_RMSE']; gy = g_user.loc[uid,'G_P_Net_Trend']
            ax2 = a_user.loc[uid,'A_RMSE']; ay = a_user.loc[uid,'A_P_Net_Trend']
            if np.isnan(gx+gy+ax2+ay): continue
            ax.plot([ax2, gx], [ay, gy], color='#4CAF50', alpha=0.10,
                    linewidth=0.7, zorder=1)
    for model_key, prefix in active:
        if model_key == MODEL_SAFE: continue  # drawn last, on top
        ud = user_means[prefix]
        ax.scatter(ud[f'{prefix}_RMSE'], ud[f'{prefix}_P_Net_Trend'],
                   color=MODEL_COLORS[model_key], alpha=0.55, s=35,
                   label=ALL_LABELS[model_key], zorder=3,
                   edgecolors='none', marker=MODEL_MARKERS[model_key])
    ax.scatter(g_user['G_RMSE'], g_user['G_P_Net_Trend'],
               color=MODEL_COLORS[MODEL_SAFE], alpha=0.65, s=40, label=LABEL_SAFE,
               zorder=4, edgecolors='none', marker='o')
    for model_key, prefix in active:
        ud = user_means[prefix]
        ax.scatter([ud[f'{prefix}_RMSE'].median()], [ud[f'{prefix}_P_Net_Trend'].median()],
                   color=MODEL_COLORS[model_key], s=220, marker='D', zorder=6,
                   edgecolors='black', linewidth=1.5)
    ax.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.5, label='Chance (0.5)')
    ax.set_xlabel('RMSE (mg/dL) — lower is better →', fontsize=12)
    ax.set_ylabel('Net Trend Accuracy (sign) — higher is better ↑', fontsize=12)
    ax.set_title('Per-user RMSE vs Net Trend Accuracy', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9 if len(active) > 3 else 11, markerscale=1.2 if len(active) > 3 else 1.5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_scatter.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200); plt.close()
    print(f'  {path}')


def _roc_curve_and_auroc(true_labels, probs, score_eps=1e-6):
    """Rank-based ROC via sklearn.roc_curve; scores clipped to [eps, 1-eps]
    to avoid sentinel-point artefacts when outputs pile up at exactly 0/1
    (common here — see the horizon-figure FAR discussion). AUROC computed
    on the original unclipped scores."""
    from sklearn.metrics import roc_auc_score, roc_curve
    probs_clipped = np.clip(probs, score_eps, 1 - score_eps)
    fprs, senss, _ = roc_curve(true_labels.astype(int), probs_clipped)
    try:
        auroc = roc_auc_score(true_labels.astype(int), probs)
    except Exception:
        order = np.argsort(fprs)
        auroc = float(np.trapz(senss[order], fprs[order]))
    return fprs, senss, auroc


def plot_roc_hypo(df, out_dir, beta_sens=0.90, beta_spec=0.90, active=None):
    """Hypoglycaemia detection ROC — 2 panels (1-hr actionable alarm window,
    4-hr planning window), ALL active models overlaid per panel with AUROC
    in the legend. This is the design that actually produced the published
    figure (originally a standalone script, dev_scripts/plot_roc_hypo.py —
    evaluate.py's own version had drifted into an unrelated one-subplot-
    per-model layout at some point and never matched it). Ported back in
    and generalized to N models via the same active/ALL_LABELS/MODEL_COLORS
    convention used throughout this file, so it no longer needs a separate
    script or a hardcoded 3-model list."""
    active = active or get_active_models(df)

    def _plot_panel(ax, horizon_label, prob_suffix, gt_suffix, panel_title):
        d = df[df['horizon'] == horizon_label].copy()
        gt_col = f'G_{gt_suffix}'
        if gt_col not in d.columns:
            ax.text(0.5, 0.5, f'{gt_col} not available',
                    ha='center', va='center', transform=ax.transAxes)
            return
        d['true_hypo'] = d[gt_col].notna()

        ax.plot([0, 1], [0, 1], color='#bbbbbb', lw=1.0, ls='--',
                zorder=2, label='Random classifier')
        ax.axhline(beta_sens, color='#27ae60', lw=1.4, ls='--', alpha=0.85,
                   zorder=3, label=f'Sensitivity = {beta_sens:.2f}')
        ax.axvline(1 - beta_spec, color='#1565C0', lw=1.4, ls='--', alpha=0.85,
                   zorder=3, label=f'Specificity = {beta_spec:.2f}')

        n_total_str = ''
        for model_key, prefix in active:
            prob_col = f'{prefix}_{prob_suffix}'
            if prob_col not in d.columns:
                continue
            paired = d[['true_hypo', prob_col]].dropna()
            if len(paired) == 0 or paired['true_hypo'].sum() == 0:
                continue
            true_hypo = paired['true_hypo'].values
            probs_arr = paired[prob_col].values
            fprs, senss, auroc = _roc_curve_and_auroc(true_hypo, probs_arr)
            if not n_total_str:
                n_total_str = f'N={len(true_hypo):,} instances  ({int(true_hypo.sum()):,} hypo)'
            ax.plot(fprs, senss, color=MODEL_COLORS[model_key],
                    linestyle=MODEL_LINESTYLES[model_key], lw=1.4, zorder=4,
                    label=f'{ALL_LABELS[model_key]}  (AUROC = {auroc:.3f})')
            # Flag large single-step jumps (heavy score ties -- e.g. MC-dropout
            # posteriors quantized to ~100 values, or AR1's coarse threshold
            # grid) with markers at both endpoints. The connecting line is a
            # correct rank-based interpolation (AUROC itself is a rank
            # statistic, unaffected by how the plot draws ties) but is a
            # straight line across two widely-spaced *achievable* points, not
            # a continuum of achievable operating points -- readers should
            # not read off an intermediate (FPR, sens) pair on a marked
            # segment as a real threshold. See caption.
            jump = np.hypot(np.diff(fprs), np.diff(senss))
            big = np.where(jump > 0.05)[0]
            if len(big):
                mark_idx = np.unique(np.concatenate([big, big + 1]))
                ax.scatter(fprs[mark_idx], senss[mark_idx],
                           color=MODEL_COLORS[model_key], s=22, zorder=5,
                           edgecolor='white', linewidth=0.6)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=11)
        ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=11)
        ax.set_title(f'{panel_title}\n{n_total_str}', fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.15, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7.2 if len(active) > 5 else 8.5, loc='lower right',
                  framealpha=0.93, edgecolor='#cccccc',
                  handlelength=2.8, labelspacing=0.4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'wspace': 0.42})
    _plot_panel(axes[0], '60 min', 'P_Hypo_Prob_60min', 'P_Hypo_Sens_60min',
               '1 hour Actionable Alarm Window')
    _plot_panel(axes[1], '240 min', 'P_Hypo_Prob', 'P_Hypo_Sens',
               '4 hour Planning Window')
    for ext in ['.pdf', '.png']:
        fig.savefig(os.path.join(out_dir, f'fig_roc_hypo{ext}'),
                    bbox_inches='tight', dpi=200 if ext == '.pdf' else 150)
    plt.close(fig)
    print('  fig_roc_hypo.pdf')


def run_sample_stability_check(g_uids, l_uids, imap, out_dir, n_users=20, seed=42):
    rng = np.random.default_rng(seed)
    common = sorted(set(g_uids) & set(l_uids))
    if not common: print('  WARNING: No common users for stability check'); return
    check_users = list(rng.choice(common, size=min(n_users,len(common)), replace=False))
    check_metrics = ['P_Net_Trend','P_Mean_Dir_Bias','P_Hypo_Sens',
                     'P_Hypo_Sens_60min','P_Hypo_Spec','P_CEG_D']
    tex_path = os.path.join(out_dir, 'tab_sample_stability.tex')
    with open(tex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]\centering\small' + '\n')
        f.write(r'\caption{Sample count stability.}' + '\n')
        f.write(r'\label{tab:sample_stability}' + '\n')
        f.write(r'\begin{tabular}{l}\toprule\bottomrule\end{tabular}\end{table}' + '\n')
    print(f'  {tex_path}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate all models (metrics, tables, figures).')
    parser.add_argument('--prior',           type=float, default=0.30)
    parser.add_argument('--workers',         type=int,   default=16)
    parser.add_argument('--beta-sens',       type=float, default=0.90)
    parser.add_argument('--beta-spec',       type=float, default=0.90)
    parser.add_argument('--out-dir',         type=str,   default=None,
                        help='Override cfg["OUT_DIR"] — write here instead, '
                             'e.g. to rerun without touching prior results.')
    args = parser.parse_args()

    from eval_config import get_config
    cfg          = get_config(prior_scale=args.prior)
    TARGET_USERS = cfg['TARGET_USERS']
    DATA_DIR     = cfg['DATA_DIR']
    FILTER_DIR   = cfg['FILTER_DIR']
    OUT_DIR      = args.out_dir or cfg['OUT_DIR']
    BATCH_LIST   = cfg['BATCH_LIST']

    INST_CSV  = os.path.join(DATA_DIR, 'non_exercise.csv')
    MEALS_CSV = os.path.join(DATA_DIR, 'meals.csv')
    BOLUS_CSV = os.path.join(DATA_DIR, 'bolus.csv')

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True)
    cache_path = imap_cache_path(INST_CSV)

    print(f'\n{"="*60}')
    print(f'EVALUATE — N={len(TARGET_USERS)} users, prior={args.prior}')
    print(f'Models: {LABEL_SAFE} | {LABEL_BASE} | {LABEL_AR1} | '
          f'{LABEL_AR1RLS} (if available) | {LABEL_TRANSFORMER} (if available)')
    print('Directional: P_Net_Trend | P_Mean_Dir_Bias')
    print(f'Output: {OUT_DIR}')
    print(f'{"="*60}\n')

    # ── Step 1: Instance type map ─────────────────────────────────────────────
    print('Step 1/5: Instance type map')
    inst_df = pd.read_csv(INST_CSV)
    tc = next((c for c in ['time_series_start','timestamp'] if c in inst_df.columns), None)
    if tc: inst_df['time_series_start'] = pd.to_datetime(inst_df[tc]).dt.tz_localize(None)
    with open(os.path.join(FILTER_DIR, 'filtered_user_instances.pkl'), 'rb') as fh:
        curated_dict = pickle.load(fh)
    all_curated = set()
    for eids in curated_dict.values(): all_curated.update(eids)
    inst_df = inst_df[inst_df['ex'].isin(all_curated)].copy()
    meals_df = pd.read_csv(MEALS_CSV)
    meals_df.rename(columns={'MLDTC':'datestamp','meal_dose':'carbs',
                              'MLDOSE':'carbs','USUBJID':'id'}, inplace=True, errors='ignore')
    if 'datestamp' in meals_df.columns:
        meals_df['datestamp'] = pd.to_datetime(meals_df['datestamp']).dt.tz_localize(None)
    if 'carbs' in meals_df.columns:
        meals_df = meals_df[meals_df['carbs'] > 0]
    bolus_df = pd.DataFrame()
    if os.path.exists(BOLUS_CSV):
        bolus_df = pd.read_csv(BOLUS_CSV)
        bolus_df.rename(columns={'USUBJID':'id','datestamp':'timestamp'},
                        inplace=True, errors='ignore')
        if 'timestamp' in bolus_df.columns:
            bolus_df['timestamp'] = pd.to_datetime(bolus_df['timestamp']).dt.tz_localize(None)
    imap = build_or_load_instance_type_map(inst_df, meals_df, bolus_df, cache_path)

    # ── Step 2: Load bundles ──────────────────────────────────────────────────
    print('\nStep 2/5: Loading inference bundles')
    batch_dirs = list(BATCH_LIST)
    print('\n  Scanning completed results on disk...')
    completed = get_completed_users(batch_dirs)
    target_set       = set(str(u) for u in TARGET_USERS)
    completed_global = completed[MODEL_SAFE] & target_set
    completed_lstm   = completed[MODEL_BASE] & target_set
    completed_ar1    = completed[MODEL_AR1]  & target_set
    eval_users_gl  = sorted(completed_global & completed_lstm, key=lambda x: int(x))
    print(f'\n  {LABEL_SAFE}: {len(completed_global)}, {LABEL_BASE}: {len(completed_lstm)}, '
          f'{LABEL_AR1}: {len(completed_ar1)}')
    print(f'  Evaluation set (Global∩LSTM): {len(eval_users_gl)} users')
    common_users = eval_users_gl or sorted(completed_global, key=lambda x: int(x))
    bundle_paths = find_bundle_paths(batch_dirs, common_users)
    n_global = len(bundle_paths[MODEL_SAFE]); n_lstm = len(bundle_paths[MODEL_BASE])
    n_ar1    = len(bundle_paths[MODEL_AR1])
    counts   = {MODEL_SAFE: n_global, MODEL_BASE: n_lstm, MODEL_AR1: n_ar1}

    # ── Additional comparators (AR1-RLS, Transformer-MC, LSTM-Ensemble,
    # LSTM-Conformal, Transformer-Conformal, Transformer-Ensemble) ──────────
    extra_batch_dirs = list(BATCH_LIST)
    extra_model_keys = [mk for mk, _ in EXTRA_MODELS]
    completed_extra = get_completed_users(extra_batch_dirs, models=extra_model_keys)
    extra_paths = find_bundle_paths(extra_batch_dirs, common_users, models=extra_model_keys)
    for mk, pfx in EXTRA_MODELS:
        completed = completed_extra[mk] & target_set & set(common_users)
        n_present = len(extra_paths[mk])
        counts[mk] = n_present
        print(f'  {ALL_LABELS[mk]}: {len(completed)} completed, {n_present} bundles found '
              f'(of {len(common_users)} evaluation users)')

    # ── Step 3: Compute metrics ───────────────────────────────────────────────
    print('\nStep 3/5: Computing metrics')
    g_paths_map = bundle_paths[MODEL_SAFE]
    l_paths_map = bundle_paths[MODEL_BASE]
    a_paths_map = bundle_paths[MODEL_AR1]
    task_args = []
    for uid in common_users:
        g_path = g_paths_map.get(uid)
        if g_path is None: continue
        l_path = l_paths_map.get(uid)
        a_path = a_paths_map.get(uid)
        extra_pairs = [(pfx, extra_paths[mk].get(uid)) for mk, pfx in EXTRA_MODELS]
        task_args.append((uid, g_path, l_path, a_path, imap, HORIZONS, extra_pairs))

    all_rows = []
    with Pool(processes=args.workers) as pool:
        for rows in tqdm(pool.imap_unordered(compute_user_metrics, task_args),
                         total=len(task_args), desc='  Computing'):
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if len(df) > 0 and isinstance(df['horizon'].iloc[0], int):
        df['horizon'] = df['horizon'].map(dict(zip(HORIZONS, HORIZON_LABELS)))

    print(f'  Total instance-horizon pairs: {len(df):,}')
    d4h = df[df['horizon'] == '240 min']
    print('\n  Directional summary (240 min, all instances):')
    dir_models = [('G',LABEL_SAFE),('L',LABEL_BASE),('A',LABEL_AR1)]
    dir_models += [(pfx, ALL_LABELS[mk]) for mk, pfx in EXTRA_MODELS if counts.get(mk, 0) > 0]
    for prefix, label in dir_models:
        nta = d4h[f'{prefix}_P_Net_Trend'].dropna().mean()
        mdb = d4h[f'{prefix}_P_Mean_Dir_Bias'].dropna().mean()
        print(f'    {label:<14}  NTA(sign)={nta:.3f}  MeanDirBias={mdb:.1f} mg/dL')

    print('\nStep 3b/5: AUROC (supplement)')
    df = compute_and_fill_auroc(df)

    print('\nStep 3c/5: Sample stability')
    run_sample_stability_check(g_paths_map.keys(), l_paths_map.keys(), imap, OUT_DIR)

    # ── Step 4: Write outputs ─────────────────────────────────────────────────
    print('\nStep 4/5: Writing outputs')
    summary_path = os.path.join(OUT_DIR, 'results_summary.txt')
    with open(summary_path, 'w') as fh:
        write_summary_txt(fh, df, len(TARGET_USERS), counts)
    print(f'  {summary_path}')

    tex_path = os.path.join(OUT_DIR, 'tables.tex')
    with open(tex_path, 'w') as fh:
        fh.write('% Generated by evaluate.py\n')
        fh.write('% Directional: P_Net_Trend | P_Mean_Dir_Bias\n\n')
        write_table1_dataset(fh, common_users, bundle_paths[MODEL_SAFE], imap)
        if common_users:
            write_table2_main(fh, df, out_dir=OUT_DIR)
            write_table_ar1_baseline(fh, df, out_dir=OUT_DIR)
            write_table3_horizons(fh, df)
            write_table4_stratified(fh, df, out_dir=OUT_DIR)
            write_table_meal_response_accuracy(fh, df, out_dir=OUT_DIR)
            write_table5_paradox(fh, df)
            write_table_false_alarm(fh, df, out_dir=OUT_DIR)
    print(f'  {tex_path}')

    df_path = os.path.join(OUT_DIR, 'metrics_df.pkl')
    df.to_pickle(df_path); print(f'  {df_path}')

    # ── Step 5: Figures ───────────────────────────────────────────────────────
    print('\nStep 5/5: Figures')
    active = get_active_models(df)
    plot_horizon_comparison(df, OUT_DIR, active=active)
    plot_instance_type_comparison(df, OUT_DIR, active=active)
    plot_user_level_metrics(df, OUT_DIR, active=active)
    plot_rmse_vs_nta(df, OUT_DIR, active=active)
    plot_roc_hypo(df, OUT_DIR, active=active)
    horizon_data, itype_data, user_data, plot_meta = extract_all_plot_data(df, active)
    save_plot_data(OUT_DIR, horizon_data, itype_data, user_data, plot_meta)

    print(f'\nDone. Results in: {OUT_DIR}')


if __name__ == '__main__':
    main()