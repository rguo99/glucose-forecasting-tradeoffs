#!/usr/bin/env python3
"""sensitivity.py
==============
Sensitivity of the results to the Bayesian Loop prior scale (0.10, 0.30,
0.50): predictive metrics from each scale's metrics_df.pkl and posterior
parameter means from each scale's fits.

Prerequisite:
    python evaluate.py --prior 0.30
    python evaluate.py --prior 0.10
    python evaluate.py --prior 0.50

Usage:
    python sensitivity.py

Outputs: <MAIN_RESULTS_DIR>/sensitivity/ (sensitivity_predictive.pdf,
sensitivity_parameters.pdf, sensitivity_tables.tex)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

ROOT = os.path.dirname(os.path.abspath(__file__))

import config
import eval_config

# CONFIG
SCALES = {
    '0.10 (tight)': 0.10,
    '0.30 (main)':  0.30,
    '0.70 (loose)': 0.50,
}
# Dict values are the PRIOR_SCALE trigger passed to evaluate.py/run_cohort.py
# (0.50 selects models/ar1.py's `sensitivity_analysis` branch via the
# `_PRIOR_SCALE >= 0.50` check). The labels are the actual effective prior
# scale that branch hardcodes for theta_CR/the ISF-CR ratio (0.70, not 0.50
# -- see models/ar1.py).

# v2: directional and safety metrics now use full-posterior P_ columns,
# matching the primary reported metrics in the main paper.
# Statistical UQ metrics (RMSE, CRPS, Coverage, Width, IS) are unchanged.
PRED_METRICS = [
    # Statistical UQ — unchanged
    ('RMSE (mg/dL)',         'RMSE',         True,  False),
    ('CRPS (mg/dL)',         'CRPS',         True,  False),
    ('Coverage (95% PI)',    'Coverage',     False, True),
    ('PI Width (mg/dL)',     'Width',        True,  False),
    ('Interval Score',       'IS',           True,  False),
    # Directional — full posterior (v2)
    ('Slope Corr (90 min)',  'P_Slope_Corr', False, False),
    ('Net Trend Acc (%)',    'P_Net_Trend',  False, True),
    ('Sign Match (%)',       'P_Sign_Match', False, True),
    # Safety — full posterior (v2)
    ('Hypo Sensitivity (%)', 'P_Hypo_Sens',  False, True),
    ('Hypo Specificity (%)', 'P_Hypo_Spec',  False, True),
    # CEG — full posterior (v2)
    ('CEG Zone A (%)',       'P_CEG_A',      False, False),
    ('CEG Zone D (%)',       'P_CEG_D',      True,  False),
]

# Inferred parameters — unchanged from v1 (S3 figure unaffected)
PARAM_METRICS = [
    ('ISF (mg/dL/U)',        'theta_ISF'),
    ('CR (g/U)',             'theta_CR'),
    ('ISF/CR Ratio',         'isf_cr_ratio'),
    ('AR1 Persistence phi',  'phi'),
    ('sigma_eta (mg/dL)',    'sigma_eta'),
    ('sigma_obs (mg/dL)',    'sigma_obs'),
    ('sigma_carb (rel.)',    'sigma_carb_obs'),
]

OUT_DIR = os.path.join(config.MAIN_RESULTS_DIR, 'sensitivity')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_metrics(path):
    full = path
    if not os.path.exists(full):
        return None
    df = pd.read_pickle(full)
    # Warn if P_ columns absent — means old evaluate.py was used
    p_cols = [c for c in df.columns if c.startswith('G_P_')]
    if not p_cols:
        print('  WARNING: No P_ columns found — metrics_df.pkl was produced by an older evaluate.py.')
        print('    Rerun: python evaluate.py --prior <scale>')
        print('    Falling back to posterior-mean columns for directional metrics.')
    return df


def load_params_for_batch(batch_name, valid_ids=None):
    """
    Load per-user posterior means for all parameters from a batch.
    If valid_ids is provided (set of str user IDs), only those users
    are included — filtering out MDI users in the n=483 batch.

    batch_name may be a single run directory or a list of them (one per
    user or shard); posterior means are aggregated across all of them.
    """
    import glob, pickle
    batch_names = [batch_name] if isinstance(batch_name, str) else list(batch_name)

    # Keyed by user id parsed from the filename, "later folder wins" -- same
    # dedup convention as parameter_analysis.py's collect_pkl_paths(). Without
    # this, a user whose results exist in more than one shard/retry directory
    # for the same scale is double-counted (seen for the main scale, which can
    # span multiple shard directories; tight/loose each use a single directory
    # and were never affected).
    pkls_by_uid = {}
    for name in batch_names:
        experiments_dir = name if os.path.isabs(name) else os.path.join(config.EXPERIMENTS_DIR, name)
        folders = sorted(glob.glob(os.path.join(experiments_dir, '*carb_uncert_global*')))
        if not folders:
            folders = sorted(glob.glob(os.path.join(experiments_dir, '*ar1*carb*')))
        if not folders:
            continue
        for folder in folders:
            for p in sorted(glob.glob(os.path.join(folder, 'results', '*.pkl'))):
                fname = os.path.basename(p)
                if fname.startswith('user_') and fname.endswith('_results.pkl'):
                    uid = fname[len('user_'):-len('_results.pkl')]
                else:
                    uid = fname
                pkls_by_uid[uid] = p
    pkls = sorted(pkls_by_uid.values())

    if not pkls:
        print(f'  WARNING: No Global folder/results found in {batch_names}')
        return {}

    param_data = {p[1]: [] for p in PARAM_METRICS}
    skipped = 0
    for pkl_path in pkls:
        try:
            with open(pkl_path, 'rb') as f:
                b = pickle.load(f)
            uid = b.get('user_id')
            if valid_ids is not None and str(uid) not in valid_ids:
                skipped += 1
                continue
            ps = b.get('param_samples', {})
            for _, key in PARAM_METRICS:
                if key in ps:
                    param_data[key].append(float(np.mean(ps[key])))
                else:
                    param_data[key].append(np.nan)
        except Exception:
            pass
    if skipped:
        print(f'    (skipped {skipped} users outside TARGET_USERS)')
    return param_data


# ============================================================================
# HELPERS
# ============================================================================

def _resolve_col(df, prefix, col):
    """
    Return the actual column name to use.
    If the requested P_ column is absent (old pkl), fall back to the
    posterior-mean version so the script degrades gracefully.
    """
    full = f'{prefix}_{col}'
    if df is not None and full not in df.columns and col.startswith('P_'):
        fallback = f'{prefix}_{col[2:]}'  # strip 'P_'
        if df is not None and fallback in df.columns:
            return fallback
    return full


def get_stat(df, prefix, col, pct, horizon='240 min'):
    """Return (mean, sem) for a metric at a given horizon."""
    if df is None:
        return np.nan, np.nan
    full_col = _resolve_col(df, prefix, col)
    if full_col not in df.columns:
        return np.nan, np.nan
    vals = df[df['horizon'] == horizon][full_col].dropna()
    scale = 100 if pct else 1
    return float(vals.mean() * scale), float(vals.sem() * scale)


def fmt(v, decimals=2):
    if np.isnan(v):
        return '—'
    return f'{v:.{decimals}f}'


# ============================================================================
# LATEX OUTPUT
# ============================================================================

def write_latex(scale_dfs, param_data_by_scale, f):
    scales = list(scale_dfs.keys())
    ncols  = len(scales)
    col_spec = 'l' + 'r' * ncols

    f.write('% SENSITIVITY ANALYSIS -- auto-generated by sensitivity.py v3\n')
    f.write('% Directional/safety metrics: full posterior P_ columns\n')
    n_users = next(iter(scale_dfs.values()))['uid'].nunique()
    f.write(f'% Parameter data filtered to TARGET_USERS (n={n_users} pump users)\n')
    f.write('% Requires: \\usepackage{booktabs}\n\n')

    for model_label, prefix in [('Bayesian Loop', 'G'), ('LSTM-MC', 'L')]:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering\small' + '\n')
        f.write(
            f'\\caption{{Sensitivity analysis: {model_label} predictive performance '
            r'across prior scales ($\sigma \in \{0.10, 0.30, 0.70\}$). '
            r'Statistical metrics (RMSE, CRPS, Coverage, PI Width, Interval Score) '
            r'use posterior mean or empirical quantiles. '
            r'Directional and safety metrics use the full posterior sample '
            r'distribution, matching the primary reported metrics in '
            r'Table~\ref{tab:main_results}. '
            r'Bold = best value per row.}' + '\n'
        )
        tag = 'global' if prefix == 'G' else 'lstm'
        f.write(f'\\label{{tab:sensitivity_{tag}}}' + '\n')
        f.write(f'\\begin{{tabular}}{{{col_spec}}}' + '\n')
        f.write(r'\toprule' + '\n')
        header = 'Metric & ' + ' & '.join(
            f'$\\sigma={s.split()[0]}$' for s in scales) + r' \\'
        f.write(header + '\n')
        f.write(r'\midrule' + '\n')

        for label, col, lower, pct in PRED_METRICS:
            means = [get_stat(df, prefix, col, pct)[0]
                     for df in scale_dfs.values()]
            valid  = [(i, v) for i, v in enumerate(means) if not np.isnan(v)]
            best_i = -1
            if valid:
                best_i = (min if lower else max)(valid, key=lambda x: x[1])[0]
            cells = []
            for i, v in enumerate(means):
                s = fmt(v)
                if i == best_i and s != '—':
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
            f.write(f'  {label} & ' + ' & '.join(cells) + r' \\' + '\n')

        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n\n')

    # Parameter table
    f.write(r'\begin{table}[htbp]' + '\n')
    f.write(r'\centering\small' + '\n')
    f.write(r'\caption{Posterior parameter estimates (median across users) '
            f'under different prior scales (N={n_users} users). '
            r'Stability of ISF and $\sigma_\eta$ '
            r'across scales indicates likelihood dominance. '
            r'CR is partially non-identifiable at all scales.}' + '\n')
    f.write(r'\label{tab:sensitivity_params}' + '\n')
    col_spec2 = 'l' + 'r' * ncols
    f.write(f'\\begin{{tabular}}{{{col_spec2}}}' + '\n')
    f.write(r'\toprule' + '\n')
    header2 = 'Parameter & ' + ' & '.join(
        f'$\\sigma={s.split()[0]}$' for s in scales) + r' \\'
    f.write(header2 + '\n')
    f.write(r'\midrule' + '\n')

    for label, key in PARAM_METRICS:
        cells = []
        for scale_label in scales:
            pdata = param_data_by_scale.get(scale_label, {})
            vals  = [v for v in pdata.get(key, []) if not np.isnan(v)]
            cells.append(fmt(np.median(vals)) if vals else '—')
        f.write(f'  {label} & ' + ' & '.join(cells) + r' \\' + '\n')

    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{table}' + '\n\n')


# ============================================================================
# FIGURES
# ============================================================================

def plot_predictive_sensitivity(scale_dfs, out_dir):
    """
    Figure S2: Bayesian Loop UQ metrics vs prior scale.
    LSTM-MC omitted — it has no prior so adds no information to a
    sensitivity analysis. Four core probabilistic metrics only.
    """
    scales_x = [float(k.split()[0]) for k in scale_dfs.keys()]

    plot_metrics = [
        ('RMSE (mg/dL)',       'RMSE',     True,  False),
        ('CRPS (mg/dL)',       'CRPS',     True,  False),
        ('Coverage (95% PI)',  'Coverage', False, True),
        ('PI Width (mg/dL)',   'Width',    True,  False),
    ]

    color  = '#2196F3'
    marker = 'o'

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, (ylabel, col, lower, pct) in zip(axes, plot_metrics):
        means, sems = [], []
        for df in scale_dfs.values():
            m, s = get_stat(df, 'G', col, pct)
            means.append(m); sems.append(s)
        means = np.array(means); sems = np.array(sems)
        valid = ~np.isnan(means)
        if valid.any():
            ax.plot(np.array(scales_x)[valid], means[valid],
                    marker=marker, color=color, label='Bayesian Loop',
                    markersize=8, linewidth=2)
            ax.fill_between(np.array(scales_x)[valid],
                            (means - sems)[valid], (means + sems)[valid],
                            alpha=0.15, color=color)

        ax.set_xlabel('Prior scale $\\sigma$')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.set_xticks(scales_x)
        ax.set_xticklabels([str(x) for x in scales_x])
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.02, '↓ better' if lower else '↑ better',
                transform=ax.transAxes, ha='right', va='bottom', color='gray')

    plt.suptitle(
        'Sensitivity Analysis: Bayesian Loop Predictive Metrics vs Prior Scale ± SEM',
        fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'sensitivity_predictive.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  {path}')


def plot_param_sensitivity(param_data_by_scale, out_dir, n_users):
    """
    Parameter distributions across prior scales (supplementary figure).
    """
    scales   = list(param_data_by_scale.keys())
    n_params = len(PARAM_METRICS)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax, (label, key) in zip(axes, PARAM_METRICS):
        data_per_scale = []
        valid_scales   = []
        for scale in scales:
            vals = [v for v in param_data_by_scale[scale].get(key, [])
                    if not np.isnan(v)]
            if vals:
                data_per_scale.append(vals)
                valid_scales.append(scale)

        if not data_per_scale:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data_per_scale, patch_artist=True, notch=False,
                        medianprops={'color': 'black', 'linewidth': 2})
        colors_box = ['#42A5F5', '#2196F3', '#1565C0']
        for patch, color in zip(bp['boxes'], colors_box[:len(data_per_scale)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(valid_scales) + 1))
        ax.set_xticklabels([f'σ={s.split()[0]}' for s in valid_scales], rotation=15)
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel('Posterior mean (per user)')
        ax.grid(True, alpha=0.3, axis='y')

    for ax in axes[n_params:]:
        ax.set_visible(False)

    plt.suptitle(
        f'Inferred Parameter Distributions Across Prior Scales (N={n_users} users)\n'
        'Stability indicates likelihood dominates prior',
        fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(out_dir, 'sensitivity_parameters.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  {path}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prior-scale sensitivity of the '
                                     'Bayesian Loop (run evaluate.py --prior 0.10/0.30/0.50 first).')
    parser.parse_args()

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    cfg = eval_config.get_config(resolve_batches_now=False)
    valid_ids = set(str(u) for u in cfg['TARGET_USERS'])
    print(f'Cohort: {len(valid_ids)} users')

    print('\nLoading predictive metrics...')
    scale_dfs = {}
    for label, scale in SCALES.items():
        path = os.path.join(eval_config.out_dir_for(scale), 'metrics_df.pkl')
        df = load_metrics(path)
        if df is None:
            print(f'  WARNING: Missing: {path}')
            print(f'    Run: python evaluate.py --prior {scale}')
        else:
            print(f'  {label}: {len(df):,} rows')
        scale_dfs[label] = df

    print('\nLoading inferred parameters from pkl bundles...')
    param_data_by_scale = {}
    for label, scale in SCALES.items():
        print(f'  Loading {label}...')
        dirs = [d for d in eval_config.resolve_batches(scale)
                if eval_config._is_bayesian_loop_dir(d)]
        param_data_by_scale[label] = load_params_for_batch(dirs, valid_ids=valid_ids)
        n = len([v for v in next(iter(param_data_by_scale[label].values()), [])
                 if not np.isnan(v)])
        print(f'    {n} users loaded')

    tex_path = os.path.join(out_dir, 'sensitivity_tables.tex')
    with open(tex_path, 'w') as f:
        write_latex(scale_dfs, param_data_by_scale, f)
    print(f'\n  {tex_path}')

    if any(df is not None for df in scale_dfs.values()):
        plot_predictive_sensitivity(scale_dfs, out_dir)

    if any(param_data_by_scale.values()):
        plot_param_sensitivity(param_data_by_scale, out_dir, n_users=len(valid_ids))

    print(f'\nDone. Results in: {out_dir}')


if __name__ == '__main__':
    main()
