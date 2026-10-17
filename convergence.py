#!/usr/bin/env python3
"""convergence.py
==============
MCMC convergence diagnostics (R-hat, ESS) for Bayesian Loop and Pure AR1,
read from each user's diagnostics/convergence_diagnostics_summary.csv.
Produces the convergence histograms, a text summary, and the LaTeX tables
(main prior plus the 0.10/0.50 prior-sensitivity fits).

Usage:
    python convergence.py

Outputs: <MAIN_RESULTS_DIR>/convergence/
"""

import argparse
import glob
import os
import sys
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

import config
import eval_config

OUT_DIR = os.path.join(config.MAIN_RESULTS_DIR, 'convergence')

SCALES = {
    'main (scale=0.30)': 0.30,
    'loose (scale=0.70)': 0.50,
    'tight (scale=0.10)': 0.10,
}
# Dict values are the PRIOR_SCALE trigger passed to evaluate.py/run_cohort.py
# (0.50 selects models/ar1.py's `sensitivity_analysis` branch via the
# `_PRIOR_SCALE >= 0.50` check). The labels are the actual effective prior
# scale that branch hardcodes for theta_CR/the ISF-CR ratio (0.70, not 0.50
# -- see models/ar1.py).


# ============================================================================
# DATA LOADING
# ============================================================================

def load_diagnostics_by_model(batch_names, valid_ids=None):
    """
    Load Bayesian Loop and Pure AR1 diagnostics separately from one or more
    batches. batch_names may be a single batch dir name/path (str, original
    behavior) or a list of them (one run directory per user or shard).

    Rows are deduped by user_id (keep last): a retried run can leave a
    diagnostics row from an earlier, incomplete attempt.

    Returns dict: {'bayesian_loop': (df, rhat_cols, ess_cols),
                   'pure_ar1':      (df, rhat_cols, ess_cols)}
    """
    if isinstance(batch_names, str):
        batch_names = [batch_names]

    all_csvs = []
    for name in batch_names:
        batch_dir = name if os.path.isabs(name) else os.path.join(config.EXPERIMENTS_DIR, name)
        all_csvs.extend(sorted(glob.glob(
            os.path.join(batch_dir, '**/diagnostics/convergence_diagnostics_summary.csv'),
            recursive=True)))

    bl_csvs  = [c for c in all_csvs
                if 'carb_uncert_global' in c or 'ar1_with_carb' in c]
    ar1_csvs = [c for c in all_csvs if 'ar1_pure' in c]

    def _process(csvs, label):
        if not csvs:
            return None
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        if 'user_id' in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset='user_id', keep='last').reset_index(drop=True)
            deduped = before - len(df)
            if deduped:
                print(f'    {label}: dropped {deduped} retry-duplicate row(s) '
                      f'(kept latest attempt) -> {len(df)}')
        if valid_ids is not None and 'user_id' in df.columns:
            before = len(df)
            df = df[df['user_id'].astype(str).isin(valid_ids)].copy()
            skipped = before - len(df)
            if skipped:
                print(f'    {label}: filtered {skipped} rows ({len(df)} remaining)')
        rhat_cols = [c for c in df.columns if c.startswith('rhat_')]
        ess_cols  = [c for c in df.columns if c.startswith('ess_')]
        for c in rhat_cols + ess_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df, rhat_cols, ess_cols

    return {
        'bayesian_loop': _process(bl_csvs,  'Bayesian Loop'),
        'pure_ar1':      _process(ar1_csvs, 'Pure AR1'),
    }


def summarize(df, rhat_cols, ess_cols):
    all_rhats = df[rhat_cols].values.flatten()
    all_rhats = all_rhats[~np.isnan(all_rhats)]
    all_ess   = df[ess_cols].values.flatten() if ess_cols else np.array([])
    all_ess   = all_ess[~np.isnan(all_ess)]
    return {
        'n_users':     len(df),
        'median_rhat': float(np.median(all_rhats)),
        'max_rhat':    float(np.max(all_rhats)),
        'pct_lt_105':  float(np.mean(all_rhats < 1.05) * 100),
        'pct_lt_101':  float(np.mean(all_rhats < 1.01) * 100),
        'median_ess':  float(np.median(all_ess)) if len(all_ess) > 0 else np.nan,
        'min_ess':     float(np.min(all_ess))    if len(all_ess) > 0 else np.nan,
        'per_param':   {
            col.replace('rhat_', ''): {
                'median_rhat': float(df[col].median()),
                'max_rhat':    float(df[col].max()),
                'pct_lt_105':  float(np.mean(df[col] < 1.05) * 100),
                'median_ess':  float(df[col.replace('rhat_', 'ess_')].median())
                               if col.replace('rhat_', 'ess_') in df.columns else np.nan,
            }
            for col in rhat_cols if col in df.columns
        }
    }


# ============================================================================
# TEXT OUTPUT
# ============================================================================

def write_txt(results_bl, results_ar1, f):
    f.write('=' * 70 + '\n')
    f.write('MCMC CONVERGENCE DIAGNOSTICS (main results)\n')
    f.write('R-hat < 1.05 required; ESS > 400 recommended\n')
    f.write('=' * 70 + '\n\n')

    for model_label, s in [('Bayesian Loop (scale=0.30)', results_bl),
                            ('Pure AR1', results_ar1)]:
        if s is None:
            continue
        f.write(f'{model_label}\n')
        f.write(f'  Users:          {s["n_users"]}\n')
        f.write(f'  Median R-hat:   {s["median_rhat"]:.4f}\n')
        f.write(f'  Max R-hat:      {s["max_rhat"]:.4f}\n')
        f.write(f'  % R-hat < 1.05: {s["pct_lt_105"]:.1f}%\n')
        f.write(f'  % R-hat < 1.01: {s["pct_lt_101"]:.1f}%\n')
        if not np.isnan(s['median_ess']):
            f.write(f'  Median ESS:     {s["median_ess"]:.0f}\n')
            f.write(f'  Min ESS:        {s["min_ess"]:.0f}\n')
        f.write('\n  Per-parameter:\n')
        for param, ps in s['per_param'].items():
            ess_str = f"{ps['median_ess']:.0f}" if not np.isnan(ps['median_ess']) else '-'
            f.write(f'    {param:<20} R-hat median={ps["median_rhat"]:.4f} '
                    f'max={ps["max_rhat"]:.4f} %<1.05={ps["pct_lt_105"]:.1f}% '
                    f'ESS={ess_str}\n')
        f.write('\n')


# ============================================================================
# LATEX OUTPUT
# ============================================================================

def write_latex(results_bl, results_ar1, multi_scale, f):
    f.write('% CONVERGENCE DIAGNOSTICS -- auto-generated by convergence.py v4\n')
    n_bl = results_bl['n_users']
    f.write(f'% Filtered to {n_bl} users via valid_ids from reference batch\n')
    f.write('% Requires: \\usepackage{booktabs}\n\n')

    # Table S1: Multi-scale summary (Bayesian Loop across prior scales)
    f.write(r'\begin{table}[!htbp]' + '\n')
    f.write(r'\centering' + '\n')
    f.write(r'\caption{MCMC convergence diagnostics for Bayesian Loop across '
            f'prior scales (N={n_bl} users, '
            r'1{,}000 warmup / 1{,}000 samples / 4 chains per user). '
            r'$\hat{R} < 1.05$ indicates convergence.}' + '\n')
    f.write(r'\label{tab:convergence}' + '\n')
    f.write(r'\begin{tabular}{lrrrrrr}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'Scale & $N$ & Median $\hat{R}$ & Max $\hat{R}$ '
            r'& \% $\hat{R}<1.05$ & \% $\hat{R}<1.01$ & Median ESS \\' + '\n')
    f.write(r'\midrule' + '\n')
    for label, s in multi_scale.items():
        ess_str = f"{s['median_ess']:.0f}" if not np.isnan(s['median_ess']) else '--'
        f.write(f'  {label} & {s["n_users"]} & {s["median_rhat"]:.4f} & '
                f'{s["max_rhat"]:.4f} & {s["pct_lt_105"]:.1f}\\% & '
                f'{s["pct_lt_101"]:.1f}\\% & {ess_str} \\\\\n')
    f.write(r'\bottomrule' + '\n\end{tabular}' + '\n\end{table}' + '\n\n')

    # Table S2: Per-parameter for Bayesian Loop main results
    PARAM_LABELS = {
        'theta_ISF':      r'$\theta_{\text{ISF}}$',
        'theta_CR':       r'$\theta_{\text{CR}}$',
        'isf_cr_ratio':   r'ISF/CR ratio',
        'phi':            r'$\phi$',
        'sigma_eta':      r'$\sigma_{\eta}$',
        'sigma_obs':      r'$\sigma_{\text{obs}}$',
        'sigma_carb_obs': r'$\sigma_{\text{carb}}$',
        'sigma_carb':     r'$\sigma_{\text{carb}}$',
    }
    f.write(r'\begin{table}[!htbp]' + '\n')
    f.write(r'\centering' + '\n')
    f.write(r'\caption{Per-parameter MCMC convergence diagnostics for Bayesian Loop '
            r'(main results, prior scale $\sigma=0.30$, '
            f'N={n_bl} users, ' r'1{,}000 warmup / 1{,}000 samples / 4 chains). '
            r'$\hat{R} < 1.05$ indicates convergence.}' + '\n')
    f.write(r'\label{tab:convergence_params}' + '\n')
    f.write(r'\begin{tabular}{lrrrr}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'Parameter & Median $\hat{R}$ & Max $\hat{R}$ & \% $<1.05$ & Median ESS \\' + '\n')
    f.write(r'\midrule' + '\n')
    for param, ps in results_bl['per_param'].items():
        ess_str = f"{ps['median_ess']:.0f}" if not np.isnan(ps['median_ess']) else '--'
        label = PARAM_LABELS.get(param, param.replace('_', r'\_'))
        f.write(f'  {label} & {ps["median_rhat"]:.4f} & {ps["max_rhat"]:.4f} & '
                f'{ps["pct_lt_105"]:.1f}\\% & {ess_str} \\\\\n')
    f.write(r'\bottomrule' + '\n\end{tabular}' + '\n\end{table}' + '\n')


# ============================================================================
# PLOTS
# ============================================================================

def _plot_one_model(df, rhat_cols, ess_cols, title, color_ess, path):
    """R-hat bar chart + ESS histogram figure for one model."""
    params   = [c.replace('rhat_', '') for c in rhat_cols]
    n_params = len(params)
    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))
    if n_params == 1:
        axes = axes.reshape(2, 1)

    for j, param in enumerate(params):
        rcol = f'rhat_{param}'
        ecol = f'ess_{param}'

        ax = axes[0, j]
        rhat_vals = df[rcol].dropna().values
        unique, counts = np.unique(np.round(rhat_vals, 4), return_counts=True)
        bar_colors = ['#d32f2f' if v >= 1.05 else '#ff9800' if v >= 1.01
                      else '#2196F3' for v in unique]
        ax.bar([str(round(u, 4)) for u in unique], counts,
               color=bar_colors, alpha=0.85, edgecolor='white')
        ax.set_title(param, fontweight='bold')
        ax.set_xlabel(r'$\hat{R}$')
        if j == 0:
            ax.set_ylabel('Users')
        ax.tick_params(axis='x', rotation=45)
        pct = np.mean(rhat_vals < 1.05) * 100
        ax.text(0.98, 0.98, f'{pct:.1f}% < 1.05',
                transform=ax.transAxes, ha='right', va='top',
                color='green', fontweight='bold')

        ax = axes[1, j]
        if ecol in df.columns:
            ess_vals = df[ecol].dropna()
            ax.hist(ess_vals, bins=30, color=color_ess, alpha=0.8, edgecolor='white')
            ax.axvline(400, color='red', linestyle='--', linewidth=1.5, label='ESS=400')
            ax.set_xlabel('ESS')
            if j == 0:
                ax.set_ylabel('Users')
                ax.legend(loc='upper right')

    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  {path}')


def plot_convergence_histograms(df_bl, rhat_bl, ess_bl,
                                df_ar1, rhat_ar1, ess_ar1,
                                n_users, out_dir):
    """Separate figures for Bayesian Loop and Pure AR1."""
    bl_path = os.path.join(out_dir, 'convergence_histograms_bayesian_loop.pdf')
    _plot_one_model(
        df_bl, rhat_bl, ess_bl,
        title=(f'MCMC Convergence Diagnostics — Bayesian Loop '
               f'(main results, scale=0.30)\n'
               f'N={n_users} users, 1000 warmup / 1000 samples / 4 chains'),
        color_ess='#FF5722',
        path=bl_path,
    )
    import shutil
    shutil.copy2(bl_path, os.path.join(out_dir, 'convergence_histograms.pdf'))
    if df_ar1 is not None:
        _plot_one_model(
            df_ar1, rhat_ar1, ess_ar1,
            title=(f'MCMC Convergence Diagnostics — Pure AR1\n'
                   f'N={n_users} users, 1000 warmup / 1000 samples / 4 chains'),
            color_ess='#4CAF50',
            path=os.path.join(out_dir, 'convergence_histograms_pure_ar1.pdf'),
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MCMC convergence diagnostics '
                                     '(Bayesian Loop and Pure AR1).')
    parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = eval_config.get_config(prior_scale=eval_config.MAIN_PRIOR_SCALE)
    valid_ids = set(str(u) for u in cfg['TARGET_USERS'])
    by_model = load_diagnostics_by_model(cfg['BATCH_LIST'], valid_ids=valid_ids)
    bl  = by_model['bayesian_loop']
    ar1 = by_model['pure_ar1']

    if bl is None:
        print('  WARNING: No Bayesian Loop diagnostics found.')
        return

    df_bl,  rhat_bl,  ess_bl  = bl
    df_ar1, rhat_ar1, ess_ar1 = ar1 if ar1 is not None else (None, [], [])

    s_bl  = summarize(df_bl,  rhat_bl,  ess_bl)
    s_ar1 = summarize(df_ar1, rhat_ar1, ess_ar1) if df_ar1 is not None else None

    print(f'  Bayesian Loop: N={s_bl["n_users"]} '
          f'median_Rhat={s_bl["median_rhat"]:.4f} '
          f'max={s_bl["max_rhat"]:.4f} '
          f'%<1.05={s_bl["pct_lt_105"]:.1f}% '
          f'median_ESS={s_bl["median_ess"]:.0f}')
    if s_ar1:
        print(f'  Pure AR1:      N={s_ar1["n_users"]} '
              f'median_Rhat={s_ar1["median_rhat"]:.4f} '
              f'max={s_ar1["max_rhat"]:.4f} '
              f'%<1.05={s_ar1["pct_lt_105"]:.1f}% '
              f'median_ESS={s_ar1["median_ess"]:.0f}')

    # Combined figure
    plot_convergence_histograms(
        df_bl,  rhat_bl,  ess_bl,
        df_ar1, rhat_ar1, ess_ar1,
        n_users=s_bl['n_users'],
        out_dir=OUT_DIR,
    )

    # Text summary
    txt_path = os.path.join(OUT_DIR, 'convergence_summary.txt')
    with open(txt_path, 'w') as f:
        write_txt(s_bl, s_ar1, f)
    print(f'  {txt_path}')

    # Multi-scale summary (Bayesian Loop at each prior scale)
    print('\nLoading all prior scales for multi-scale summary table...')
    multi_scale = {}
    for label, scale in SCALES.items():
        if scale == eval_config.MAIN_PRIOR_SCALE:
            df_ms, rhat_ms, ess_ms = df_bl, rhat_bl, ess_bl
        else:
            dirs = [d for d in eval_config.resolve_batches(scale)
                    if eval_config._is_bayesian_loop_dir(d)]
            bl_ms = load_diagnostics_by_model(dirs, valid_ids=valid_ids)['bayesian_loop']
            if bl_ms is None:
                continue
            df_ms, rhat_ms, ess_ms = bl_ms
        multi_scale[label] = summarize(df_ms, rhat_ms, ess_ms)
        s = multi_scale[label]
        print(f'  {label}: N={s["n_users"]} '
              f'median_Rhat={s["median_rhat"]:.4f} '
              f'%<1.05={s["pct_lt_105"]:.1f}% '
              f'median_ESS={s["median_ess"]:.0f}')

    # LaTeX tables
    tex_path = os.path.join(OUT_DIR, 'convergence_tables.tex')
    with open(tex_path, 'w') as f:
        write_latex(s_bl, s_ar1, multi_scale, f)
    print(f'  {tex_path}')


if __name__ == '__main__':
    main()