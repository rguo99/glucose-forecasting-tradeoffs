#!/usr/bin/env python3
"""parameter_analysis.py
=====================
Population-level summary of the Bayesian Loop posteriors (main prior scale).
Loads every analysed user's posterior samples and produces:
  1. per-user posterior-mean distributions (inter-patient variability)
  2. credible-interval widths (identifiability)
  3. LaTeX table of parameter summary statistics (parameter_table.tex)
  4. pooled posterior KDEs (kde_parameter_distributions.pdf)

Usage:
    python parameter_analysis.py --prior 0.30
"""

import glob
import os
import sys
import pickle
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

# CONFIG
import config
import eval_config

OUT_DIR = os.path.join(config.MAIN_RESULTS_DIR, 'parameter_analysis')

# Parameters to show and their display info
# (key, display_label, units, physiological_note)
PARAMS = [
    ('theta_ISF',      r'ISF ($\theta_{ISF}$)',      'mg/dL/U',
     'Limited identifiability\non real data'),
    ('theta_CR',       r'CR ($\theta_{CR}$)',         'g/U',
     'Limited identifiability\non real data'),
    ('isf_cr_ratio',   'ISF/CR Ratio',                'dimensionless',
     'Ratio more identifiable\nthan individual params'),
    ('phi',            r'AR(1) Persistence ($\phi$)', 'dimensionless',
     'Near random walk\n(phi → 0.99)'),
    ('sigma_eta',      r'Process Noise ($\sigma_\eta$)', 'mg/dL',
     'Residual model\nmismatch'),
    ('sigma_obs',      r'Obs. Noise ($\sigma_{obs}$)', 'mg/dL',
     'Collapsed near zero\n(AR1 absorbs noise)'),
    ('sigma_carb_obs', r'Carb Error ($\sigma_{carb}$)', 'relative',
     'Identifiable: 33%\nmedian carb error'),
    ('carb_adjustment', 'Carb Adjustment',             'relative',
     'Positive = overreport\nNegative = underreport'),
]


# ============================================================================
# DATA LOADING
# ============================================================================

def find_bayesian_loop_results_dirs(batch_dirs):
    """All Bayesian Loop '*carb_uncert_global*/results' dirs in the given run dirs."""
    results_dirs = []
    for d in batch_dirs:
        for folder in sorted(glob.glob(os.path.join(d, '*carb_uncert_global*'))):
            rd = os.path.join(folder, 'results')
            if os.path.isdir(rd):
                results_dirs.append(rd)
    if not results_dirs:
        raise FileNotFoundError('No *carb_uncert_global*/results dirs found in the model-run dirs')
    print(f'  Found {len(results_dirs)} Bayesian Loop results dirs')
    return results_dirs


def collect_pkl_paths(results_dirs):
    """
    Merge '*.pkl' paths across one or more results dirs, keyed by the user id
    parsed from the filename ('user_<uid>_results.pkl'). Later dirs in the
    list win on duplicate user ids -- same "later folder wins" convention as
    evaluate.py's find_bundle_paths(); a user found in two run directories
    is an error.
    """
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    paths = {}
    for d in results_dirs:
        for p in sorted(glob.glob(os.path.join(d, '*.pkl'))):
            fname = os.path.basename(p)
            if fname.startswith('user_') and fname.endswith('_results.pkl'):
                uid = fname[len('user_'):-len('_results.pkl')]
            else:
                uid = fname
            if uid in paths:
                raise RuntimeError(f'Bayesian Loop user {uid} has results in more than '
                                   f'one run directory:\n  {paths[uid]}\n  {p}')
            paths[uid] = p
    return sorted(paths.values())


def load_all_params(results_dirs, valid_ids=None):
    """
    Load per-user parameter statistics from pkl bundles.
    results_dirs may be a single directory (str) or a list of directories
    (one results dir per user or shard) -- see
    collect_pkl_paths() for the merge/dedup logic.
    If valid_ids is provided (a set of str user IDs), only those users are loaded.
    Returns dict: param_key -> list of per-user posterior means
    Also returns: param_key -> list of per-user CI widths (95th - 5th percentile)
    """
    pkls = collect_pkl_paths(results_dirs)
    print(f'  Found {len(pkls)} pkl bundles on disk...')
    if valid_ids is not None:
        print(f'  Filtering to {len(valid_ids)} TARGET_USERS...')

    param_means       = {k: [] for k, *_ in PARAMS}
    param_widths      = {k: [] for k, *_ in PARAMS}
    param_samples_all = {k: [] for k, *_ in PARAMS}
    user_ids = []
    skipped  = 0

    for path in pkls:
        try:
            with open(path, 'rb') as f:
                b = pickle.load(f)
            uid = b.get('user_id')

            # Filter to TARGET_USERS if supplied
            if valid_ids is not None and str(uid) not in valid_ids:
                skipped += 1
                continue

            ps = b.get('param_samples', {})
            user_ids.append(uid)
            for key, *_ in PARAMS:
                if key in ps:
                    samples = np.asarray(ps[key]).flatten()
                    samples = samples[np.isfinite(samples)]
                    if len(samples) > 0:
                        param_means[key].append(float(np.mean(samples)))
                        param_widths[key].append(
                            float(np.percentile(samples, 97.5) -
                                  np.percentile(samples, 2.5)))
                        n_keep = min(200, len(samples))
                        idx = np.linspace(0, len(samples)-1, n_keep, dtype=int)
                        param_samples_all[key].extend(samples[idx].tolist())
                    else:
                        param_means[key].append(np.nan)
                        param_widths[key].append(np.nan)
                else:
                    param_means[key].append(np.nan)
                    param_widths[key].append(np.nan)
        except Exception as e:
            print(f'  Warning: {os.path.basename(path)}: {e}')

    print(f'  Loaded {len(user_ids)} users'
          + (f' (skipped {skipped} outside TARGET_USERS)' if skipped else ''))
    return param_means, param_widths, param_samples_all, user_ids


# ============================================================================
# LATEX TABLE
# ============================================================================

def write_latex(param_means, param_widths, f, n_users):
    f.write('% PARAMETER ANALYSIS -- auto-generated by parameter_analysis.py\n')
    f.write('% Requires: \\usepackage{booktabs}\n\n')

    f.write(r'\begin{table}[!htbp]' + '\n')
    f.write(r'\centering\small' + '\n')
    f.write(r'\caption{Population-level posterior parameter estimates across '
            f'N={n_users} users (prior scale $\\sigma=0.30$). '
            r'Values are posterior means per user; summary statistics are '
            r"computed across users. CI width = the median, across users, of each user's "
            r'95\% credible interval width.}' + '\n')
    f.write(r'\label{tab:parameters}' + '\n')
    f.write(r'\begin{adjustbox}{max width=\linewidth}' + '\n')
    f.write(r'\begin{tabular}{llrrrrrrr}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'Parameter & Units & $N$ & Mean & Median & SD & '
            r'5\textsuperscript{th} & 95\textsuperscript{th} & '
            r'Median CI Width \\' + '\n')
    f.write(r'\midrule' + '\n')

    for key, label, units, note in PARAMS:
        vals = np.array([v for v in param_means[key] if not np.isnan(v)])
        widths = np.array([v for v in param_widths[key] if not np.isnan(v)])
        if len(vals) == 0:
            f.write(f'  {label} & {units} & — & — & — & — & — & — & — \\\\\n')
            continue
        n   = len(vals)
        mn  = np.mean(vals)
        med = np.median(vals)
        sd  = np.std(vals)
        p5  = np.percentile(vals, 5)
        p95 = np.percentile(vals, 95)
        ciw = np.median(widths) if len(widths) > 0 else np.nan
        ciw_str = f'{ciw:.3f}' if not np.isnan(ciw) else '—'
        f.write(f'  {label} & {units} & {n} & {mn:.3f} & {med:.3f} & '
                f'{sd:.3f} & {p5:.3f} & {p95:.3f} & {ciw_str} \\\\\n')

    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{adjustbox}' + '\n')
    f.write(r'\end{table}' + '\n\n')


# ============================================================================
# FIGURES
# ============================================================================

def plot_parameter_distributions(param_means, param_widths, out_dir, n_users=395):
    """
    
    - Removed gray column notes.
    - Moved median values to the top of each violin to prevent overlap with bars.
    - Optimized font sizes and padding for high-resolution output.
    """
    n_params = len(PARAMS)
    
    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 28,
        'axes.labelsize': 28,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
    })

    fig, axes = plt.subplots(2, n_params, figsize=(5 * n_params, 12))

    color_main  = '#2196F3'
    color_width = '#FF5722'

    for j, (key, label, units, _) in enumerate(PARAMS):
        # Row 0: posterior means
        ax = axes[0, j]
        vals = [v for v in param_means[key] if not np.isnan(v)]
        if vals:
            vp = ax.violinplot([vals], positions=[0], showmedians=True,
                               showextrema=True)
            for body in vp['bodies']:
                body.set_facecolor(color_main)
                body.set_alpha(0.6)
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(3)
            
            # Place median text at the top of the violin range
            y_max = ax.get_ylim()[1]
            ax.text(0.5, 0.92, f'med={np.median(vals):.2f}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=24, color='black', fontweight='bold')
        
        ax.set_title(label, fontweight='bold', pad=15)
        ax.set_xticks([])
        # Only show units on the left-most plot to reduce clutter
        ax.set_ylabel(units if j == 0 else '', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')

        # Row 1: CI widths
        ax = axes[1, j]
        widths = [v for v in param_widths[key] if not np.isnan(v)]
        if widths:
            vp = ax.violinplot([widths], positions=[0], showmedians=True,
                               showextrema=True)
            for body in vp['bodies']:
                body.set_facecolor(color_width)
                body.set_alpha(0.6)
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(3)
            
            # Place median text at the top of the violin range
            ax.text(0.5, 0.92, f'med={np.median(widths):.2f}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=28, color='black', fontweight='bold')
        
        ax.set_xticks([])
        ax.set_ylabel('95% CI Width' if j == 0 else '', fontsize=28)
        ax.grid(True, alpha=0.3, axis='y')

    # Specific labels for the first column
    axes[0, 0].set_ylabel('Posterior Mean\n(per user)', fontsize=28, fontweight='bold')
    axes[1, 0].set_ylabel('95% CI Width\n(per user)', fontsize=28, fontweight='bold')

    plt.suptitle(f'Population-Level Parameter Distributions (N={n_users} pump users)',
                 fontsize=24, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    path = os.path.join(out_dir, 'parameter_distributions.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  {path}')

def plot_carb_adjustment(param_means, out_dir):
    """
    Histogram of carb_adjustment values across users.
    Positive = overreporting, negative = underreporting.
    """
    vals = np.array([v for v in param_means['carb_adjustment'] if not np.isnan(v)])
    if len(vals) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of carb_adjustment
    ax = axes[0]
    ax.hist(vals, bins=40, color='#2196F3', alpha=0.8, edgecolor='white')
    ax.axvline(0, color='black', linewidth=1.5, linestyle='-', label='No adjustment')
    ax.axvline(np.median(vals), color='red', linewidth=2, linestyle='--',
               label=f'Median = {np.median(vals):.2f}')
    ax.set_xlabel('Carb Adjustment (positive = overreport)')
    ax.set_ylabel('Users')
    ax.set_title('Carb Reporting Bias Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    pct_over  = np.mean(vals > 0) * 100
    pct_under = np.mean(vals < 0) * 100
    ax.text(0.98, 0.98, f'{pct_over:.0f}% overreport\n{pct_under:.0f}% underreport',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: histogram of sigma_carb_obs
    ax = axes[1]
    sc_vals = np.array([v for v in param_means['sigma_carb_obs'] if not np.isnan(v)])
    ax.hist(sc_vals, bins=40, color='#FF5722', alpha=0.8, edgecolor='white')
    ax.axvline(np.median(sc_vals), color='red', linewidth=2, linestyle='--',
               label=f'Median = {np.median(sc_vals):.2f}')
    ax.set_xlabel(r'Carb Reporting Noise ($\sigma_{carb}$, relative)')
    ax.set_ylabel('Users')
    ax.set_title('Carb Reporting Error Magnitude', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.98,
            f'Median: {np.median(sc_vals):.2f}\n'
            f'(= {np.median(sc_vals)*100:.0f}% relative error)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Carbohydrate Reporting Error Across T1DEXI Pump Users',
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'carb_adjustment.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  {path}')


def plot_identifiability_scatter(param_means, param_widths, out_dir):
    """
    Scatter: posterior mean vs CI width per user for ISF and sigma_carb.
    Wide CI = low identifiability. Shows which params are identifiable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (key, label, units, _) in zip(axes, [
        ('theta_ISF',      r'ISF ($\theta_{ISF}$)', 'mg/dL/U', ''),
        ('sigma_carb_obs', r'$\sigma_{carb}$',      'relative', ''),
    ]):
        means  = np.array(param_means[key])
        widths = np.array(param_widths[key])
        valid  = ~(np.isnan(means) | np.isnan(widths))
        ax.scatter(means[valid], widths[valid], alpha=0.4, s=20,
                   color='#2196F3', linewidths=0)
        ax.set_xlabel(f'Posterior Mean ({units})')
        ax.set_ylabel('95% CI Width')
        ax.set_title(f'{label}: Mean vs Uncertainty', fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Correlation
        if valid.sum() > 10:
            r = np.corrcoef(means[valid], widths[valid])[0, 1]
            ax.text(0.98, 0.98, f'r = {r:.2f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=12, fontweight='bold')

    plt.suptitle('Parameter Identifiability: Posterior Mean vs Uncertainty',
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'identifiability_scatter.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  {path}')



def plot_kde_distributions(param_samples_all, out_dir, n_users):
    """
    KDE of pooled posterior samples across all users.
    Shows the population-level posterior distribution for each parameter.
    Useful for assessing whether posteriors concentrate relative to prior.
    """
    from scipy.stats import gaussian_kde

    params_info = [
        ('theta_ISF',      'ISF (mg/dL/U)',         (0,    50)),
        ('theta_CR',       'CR (g/U)',               (0,    25)),
        ('phi',            'AR(1) phi',              (0.985, 0.9905)),
        ('sigma_eta',      'sigma_eta (mg/dL)',      (0,    15)),
        ('sigma_obs',      'sigma_obs (mg/dL)',      (0,   0.5)),
        ('sigma_carb_obs', 'sigma_carb (relative)',  (0.1,  0.7)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (key, label, xlim) in zip(axes, params_info):
        samples = np.array(param_samples_all.get(key, []))
        samples = samples[np.isfinite(samples)]
        samples = samples[(samples >= xlim[0]) & (samples <= xlim[1])]
        if len(samples) < 10:
            ax.set_visible(False)
            continue

        kde = gaussian_kde(samples, bw_method=0.15)
        x = np.linspace(xlim[0], xlim[1], 300)
        ax.fill_between(x, kde(x), alpha=0.5, color='#2196F3')
        ax.plot(x, kde(x), color='#1565C0', linewidth=2, label='Posterior')

        # Overlay prior if available
        prior_kdes = {
            'theta_ISF':      None,  # complex derived param, skip
            'theta_CR':       ('lognormal', 2.30, 0.30),
            'isf_cr_ratio':   None,
            'phi':            None,  # Beta-derived, skip
            'sigma_eta':      ('lognormal', 1.10, 0.50),
            'sigma_obs':      ('lognormal', 2.85, 1.00),
            'sigma_carb_obs': ('lognormal', -1.60, 0.30),
        }
        prior_spec = prior_kdes.get(key)
        if prior_spec is not None:
            dist_type, loc, scale = prior_spec
            if dist_type == 'lognormal':
                from scipy.stats import lognorm
                prior_samples = lognorm.rvs(s=scale, scale=np.exp(loc), size=10000)
                prior_samples = prior_samples[(prior_samples >= xlim[0]) & (prior_samples <= xlim[1])]
                if len(prior_samples) > 10:
                    prior_kde = gaussian_kde(prior_samples, bw_method=0.15)
                    ax.plot(x, prior_kde(x), color='gray', linewidth=2,
                            linestyle='--', alpha=0.8, label='Prior')

        med = np.median(samples)
        ax.axvline(med, color='red', linestyle='--', linewidth=1.5,
                   label=f'Median={med:.3f}')

        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(label, fontweight='bold')
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.text(0.02, 0.98,
                f'N={n_users} users\nMean={np.mean(samples):.3f}\nStd={np.std(samples):.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(
        f'Population-Level Posterior Distributions (N={n_users} pump users)\n'
        f'Pooled posterior samples across all users -- prior scale sigma=0.30',
        fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'kde_parameter_distributions.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  {path}')

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=float, default=0.30)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load TARGET_USERS from eval_config (same cohort as evaluate.py)
    from eval_config import get_config
    cfg = get_config(prior_scale=args.prior)
    valid_ids = set(str(u) for u in cfg['TARGET_USERS'])
    print(f'Cohort: {len(valid_ids)} users')
    results_dirs = find_bayesian_loop_results_dirs(cfg['BATCH_LIST'])

    param_means, param_widths, param_samples_all, user_ids = load_all_params(
        results_dirs, valid_ids=valid_ids)

    n_users = len(user_ids)

    # Print summary to console
    print(f'\n{"Parameter":<22} {"N":<6} {"Mean":<10} {"Median":<10} '
          f'{"5th":<10} {"95th":<10} {"Std":<10}')
    print('-' * 78)
    for key, label, units, note in PARAMS:
        v = np.array([x for x in param_means[key] if not np.isnan(x)])
        if len(v) == 0:
            continue
        print(f'{key:<22} {len(v):<6} {np.mean(v):<10.3f} {np.median(v):<10.3f} '
              f'{np.percentile(v,5):<10.3f} {np.percentile(v,95):<10.3f} '
              f'{np.std(v):<10.3f}')

    # LaTeX table — N comes from actual loaded users, not hardcoded
    tex_path = os.path.join(OUT_DIR, 'parameter_table.tex')
    with open(tex_path, 'w') as f:
        write_latex(param_means, param_widths, f, n_users=n_users)
    print(f'\n  {tex_path}')

    # Figures
    plot_parameter_distributions(param_means, param_widths, OUT_DIR, n_users=n_users)
    plot_kde_distributions(param_samples_all, OUT_DIR, n_users)
    plot_carb_adjustment(param_means, OUT_DIR)
    plot_identifiability_scatter(param_means, param_widths, OUT_DIR)

    print(f'\nDone. Results in: {OUT_DIR}')


if __name__ == '__main__':
    main()