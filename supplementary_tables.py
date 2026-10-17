#!/usr/bin/env python3
"""
supplementary_tables.py
=======================
Builds the supplementary tables that summarize data and run-time properties
rather than model metrics:

  tab_completeness_prevalence.tex  severity of likely missing / under-reported
                                   carbohydrates in the pre-curation sample and
                                   the retained test and train sets
                                   (from completeness_analysis.py's per-instance
                                   scores, restricted to the analysed cohort)
  tab_glucose_composition.tex      share of hypo-, hyper- and in-range CGM
                                   readings in the 4-hour forecast windows of
                                   the train and test instances
  tab_scalability.tex              MCMC run time vs number of training
                                   instances on CPU and GPU (from
                                   scalability_timing.py's results)

Outputs go to <RESULTS_DIR>/eval_results/supplementary_tables/.

Usage:
  python supplementary_tables.py                 # all tables whose inputs exist
  python supplementary_tables.py --only scalability
"""
import argparse
import glob
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import eval_config

OUT_DIR = os.path.join(config.EVAL_RESULTS_DIR, 'supplementary_tables')
COMPLETENESS_DIR = os.path.join(config.EVAL_RESULTS_DIR, 'completeness_analysis')
SEVERITY_ORDER = ['CLEAN', 'MILD', 'MODERATE', 'SEVERE']
HYPO_MGDL, HYPER_MGDL = 70.0, 180.0
FORECAST_STEPS = 48          # 4-hour forecast window, 5-minute steps
SCALABILITY_USER = '37'
CPU_CORES = 2                # cores used for the CPU timing (slurm/scalability_timing.sbatch)


def latex_int(n):
    """Integer with LaTeX thousands separators, e.g. 59134 -> 59{,}134."""
    return f'{int(n):,}'.replace(',', '{,}')


def analysed_users():
    """User ids of the analysed cohort (eval_config.load_eligible_users)."""
    return eval_config.load_eligible_users(config.data_dir(), config.filter_dir(), verbose=False)


# ---------------------------------------------------------------------------
# Completeness prevalence
# ---------------------------------------------------------------------------

def completeness_prevalence_table(users):
    """Severity-grade percentages for the three scored instance sets of
    completeness_analysis.py, each restricted to `users`."""
    sets = [('full_pool', 'Pre-curation sample'),
            ('retained_test', 'Retained test set'),
            ('retained_train', 'Retained train set')]
    keep = set(str(u) for u in users)
    stats = []
    for key, _ in sets:
        path = os.path.join(COMPLETENESS_DIR, f'{key}_scored_instances.csv')
        df = pd.read_csv(path, dtype={'uid': str})
        df = df[df['uid'].isin(keep)]
        pct = df['severity'].value_counts(normalize=True) * 100
        stats.append(dict(n=len(df), users=df['uid'].nunique(),
                          pct={s: pct.get(s, 0.0) for s in SEVERITY_ORDER}))

    n_users = len(keep)
    lines = [
        '\\begin{table}[!htbp]',
        '\\centering',
        '\\caption{Prevalence of a heuristic signature of likely missing or under-reported '
        f'carbohydrates for the same $N={n_users}$ users throughout (severity grades defined in '
        'Section~\\ref{app:completeness-methods}). \\emph{Pre-curation sample}: every 48th '
        'candidate window (about 4 hours apart) across each user\'s entire record, spanning both '
        'the training and test periods, before any curation filter is applied. It is shown to '
        'illustrate how common the signature is in the raw data and what curation removes; the '
        '\\textsc{severe} grade matches the curation exclusion rule, so it is absent from both '
        'retained sets by construction. \\emph{Retained test set} and \\emph{retained train set}: '
        'the held-out test instances evaluated throughout the paper and the instances used to fit '
        'each user\'s models.}',
        '\\label{tab:completeness_prevalence}',
        '\\begin{adjustbox}{max width=\\linewidth}',
        '\\begin{tabular}{lrrr}',
        '\\toprule',
        ' & ' + ' & '.join(label for _, label in sets) + ' \\\\',
        ' & ' + ' & '.join(f'($n={latex_int(s["n"])}$, {s["users"]} users)' for s in stats) + ' \\\\',
        '\\midrule',
    ]
    for sev in SEVERITY_ORDER:
        lines.append(f'  {sev} & ' + ' & '.join(f'{s["pct"][sev]:.2f}\\%' for s in stats) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}']
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Glucose composition of the forecast windows
# ---------------------------------------------------------------------------

def _forecast_window_readings(args):
    """CGM readings of the 4-hour forecast window of every train and test
    instance of one user: the FORECAST_STEPS readings starting at the
    forecast origin (the instance's 48th reading)."""
    uid, data_dir, filter_dir = args
    with open(os.path.join(filter_dir, 'per_user', f'user_{uid}.pkl'), 'rb') as f:
        pu = pickle.load(f)
    cgm = pd.read_parquet(os.path.join(data_dir, 'by_user', 'cgm', f'{uid}.parquet'),
                          columns=['ex', 'cgm', 'datestamp'])
    cgm['datestamp'] = pd.to_datetime(cgm['datestamp'])
    out = {}
    for split in ('train', 'test'):
        sub = cgm[cgm['ex'].isin(set(pu[f'{split}_ids']))]
        vals = []
        for _, g in sub.groupby('ex'):
            g = g.sort_values('datestamp')
            if len(g) < 48:
                continue
            t0 = g['datestamp'].iloc[47]          # forecast origin: 48th reading
            window = g.loc[g['datestamp'] >= t0, 'cgm'].values
            if len(window) >= FORECAST_STEPS:
                vals.append(window[:FORECAST_STEPS].astype(float))
        out[split] = np.concatenate(vals) if vals else np.array([])
    return out


def glucose_composition_table(users, n_workers):
    """Hypo (<70), hyper (>180) and in-range shares of the forecast-window
    CGM readings of the train and test instances of `users`."""
    args = [(u, config.data_dir(), config.filter_dir()) for u in users]
    with Pool(n_workers) as pool:
        res = pool.map(_forecast_window_readings, args)
    stats = {}
    for split in ('train', 'test'):
        v = np.concatenate([r[split] for r in res])
        n = len(v)
        hypo, hyper = (v < HYPO_MGDL).sum(), (v > HYPER_MGDL).sum()
        stats[split] = dict(n=n, hypo=100 * hypo / n, hyper=100 * hyper / n,
                            inrange=100 * (n - hypo - hyper) / n)
    lines = [
        '\\begin{table}[!tp]',
        '\\centering',
        '\\small',
        '\\caption{Composition of the 5-minute CGM readings within the training and test windows '
        f'for the $N={len(users)}$ evaluated users. ``Hypo\'\' and ``Hyper\'\' are pointwise '
        'thresholds ($<$70~mg/dL and $>$180~mg/dL respectively) applied to every individual '
        '5-minute reading, not to whole instances (contrast with '
        'Table~\\ref{tab:completeness_prevalence}\'s instance-level severity grading). }',
        '\\label{tab:glucose_composition}',
        '\\begin{tabular}{lrr}',
        '\\toprule',
        f' & Train ({latex_int(stats["train"]["n"])} readings) & '
        f'Test ({latex_int(stats["test"]["n"])} readings) \\\\',
        '\\midrule',
        f'Hypoglycemic ($<$70~mg/dL) & {stats["train"]["hypo"]:.2f}\\% & {stats["test"]["hypo"]:.2f}\\% \\\\',
        f'Hyperglycemic ($>$180~mg/dL) & {stats["train"]["hyper"]:.2f}\\% & {stats["test"]["hyper"]:.2f}\\% \\\\',
        f'In range & {stats["train"]["inrange"]:.2f}\\% & {stats["test"]["inrange"]:.2f}\\% \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# MCMC run-time scalability
# ---------------------------------------------------------------------------

def _gpu_name(batch_dir):
    """GPU model recorded in a timing run's result bundles, without vendor prefix."""
    for path in glob.glob(os.path.join(batch_dir, 'n*', '*', 'results', '*_results.pkl')):
        with open(path, 'rb') as f:
            name = (pickle.load(f).get('env_provenance') or {}).get('gpu_name')
        if name:
            return name.replace('NVIDIA ', '').replace('GeForce ', '')
    return 'GPU'


def scalability_table(cpu_dir, gpu_dir):
    """Run time (CPU and GPU) and predictive metrics by training-set size."""
    cpu = pd.read_csv(os.path.join(cpu_dir, 'scalability_results.csv')).set_index('n_train')
    gpu = pd.read_csv(os.path.join(gpu_dir, 'scalability_results.csv')).set_index('n_train')
    with open(os.path.join(cpu_dir, 'scalability_fit.json')) as f:
        fit_cpu = json.load(f)
    with open(os.path.join(gpu_dir, 'scalability_fit.json')) as f:
        fit_gpu = json.load(f)
    gpu_name = _gpu_name(gpu_dir)
    metrics = ['RMSE', 'Coverage', 'Width']
    same = np.allclose(cpu.loc[gpu.index, metrics].values, gpu[metrics].values)
    n_test = int(cpu['n_test'].iloc[0])

    def fit_str(fit):
        return (f'$t = {fit["slope_sec_per_train_instance"]:.1f} \\times N_{{\\text{{train}}}} '
                f'{fit["intercept_sec"]:+.1f}$~s ($R^2={fit["r2"]:.3f}$)')

    metric_note = ('Predictive metrics are identical on both devices'
                   if same else 'Predictive metrics (shown for CPU) differ slightly between devices')
    lines = [
        '\\begin{table}[!htbp]',
        '\\centering',
        f'\\caption{{Runtime and predictive metrics by training set size (user {fit_cpu["user_id"]}, '
        f'fixed test set of {n_test} instances), with all {_number_word(len(cpu))} sizes run '
        f'sequentially on each of two devices: CPU ({CPU_CORES} cores, the configuration used for '
        f'all main-cohort fits) and GPU (NVIDIA {gpu_name}). Runtime scales linearly: CPU '
        f'$\\approx {fit_cpu["slope_sec_per_train_instance"]:.1f}$~s per training instance '
        f'($R^2={fit_cpu["r2"]:.3f}$), GPU $\\approx {fit_gpu["slope_sec_per_train_instance"]:.1f}$~s '
        f'($R^2={fit_gpu["r2"]:.3f}$). {metric_note} and show no consistent trend with additional '
        'training data.}',
        '\\label{tab:scalability}',
        '\\small',
        '\\begin{tabular}{rrrrrrrrr}',
        '\\toprule',
        f' & & \\multicolumn{{2}}{{c}}{{CPU ({CPU_CORES} cores)}} & '
        f'\\multicolumn{{2}}{{c}}{{GPU ({gpu_name})}} & & & \\\\',
        '$N_{\\text{train}}$ & $N_{\\text{test}}$ & Runtime (min) & s/instance & Runtime (min) & '
        's/instance & RMSE & Coverage & Width \\\\',
        '\\midrule',
    ]
    for n in cpu.index:
        c, g = cpu.loc[n], gpu.loc[n]
        lines.append(
            f'{n} & {int(c["n_test"])} & {c["runtime_min"]:.1f} & {c["runtime_sec"] / n:.2f} & '
            f'{g["runtime_min"]:.1f} & {g["runtime_sec"] / n:.2f} & {c["RMSE"]:.1f} & '
            f'{c["Coverage"]:.3f} & {c["Width"]:.1f} \\\\')
    lines += [
        '\\midrule',
        f'\\multicolumn{{9}}{{l}}{{\\small Linear fits: CPU {fit_str(fit_cpu)}; GPU {fit_str(fit_gpu)}}} \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def _number_word(n):
    words = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
    return words.get(n, str(n))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', choices=['prevalence', 'composition', 'scalability'], action='append')
    ap.add_argument('--cpu-dir', default=os.path.join(
        config.EXPERIMENTS_DIR, f'scalability_real_user{SCALABILITY_USER}_cpu'))
    ap.add_argument('--gpu-dir', default=os.path.join(
        config.EXPERIMENTS_DIR, f'scalability_real_user{SCALABILITY_USER}_gpu'))
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    args = ap.parse_args()
    selected = args.only or ['prevalence', 'composition', 'scalability']
    os.makedirs(OUT_DIR, exist_ok=True)
    users = analysed_users() if {'prevalence', 'composition'} & set(selected) else None

    builders = {
        'prevalence': ('tab_completeness_prevalence.tex', lambda: completeness_prevalence_table(users)),
        'composition': ('tab_glucose_composition.tex', lambda: glucose_composition_table(users, args.workers)),
        'scalability': ('tab_scalability.tex', lambda: scalability_table(args.cpu_dir, args.gpu_dir)),
    }
    missing = []
    for key in selected:
        name, build = builders[key]
        try:
            tex = build()
        except FileNotFoundError as e:
            print(f'  skipped {name}: input not found ({e.filename})')
            if key != 'scalability':   # the timing runs are optional (WITH_TIMING)
                missing.append(name)
            continue
        with open(os.path.join(OUT_DIR, name), 'w') as f:
            f.write(tex)
        print(f'  wrote {os.path.join(OUT_DIR, name)}')
    if missing:
        raise SystemExit(f'missing inputs for: {", ".join(missing)}')


if __name__ == '__main__':
    main()
