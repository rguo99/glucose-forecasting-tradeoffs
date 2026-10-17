#!/usr/bin/env python3
"""heterogeneity_drivers.py
========================
Candidate drivers of per-user heterogeneity in 4-hour hypoglycemia
sensitivity/specificity. For each model, users whose weaker metric falls
below 0.10 are compared with the rest on:
    1. hypo-event / non-event counts (small-N artifact)
    2. MCMC convergence: max R-hat, ESS of theta_ISF/theta_CR (Bayesian Loop only)
    3. mean directional bias and RMSE
    4. insulin delivery device (AID vs non-AID)
    5. age, sex
    6. HbA1c
    7. number of exercise sessions (T1DEXI PR domain)
    8. glycemic variability (%CV of the user's full CGM record)
using Spearman correlation with the continuous metric and Mann-Whitney /
chi-square tests between groups.

Inputs:
    <OUT_DIR>/metrics_df.pkl (evaluate.py)
    convergence_diagnostics_summary.csv of each user's Bayesian Loop fit
    $T1DEXI_DATA_DIR/netiob_cleaned_data/user_device_data.parquet
    $T1DEXI_RAW_DIR/{DM,LB,PR}.parquet
    $T1DEXI_DATA_DIR/processed_data/non_exercise_8h/by_user/cgm/<uid>.parquet

Outputs (<OUT_DIR>/heterogeneity_analysis/):
    driver_tests.csv, driver_tests_all_models.csv, glucose_variability.csv,
    model_comparison.csv, per_user_covariates.csv, summary.txt,
    tab_heterogeneity_drivers.tex, tab_heterogeneity_drivers_all_models.tex

Usage:
    python heterogeneity_drivers.py
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
from scipy import stats

import eval_config
from eval_config import get_config

import glob

import config

# Set in main(): Bayesian Loop run dirs of the main analysis.
BATCH_DIRS = []

LOW_SPEC_THRESHOLD = 0.10


def load_per_user_metrics(out_dir):
    """4-hr-horizon per-user sens/spec/bias/RMSE for all three models."""
    df_path = os.path.join(out_dir, 'metrics_df.pkl')
    if not os.path.exists(df_path):
        raise FileNotFoundError(f'{df_path} not found. Run evaluate.py first.')
    df = pd.read_pickle(df_path)
    d4h = df[df['horizon'] == '240 min'].copy()

    per_model = {}
    # All nine models evaluate.py can emit; models absent from the pkl are skipped.
    _ALL = [('G', 'Bayesian Loop'), ('L', 'LSTM_MC'), ('A', 'AR1Pure'),
            ('R', 'Adaptive AR1-RLS'), ('T', 'Transformer-MC'),
            ('E', 'LSTM-Ensemble'), ('C', 'LSTM-Conformal'),
            ('F', 'Transformer-Conformal'), ('N', 'Transformer-Ensemble')]
    _present = [(p_, l_) for p_, l_ in _ALL
                if f'{p_}_RMSE' in d4h.columns and d4h[f'{p_}_RMSE'].notna().any()]
    _absent = [l_ for p_, l_ in _ALL if (p_, l_) not in _present]
    print(f'  Models present: {len(_present)} ' + ', '.join(l_ for _, l_ in _present))
    if _absent:
        print('    skipped (no data in this pkl): ' + ', '.join(_absent))
    for prefix, label in _present:
        sens = d4h.groupby('uid')[f'{prefix}_P_Hypo_Sens'].agg(['mean', 'count'])
        spec = d4h.groupby('uid')[f'{prefix}_P_Hypo_Spec'].agg(['mean', 'count'])
        sens = sens[sens['count'] > 0].rename(columns={'mean': 'sens', 'count': 'n_hypo'})
        spec = spec[spec['count'] > 0].rename(columns={'mean': 'spec', 'count': 'n_nonhypo'})
        rmse = d4h.groupby('uid')[f'{prefix}_RMSE'].mean().rename('rmse')
        mdb = d4h.groupby('uid')[f'{prefix}_P_Mean_Dir_Bias'].mean().rename('mdb')
        m = sens.join(spec, how='outer').join(rmse).join(mdb)
        m.index = m.index.astype(int)
        per_model[label] = m
    return per_model


def load_convergence(valid_index):
    """Per-user MCMC diagnostics of the Bayesian Loop fits used in the
    evaluation (the diagnostics file next to each user's results pickle)."""
    rows = []
    for d in BATCH_DIRS:
        for folder in sorted(glob.glob(os.path.join(d, '*carb_uncert_global*'))):
            csv = os.path.join(folder, 'diagnostics', 'convergence_diagnostics_summary.csv')
            if not os.path.exists(csv):
                continue
            c = pd.read_csv(csv)
            has_result = [os.path.exists(os.path.join(folder, 'results', f'user_{u}_results.pkl'))
                          for u in c['user_id']]
            rows.append(c[has_result])
    if not rows:
        raise FileNotFoundError('No Bayesian Loop convergence diagnostics found in the model-run dirs')
    conv = pd.concat(rows, ignore_index=True)
    dup = conv['user_id'][conv['user_id'].duplicated()].unique()
    if len(dup):
        raise RuntimeError(f'Users with diagnostics from more than one Bayesian Loop run: {list(dup)}')
    conv = conv.set_index('user_id')
    conv.index = conv.index.astype(int)
    cols = ['max_rhat', 'ess_theta_ISF', 'ess_theta_CR', 'mean_theta_ISF', 'mean_theta_CR']
    return conv[cols].reindex(valid_index)


def load_device(valid_index):
    dev = pd.read_parquet(config.device_parquet())
    dev['USUBJID'] = dev['USUBJID'].astype(int)
    dev = dev.set_index('USUBJID')
    return dev['device_type'].reindex(valid_index)


def load_demographics_and_labs(valid_index):
    dm = pd.read_parquet(os.path.join(config.raw_dir(), 'DM.parquet'),
                          columns=['USUBJID', 'AGE', 'SEX'])
    dm['USUBJID'] = dm['USUBJID'].astype(int)
    dm = dm.set_index('USUBJID')

    lb = pd.read_parquet(os.path.join(config.raw_dir(), 'LB.parquet'),
                          columns=['USUBJID', 'LBTESTCD', 'LBSTRESN'])
    hba1c = lb[lb['LBTESTCD'] == 'HBA1C'].copy()
    hba1c['USUBJID'] = hba1c['USUBJID'].astype(int)
    hba1c = hba1c.groupby('USUBJID')['LBSTRESN'].mean().rename('hba1c')

    pr = pd.read_parquet(os.path.join(config.raw_dir(), 'PR.parquet'), columns=['USUBJID'])
    pr['USUBJID'] = pr['USUBJID'].astype(int)
    n_exercise = pr.groupby('USUBJID').size().rename('n_exercise_events')

    out = dm[['AGE', 'SEX']].join(hba1c).join(n_exercise)
    out['n_exercise_events'] = out['n_exercise_events'].fillna(0)
    return out.reindex(valid_index)


def compute_glucose_variability(valid_index, cache_path=None):
    """Per-user %CV (coefficient of variation, SD/mean*100) of each user's
    full raw CGM history — a standard glycemic-variability metric (ADA/ATTD
    consensus target: %CV < 36%), added as a candidate heterogeneity driver
    alongside the all-model generalization.

    Reads one parquet per user from CGM_BY_USER_DIR rather than the combined
    cgm.parquet (which is small on disk, ~170MB, but explodes in memory when
    loaded whole in this environment). Each per-user file is heavily
    row-duplicated (repeated CGM readings joined against higher-frequency
    tables upstream) -- verified by inspection that de-duplicating on
    (datestamp, cgm) recovers the expected ~5-minute-interval reading count
    (e.g. one user: 755,377 raw rows -> 7,917 unique readings, ~27.5 days at
    5-min spacing), so that de-dup is applied before computing SD/mean.

    This is a per-user scan (up to ~395 users x up to ~750k raw rows each)
    so results are cached to cache_path and reused on subsequent runs;
    delete the cache file to force a rescan (e.g. if the CGM data changes)."""
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0)
        cached.index = cached.index.astype(int)
        return cached.reindex(valid_index)

    rows = []
    for uid in valid_index:
        path = os.path.join(config.data_dir(), 'by_user', 'cgm', f'{uid}.parquet')
        if not os.path.exists(path):
            continue
        cgm = pd.read_parquet(path, columns=['datestamp', 'cgm'])
        cgm = cgm.drop_duplicates(subset=['datestamp', 'cgm'])
        vals = cgm['cgm'].dropna()
        if len(vals) < 10:
            continue
        mean, sd = vals.mean(), vals.std()
        rows.append(dict(uid=uid, glucose_mean=mean, glucose_sd=sd,
                          glucose_cv_pct=100 * sd / mean, n_cgm_readings=len(vals)))
    out = pd.DataFrame(rows).set_index('uid') if rows else pd.DataFrame(
        columns=['glucose_mean', 'glucose_sd', 'glucose_cv_pct', 'n_cgm_readings'])
    if cache_path:
        out.to_csv(cache_path)
    return out.reindex(valid_index)


def pick_target_metric(model_label, model_df):
    """Which of sensitivity/specificity to test drivers against for this
    model: whichever the model's own row in model_comparison.csv shows
    collapsing (< 0.10) more often
    that every model has heterogeneity in the metric complementary to its
    own aggregate operating point. Ties broken toward specificity."""
    row = model_df.set_index('model').loc[model_label]
    return 'spec' if row['pct_spec_below_0p1'] >= row['pct_sens_below_0p1'] else 'sens'


def run_driver_tests(metrics, target_col='spec', include_mcmc=True, gv=None):
    """Test each candidate driver against low-target-metric (<0.1) status
    and against the continuous target-metric value. `target_col` is 'spec'
    or 'sens' (see pick_target_metric); `include_mcmc` should be False for
    every model except Bayesian Loop, the only one with an MCMC posterior.
    `gv`, if given, is compute_glucose_variability's output, joined in as
    an additional candidate driver. Returns a results DataFrame."""
    idx = metrics.index
    dev = load_device(idx)
    demo = load_demographics_and_labs(idx)

    m = metrics.join(dev).join(demo)
    if gv is not None:
        m = m.join(gv)
    if include_mcmc:
        m = m.join(load_convergence(idx))
    m['low_target'] = m[target_col] < LOW_SPEC_THRESHOLD
    m['is_aid'] = (m['device_type'] == 'AID').astype(float)
    m['is_male'] = (m['SEX'] == 'M').astype(float)

    rows = []

    def add_continuous(name, col):
        if col not in m.columns:
            return
        sub = m[[col, target_col, 'low_target']].dropna()
        if len(sub) < 10 or sub['low_target'].nunique() < 2:
            rows.append(dict(driver=name, kind='continuous', n=len(sub),
                              spearman_r=np.nan, spearman_p=np.nan, mwu_p=np.nan))
            return
        r, p = stats.spearmanr(sub[target_col], sub[col])
        med_low = sub.loc[sub['low_target'], col].median()
        med_rest = sub.loc[~sub['low_target'], col].median()
        u, p_mwu = stats.mannwhitneyu(sub.loc[sub['low_target'], col],
                                       sub.loc[~sub['low_target'], col])
        rows.append(dict(driver=name, kind='continuous', n=len(sub),
                          spearman_r=r, spearman_p=p,
                          median_low_spec_group=med_low, median_rest=med_rest,
                          mwu_p=p_mwu))

    def add_categorical(name, col):
        sub = m[[col, 'low_target']].dropna()
        tab = pd.crosstab(sub[col], sub['low_target'])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            rows.append(dict(driver=name, kind='categorical', n=len(sub), chi2_p=np.nan))
            return
        chi2, p, dof, exp = stats.chi2_contingency(tab)
        rows.append(dict(driver=name, kind='categorical', n=len(sub),
                          chi2_p=p, contingency_table=tab.to_dict()))

    # 1. small-N artifact (denominator size, both for sens and spec)
    add_continuous('n_hypo_events (sens denominator, small-N artifact test)', 'n_hypo')
    add_continuous('n_nonhypo_instances (spec denominator, small-N artifact test)', 'n_nonhypo')
    # 2. MCMC convergence (Bayesian Loop only)
    if include_mcmc:
        add_continuous('max_rhat (MCMC convergence)', 'max_rhat')
        add_continuous('ess_theta_ISF (MCMC effective sample size)', 'ess_theta_ISF')
        add_continuous('ess_theta_CR (MCMC effective sample size)', 'ess_theta_CR')
    # 3. directional bias / RMSE (proximate mechanism)
    add_continuous('mdb (mean directional bias, mg/dL)', 'mdb')
    add_continuous('rmse (mg/dL)', 'rmse')
    # 4. AID vs non-AID
    add_categorical('device_type (AID vs non-AID)', 'device_type')
    # 5. demographics
    add_continuous('age', 'AGE')
    add_categorical('sex', 'SEX')
    # 6. HbA1c
    add_continuous('hba1c', 'hba1c')
    # 7. exercise
    add_continuous('n_exercise_events', 'n_exercise_events')
    # 8. glucose variability
    if gv is not None:
        add_continuous('glucose_cv_pct (%CV, glycemic variability)', 'glucose_cv_pct')

    return pd.DataFrame(rows), m


def run_model_comparison(per_model):
    rows = []
    for label, m in per_model.items():
        n_spec = len(m['spec'].dropna())
        n_sens = len(m['sens'].dropna())
        rows.append(dict(
            model=label,
            n_users_spec=n_spec,
            median_spec=m['spec'].median(),
            mean_spec=m['spec'].mean(),
            pct_spec_below_0p1=100 * (m['spec'] < 0.1).sum() / n_spec,
            n_users_sens=n_sens,
            median_sens=m['sens'].median(),
            pct_sens_below_0p1=100 * (m['sens'] < 0.1).sum() / n_sens,
        ))
    return pd.DataFrame(rows)


_MODEL_DISPLAY = {'LSTM_MC': 'LSTM-MC', 'AR1Pure': 'Pure AR1'}


def _verdict(row):
    """Verdict used in tab_heterogeneity_drivers: driver if p < 0.01 and
    |r| >= 0.3 (categorical: chi-square p < 0.01); weak signal if p < 0.01 and
    |r| < 0.3; ruled out otherwise. For continuous covariates p is the smaller
    of the Spearman and Mann-Whitney p-values."""
    if row['kind'] == 'continuous':
        p = min(row['spearman_p'], row['mwu_p'])
        if p >= 0.01:
            return 'Ruled out', p
        return ('Driver' if abs(row['spearman_r']) >= 0.3 else 'Weak signal'), p
    p = row['chi2_p']
    return ('Driver' if p < 0.01 else 'Ruled out'), p


def _bonferroni_sentence(driver_df):
    """Caption sentence on Bonferroni correction across all tests in the table
    (two per continuous covariate, one per categorical covariate)."""
    n_tests = int(2 * (driver_df['kind'] == 'continuous').sum()
                  + (driver_df['kind'] == 'categorical').sum())
    alpha = 0.05 / n_tests
    flagged = [r for _, r in driver_df.iterrows() if _verdict(r)[0] != 'Ruled out']
    survive = all((r['spearman_p'] < alpha and r['mwu_p'] < alpha) if r['kind'] == 'continuous'
                  else r['chi2_p'] < alpha for r in flagged)
    if survive:
        return (f'All drivers and weak signals remain significant after Bonferroni correction '
                f'across all {n_tests} tests ($p<0.05/{n_tests}$), on both the Spearman and MWU '
                f'tests for continuous covariates.')
    return (f'Not all drivers and weak signals remain significant after Bonferroni correction '
            f'across all {n_tests} tests ($p<0.05/{n_tests}$).')


def write_latex_table(tex_path, driver_df, model_df):
    """Bayesian Loop driver table (tab:heterogeneity_drivers) and the
    cross-model collapse-rate table (tab:heterogeneity_model_comparison)."""
    label_map = {
        'n_hypo_events (sens denominator, small-N artifact test)': 'Hypo-event count (n)',
        'n_nonhypo_instances (spec denominator, small-N artifact test)': 'Non-hypo instance count (n)',
        'max_rhat (MCMC convergence)': 'Max $\\hat{R}$',
        'ess_theta_ISF (MCMC effective sample size)': 'ESS ($\\theta_{\\text{ISF}}$)',
        'ess_theta_CR (MCMC effective sample size)': 'ESS ($\\theta_{\\text{CR}}$)',
        'mdb (mean directional bias, mg/dL)': 'Mean directional bias (mg/dL)',
        'rmse (mg/dL)': 'RMSE (mg/dL)',
        'device_type (AID vs non-AID)': 'Device type (AID vs.\\ non-AID)',
        'age': 'Age',
        'sex': 'Sex',
        'hba1c': 'HbA1c',
        'n_exercise_events': 'Exercise event count (n)',
        'glucose_cv_pct (%CV, glycemic variability)': 'Glucose \\%CV (glycemic variability)',
    }
    n_users, n_hypo = _cohort_counts(model_df, 'Bayesian Loop')
    with open(tex_path, 'w') as f:
        f.write('\\begin{table}[!htbp]\n\\centering\n')
        f.write('\\caption{Candidate drivers of per-user hypoglycemia specificity '
                f'heterogeneity under Bayesian Loop ($N={n_users}$ users, or $N={n_hypo}$ for '
                f'hypo-event-count, which is undefined for the {n_users - n_hypo} users with zero '
                'true-hypo test instances; 4-hour horizon). For continuous covariates we ran two '
                'tests, a Spearman correlation with per-user specificity and a Mann-Whitney $U$ '
                '(MWU) test comparing users with specificity below 0.10 against the rest, and '
                'report the smaller $p$-value; for categorical covariates, a $\\chi^2$ test of '
                'association with low-specificity status. Verdicts: driver if $p<0.01$ and '
                '$|r|\\geq 0.3$ (categorical: $\\chi^2$ $p<0.01$); weak signal if $p<0.01$ and '
                '$|r|<0.3$; ruled out if $p\\geq 0.01$. ' + _bonferroni_sentence(driver_df) + '}\n')
        f.write('\\label{tab:heterogeneity_drivers}\n')
        f.write('\\begin{tabular}{lrrrr}\n\\toprule\n')
        f.write('  Candidate driver & $n$ & Spearman $r$ & $p$ (Spearman/MWU/$\\chi^2$) & Verdict \\\\\n\\midrule\n')
        for _, row in driver_df.iterrows():
            name = label_map.get(row['driver'], row['driver'])
            n = int(row['n'])
            verdict, p = _verdict(row)
            r = f"{row['spearman_r']:.3f}" if row['kind'] == 'continuous' else '--'
            p_str = f'{p:.2e}' if p < 0.001 else f'{p:.3f}'
            f.write(f'  {name} & {n} & {r} & {p_str} & {verdict} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')

        f.write('\\begin{table}[!htbp]\n\\centering\n')
        f.write('\\caption{Per-user hypoglycemia sensitivity/specificity collapse rate '
                '(fraction of users below 0.10) is not unique to Bayesian Loop: each model shows '
                'severe heterogeneity in whichever of sensitivity/specificity it trades away in '
                'aggregate (4-hour horizon).}\n')
        f.write('\\label{tab:heterogeneity_model_comparison}\n')
        f.write('\\begin{tabular}{lrrr}\n\\toprule\n')
        f.write('  Model & Median Spec. & \\% Spec.$<0.10$ & \\% Sens.$<0.10$ \\\\\n\\midrule\n')
        for _, row in model_df.iterrows():
            name = _MODEL_DISPLAY.get(row['model'], row['model'])
            f.write(f'  {name} & {row["median_spec"]:.2f} & '
                    f'{row["pct_spec_below_0p1"]:.1f}\\% & {row["pct_sens_below_0p1"]:.1f}\\% \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    print(f'  {tex_path}')


_ALL_MODEL_DRIVER_ROWS = [
    ('n_hypo_events (sens denominator, small-N artifact test)', 'Hypo/non-target-event count (n)', 'continuous'),
    ('n_nonhypo_instances (spec denominator, small-N artifact test)', 'Non-hypo/target instance count (n)', 'continuous'),
    ('max_rhat (MCMC convergence)', 'Max $\\hat{R}$ (MCMC; BLoop only)', 'continuous'),
    ('ess_theta_ISF (MCMC effective sample size)', 'ESS $\\theta_{\\text{ISF}}$ (MCMC; BLoop only)', 'continuous'),
    ('ess_theta_CR (MCMC effective sample size)', 'ESS $\\theta_{\\text{CR}}$ (MCMC; BLoop only)', 'continuous'),
    ('mdb (mean directional bias, mg/dL)', 'Mean directional bias', 'continuous'),
    ('rmse (mg/dL)', 'RMSE', 'continuous'),
    ('device_type (AID vs non-AID)', 'Device type (AID vs.\\ non-AID)', 'categorical'),
    ('age', 'Age', 'continuous'),
    ('sex', 'Sex', 'categorical'),
    ('hba1c', 'HbA1c', 'continuous'),
    ('n_exercise_events', 'Exercise event count (n)', 'continuous'),
    ('glucose_cv_pct (%CV, glycemic variability)', 'Glucose \\%CV (glycemic variability)', 'continuous'),
]

_MODEL_ABBREV = {
    'Bayesian Loop': 'BLoop', 'LSTM_MC': 'LSTM-MC', 'LSTM-Ensemble': 'LSTM-Ens',
    'LSTM-Conformal': 'LSTM-Conf', 'Transformer-MC': 'Tform-MC',
    'Transformer-Ensemble': 'Tform-Ens', 'Transformer-Conformal': 'Tform-Conf',
    'AR1Pure': 'AR1', 'Adaptive AR1-RLS': 'AR1-RLS',
}


def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def write_latex_table_all_models(tex_path, all_driver_df, model_df, model_order):
    """Wide table: one row per candidate driver, one column per model,
    generalizing tab_heterogeneity_drivers (Bayesian Loop only) to all
    nine models. Each model is tested against whichever of
    sensitivity/specificity its own model_comparison.csv row shows
    collapsing more often (pick_target_metric) -- the column header notes
    which. Continuous drivers show signed Spearman r (vs.\\ the target
    metric) with significance stars; categorical drivers show the chi-sq
    p-value. MCMC rows are N/A for every model but Bayesian Loop (only
    model with an MCMC posterior)."""
    target_by_model = {mdl: pick_target_metric(mdl, model_df) for mdl in model_order}
    n_users, n_hypo = _cohort_counts(model_df)
    with open(tex_path, 'w') as f:
        f.write('\\begin{table}[!htbp]\n\\centering\n\\scriptsize\n')
        f.write('\\caption{Candidate drivers of per-user heterogeneity tested against all '
                f'nine models ($N={n_users}$ users, or $N={n_hypo}$ for the hypo-event-count driver; '
                '4-hour horizon), generalizing Table~\\ref{tab:heterogeneity_drivers} beyond '
                'Bayesian Loop. Each model column is tested against whichever of sensitivity '
                'or specificity that model\'s own collapse rate (Table~\\ref{tab:heterogeneity_model_comparison}) '
                'is worse on -- noted in the column header -- per the paper\'s framing that '
                'every model has heterogeneity in the metric complementary to its own aggregate '
                'operating point. Continuous drivers: signed Spearman $r$ against the target '
                'metric ($*p<0.05$, $**p<0.01$, $***p<0.001$). Categorical drivers: '
                '$\\chi^2$ $p$-value against low-target-metric ($<0.10$) status. '
                'N/A = not applicable (MCMC diagnostics only exist for Bayesian Loop\'s posterior). '
                '-- = insufficient data (e.g.\\ too few low-target-metric users for that model '
                'to run the test).}\n')
        f.write('\\label{tab:heterogeneity_drivers_all_models}\n')
        f.write('\\begin{adjustbox}{max width=\\linewidth}\n')
        f.write('\\begin{tabular}{l' + 'r' * len(model_order) + '}\n\\toprule\n')
        header = '  Candidate driver & ' + ' & '.join(
            f'{_MODEL_ABBREV.get(mdl, mdl)} ({target_by_model[mdl].capitalize()})'
            for mdl in model_order) + ' \\\\\n'
        f.write(header)
        f.write('\\midrule\n')
        for key, label, kind in _ALL_MODEL_DRIVER_ROWS:
            cells = []
            for mdl in model_order:
                sub = all_driver_df[(all_driver_df['model'] == mdl) & (all_driver_df['driver'] == key)]
                if len(sub) == 0:
                    cells.append('N/A')
                    continue
                row = sub.iloc[0]
                if kind == 'continuous':
                    if pd.isna(row.get('spearman_r')):
                        cells.append('--')
                    else:
                        cells.append(f"{row['spearman_r']:.2f}{_stars(row['spearman_p'])}")
                else:
                    if pd.isna(row.get('chi2_p')):
                        cells.append('--')
                    else:
                        p = row['chi2_p']
                        p_str = f'{p:.1e}' if p < 0.001 else f'{p:.3f}'
                        cells.append(f'{p_str}{_stars(p)}')
            f.write(f'  {label} & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n')
    print(f'  {tex_path}')


def _cohort_counts(model_df, model=None):
    """Users in the specificity and sensitivity analyses (the latter excludes
    users with no true-hypoglycemia test instance), for table captions.
    Uses one model's row if `model` is given, else the largest counts."""
    rows = model_df[model_df['model'] == model] if model else model_df
    return int(rows['n_users_spec'].max()), int(rows['n_users_sens'].max())


# Rows of the driver-direction summary: (driver key in driver_tests_all_models,
# row label, kind, words). For continuous drivers `words` = (word for low
# values, word for high values); for categorical drivers it maps category
# value -> word. Mean directional bias, RMSE and the MCMC diagnostics are
# omitted (they describe each model's own errors, not patient covariates),
# and the non-hypo instance count mirrors the hypo-event count.
_DIRECTION_ROWS = [
    ('sex', 'Sex', 'categorical', {'M': 'Male', 'F': 'Female'}),
    ('age', 'Age', 'continuous', ('Younger', 'Older')),
    ('hba1c', 'HbA1c', 'continuous', ('Lower', 'Higher')),
    ('n_exercise_events', 'Exercise event count', 'continuous', ('Fewer', 'More')),
    ('device_type (AID vs non-AID)', 'Device type (AID vs.\\ non-AID)', 'categorical',
     {'AID': 'AID', 'non-AID': 'non-AID'}),
    ('glucose_cv_pct (%CV, glycemic variability)', 'Glucose \\%CV (glycemic variability)',
     'continuous', ('Lower', 'Higher')),
    ('n_hypo_events (sens denominator, small-N artifact test)', 'Hypoglycemia prevalence',
     'continuous', ('Lower', 'Higher')),
]
_DIRECTION_ABBREV = dict(_MODEL_ABBREV, AR1Pure='Pure AR1')


def _collapse_direction(row, kind, words):
    """Covariate value associated with MORE collapse (low target metric) for
    one significant (model, driver) test, or None if not significant."""
    if kind == 'continuous':
        if pd.isna(row.get('spearman_p')) or row['spearman_p'] >= 0.05:
            return None
        # r > 0: lower covariate goes with a lower target metric (collapse).
        return words[0] if row['spearman_r'] > 0 else words[1]
    if pd.isna(row.get('chi2_p')) or row['chi2_p'] >= 0.05:
        return None
    tab = row['contingency_table']
    if isinstance(tab, str):
        tab = ast.literal_eval(tab)
    low, rest = tab.get(True, {}), tab.get(False, {})
    rate = {c: low.get(c, 0) / (low.get(c, 0) + rest.get(c, 0))
            for c in set(low) | set(rest) if low.get(c, 0) + rest.get(c, 0) > 0}
    return words.get(max(rate, key=rate.get))


def write_latex_table_directions(tex_path, all_driver_df, model_df, model_order):
    """Summary of the significant (p < 0.05) patient covariates in the
    all-model driver table: for each driver, which covariate value goes with
    more sensitivity collapse and which with more specificity collapse, and
    for which models. Each model contributes to the column of the metric it
    was tested against (pick_target_metric)."""
    target_by_model = {mdl: pick_target_metric(mdl, model_df) for mdl in model_order}
    lines = []
    for key, label, kind, words in _DIRECTION_ROWS:
        cols = {'sens': {}, 'spec': {}}
        n_sig = 0
        for mdl in model_order:
            sub = all_driver_df[(all_driver_df['model'] == mdl) & (all_driver_df['driver'] == key)]
            if len(sub) == 0:
                continue
            word = _collapse_direction(sub.iloc[0], kind, words)
            if word is None:
                continue
            n_sig += 1
            cols[target_by_model[mdl]].setdefault(word, []).append(_DIRECTION_ABBREV.get(mdl, mdl))
        if n_sig == 0:
            continue
        cells = []
        for target in ('sens', 'spec'):
            if not cols[target]:
                cells.append('---')
                continue
            parts = []
            for word, models in cols[target].items():
                who = f'{models[0]} only' if n_sig == 1 else ', '.join(models)
                parts.append(f'{word} ({who})')
            cells.append('; '.join(parts))
        lines.append(f'{label} & {cells[0]} & {cells[1]} \\\\')
    with open(tex_path, 'w') as f:
        f.write('\\begin{table}[!htbp]\n\\centering\n\\small\n')
        f.write('\\caption{Direction of each significant candidate driver from '
                'Table~\\ref{tab:heterogeneity_drivers} (Bayesian Loop) and '
                'Table~\\ref{tab:heterogeneity_drivers_all_models} (all nine models), '
                'summarizing which covariate value is associated with \\emph{more} '
                'sensitivity collapse versus \\emph{more} specificity collapse for each '
                'affected model. Only drivers reaching $p<0.05$ for at least one model are '
                'shown; Mean Directional Bias and RMSE are omitted here as their direction is '
                "a property of each model's own error rather than an external patient "
                'covariate (discussed in the main text and '
                'Section~\\ref{app:heterogeneity_drivers} directly). Model abbreviations '
                'follow Table~\\ref{tab:heterogeneity_drivers_all_models}.}\n')
        f.write('\\label{tab:heterogeneity_driver_directions}\n')
        f.write('\\begin{adjustbox}{max width=\\linewidth}\n')
        f.write('\\begin{tabular}{lll}\n\\toprule\n')
        f.write('Candidate driver & Associated with \\emph{more} sensitivity collapse & '
                'Associated with \\emph{more} specificity collapse \\\\\n\\midrule\n')
        f.write('\n'.join(lines) + '\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n')
    print(f'  {tex_path}')


def write_summary(txt_path, driver_df, model_df, n_matched):
    with open(txt_path, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('HETEROGENEITY DRIVER ANALYSIS\n')
        f.write(f'Bayesian Loop, 4-hr horizon, n={n_matched} users, '
                f'low-spec threshold={LOW_SPEC_THRESHOLD}\n')
        f.write('=' * 70 + '\n\n')
        f.write('--- Cross-model sens/spec collapse rate (is this Bayesian-Loop-specific?) ---\n')
        f.write(model_df.to_string(index=False) + '\n\n')
        f.write('--- Candidate driver tests ---\n')
        cols = ['driver', 'kind', 'n', 'spearman_r', 'spearman_p',
                'median_low_spec_group', 'median_rest', 'mwu_p', 'chi2_p']
        present = [c for c in cols if c in driver_df.columns]
        f.write(driver_df[present].to_string(index=False) + '\n')
    print(f'  {txt_path}')


def main():
    global BATCH_DIRS
    parser = argparse.ArgumentParser(description='Candidate drivers of per-user '
                                     'hypoglycemia sensitivity/specificity heterogeneity.')
    parser.parse_args()

    cfg = get_config()
    BATCH_DIRS = [d for d in cfg['BATCH_LIST'] if eval_config._is_bayesian_loop_dir(d)]
    out_dir = os.path.join(cfg['OUT_DIR'], 'heterogeneity_analysis')
    os.makedirs(out_dir, exist_ok=True)

    per_model = load_per_user_metrics(cfg['OUT_DIR'])
    model_df = run_model_comparison(per_model)
    print(model_df.to_string(index=False))

    # Union of every model's user index -- glucose variability is a per-user,
    # model-independent covariate, computed once and reused for all 9 tests.
    all_uids = sorted(set().union(*(m.index for m in per_model.values())))
    gv_cache = os.path.join(out_dir, 'glucose_variability.csv')
    gv = compute_glucose_variability(all_uids, cache_path=gv_cache)
    print(f'  glucose variability: {gv["glucose_cv_pct"].notna().sum()}/{len(all_uids)} users '
          f'(median %CV = {gv["glucose_cv_pct"].median():.1f})')

    # Bayesian-Loop-only detailed table (unchanged behavior/output files).
    driver_df, merged = run_driver_tests(per_model['Bayesian Loop'], target_col='spec',
                                          include_mcmc=True, gv=gv)
    print()
    print(driver_df[['driver', 'n', 'spearman_r', 'spearman_p', 'mwu_p', 'chi2_p']]
          .to_string(index=False))

    driver_df.to_csv(os.path.join(out_dir, 'driver_tests.csv'), index=False)
    model_df.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)
    merged.to_csv(os.path.join(out_dir, 'per_user_covariates.csv'))
    write_summary(os.path.join(out_dir, 'summary.txt'), driver_df, model_df, len(merged))
    write_latex_table(os.path.join(out_dir, 'tab_heterogeneity_drivers.tex'), driver_df, model_df)
    print(f'  {out_dir}/driver_tests.csv')
    print(f'  {out_dir}/model_comparison.csv')
    print(f'  {out_dir}/per_user_covariates.csv')

    # All-model generalization: same battery, run once per model against
    # that model's own weaker metric (pick_target_metric).
    all_rows = []
    for mdl in model_df['model']:
        target = pick_target_metric(mdl, model_df)
        d_df, _ = run_driver_tests(per_model[mdl], target_col=target,
                                    include_mcmc=(mdl == 'Bayesian Loop'), gv=gv)
        d_df.insert(0, 'model', mdl)
        d_df.insert(1, 'target_metric', target)
        all_rows.append(d_df)
    all_driver_df = pd.concat(all_rows, ignore_index=True)
    all_driver_df.to_csv(os.path.join(out_dir, 'driver_tests_all_models.csv'), index=False)
    write_latex_table_all_models(
        os.path.join(out_dir, 'tab_heterogeneity_drivers_all_models.tex'),
        all_driver_df, model_df, list(model_df['model']))
    write_latex_table_directions(
        os.path.join(out_dir, 'tab_heterogeneity_driver_directions.tex'),
        all_driver_df, model_df, list(model_df['model']))
    print(f'  {out_dir}/driver_tests_all_models.csv')
    print(f'  {out_dir}/tab_heterogeneity_drivers_all_models.tex')


if __name__ == '__main__':
    main()
