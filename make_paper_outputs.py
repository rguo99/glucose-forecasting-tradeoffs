#!/usr/bin/env python3
"""
make_paper_outputs.py
=====================
Run every analysis step for the paper, in order, and collect the figures and
tables under <RESULTS_DIR>/paper/{figures,tables}/ using the file names the
manuscript uses.

Prerequisites: all model runs (run_cohort.py, see README) and the synthetic
parameter-recovery run (run_experiments.py --synthetic) have finished.

Usage:
    python make_paper_outputs.py              # run all steps, then collect
    python make_paper_outputs.py --collect-only
"""

import argparse
import os
import shutil
import subprocess
import sys

import config
import eval_config

PROJECT_ROOT = config.REPO_ROOT

# Step order matters: later steps read outputs of earlier ones.
STEPS = [
    ["evaluate.py", "--prior", "0.30"],
    ["evaluate.py", "--prior", "0.10"],
    ["evaluate.py", "--prior", "0.50"],
    ["parameter_analysis.py", "--prior", "0.30"],
    ["convergence.py"],
    ["compute_parameter_recovery.py", "--n", "50"],
    ["sensitivity.py"],
    ["tier_table.py", "--prior", "0.30"],
    ["scorecard_figure.py", "--prior", "0.30"],
    ["qualitative_figure.py", "--prior", "0.30"],
    ["heterogeneity_drivers.py"],
    ["completeness_analysis.py"],
    ["supplementary_tables.py"],
    ["extra_tables.py"],
    ["paper_tables.py"],
]

TABLE_LABEL_MAP = {
    "tab:dataset":            "tab_dataset.tex",
    "tab:convergence":        "tab_convergence.tex",
    "tab:convergence_params": "tab_convergence_params.tex",
    "tab:param_recovery":     "tab_param_recovery.tex",
    "tab:reporting_bias":     "tab_param_recovery_bias.tex",
}


def split_tables_file(src_path, tab_dir, label_map):
    if not os.path.exists(src_path):
        return []
    import re as _re
    with open(src_path, "r") as fh:
        src = fh.read()
    PARAM_LATEX = [
        ("theta_ISF",     r"$\theta_{\text{ISF}}$"),
        ("theta_CR",      r"$\theta_{\text{CR}}$"),
        ("isf_cr_ratio",  r"ISF/CR ratio"),
        ("phi",           r"$\phi$"),
        ("sigma_eta",     r"$\sigma_{\eta}$"),
        ("sigma_obs",     r"$\sigma_{\text{obs}}$"),
        ("sigma_carb_obs",r"$\sigma_{\text{carb}}$"),
        ("sigma_carb",    r"$\sigma_{\text{carb}}$"),
    ]
    written = []
    pat = _re.compile(r'(\\begin\{table\*?\}.*?\\end\{table\*?\})', _re.DOTALL)
    for m in pat.finditer(src):
        block = m.group(1).strip()
        lm = _re.search(r'\\label\{([^}]+)\}', block)
        if not lm:
            continue
        label = lm.group(1)
        if label not in label_map:
            continue
        for plain, latex in PARAM_LATEX:
            block = _re.sub(
                r'(?<=  )' + _re.escape(plain) + r'(?= &)',
                lambda m, s=latex: s, block
            )
        out_name = label_map[label]
        out_path = os.path.join(tab_dir, out_name)
        os.makedirs(tab_dir, exist_ok=True)
        block = block.replace("\\\\%%", "\\\\%")
        with open(out_path, "w") as fh:
            fh.write(block)
            fh.write("\n")
        written.append((label, out_name))
    return written


def manifest():
    """(source, destination name) for every generated paper figure and table."""
    OUT = eval_config.out_dir_for()
    MR = config.MAIN_RESULTS_DIR
    SUP = os.path.join(config.EVAL_RESULTS_DIR, "supplementary_tables")
    figures = [
        (f"{OUT}/scorecard/scorecard.pdf",                         "fig_scorecard.pdf"),
        (f"{OUT}/paper_outputs/figures/fig_tier_heatmap.pdf",      "fig_tier_heatmap.pdf"),
        (f"{OUT}/fig_roc_hypo.pdf",                                "fig_roc_hypo.pdf"),
        (f"{OUT}/fig_horizons.pdf",                                "fig_horizons.pdf"),
        (f"{OUT}/instance_type_comparison.pdf",                    "fig_stratified.pdf"),
        (f"{OUT}/fig_user_level_metric_distributions.pdf",         "fig_user_level_metric_distributions.pdf"),
        (f"{OUT}/main_vis_figure/qualitative.pdf",                 "fig_qualitative.pdf"),
        (f"{MR}/convergence/convergence_histograms_bayesian_loop.pdf", "convergence_histograms_bayesian_loop.pdf"),
        (f"{MR}/convergence/convergence_histograms_pure_ar1.pdf",  "convergence_histograms_pure_ar1.pdf"),
        (f"{MR}/parameter_analysis/kde_parameter_distributions.pdf", "kde_parameter_distributions.pdf"),
        (f"{MR}/sensitivity/sensitivity_predictive.pdf",           "sensitivity_predictive.pdf"),
        (f"{MR}/sensitivity/sensitivity_parameters.pdf",           "sensitivity_parameters.pdf"),
    ]
    tables = [
        (f"{OUT}/paper_tables/tab_main_results.tex",               "tab_main_results.tex"),
        (f"{OUT}/tables/tab_main_results_30min.tex",               "tab_main_results_30min.tex"),
        (f"{OUT}/tables/tab_instance_percentiles_body.tex",        "tab_instance_percentiles.tex"),
        (f"{OUT}/paper_tables/tab_completeness_stratified.tex",    "tab_completeness_stratified.tex"),
        (f"{MR}/parameter_analysis/parameter_table.tex",           "parameter_table.tex"),
        (f"{OUT}/heterogeneity_analysis/tab_heterogeneity_drivers.tex", "tab_heterogeneity_drivers.tex"),
        (f"{OUT}/heterogeneity_analysis/tab_heterogeneity_drivers_all_models.tex",
         "tab_heterogeneity_drivers_all_models.tex"),
        (f"{OUT}/heterogeneity_analysis/tab_heterogeneity_driver_directions.tex",
         "tab_heterogeneity_driver_directions.tex"),
        (f"{SUP}/tab_completeness_prevalence.tex",                "tab_completeness_prevalence.tex"),
        (f"{SUP}/tab_glucose_composition.tex",                    "tab_glucose_composition.tex"),
        (f"{SUP}/tab_scalability.tex",                            "tab_scalability.tex"),
    ]
    split_sources = {
        os.path.join(OUT, "tables.tex"): ["tab:dataset"],
        os.path.join(MR, "convergence", "convergence_tables.tex"): ["tab:convergence", "tab:convergence_params"],
        os.path.join(MR, "synthetic_n50", "parameter_recovery_tables.tex"): ["tab:param_recovery", "tab:reporting_bias"],
    }
    return figures, tables, split_sources


def run_all(strict):
    failed = []
    for i, step in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] python {' '.join(step)}\n{'-'*60}", flush=True)
        r = subprocess.run([sys.executable] + step, cwd=PROJECT_ROOT)
        if r.returncode != 0:
            failed.append(step[0])
            if strict:
                sys.exit(f"{step[0]} failed (--strict).")
    return failed


def collect():
    dest = os.path.join(config.RESULTS_DIR, "paper")
    fig_dir, tab_dir = os.path.join(dest, "figures"), os.path.join(dest, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    figures, tables, split_sources = manifest()
    missing = []
    for src, labels in split_sources.items():
        written = split_tables_file(src, tab_dir, {k: TABLE_LABEL_MAP[k] for k in labels})
        if len(written) < len(labels):
            missing.append(src)
    for src, name in figures:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(fig_dir, name))
        else:
            missing.append(src)
    for src, name in tables:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tab_dir, name))
        else:
            missing.append(src)
    print(f"\nCollected into {dest}")
    for m in missing:
        print(f"  missing: {m}")
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--strict", action="store_true", help="stop at the first failing step")
    args = ap.parse_args()
    failed = [] if args.collect_only else run_all(args.strict)
    missing = collect()
    if failed or missing:
        print(f"\nFailed steps: {failed}" if failed else "")
        sys.exit(1)


if __name__ == "__main__":
    main()
