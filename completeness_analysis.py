#!/usr/bin/env python3
"""
completeness_analysis.py
=========================
Prevalence of missing/inaccurate meal (carbohydrate) reporting, and model
performance stratified by data completeness. No model inference is run.

Inputs:
  - DATA_DIR/non_exercise.csv and by_user/{cgm,meals,bolus,basal}/*.parquet
  - FILTER_DIR/per_user/user_{uid}.pkl     (curated train/test ids, in order)
  - <OUT_DIR of evaluate.py>/metrics_df.pkl (per-instance, per-horizon metrics;
    the instance id is recovered positionally from per_user test_ids, which
    matches evaluate.py's compute_user_metrics row order: 6 consecutive rows
    per instance, one per horizon)

Completeness proxy (graded, not a hard cutoff):
  - unexplained_rise = max(future_cgm) - future_cgm[t0]   (same window and
    definition as the curation step's unexplained-rise check, mg/dL)
  - total_carbs logged in [-30, +60] min around the forecast origin
  - severity bin from the 2D (rise, carbs) surface (see SEVERITY_BINS)
  - corroborating (non-gating) flags:
      * bolus_no_carb: bolus above a threshold in the window with no carb
        logged within 30 min (thresholds in BOLUS_TH_SWEEP)
      * meal_window: whether the forecast origin falls in a canonical
        breakfast/lunch/dinner window
      * basal_suppressed: whether AID basal in the 60 min preceding the rise
        is < 50% of that user's median basal rate. Not used to override the
        score: a successfully compensated missing meal shows a smaller rise
        and is therefore undercounted by any input/CGM-based proxy.

Outputs (OUT_DIR = <RESULTS_DIR>/eval_results/completeness_analysis):
  - prevalence_full_pool.csv       : severity-bin counts over the full
                                     CGM-eligible instance pool (pre-curation)
  - prevalence_retained_test.csv   : severity-bin counts over the retained
                                     test instances of the cohort
  - prevalence_retained_train.csv  : the same for the retained training instances
  - full_pool / retained_test / retained_train _scored_instances.csv :
                                     per-instance scores (read by
                                     supplementary_tables.py for the
                                     prevalence table)
  - stratified_metrics_4h.csv      : headline 4-h metrics by severity bin
  - summary.txt                    : plain-text headline numbers

Usage:
  python completeness_analysis.py            # uses cached instance scans if present
  python completeness_analysis.py --rescan   # force the per-instance scan (~40 min)
"""
import os
import pickle
import numpy as np
import pandas as pd

import config
import eval_config

DATA_DIR = CURATED_DIR = METRICS_PKL = None   # set in main()
OUT_DIR = os.path.join(config.EVAL_RESULTS_DIR, 'completeness_analysis')
# Users whose retained test instances are scored; set in main().
TARGET_USERS = None

MIN_CGM_POINTS  = 96
MEAL_WINDOW_MIN = -30
MEAL_WINDOW_MAX = 60
# Single BOLUS_TH retired in favor of a sweep (see BOLUS_TH_SWEEP below) --
# 4U alone implies only detecting ~40-80g-carb-equivalent meals under typical
# 1U:10-20g insulin-to-carb ratios, inconsistent with curate_data.py's own
# 10g MEAL bar, and undercounts small/moderate missed meals. A single lower
# threshold risks misclassifying correction-only boluses (no meal, correctly
# no carb entry) as missed-meal signals. Reporting a sweep instead of picking
# one number is the defensible choice given no per-user ICR data is wired
# into this script.
BOLUS_TH_SWEEP = [1.0, 2.0, 4.0]

# non_exercise.csv contains ~22.6k overlapping 5-min-shifted candidate windows
# per user on average (11M rows total, 487 users); scoring every overlapping
# window is redundant (adjacent windows differ by one 5-min step) and not
# computationally tractable. We subsample the
# full pre-curation pool to non-overlapping ~4h-spaced candidates (stride=48
# rows = 48*5min = 4h), matching curate_data.py's own MIN_INSTANCE_GAP=4h
# de-overlap logic, before computing prevalence over that pool. The retained
# curated test set (150/user) is scored exhaustively (no subsampling).
FULL_POOL_STRIDE = 48

# (rise_lo, rise_hi, carbs_lo, carbs_hi) -> bin label, ordered most-to-least severe
# replaces the disconnected 40 mg/dL exclusion + 5g/10g meal thresholds with a
# single consistent grid; instances with rise>40 & carbs<5 are the instances
# that would have been hard-excluded during curation (SEVERE).
#
# SEVERE/MODERATE use a strict ">" on the 40 mg/dL threshold, matching the
# curation step's unexplained-rise check (rise > UNEXPLAINED_RISE_TH) exactly,
# so that SEVERE mirrors curation's exclusion rule (and is ~0 in the retained
# test set). Ties at exactly 40.0 are common in this CGM data.
SEVERITY_ORDER = ["SEVERE", "MODERATE", "MILD", "CLEAN"]


def severity_bin(rise, carbs):
    if pd.isna(rise):
        return None
    if rise > 40.0 and carbs < 5.0:
        return "SEVERE"        # would be hard-excluded by curate_data.py
    if rise > 40.0 and carbs < 10.0:
        return "MODERATE"      # the 5-10g gray zone (neither excluded nor MEAL)
    if rise >= 20.0 and carbs < 10.0:
        return "MILD"          # smaller unexplained rise, still under-logged
    return "CLEAN"


def meal_window_flag(t0):
    t = t0.time()
    return (
        (pd.Timestamp("06:00").time() <= t < pd.Timestamp("11:00").time()) or
        (pd.Timestamp("11:00").time() <= t < pd.Timestamp("16:00").time()) or
        (pd.Timestamp("16:00").time() <= t < pd.Timestamp("21:00").time())
    )


def load_user_frames(uid, by_user_dir):
    frames = {}
    for name in ["cgm", "meals", "bolus", "basal"]:
        fp = os.path.join(by_user_dir, name, f"{uid}.parquet")
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            frames[name] = {eid: g.reset_index(drop=True) for eid, g in df.groupby("ex")}
        else:
            frames[name] = {}
    return frames


def score_instance(eid, frames, user_median_basal):
    cgm = frames.get("cgm", {}).get(eid, pd.DataFrame())
    if cgm.empty or len(cgm) < MIN_CGM_POINTS:
        return None
    tcol = next((c for c in ["timestamp", "datestamp", "time"] if c in cgm.columns), None)
    if not tcol:
        return None
    cgm = cgm.copy()
    cgm[tcol] = pd.to_datetime(cgm[tcol], format="mixed")
    cgm = cgm.sort_values(tcol)
    if len(cgm) <= 47:
        return None
    t0 = cgm.iloc[47][tcol]
    future = cgm[cgm[tcol] >= t0]["cgm"].values
    if len(future) < 48:
        return None
    rise = float(np.max(future) - future[0])

    meals = frames.get("meals", {}).get(eid, pd.DataFrame())
    total_carbs = 0.0
    if not meals.empty:
        mcol = next((c for c in ["timestamp", "datestamp", "time"] if c in meals.columns), None)
        if mcol:
            meals = meals.copy()
            meals[mcol] = pd.to_datetime(meals[mcol])
            meals["rel_min"] = (meals[mcol] - t0).dt.total_seconds() / 60.0
            rel = meals[(meals["rel_min"] >= MEAL_WINDOW_MIN) & (meals["rel_min"] <= MEAL_WINDOW_MAX)]
            total_carbs = float(rel["meal_dose"].sum()) if "meal_dose" in meals.columns else 0.0

    bolus = frames.get("bolus", {}).get(eid, pd.DataFrame())
    bolus_no_carb_sweep = {th: False for th in BOLUS_TH_SWEEP}
    if not bolus.empty:
        bcol = next((c for c in ["timestamp", "datestamp", "time"] if c in bolus.columns), None)
        if bcol and "bolus" in bolus.columns:
            bolus = bolus.copy()
            bolus[bcol] = pd.to_datetime(bolus[bcol])
            for th in BOLUS_TH_SWEEP:
                big = bolus[bolus["bolus"] > th]
                for _, brow in big.iterrows():
                    bt = brow[bcol]
                    hits = meals[(meals[mcol] >= bt) & (meals[mcol] <= bt + pd.Timedelta(minutes=30))] \
                        if (not meals.empty and mcol) else pd.DataFrame()
                    has_carb_hit = (not hits.empty) and (hits.get("meal_dose", pd.Series(dtype=float)).gt(0).any())
                    if not has_carb_hit:
                        bolus_no_carb_sweep[th] = True
                        break
    # backward-compat single flag, pinned to the original 4U threshold
    bolus_no_carb = bolus_no_carb_sweep[4.0]

    basal = frames.get("basal", {}).get(eid, pd.DataFrame())
    basal_suppressed = None
    if not basal.empty and user_median_basal and user_median_basal > 0:
        bacol = next((c for c in ["timestamp", "datestamp", "time"] if c in basal.columns), None)
        if bacol and "basal_units" in basal.columns:
            basal = basal.copy()
            basal[bacol] = pd.to_datetime(basal[bacol])
            pre = basal[(basal[bacol] >= t0 - pd.Timedelta(minutes=60)) & (basal[bacol] < t0)]
            if not pre.empty:
                basal_suppressed = bool(pre["basal_units"].median() < 0.5 * user_median_basal)

    row = {
        "eid": eid,
        "rise": rise,
        "total_carbs": total_carbs,
        "severity": severity_bin(rise, total_carbs),
        "bolus_no_carb": bolus_no_carb,
        "meal_window": meal_window_flag(t0),
        "basal_suppressed": basal_suppressed,
    }
    for th, val in bolus_no_carb_sweep.items():
        row[f"bolus_no_carb_{th:g}u"] = val
    return row


FULL_POOL_CSV = os.path.join(OUT_DIR, "full_pool_scored_instances.csv")
RETAINED_CSV = os.path.join(OUT_DIR, "retained_test_scored_instances.csv")


def scan_instances():
    """Expensive phase: per-instance completeness scoring (rise/carbs/bolus
    sweep/basal) from raw CGM/meal/bolus/basal parquet, for the full
    pre-curation pool and the retained curated test and train sets. Depends ONLY on raw
    sensor data + curate_data.py's fixed test_ids -- NOT on any model's
    metrics_df.pkl, which is why this is cacheable to CSV and does not need
    to rerun every time a new model/baseline is added (~40 min for 487
    users; the model-stratification step that consumes this output takes
    seconds). Writes FULL_POOL_CSV / RETAINED_CSV / RETAINED_TRAIN_CSV and
    returns the three DataFrames.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    by_user_dir = os.path.join(DATA_DIR, "by_user")
    per_user_dir = os.path.join(CURATED_DIR, "per_user")

    print("Loading full instance pool metadata...")
    non_ex = pd.read_csv(os.path.join(DATA_DIR, "non_exercise.csv"))
    non_ex["ex"] = non_ex["ex"].astype(int)
    users = sorted(non_ex["id"].unique())
    print(f"  {len(users)} users, {len(non_ex)} total instances (pre-curation pool)")

    full_pool_rows = []
    retained_test_rows = []
    retained_train_rows = []

    for ui, uid in enumerate(users):
        uid_s = str(uid)
        user_rows = non_ex.loc[non_ex["id"] == uid].sort_values("time_series_start")
        eids_time_ordered = user_rows["ex"].tolist()
        # de-overlap the pre-curation pool to ~4h-spaced candidates (see
        # FULL_POOL_STRIDE note above) before scoring, to keep this tractable
        sampled_eids = set(eids_time_ordered[::FULL_POOL_STRIDE])

        pu_path = os.path.join(per_user_dir, f"user_{uid}.pkl")
        test_ids, train_ids = [], []
        if os.path.exists(pu_path):
            with open(pu_path, "rb") as f:
                pu = pickle.load(f)
            test_ids = pu.get("test_ids", [])
            train_ids = pu.get("train_ids", [])

        # always score the actual retained test and train instances
        # exhaustively, plus the strided sample for the pre-curation
        # prevalence estimate
        eids_to_score = sampled_eids | set(test_ids) | set(train_ids)

        frames = load_user_frames(uid, by_user_dir)

        basal_all = frames.get("basal", {})
        if basal_all:
            all_basal_vals = pd.concat(
                [g["basal_units"] for g in basal_all.values() if "basal_units" in g.columns],
                ignore_index=True) if any("basal_units" in g.columns for g in basal_all.values()) else pd.Series(dtype=float)
            user_median_basal = float(all_basal_vals.median()) if len(all_basal_vals) else None
        else:
            user_median_basal = None

        scored = {}
        for eid in eids_to_score:
            s = score_instance(eid, frames, user_median_basal)
            if s is None:
                continue
            scored[eid] = s
            if eid in sampled_eids:
                full_pool_rows.append({"uid": uid_s, **s})

        in_cohort = TARGET_USERS is None or int(uid) in TARGET_USERS
        if in_cohort:
            for eid in test_ids:
                s = scored.get(eid)
                if s is not None:
                    retained_test_rows.append({"uid": uid_s, **s})
            for eid in train_ids:
                s = scored.get(eid)
                if s is not None:
                    retained_train_rows.append({"uid": uid_s, **s})

        if ui % 20 == 0 or ui == len(users) - 1:
            print(f"  [{ui+1}/{len(users)}] user {uid}: pool_sampled={len(sampled_eids)} "
                  f"scored={len(scored)} test_scored={sum(1 for e in test_ids if e in scored)}",
                  flush=True)

    full_pool_df = pd.DataFrame(full_pool_rows)
    retained_df = pd.DataFrame(retained_test_rows)
    retained_train_df = pd.DataFrame(retained_train_rows)
    full_pool_df.to_csv(FULL_POOL_CSV, index=False)
    retained_df.to_csv(RETAINED_CSV, index=False)
    retained_train_df.to_csv(RETAINED_TRAIN_CSV, index=False)
    return full_pool_df, retained_df, retained_train_df


def recover_eid_column(metrics_df):
    """Cheap phase: recover each metrics_df row's instance id by replaying
    curate_data.py's own test_ids order per user (small per-user pkl reads
    only, no raw parquet parsing -- unlike scan_instances() above). Verified
    Exact match for all 395 users (n_inst == len(test_ids) for
    every user, 0% fallback rate in the merge that consumes this column).
    """
    per_user_dir = os.path.join(CURATED_DIR, "per_user")
    metrics_eid_col = np.full(len(metrics_df), None, dtype=object)
    for uid_s in metrics_df["uid"].unique():
        pu_path = os.path.join(per_user_dir, f"user_{uid_s}.pkl")
        if not os.path.exists(pu_path):
            continue
        with open(pu_path, "rb") as f:
            test_ids = pickle.load(f).get("test_ids", [])
        block_idx = metrics_df.index[metrics_df["uid"] == uid_s].to_numpy()
        n_inst = len(block_idx) // 6
        for i in range(min(n_inst, len(test_ids))):
            metrics_eid_col[block_idx[i * 6:(i + 1) * 6]] = test_ids[i]
    return metrics_eid_col


def stratify(metrics_pkl_path, full_pool_df, retained_df, retained_train_df=None):
    """Fast phase: load a given metrics_df.pkl (whichever models are
    currently available), recover eids, merge with the cached completeness
    scores, and write prevalence + stratified-metrics outputs. Rerun this
    (not scan_instances()) whenever a new model's results land -- takes
    seconds, since it does no raw-data rescanning.
    """
    print("Loading metrics_df.pkl...")
    with open(metrics_pkl_path, "rb") as f:
        metrics_df = pickle.load(f)
    metrics_df = metrics_df.copy()
    metrics_df["uid"] = metrics_df["uid"].astype(str)

    # Prefer the eid that evaluate.py now PERSISTS in metrics_df.pkl over
    # re-deriving it positionally. Two independent derivations of the same
    # join key existed here, and this line used to unconditionally discard
    # the authoritative one -- if they ever disagreed (partial/interrupted
    # run, any change to row ordering) the severity scores would silently
    # join to the wrong instances. Native column wins; positional recovery
    # is now only a fallback for older pkls written before eid was
    # persisted. When both are available we cross-check and report, which
    # also retroactively validates the positional path used by the earlier
    _native = ("eid" in metrics_df.columns) and metrics_df["eid"].notna().all()
    if _native:
        _recovered = recover_eid_column(metrics_df)
        _native_vals = metrics_df["eid"].astype(str).values
        _rec_vals = pd.Series(_recovered).astype(str).values
        _cmp = pd.notna(_recovered)
        _agree = int((_native_vals[_cmp] == _rec_vals[_cmp]).sum())
        _n_cmp = int(_cmp.sum())
        if _n_cmp and _agree == _n_cmp:
            print(f"  eid: using persisted column; positional recovery agrees "
                  f"on all {_n_cmp:,} comparable rows")
        else:
            print(f"  WARNING: eid: persisted vs positional recovery DISAGREE on "
                  f"{_n_cmp - _agree:,}/{_n_cmp:,} rows — using the persisted "
                  f"column (authoritative). Investigate before trusting the "
                  f"stratified numbers.")
    else:
        print("  eid: not persisted in this pkl — falling back to positional recovery")
        metrics_df["eid"] = recover_eid_column(metrics_df)

    def prevalence_table(df):
        counts = df["severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0).astype(int)
        pct = (counts / counts.sum() * 100).round(2)
        out = pd.DataFrame({"n": counts, "pct": pct})
        for th in BOLUS_TH_SWEEP:
            col = f"bolus_no_carb_{th:g}u"
            if col in df.columns:
                out.loc[f"{col}_pct"] = ["", round(100 * df[col].mean(), 2)]
        n_basal_known = df["basal_suppressed"].notna().sum()
        if n_basal_known:
            out.loc["basal_suppressed_pct_of_known"] = ["", round(100 * df["basal_suppressed"].mean(skipna=True), 2)]
        return out

    prev_full = prevalence_table(full_pool_df)
    prev_test = prevalence_table(retained_df)
    prev_full.to_csv(os.path.join(OUT_DIR, "prevalence_full_pool.csv"))
    prev_test.to_csv(os.path.join(OUT_DIR, "prevalence_retained_test.csv"))
    if retained_train_df is not None and len(retained_train_df):
        prevalence_table(retained_train_df).to_csv(
            os.path.join(OUT_DIR, "prevalence_retained_train.csv"))

    # Normalize the join key dtype on BOTH sides. metrics_df's persisted eid
    # is object/str; retained_df's comes back from CSV as int64, and pandas
    # refuses an object<->int64 merge. Casting both to str also makes the
    # join independent of however the CSV round-tripped.
    _ret = retained_df[["uid", "eid", "severity"]].copy()
    _ret["eid"] = _ret["eid"].astype(str)
    _ret["uid"] = _ret["uid"].astype(str)
    metrics_df["eid"] = metrics_df["eid"].astype(str)
    _before = len(metrics_df)
    merged = metrics_df.merge(_ret, on=["uid", "eid"], how="left")
    assert len(merged) == _before, (
        f"merge changed row count {_before} -> {len(merged)}; duplicate "
        f"(uid, eid) keys on the severity side would silently fan out rows")
    _matched = merged["severity"].notna().sum()
    print(f"  severity join: {_matched:,}/{len(merged):,} rows matched "
          f"({100*_matched/len(merged):.1f}%)")
    merged["severity"] = merged["severity"].fillna("CLEAN")
    d4h = merged[merged["horizon"] == "240 min"]

    metric_cols = [
        "RMSE", "P_Mean_Dir_Bias", "P_Hypo_Sens", "P_Hypo_Spec",
        "P_Hyper_Sens", "P_Hyper_Spec",
    ]
    strat_rows = []
    # Every model evaluate.py can emit, in the order it reports them. Models
    # whose columns aren't in this pkl are SKIPPED (not emitted as NaN rows),
    # so this list is safe to keep ahead of what's actually been trained --
    # it self-adapts as new baselines land instead of silently omitting them,
    # which is what happened when this loop was hardcoded to 5 models while
    # the pkl had already grown to 9.
    ALL_MODEL_PREFIXES = [
        ("G", "Bayesian Loop"), ("L", "LSTM-MC"), ("A", "Pure AR1"),
        ("R", "Adaptive AR1-RLS"), ("T", "Transformer-MC"),
        ("E", "LSTM-Ensemble"), ("C", "LSTM-Conformal"),
        ("F", "Transformer-Conformal"), ("N", "Transformer-Ensemble"),
    ]
    _present = [(p_, l_) for p_, l_ in ALL_MODEL_PREFIXES
                if f"{p_}_RMSE" in d4h.columns and d4h[f"{p_}_RMSE"].notna().any()]
    _absent = [l_ for p_, l_ in ALL_MODEL_PREFIXES if (p_, l_) not in _present]
    print(f"  Stratified metrics: {len(_present)} models present "
          f"({', '.join(l_ for _, l_ in _present)})")
    if _absent:
        print(f"    skipped (no data in this pkl): {', '.join(_absent)}")
    for prefix, label in _present:
        for sev in SEVERITY_ORDER:
            sub = d4h[d4h["severity"] == sev]
            row = {"model": label, "severity": sev, "n": len(sub)}
            for mc in metric_cols:
                col = f"{prefix}_{mc}"
                row[mc] = round(sub[col].mean(), 3) if col in sub.columns and len(sub) else np.nan
            strat_rows.append(row)
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(os.path.join(OUT_DIR, "stratified_metrics_4h.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("=== Completeness / Missing-Meal Prevalence Analysis ===\n\n")
        f.write(f"Full CGM-eligible pool: {len(full_pool_df)} scored instances "
                f"across {full_pool_df['uid'].nunique()} users\n")
        f.write(prev_full.to_string() + "\n\n")
        f.write(f"Retained curated test set: {len(retained_df)} scored instances "
                f"across {retained_df['uid'].nunique()} users\n")
        f.write(prev_test.to_string() + "\n\n")
        f.write("4h-horizon metrics stratified by residual completeness severity "
                "(retained test set only; all 5 evaluated models -- Bayesian Loop, "
                "LSTM-MC, Pure AR1, Adaptive AR1-RLS, Transformer-MC; SEVERE would "
                "already be excluded by curation and should be ~0 here -- reported "
                "as a sanity check):\n")
        f.write(strat_df.to_string(index=False) + "\n")
        f.write(f"\n(model source: {metrics_pkl_path})\n")

    print("Done. Outputs in", OUT_DIR)


def main():
    import argparse
    global DATA_DIR, CURATED_DIR, METRICS_PKL, FULL_POOL_CSV, RETAINED_CSV, RETAINED_TRAIN_CSV, TARGET_USERS
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument(
        "--cohort", choices=["eligible", "analysed"], default="eligible",
        help="Users whose retained test instances are scored for prevalence. "
             "'eligible' (default, as in the paper) uses all eligible pump users, "
             "including the two users excluded from model fitting for duplicate "
             "CGM timestamps; 'analysed' uses only the analysed cohort.")
    parser.add_argument(
        "--metrics-pkl", default=None,
        help="metrics_df.pkl to stratify against (default: evaluate.py's main output).")
    parser.add_argument(
        "--rescan", action="store_true",
        help="force re-running the ~40 min per-instance scan even if cached "
             "full_pool/retained CSVs already exist")
    args = parser.parse_args()

    DATA_DIR    = config.data_dir()
    CURATED_DIR = config.filter_dir()
    METRICS_PKL = args.metrics_pkl or os.path.join(eval_config.out_dir_for(), "metrics_df.pkl")
    os.makedirs(OUT_DIR, exist_ok=True)
    FULL_POOL_CSV = os.path.join(OUT_DIR, "full_pool_scored_instances.csv")
    RETAINED_CSV  = os.path.join(OUT_DIR, "retained_test_scored_instances.csv")
    RETAINED_TRAIN_CSV = os.path.join(OUT_DIR, "retained_train_scored_instances.csv")

    users = eval_config.load_eligible_users(
        DATA_DIR, CURATED_DIR, exclude_duplicate_cgm=(args.cohort == "analysed"))
    TARGET_USERS = set(int(u) for u in users)
    print(f"  cohort filter ({args.cohort}): {len(TARGET_USERS)} users")

    cached = all(os.path.exists(p) for p in (FULL_POOL_CSV, RETAINED_CSV, RETAINED_TRAIN_CSV))
    if args.rescan or not cached:
        full_pool_df, retained_df, retained_train_df = scan_instances()
    else:
        print(f"Using cached instance scores in {OUT_DIR} "
              f"(pass --rescan to force a fresh scan)")
        full_pool_df = pd.read_csv(FULL_POOL_CSV, dtype={"uid": str})
        retained_df = pd.read_csv(RETAINED_CSV, dtype={"uid": str})
        retained_train_df = pd.read_csv(RETAINED_TRAIN_CSV, dtype={"uid": str})

    stratify(METRICS_PKL, full_pool_df, retained_df, retained_train_df)


if __name__ == "__main__":
    main()
