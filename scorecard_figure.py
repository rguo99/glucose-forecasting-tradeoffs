#!/usr/bin/env python3
"""scorecard_figure.py
===================
Single-instance scorecard figure: Bayesian Loop and LSTM-MC forecasts for one
test instance (figure_selection.json, "scorecard") with the insulin/carb
event strip, and a per-metric win/lose scorecard for that instance.

Usage:
    python scorecard_figure.py --prior 0.30

Output: <OUT_DIR>/scorecard/scorecard.pdf (+ .png)
"""

import argparse
import re
import glob
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

BY_USER_ROOT = None
INSTANCE = None  # resolved in main() from figure_selection.json

COLOR_LSTM    = "#d62728"
COLOR_BLOOP   = "#1f77b4"
COLOR_HISTORY = "black"
COLOR_TRUTH   = "#444444"
COLOR_HYPO    = "#c0392b"
COLOR_HYPER   = "#e67e22"
COLOR_BASAL   = "#2ca02c"
COLOR_BOLUS   = "#1f77b4"
COLOR_CARBS   = "#d95f02"

WIN_COLOR  = "#27ae60"
LOSE_COLOR = "#e74c3c"
TIE_BG     = "#dde3ea"
NA_BG      = "#f0f0f0"

# =============================================================================
# METRIC COMPUTATIONS  (unchanged)
# =============================================================================

def clarke_zone(obs, pred):
    # Standard Clarke et al. (1987) zones, same rule as evaluate.clarke_zones
    # (precedence A, E, C, D, else B).
    if abs(pred - obs) <= 0.20 * obs or (obs <= 70 and pred <= 70): return 'A'
    if (obs >= 180 and pred <= 70) or (obs <= 70 and pred >= 180): return 'E'
    if (70 <= obs <= 290 and pred >= obs + 110) or (130 <= obs <= 180 and pred <= 1.4 * obs - 182): return 'C'
    if (obs >= 240 and 70 <= pred <= 180) or (obs <= 175 / 3 and 70 <= pred <= 180) \
            or (175 / 3 <= obs <= 70 and pred >= 1.2 * obs): return 'D'
    return 'B'

def compute_clarke_prob(obs_arr, samples):
    mp = samples.mean(axis=0)
    zones_mean = [clarke_zone(float(o), float(p)) for o, p in zip(obs_arr, mp)]
    counts = {z: 0 for z in 'ABCDE'}
    total = samples.shape[0] * samples.shape[1]
    for s in range(samples.shape[0]):
        for t in range(samples.shape[1]):
            counts[clarke_zone(float(obs_arr[t]), float(samples[s, t]))] += 1
    fracs = {z: counts[z] / total for z in 'ABCDE'}
    return zones_mean, fracs

def compute_nta(obs_arr, samples):
    obs_sign = np.sign(float(obs_arr[-1]) - float(obs_arr[0]))
    if obs_sign == 0:
        return None
    sample_signs = np.sign(samples[:, -1] - samples[:, 0])
    return float(np.mean(sample_signs == obs_sign))

def compute_crps(obs_arr, samples):
    crps_vals = []
    for t in range(samples.shape[1]):
        o = float(obs_arr[t])
        s = samples[:, t].astype(float)
        term1 = float(np.mean(np.abs(s - o)))
        term2 = float(np.mean(np.abs(s[:, None] - s[None, :]))) / 2.0
        crps_vals.append(term1 - term2)
    return float(np.mean(crps_vals))

def compute_interval_score(obs_arr, lo, hi, alpha=0.05):
    width      = hi - lo
    penalty_lo = (2 / alpha) * np.maximum(lo - obs_arr, 0)
    penalty_hi = (2 / alpha) * np.maximum(obs_arr - hi,  0)
    return float(np.mean(width + penalty_lo + penalty_hi))

def compute_trajectory_corr(obs_arr, samples):
    """Pearson r between the posterior-mean trajectory and the observed
    trajectory, matching the paper's Table~1 Trajectory Correlation metric
    (a magnitude/shape-aware companion to Net Trend Accuracy). Unlike
    Table 1's *pooled* value (correlation computed once over all instances'
    concatenated timesteps), this is a genuine per-instance quantity: one
    trajectory's predicted-mean vs.\ observed values, both length-48 here."""
    pred_mean = samples.mean(axis=0)
    if np.std(obs_arr) == 0 or np.std(pred_mean) == 0:
        return None
    return float(np.corrcoef(pred_mean, obs_arr)[0, 1])


def compute_all_metrics(obs, samp):
    mp = samp.mean(axis=0)
    lo = np.percentile(samp, 2.5,  axis=0)
    hi = np.percentile(samp, 97.5, axis=0)

    rmse     = float(np.sqrt(np.mean((obs - mp) ** 2)))
    crps     = compute_crps(obs, samp)
    coverage = float(np.mean((obs >= lo) & (obs <= hi)))
    pi_width = float(np.mean(hi - lo))
    is_score = compute_interval_score(obs, lo, hi)
    nta      = compute_nta(obs, samp)
    traj_corr = compute_trajectory_corr(obs, samp)
    mdb      = float(np.mean(samp[:, -1] - obs[-1]))

    # Zone fractions over the timesteps of the posterior-mean trajectory
    # (as in the main results table).
    zones_mean, _ = compute_clarke_prob(obs, samp)
    fracs = {z: zones_mean.count(z) / len(zones_mean) for z in 'ABCDE'}

    p_hypo  = float(np.mean(np.min(samp, axis=1) < 70))
    p_hyper = float(np.mean(np.max(samp, axis=1) > 180))

    is_true_hypo      = bool(np.any(obs < 70))
    is_true_non_hypo  = not is_true_hypo
    is_true_hyper     = bool(np.any(obs > 180))
    is_true_non_hyper = not is_true_hyper

    return {
        'rmse':       rmse,
        'crps':       crps,
        'coverage':   coverage,
        'pi_width':   pi_width,
        'is':         is_score,
        'nta':        nta,
        'traj_corr':  traj_corr,
        'mdb':        mdb,
        'ceg_a':      fracs['A'],
        'ceg_b':      fracs['B'],
        'ceg_c':      fracs['C'],
        'ceg_d':      fracs['D'],
        'ceg_e':      fracs['E'],
        'hypo_sens':  p_hypo        if is_true_hypo      else None,
        'hypo_spec':  (1 - p_hypo)  if is_true_non_hypo  else None,
        'hyper_sens': p_hyper       if is_true_hyper      else None,
        'hyper_spec': (1 - p_hyper) if is_true_non_hyper  else None,
        # Raw probabilities + event flags, exposed so draw_figure() can
        # compare against each model's own population-calibrated alarm
        # threshold (see get_alarm_thresholds()) to fill in hypo_alarm/
        # hyper_alarm below -- compute_all_metrics() itself has no access
        # to the population-level metrics_df, so it can't do this alone.
        'p_hypo':          p_hypo,
        'p_hyper':         p_hyper,
        'is_true_hypo':    is_true_hypo,
        'is_true_non_hyper': is_true_non_hyper,
        'hypo_alarm':  None,   # filled in by draw_figure()
        'hyper_alarm': None,   # filled in by draw_figure()
    }


def get_alarm_thresholds(metrics_pkl_path, horizon='240 min'):
    """Population-level thresh90 (the probability threshold achieving 90%
    sensitivity cohort-wide, i.e.\ the same operating point used for
    Table~1's False-Alarm-Rate/PPV rows and the horizon figure) for
    Bayesian Loop ('G') and LSTM-MC ('L'), Hypo and Hyper. Computed fresh
    from metrics_df.pkl rather than hardcoded, so it always matches
    whatever cohort/data the scorecard itself was pointed at."""
    import sys as _sys
    _sys.path.insert(0, PROJECT_ROOT)
    import evaluate as ev
    df = pd.read_pickle(metrics_pkl_path)
    d = df[df['horizon'] == horizon]
    out = {}
    for prefix in ('G', 'L'):
        for metric in ('Hypo', 'Hyper'):
            scores = d[f'{prefix}_P_{metric}_Prob'].values
            labels = d[f'{prefix}_P_{metric}_Sens'].notna().astype(float).values
            stats = ev._operating_point_stats(scores, labels, target_sens=0.90)
            out[(prefix, metric)] = stats['thresh90']
    return out

# =============================================================================
# BUNDLE LOADING  (unchanged)
# =============================================================================

def _clean(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s

def _eid_variants(e):
    s = str(e).strip()
    vs = {s}
    if s.endswith(".0"): vs.add(s[:-2])
    try: vs.add(str(int(float(s)))); vs.add(str(int(float(s))) + ".0")
    except: pass
    return vs

def _label(folder):
    n = folder.lower()
    if "lstm_mc" in n: return "LSTM_MC"
    if "ar1" in n:     return "Global" if "carb_uncert" in n else "AR1"
    return None

def _load_bundle(path):
    with open(path, "rb") as f: d = pickle.load(f)
    ap = d.get("all_preds", d.get("all_preds_cpu", {}))
    rs = np.asarray(ap.get("obs", ap.get("pred_mean")))
    if rs.ndim == 2: rs = rs[np.newaxis]
    ro = np.asarray(d.get("obs_glucose", d.get("observed_glucose_array")))
    ti = np.asarray(d.get("test_indices", d.get("test_idx", [])))
    ri = np.asarray(d.get("instance_ids", []))
    if rs.shape[1] == len(ro):
        s = rs[:, ti, :]; o = ro[ti]
        ids = ri[ti] if len(ri) > 0 else np.arange(len(ti))
    elif rs.shape[1] == len(ti):
        s = rs; o = ro[ti]
        ids = ri[ti] if len(ri) == len(ro) else ri
    else:
        m = min(len(ti), rs.shape[1])
        s = rs[:, :m, :]; o = ro[ti[:m]]
        ids = ri[ti[:m]] if len(ri) > 0 else np.arange(m)
    return {"obs": o, "samples": s, "test_idx": ti, "test_ids": ids}

def load_models(batch_dirs, uid):
    data = {}
    for bd in batch_dirs:
        if not os.path.exists(bd): continue
        for folder in sorted(glob.glob(os.path.join(bd, "20*"))):
            if not os.path.isdir(folder): continue
            lbl = _label(folder)
            if lbl not in ("LSTM_MC", "Global"): continue
            p = os.path.join(folder, "results", f"user_{uid}_results.pkl")
            if os.path.exists(p):
                try:
                    data[lbl] = _load_bundle(p)
                except Exception as e:
                    print(f"  Warning: {lbl} uid={uid}: {e}")
    return data

# =============================================================================
# DATA LOADING  (unchanged)
# =============================================================================

def _norm_df(df):
    th = [c for c in df.columns if any(
        x in c.lower() for x in ["timestamp","datestamp","time","start"])]
    if th:
        df = df.rename(columns={th[0]: "timestamp"})
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = (pd.to_datetime(df["timestamp"], unit="s")
                               if df["timestamp"].max() > 1e8
                               else pd.to_datetime(df["timestamp"], unit="m",
                                                   origin="2026-01-01"))
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    ex_cols = [c for c in df.columns if any(x in c.lower() for x in ["ex","instance"])]
    if ex_cols:
        df = df.rename(columns={ex_cols[0]: "ex"})
        df["_eid"] = df["ex"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df

def load_cgm(uid, eid, data_path=None):
    target = {re.sub(r"\.0$", "", str(eid).strip())}
    try: target.add(str(int(float(eid))))
    except: pass
    if BY_USER_ROOT:
        path = os.path.join(BY_USER_ROOT, "cgm", f"{uid}.parquet")
        if os.path.exists(path):
            df = _norm_df(pd.read_parquet(path))
            if "cgm" not in df.columns:
                vh = [c for c in df.columns if "cgm" in c.lower() or "glucose" in c.lower()]
                if vh: df = df.rename(columns={vh[0]: "cgm"})
            if "_eid" in df.columns:
                sub = df[df["_eid"].isin(target)]
                if not sub.empty:
                    return sub.sort_values("timestamp").reset_index(drop=True)
    if data_path:
        for f in glob.glob(os.path.join(data_path, "cgm*.csv")):
            df = _norm_df(pd.read_csv(f))
            if "cgm" not in df.columns:
                vh = [c for c in df.columns if "cgm" in c.lower() or "glucose" in c.lower()]
                if vh: df = df.rename(columns={vh[0]: "cgm"})
            if "_eid" in df.columns:
                sub = df[df["_eid"].isin(target)]
                if not sub.empty:
                    return sub.sort_values("timestamp").reset_index(drop=True)
    return None

def load_events(uid, eid, data_path=None):
    target = {re.sub(r"\.0$", "", str(eid).strip())}
    try: target.add(str(int(float(eid))))
    except: pass
    streams = {"basal": "basal_units", "bolus": "bolus", "meals": "carbs"}
    result = {}
    for stream, val_col in streams.items():
        df = None
        if BY_USER_ROOT:
            path = os.path.join(BY_USER_ROOT, stream, f"{uid}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
        if df is None and data_path:
            files = glob.glob(os.path.join(data_path, f"{stream}*.csv"))
            if files: df = pd.read_csv(files[0])
        if df is None:
            result[stream] = pd.DataFrame(); continue
        df = _norm_df(df)
        if "_eid" in df.columns:
            df = df[df["_eid"].isin(target)]
        if val_col not in df.columns:
            vh = [c for c in df.columns if any(x in c.lower()
                  for x in [stream, "rate", "dose", "amount", "carb"])]
            if vh: df = df.rename(columns={vh[0]: val_col})
        result[stream] = df.sort_values("timestamp").reset_index(drop=True) if not df.empty else pd.DataFrame()
    return result

def make_window(cgm_df, obs):
    if cgm_df is not None and len(cgm_df) >= 48:
        anchor_t = cgm_df["timestamp"].iloc[47]
        start_t  = cgm_df["timestamp"].iloc[0]
        end_t    = anchor_t + pd.Timedelta(hours=4)
        t_pred   = [anchor_t + pd.Timedelta(minutes=m)
                    for m in np.arange(5, 245, 5)[:48]]
        return cgm_df, anchor_t, start_t, end_t, t_pred
    anchor_t = pd.Timestamp("2024-01-15 12:00:00")
    hist_ts  = [anchor_t - pd.Timedelta(minutes=5*(47-i)) for i in range(48)]
    vals = list(obs[:48]) if len(obs) >= 48 else list(obs) + [float(obs[-1])]*(48-len(obs))
    sub  = pd.DataFrame({"timestamp": hist_ts, "cgm": vals})
    t_pred = [anchor_t + pd.Timedelta(minutes=m) for m in np.arange(5, 245, 5)[:48]]
    return sub, anchor_t, hist_ts[0], anchor_t + pd.Timedelta(hours=4), t_pred

# =============================================================================
# EVENT STRIP  — larger fonts
# =============================================================================

def draw_events(ax, events, start_t, end_t):
    ax.set_xlim(start_t, end_t)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4)

    # Basal: convert 5-min units to U/hr, track max
    basal_df = events.get("basal", pd.DataFrame())
    basal_max = 0.1
    if not basal_df.empty and "basal_units" in basal_df.columns:
        bdf = basal_df[(basal_df["timestamp"] >= start_t) &
                       (basal_df["timestamp"] <= end_t)].copy()
        if not bdf.empty:
            bdf["basal_uhr"] = bdf["basal_units"] * 12.0 if bdf["basal_units"].median() < 0.5 else bdf["basal_units"]
            basal_max = max(bdf["basal_uhr"].max(), 0.1)
            ax.step(bdf["timestamp"], bdf["basal_uhr"],
                    where="post", color=COLOR_BASAL, lw=1.4, alpha=0.8)

    # Bolus: treat each bolus U as a spike on the basal axis.
    #    We scale so that the largest bolus reaches ~80% of y-top,
    #    then set y-top to accommodate both basal and bolus comfortably.
    bolus_df = events.get("bolus", pd.DataFrame())
    bolus_max_u = 0.0
    if not bolus_df.empty and "bolus" in bolus_df.columns:
        bols = bolus_df[(bolus_df["timestamp"] >= start_t) &
                        (bolus_df["timestamp"] <= end_t)].copy()
        bols = bols[bols["bolus"] > 0]
        if not bols.empty:
            bolus_max_u = bols["bolus"].max()

    y_top = basal_max * 1.35
    if bolus_max_u > 0:
        bolus_scale = (y_top * 0.75) / bolus_max_u   # tallest spike = 75% of y_top
    else:
        bolus_scale = 1.0

    ax.set_ylim(0, y_top)
    ax.set_ylabel("Basal\n(U/hr)", fontsize=16, color=COLOR_BASAL)
    ax.tick_params(axis="y", labelsize=15, colors=COLOR_BASAL)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=False))

    # Draw bolus spikes (on basal axis)
    if not bolus_df.empty and "bolus" in bolus_df.columns:
        bols = bolus_df[(bolus_df["timestamp"] >= start_t) &
                        (bolus_df["timestamp"] <= end_t)].copy()
        for _, row in bols.iterrows():
            v = float(row["bolus"])
            if pd.isna(v) or v <= 0: continue
            h = v * bolus_scale
            ax.vlines(row["timestamp"], 0, h, color=COLOR_BOLUS, lw=2.2, alpha=0.85)
            ax.scatter(row["timestamp"], h, color=COLOR_BOLUS, s=50, marker='o', zorder=4)
            ax.text(row["timestamp"], h + y_top * 0.04, f"{v:.1f}U",
                    color=COLOR_BOLUS, fontsize=14, ha="center", fontweight="bold")

    # Draw meal markers (annotate only, no axis scaling)
    meals_df = events.get("meals", pd.DataFrame())
    if not meals_df.empty and "carbs" in meals_df.columns:
        mdf = meals_df[(meals_df["timestamp"] >= start_t) &
                       (meals_df["timestamp"] <= end_t)].copy()
        for _, row in mdf.iterrows():
            v = float(row["carbs"])
            if pd.isna(v) or v <= 0: continue
            h = y_top * 0.55   # fixed mid-height tick for meals
            ax.vlines(row["timestamp"], 0, h, color=COLOR_CARBS, lw=2.2, alpha=0.85)
            ax.scatter(row["timestamp"], h, color=COLOR_CARBS, s=50, marker='s', zorder=4)
            ax.text(row["timestamp"], h + y_top * 0.04, f"{v:.0f}g",
                    color=COLOR_CARBS, fontsize=14, ha="center", fontweight="bold")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="x", labelsize=17)

# =============================================================================
# SCORECARD PANEL  — larger fonts throughout
# =============================================================================

def draw_scorecard(ax, met_l, met_b):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    FMT_MG  = lambda v: f"{v:.1f}"
    FMT_INT = lambda v: f"{v:.0f}"
    FMT_PCT = lambda v: f"{v*100:.0f}%"
    FMT_MDB = lambda v: f"{v:+.0f}"
    FMT_R   = lambda v: f"{v:.2f}"
    FMT_YN  = lambda v: "Yes" if v else "No"

    ALL_ROWS = [
        ('S', 'RMSE (mg/dL)',                 'rmse',       False,  FMT_MG),
        ('S', 'Coverage (95% PI)',             'coverage',   True,   FMT_PCT),
        ('S', 'PI Width (mg/dL)',              'pi_width',   False,  FMT_INT),
        ('S', 'Interval Score',                'is',         False,  FMT_INT),
        ('S', 'CRPS (mg/dL)',                  'crps',       False,  FMT_MG),
        ('C', 'Net Trend Accuracy',            'nta',        True,   FMT_PCT),
        ('C', 'Trajectory Correlation (r)',    'traj_corr',  True,   FMT_R),
        ('C', 'Mean Directional Bias (mg/dL)', 'mdb',        'abs',  FMT_MDB),
        ('C', 'Hypoglycemia Sensitivity',      'hypo_sens',  True,   FMT_PCT),
        ('C', 'Hypoglycemia Specificity',      'hypo_spec',  True,   FMT_PCT),
        ('C', 'Hyperglycemia Sensitivity',     'hyper_sens', True,   FMT_PCT),
        ('C', 'Hyperglycemia Specificity',     'hyper_spec', True,   FMT_PCT),
        ('C', 'CEG Zone A',                    'ceg_a',      True,   FMT_PCT),
        ('C', 'CEG Zone B',                    'ceg_b',      True,   FMT_PCT),
        ('C', 'CEG Zone C',                    'ceg_c',      False,  FMT_PCT),
        ('C', 'CEG Zone D',                    'ceg_d',      False,  FMT_PCT),
        ('C', 'CEG Zone E',                    'ceg_e',      False,  FMT_PCT),
    ]

    rows = [(cat, name, key, hib, fmt)
            for cat, name, key, hib, fmt in ALL_ROWS
            if not (met_b.get(key) is None and met_l.get(key) is None)]

    cat_colors = {
        'S': ('#2c3e50', 'STATISTICAL METRICS'),
        'C': ('#1b4f1b', 'CLINICAL & DECISION-RELEVANT METRICS'),
    }

    L, R  = 0.015, 0.985
    TOP   = 0.980
    BOT   = 0.042
    W     = R - L

    cats_present = []
    for cat, *_ in rows:
        if cat not in cats_present:
            cats_present.append(cat)
    n_sec_hdrs = len(cats_present)

    total_slots = len(rows) + n_sec_hdrs + 1
    row_h = (TOP - BOT) / total_slots

    x_name  = L + 0.010
    x_b_ctr = L + W * 0.610   # pulled left a touch more
    x_l_ctr = L + W * 0.870   # pushed further right for more separation
    cell_w  = W * 0.175        # was 0.148 — slightly wider pills

    def rect(x, y, w, h, color, alpha=1.0, zorder=2):
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="square,pad=0",
                           facecolor=color, edgecolor='none',
                           alpha=alpha, transform=ax.transAxes,
                           zorder=zorder, clip_on=False)
        ax.add_patch(p)

    def pill(xc, y_top, rh, txt, bg, tc, fs=23.0):
        rect(xc - cell_w/2, y_top - rh*0.84, cell_w, rh*0.74, bg, zorder=4)
        ax.text(xc, y_top - rh*0.47, txt,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=tc, zorder=5)

    # Column header
    y = TOP
    # Use a very dark navy/charcoal for the primary header
    rect(L, y - row_h*0.95, W, row_h*0.95, '#212f3d', alpha=1.0) 
    
    ax.text(x_name, y - row_h*0.475, 'Metric',
            transform=ax.transAxes, ha='left', va='center',
            fontsize=24, fontweight='bold', color='white', zorder=5)
    
    # Brighter blue/red for maximum contrast against the dark navy
    ax.text(x_b_ctr, y - row_h*0.475, 'Bayesian Loop',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=22, fontweight='bold', color='#5dade2', zorder=5)
            
    ax.text(x_l_ctr, y - row_h*0.475, 'LSTM-MC',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=22, fontweight='bold', color='#ec7063', zorder=5)
    y -= row_h * 0.95

    # Data Rows & Section Headers
    prev_cat = None
    row_i = 0
    
    # Distinct colors for section headers that won't compete with the top bar
    cat_colors = {
        'S': ('#5d6d7e', 'STATISTICAL METRICS'),         # Muted Slate Gray
        'C': ('#1e5128', 'CLINICAL & DECISION METRICS'),  # Brighter Forest Green
    }

    for cat, name, key, hib, fmt in rows:
        if cat != prev_cat:
            hdr_color, label = cat_colors[cat]
            rect(L, y - row_h*0.78, W, row_h*0.78, hdr_color, alpha=0.85)
            ax.text((L+R)/2, y - row_h*0.39, label,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=20, fontweight='bold', color='white',
                    style='italic', zorder=5)
            y -= row_h * 0.78
            prev_cat = cat

        v_b = met_b.get(key)
        v_l = met_l.get(key)

        # Winner logic
        if hib is None:
            win_b = win_l = False
        elif hib == 'abs':
            tol = 0.5
            win_b = (v_b is not None and v_l is not None and abs(v_b) < abs(v_l) - tol)
            win_l = (v_b is not None and v_l is not None and abs(v_l) < abs(v_b) - tol)
        else:
            if v_b is None or v_l is None:
                win_b = win_l = False
            else:
                scale = max(abs(v_b), abs(v_l))
                tol = 0.01 if scale <= 1.0 else 1.0
                win_b = (v_b > v_l + tol) if hib else (v_b < v_l - tol)
                win_l = (v_l > v_b + tol) if hib else (v_l < v_b - tol)

        # Row stripe
        stripe = '#f5f8fa' if row_i % 2 == 0 else '#ffffff'
        rect(L, y - row_h*0.90, W, row_h*0.90, stripe)
        ax.plot([L, R], [y - row_h*0.90]*2,
                transform=ax.transAxes, color='#dde3ea', lw=0.5, zorder=3)

        # Metric name
        ax.text(x_name, y - row_h*0.45, name,
                transform=ax.transAxes, ha='left', va='center',
                fontsize=21, color='#1a252f', zorder=5)

        # Value pills
        for xc, v, is_win, is_lose in [
            (x_b_ctr, v_b, win_b, win_l),
            (x_l_ctr, v_l, win_l, win_b),
        ]:
            if v is None:
                pill(xc, y, row_h, '—', NA_BG, '#aaaaaa', fs=19.0)
            else:
                txt = fmt(v)
                if is_win:    bg, tc = WIN_COLOR,  'white'
                elif is_lose: bg, tc = LOSE_COLOR, 'white'
                else:         bg, tc = TIE_BG,     '#2c3e50'
                pill(xc, y, row_h, txt, bg, tc, fs=23.0)

        y -= row_h * 0.90
        row_i += 1

    # Win/lose legend
    leg_y = BOT + 0.004
    for xpos, col, lbl in [
        (L + 0.01,  WIN_COLOR,  '▮  Wins metric'),
        (L + 0.36,  LOSE_COLOR, '▮  Loses metric'),
        (L + 0.68,  '#aaaaaa',  '▮  Comparable'),
    ]:
        ax.text(xpos, leg_y, lbl,
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=18, color=col, fontweight='bold', zorder=5)

    # Outer border
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((L, BOT), W, TOP-BOT,
                                boxstyle="round,pad=0.004",
                                facecolor='none', edgecolor='#cccccc',
                                linewidth=1.2, transform=ax.transAxes,
                                zorder=6, clip_on=False))

    ax.text(-0.03, 1.02, "(b)", transform=ax.transAxes,
            fontsize=30, fontweight='bold', va='bottom', ha='left')

# =============================================================================
# MAIN FIGURE
# =============================================================================

def draw_figure(model_data, data_path, out_path, metrics_pkl_path=None):
    uid = INSTANCE["uid"]
    eid = INSTANCE["eid"]

    def _get(key):
        bundle = model_data.get(key)
        if bundle is None: return None
        tids  = [_clean(x) for x in bundle["test_ids"]]
        match = next((f for f in _eid_variants(eid) if f in tids), None)
        if match is None: return None
        idx  = tids.index(match)
        obs  = bundle["obs"][idx]
        samp = bundle["samples"][:, idx, :]
        mp   = samp.mean(0)
        lo, hi = np.percentile(samp, [2.5, 97.5], axis=0)
        return obs, mp, lo, hi, samp

    res_l = _get("LSTM_MC")
    res_b = _get("Global")
    if res_l is None or res_b is None:
        print("ERROR: could not find instance in one or both model bundles.")
        return

    obs,  mp_l, lo_l, hi_l, samp_l = res_l
    _obs, mp_b, lo_b, hi_b, samp_b = res_b

    met_l = compute_all_metrics(obs, samp_l)
    met_b = compute_all_metrics(obs, samp_b)

    if metrics_pkl_path is not None and os.path.exists(metrics_pkl_path):
        thresh = get_alarm_thresholds(metrics_pkl_path)
        # Hypo alarm: only meaningful as a "did it detect" question when a
        # true hypo occurred (same masking as hypo_sens above).
        if met_l['is_true_hypo']:
            met_l['hypo_alarm']  = met_l['p_hypo']  >= thresh[('L', 'Hypo')]
        if met_b['is_true_hypo']:
            met_b['hypo_alarm']  = met_b['p_hypo']  >= thresh[('G', 'Hypo')]
        # Hyper alarm: only meaningful as a "did it false-alarm" question
        # when no true hyper occurred (same masking as hyper_spec above).
        if met_l['is_true_non_hyper']:
            met_l['hyper_alarm'] = met_l['p_hyper'] >= thresh[('L', 'Hyper')]
        if met_b['is_true_non_hyper']:
            met_b['hyper_alarm'] = met_b['p_hyper'] >= thresh[('G', 'Hyper')]

    for label, m in [("LSTM", met_l), ("BLoop", met_b)]:
        print(f"  {label}: RMSE={m['rmse']:.1f}  NTA={m['nta']:.2f}  "
              f"MDB={m['mdb']:+.1f}  "
              f"HypoSens={m['hypo_sens']}  HypoSpec={m['hypo_spec']}")

    cgm_df = load_cgm(uid, eid, data_path)
    hist_df, anchor_t, start_t, end_t, t_pred = make_window(cgm_df, obs)
    events = load_events(uid, eid, data_path)

    y_lo = min(min(obs.min(), lo_b.min(), lo_l.min()) - 8, 52)
    y_hi = max(max(obs.max(), hi_b.max(), hi_l.max()) + 12, 130)
    y_lo = max(y_lo, 30)

    # Figure: 2 panels — TALLER for readability
    fig = plt.figure(figsize=(28, 15.0))   # was (24, 13)
    fig.patch.set_facecolor("white")

    gs = GridSpec(1, 2, figure=fig,
                  width_ratios=[3.4, 1.9],
                  wspace=0.08,
                  left=0.05, right=0.98,
                  top=0.88, bottom=0.14)

    gs_ts = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                    height_ratios=[5, 1], hspace=0.07)
    ax_ts = fig.add_subplot(gs_ts[0, 0])
    ax_ev = fig.add_subplot(gs_ts[1, 0])
    ax_sc = fig.add_subplot(gs[0, 1])

    # (a) Time-series
    ax = ax_ts
    ax.axhspan(70, 180,  color="#e8f5e9", alpha=0.25, zorder=0)
    ax.axhspan(y_lo, 70, color="#ffebee", alpha=0.30, zorder=0)
    ax.axhline(70,  color=COLOR_HYPO,  lw=1.8, ls='--', alpha=0.85)
    ax.axhline(180, color=COLOR_HYPER, lw=1.2, ls='--', alpha=0.45)
    ax.axvline(anchor_t, color='gray', lw=1.0, ls=':', alpha=0.55)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(start_t, end_t)

    hist = hist_df[hist_df["timestamp"] <= anchor_t]
    ax.plot(hist["timestamp"], hist["cgm"],
            color=COLOR_HISTORY, lw=2.5, zorder=5)
    ax.plot(t_pred, obs,
            color=COLOR_TRUTH, lw=2.2, ls='--', alpha=0.85, zorder=5)

    ax.fill_between(t_pred, lo_b, hi_b, color=COLOR_BLOOP, alpha=0.14, zorder=2)
    ax.plot(t_pred, mp_b, color=COLOR_BLOOP, lw=2.4, zorder=3)
    ax.fill_between(t_pred, lo_l, hi_l, color=COLOR_LSTM,  alpha=0.14, zorder=2)
    ax.plot(t_pred, mp_l, color=COLOR_LSTM,  lw=2.4, zorder=3)

    ax.set_ylabel("Glucose (mg/dL)", fontsize=22, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.tick_params(labelsize=19, labelbottom=False)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(-0.08, 1.05, "(a)", transform=ax.transAxes,
            fontsize=30, fontweight='bold', va='bottom', ha='left')

    draw_events(ax_ev, events, start_t, end_t)

    # (b) Scorecard
    draw_scorecard(ax_sc, met_l, met_b)

    # Legend
    handles = [
        Line2D([0],[0], color=COLOR_HISTORY, lw=2.2, label="CGM History"),
        Line2D([0],[0], color=COLOR_TRUTH,   lw=2.2, ls='--', label="Ground Truth"),
        Line2D([0],[0], color=COLOR_LSTM,    lw=2.2, label="LSTM-MC Posterior Mean"),
        mpatches.Patch(color=COLOR_LSTM,  alpha=0.28, label="LSTM-MC 95% PI"),
        Line2D([0],[0], color=COLOR_BLOOP,   lw=2.2, label="Bayesian Loop Posterior Mean"),
        mpatches.Patch(color=COLOR_BLOOP, alpha=0.28, label="Bayesian Loop 95% PI"),
        Line2D([0],[0], color=COLOR_BASAL, lw=1.6, label="Basal (U/hr)"),
        Line2D([0],[0], color=COLOR_BOLUS, lw=0, marker='o', ms=8, label="Bolus (U)"),
        Line2D([0],[0], color=COLOR_CARBS, lw=0, marker='s', ms=8, label="Meal (g carbs)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, fontsize=20, bbox_to_anchor=(0.33, 0.00))

    for ext in (".pdf", ".png"):
        p = out_path.replace(".pdf", ext)
        fig.savefig(p, bbox_inches="tight",
                    dpi=300 if ext == ".pdf" else 200)
        print(f"  Saved: {p}")
    plt.close(fig)


# =============================================================================
# MAIN  (unchanged)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-instance scorecard figure (Bayesian Loop vs LSTM-MC).")
    parser.add_argument("--prior", type=float, default=0.30)
    parser.add_argument("--out",   type=str,   default=None)
    args = parser.parse_args()

    from eval_config import get_config
    import figure_selection
    global INSTANCE
    cfg        = get_config(prior_scale=args.prior)
    DATA_DIR   = cfg["DATA_DIR"]
    batch_dirs = list(cfg["BATCH_LIST"])
    INSTANCE   = figure_selection.resolve("scorecard", DATA_DIR, cfg["INST_CSV_NAME"])["INSTANCE"]

    import sys as _sys
    _mod = _sys.modules[__name__]
    by_user = os.path.join(DATA_DIR, "by_user")
    if os.path.isdir(by_user):
        _mod.BY_USER_ROOT = by_user

    out_dir  = os.path.join(cfg["OUT_DIR"], "scorecard")
    out_path = args.out or os.path.join(out_dir, "scorecard.pdf")
    os.makedirs(out_dir, exist_ok=True)

    uid = INSTANCE["uid"]
    print(f"\nLoading LSTM + Bayesian Loop for uid={uid}, eid={INSTANCE['eid']}...")
    model_data = load_models(batch_dirs, uid)
    print(f"  Loaded: {list(model_data.keys())}")

    for key, label in [("LSTM_MC","LSTM"), ("Global","Bayesian Loop")]:
        if key not in model_data:
            print(f"  ERROR: {label} not found for uid={uid}")
            sys.exit(1)

    metrics_pkl_path = os.path.join(cfg["OUT_DIR"], "metrics_df.pkl")

    print("\nDrawing figure...")
    draw_figure(model_data, DATA_DIR, out_path, metrics_pkl_path=metrics_pkl_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
