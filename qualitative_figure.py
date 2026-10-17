#!/usr/bin/env python3
"""qualitative_figure.py
=====================
Representative-instance figure: forecasts of all nine models (rows) on four
test instances (columns: meal, carbohydrate, correction bolus, stable), with
basal, bolus and meal events on a secondary axis.

The four instances are listed in figure_selection.json; --reselect instead
runs the automatic selection (select_fig_a) on the per-instance pool of the
three core models. The pool (master_pool.pkl) and the resolved selection are
cached under <OUT_DIR>/main_vis_figure/selection_cache/ and are also used by
extra_tables.py.

Usage:
    python qualitative_figure.py --prior 0.30

Output: <OUT_DIR>/main_vis_figure/qualitative.pdf (+ .png)
"""

import argparse
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
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# VISUAL IDENTITY
# =============================================================================

COLOR_GLOBAL  = "#1f77b4"   # Bayesian Loop
COLOR_LSTM    = "#d62728"   # LSTM
COLOR_AR1     = "#4CAF50"   # Pure AR1
COLOR_AR1RLS      = "#9C27B0"   # Adaptive AR1-RLS
COLOR_TRANSFORMER = "#FF9800"   # Transformer-MC
COLOR_LSTM_ENSEMBLE  = "#009688"   # LSTM-Ensemble
COLOR_LSTM_CONFORMAL = "#795548"   # LSTM-Conformal
COLOR_TRF_CONFORMAL  = "#3F51B5"   # Transformer-Conformal
COLOR_TRF_ENSEMBLE   = "#C2185B"   # Transformer-Ensemble — crimson, deliberately not teal (LSTM-Ensemble is #009688)
COLOR_HISTORY = "black"
COLOR_TRUTH   = "#444444"
COLOR_BASAL   = "#2ca02c"
COLOR_BOLUS   = "#1f77b4"
COLOR_CARBS   = "#d95f02"
COLOR_HYPO    = "#d62728"

GLUCOSE_LO = 70
GLUCOSE_HI = 180

LABEL_GLOBAL = "Bayesian Loop"
LABEL_LSTM   = "LSTM-MC"
LABEL_AR1    = "Pure AR1"
LABEL_AR1RLS      = "Adaptive AR1-RLS"
LABEL_TRANSFORMER = "Transformer-MC"
LABEL_LSTM_ENSEMBLE  = "LSTM-Ensemble"
LABEL_LSTM_CONFORMAL = "LSTM-Conformal"
LABEL_TRF_CONFORMAL  = "Transformer-Conformal"
LABEL_TRF_ENSEMBLE   = "Transformer-Ensemble"

THRESH       = 5.0   # mg/dL flat-zone, matching Net Trend Accuracy definition

ITYPE_LABELS = {
    "MEAL":          "Meal Instance  (\u226540g carbohydrates)",
    "CARB_INSTANCE": "Carb Instance  (10\u201339g carbohydrates)",
    "CORRECTION":    "Correction Instance  (\u22650.5U bolus, no meal)",
    "STABLE":        "Stable Instance  (basal equilibrium)",
}
ITYPE_ORDER = ["MEAL", "CARB_INSTANCE", "CORRECTION", "STABLE"]

# Manual per-column y-limit overrides for Fig A, applied on top of the
# data-driven (min-20, max+30) range computed in draw_fig_a. `None` in
# either slot keeps the auto-computed value for that bound. These exist to
# let a specific column be zoomed in/out for legibility (e.g. tightening the
# top to make a shallow slope more visible, or dropping the floor to reveal
# a wide interval's lower tail) -- any override that's tighter than the true
# data range will visibly clip whatever trace/PI extends past it.
YLIM_OVERRIDE = {
    "MEAL":          (None, 230),   # tighten top: true auto top is 371 (driven
                                     # by a wide-interval model's PI, not by
                                     # ground truth/Bayesian Loop, which stay
                                     # under ~200) -- zooms in to make the
                                     # Bayesian Loop row's rising slope more
                                     # visible; clips whichever model's PI
                                     # exceeds 230.
    "CARB_INSTANCE": (0, None),      # loosen floor: true auto bottom is
                                     # clamped at 30 even though real data
                                     # goes down to -76.9 -- 0 is the physical
                                     # floor (glucose can't be negative) so
                                     # stop there rather than at the raw
                                     # data min; still reveals much more
                                     # lower-tail PI than the old 30 clamp,
                                     # and still clips a bit of the widest
                                     # models' sub-zero PI tail.
}

# Two-line variants used as COLUMN headers in the transposed Fig A grid, where
# horizontal room per panel is much tighter than a full-width row title.
ITYPE_HEADERS = {
    "MEAL":          "Meal Instance\n(≥40 g carbohydrates)",
    "CARB_INSTANCE": "Carb Instance\n(10–39 g carbohydrates)",
    "CORRECTION":    "Correction Instance\n(≥0.5 U bolus, no meal)",
    "STABLE":        "Stable Instance\n(basal equilibrium)",
}


# =============================================================================
# METRIC HELPERS
# =============================================================================

def _clean(v):
    s = str(v)
    return s[:-2] if s.endswith(".0") else s

def _rmse(obs, mp):
    return float(np.sqrt(np.mean((obs - mp) ** 2)))

def _slope(obs, mp, steps=18):
    n = min(steps, len(obs) - 1)
    if n < 2:
        return np.nan
    t   = np.arange(n + 1, dtype=float)
    t_c = t - t.mean()
    denom       = np.dot(t_c, t_c) + 1e-12
    beta_obs    = np.dot(t_c, obs[:n + 1]) / denom
    beta_pred   = np.dot(t_c,  mp[:n + 1]) / denom
    scale = max(abs(beta_obs), abs(beta_pred), 1e-6)
    return float(np.clip((beta_obs * beta_pred) / (scale ** 2), -1.0, 1.0))

def _slope4h(obs, mp):
    return _slope(obs, mp, steps=47)

def _is(obs, lo, hi, alpha=0.05):
    return float(np.mean((hi-lo) + (2/alpha)*(np.maximum(lo-obs,0)+np.maximum(obs-hi,0))))

def _cov(obs, lo, hi):
    return float(np.mean((obs >= lo) & (obs <= hi)))

def _compute(obs, samples):
    mp = samples.mean(0)
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "RMSE":      _rmse(obs, mp),
        "slope90":   _slope(obs, mp, 18),
        "slope4h":   _slope4h(obs, mp),
        "IS":        _is(obs, lo, hi),
        "cov":       _cov(obs, lo, hi),
        "width":     float(np.mean(hi - lo)),
        "net_delta": float(mp[-1] - mp[0]),
        "obs_net":   float(obs[-1] - obs[0]),
        "obs_std":   float(np.std(obs)),
        "obs_min":   float(np.min(obs)),
        "obs_max":   float(np.max(obs)),
        "pred_hypo": bool(np.any(lo < GLUCOSE_LO)),
        "true_hypo": bool(np.any(obs < GLUCOSE_LO)),
        "true_hyper":bool(np.any(obs > GLUCOSE_HI)),
    }


# =============================================================================
# BUNDLE LOADING
# =============================================================================

def _label(folder):
    n = folder.lower()
    if "transformer_conformal" in n: return "TransformerConformal"
    if "transformer_ensemble" in n: return "TransformerEnsemble"
    if "transformer_mc"  in n: return "TransformerMC"
    if "lstm_conformal"  in n: return "LSTMConformal"
    if "lstm_ensemble"   in n: return "LSTMEnsemble"
    if "lstm_mc"         in n: return "LSTM_MC"
    if "ar1_rls"         in n: return "AR1RLS"
    if "ar1_pure"        in n: return "AR1Pure"
    if "iid_noise_fixed" in n or "iid_fixed" in n: return "IIDFixed"
    if "iid_noise"       in n: return "IID"
    if "ar1"             in n: return "Global" if "carb_uncert" in n else "AR1"
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

def load_bundles(batch_dirs, users,
                 models=("Global", "LSTM_MC", "AR1Pure")):
    data = {m: {} for m in models}
    tasks = []
    for bd in batch_dirs:
        if not os.path.exists(bd):
            print(f"  Warning: missing {bd}"); continue
        for folder in sorted(glob.glob(os.path.join(bd, "20*"))):
            if not os.path.isdir(folder): continue
            lbl = _label(folder)
            if lbl not in data: continue
            for uid in users:
                p = os.path.join(folder, "results", f"user_{uid}_results.pkl")
                if os.path.exists(p): tasks.append((lbl, uid, p))

    def _load(lbl, uid, path):
        try:    return lbl, uid, _load_bundle(path)
        except: return lbl, uid, None

    # A (model, user) pair must come from exactly one run directory
    # (the same rule as evaluate.find_bundle_paths).
    _seen = {}
    for lbl, uid, pth in tasks:
        if (lbl, uid) in _seen and _seen[(lbl, uid)] != pth:
            raise RuntimeError(f"{lbl} user {uid} has results in more than one run "
                               f"directory:\n  {_seen[(lbl, uid)]}\n  {pth}")
        _seen[(lbl, uid)] = pth

    print(f"  Loading {len(tasks)} bundles in parallel...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(_load, *t) for t in tasks]):
            lbl, uid, b = fut.result()
            if b: data[lbl][uid] = b

    for m in models:
        print(f"    {m}: {len(data[m])} users loaded")
    return data


# =============================================================================
# MASTER POOL
# =============================================================================

def _build_user_rows(uid, bg_b, bl_b, ba_b, imap):
    """Compute all rows for one user — called in parallel by build_pool."""
    rows = []
    n = min(bg_b["obs"].shape[0], bl_b["obs"].shape[0],
            bg_b["samples"].shape[1], bl_b["samples"].shape[1])
    for i in range(n):
        eid   = _clean(bg_b["test_ids"][i]) if i < len(bg_b["test_ids"]) else str(i)
        itype = imap.get(eid, "STABLE")
        mg = _compute(bg_b["obs"][i], bg_b["samples"][:, i, :])
        ml = _compute(bl_b["obs"][i], bl_b["samples"][:, i, :])

        if (ba_b is not None and i < ba_b["obs"].shape[0]
                and i < ba_b["samples"].shape[1]):
            ma = _compute(ba_b["obs"][i], ba_b["samples"][:, i, :])
        else:
            ma = {k: np.nan for k in mg}

        g_dir = np.sign(mg["net_delta"]); l_dir = np.sign(ml["net_delta"])
        o_dir = np.sign(mg["obs_net"])
        rows.append({
            "uid": uid, "eid": eid, "itype": itype,
            "obs_std": mg["obs_std"], "obs_min": mg["obs_min"],
            "obs_max": mg["obs_max"], "obs_net": mg["obs_net"],
            "true_hypo": mg["true_hypo"], "true_hyper": mg["true_hyper"],
            "sign_diverge": (g_dir != l_dir and abs(mg["net_delta"]) > 5
                             and abs(ml["net_delta"]) > 5),
            "global_right": (g_dir == o_dir and abs(mg["obs_net"]) > 5),
            "g_rmse":     mg["RMSE"],    "g_slope90":  mg["slope90"],
            "g_slope4h":  mg["slope4h"], "g_is":       mg["IS"],
            "g_cov":      mg["cov"],     "g_pred_hypo":mg["pred_hypo"],
            "g_net_delta":mg["net_delta"],
            "l_rmse":     ml["RMSE"],    "l_slope90":  ml["slope90"],
            "l_slope4h":  ml["slope4h"], "l_is":       ml["IS"],
            "l_cov":      ml["cov"],     "l_pred_hypo":ml["pred_hypo"],
            "l_net_delta":ml["net_delta"],
            "a_rmse":     ma["RMSE"],    "a_slope90":  ma["slope90"],
            "a_slope4h":  ma["slope4h"], "a_is":       ma["IS"],
            "a_cov":      ma["cov"],     "a_pred_hypo":ma.get("pred_hypo", np.nan),
            "a_net_delta":ma["net_delta"],
        })
    return rows


def find_bundle_paths(batch_dirs, users, models=("Global", "LSTM_MC", "AR1Pure")):
    """Like load_bundles, but returns file paths instead of loaded bundles.
    Used by build_pool_lazy to avoid holding the whole cohort's bundles
    resident in memory at once -- eagerly loading just G/L/A for 395 users
    (what load_bundles+build_pool did) OOM'd at 64GB, the same root cause
    already fixed in evaluate.py's find_bundle_paths."""
    paths = {m: {} for m in models}
    for bd in batch_dirs:
        if not os.path.exists(bd): continue
        for folder in sorted(glob.glob(os.path.join(bd, "20*"))):
            if not os.path.isdir(folder): continue
            lbl = _label(folder)
            if lbl not in paths: continue
            for uid in users:
                p = os.path.join(folder, "results", f"user_{uid}_results.pkl")
                if os.path.exists(p): paths[lbl][uid] = p
    return paths


def _build_user_rows_lazy(uid, g_path, l_path, a_path, imap):
    """Loads this one user's G/L/A bundles here, inside the worker, rather
    than receiving already-loaded bundles from a shared dict -- the bundles
    go out of scope (and get garbage collected) as soon as this user's rows
    are computed, instead of staying resident for the whole pool build."""
    bg_b = _load_bundle(g_path)
    bl_b = _load_bundle(l_path)
    ba_b = _load_bundle(a_path) if a_path else None
    return _build_user_rows(uid, bg_b, bl_b, ba_b, imap)


def build_pool_lazy(batch_dirs, users, imap):
    """Like build_pool, but loads each user's G/L/A bundles lazily inside
    the worker instead of eagerly loading the whole cohort into memory
    first. Same output as build_pool -- purely a loading-strategy change."""
    paths = find_bundle_paths(batch_dirs, users, models=("Global", "LSTM_MC", "AR1Pure"))
    common = sorted(set(paths["Global"]) & set(paths["LSTM_MC"]))
    print(f"  Building pool for {len(common)} users (parallel, lazy-loaded)...")

    all_rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {
            ex.submit(_build_user_rows_lazy, uid, paths["Global"][uid],
                     paths["LSTM_MC"][uid], paths.get("AR1Pure", {}).get(uid), imap): uid
            for uid in common
        }
        for fut in as_completed(futures):
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"  Warning: user {futures[fut]} failed: {e}")
            completed += 1
            if completed % 50 == 0:
                print(f"    {completed}/{len(common)} users done...")
    return pd.DataFrame(all_rows)


def build_pool(all_data, imap):
    """Build master pool in parallel — one task per user."""
    bg = all_data.get("Global", {})
    bl = all_data.get("LSTM_MC", {})
    ba = all_data.get("AR1Pure", {})
    common = sorted(set(bg) & set(bl))
    print(f"  Building pool for {len(common)} users (parallel)...")

    all_rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=min(32, len(common))) as ex:
        futures = {
            ex.submit(_build_user_rows, uid, bg[uid], bl[uid], ba.get(uid), imap): uid
            for uid in common
        }
        for fut in as_completed(futures):
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"  Warning: user {futures[fut]} failed: {e}")
            completed += 1
            if completed % 50 == 0:
                print(f"    {completed}/{len(common)} users done...")

    print(f"  Pool: {len(all_rows):,} instances")
    return pd.DataFrame(all_rows)


# =============================================================================
# CSV EVENT LOADING  (original — used for Figs B & C and as fallback)
# =============================================================================

def load_events(data_path, eids):
    target = set(_clean(e) for e in eids)
    result = {}
    for key, prefix in [("cgm","cgm"),("basal","basal"),
                         ("bolus","bolus"),("meals","meals")]:
        files = glob.glob(os.path.join(data_path, f"{prefix}*.csv"))
        if not files: result[key] = {}; continue
        print(f"  Reading {key} -> filtering to {len(target)} eids...")
        df = pd.read_csv(files[0])
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
        ex = [c for c in df.columns if any(x in c.lower() for x in ["ex","instance"])]
        if not ex: result[key] = {}; continue
        df = df.rename(columns={ex[0]: "ex"})
        df["_eid"] = df["ex"].astype(str).str.replace(r"\.0$", "", regex=True)
        df = df[df["_eid"].isin(target)]
        vm = {"basal":"basal_rate","bolus":"bolus","meals":"carbs","cgm":"cgm"}
        vh = [c for c in df.columns if any(x in c.lower() for x in [key,"rate","dose","amount"])]
        if vh: df = df.rename(columns={vh[0]: vm[key]})
        result[key] = {eid: g.sort_values("timestamp").reset_index(drop=True)
                       for eid, g in df.groupby("_eid")}
        print(f"    {key}: {len(df):,} rows, {len(result[key])} eids")
    return result


# =============================================================================
# FAST BY-USER PARQUET EVENT LOADER
# =============================================================================

def _norm_timestamp(df):
    """Detect and parse the timestamp column in-place, return df."""
    th = [c for c in df.columns if any(
        x in c.lower() for x in ["timestamp", "datestamp", "time", "start"])]
    if not th:
        return df
    df = df.rename(columns={th[0]: "timestamp"})
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="s")
            if df["timestamp"].max() > 1e8
            else pd.to_datetime(df["timestamp"], unit="m", origin="2026-01-01")
        )
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _norm_eid_col(df):
    """Rename the instance/ex column to _eid and strip trailing .0"""
    ex_cols = [c for c in df.columns
               if any(x in c.lower() for x in ["ex", "instance"])]
    if not ex_cols:
        return df
    df = df.rename(columns={ex_cols[0]: "_eid_raw"})
    df["_eid"] = df["_eid_raw"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def _load_parquet_stream(by_user_root, stream, uid, val_col_hint, eid_filter):
    """
    Load one stream from by_user/<stream>/<uid>.parquet.
    Returns a DataFrame filtered to eid_filter with columns [timestamp, <val_col>, _eid].
    Returns None if the file doesn't exist.
    """
    path = os.path.join(by_user_root, stream, f"{uid}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df = _norm_timestamp(df)
    df = _norm_eid_col(df)
    if "_eid" in df.columns:
        df = df[df["_eid"].isin(eid_filter)]
    # Normalise value column name
    if val_col_hint not in df.columns:
        candidates = [c for c in df.columns
                      if any(x in c.lower()
                             for x in [stream, "rate", "dose", "amount", "carb", "glucose"])]
        if candidates:
            df = df.rename(columns={candidates[0]: val_col_hint})
    return df.sort_values("timestamp").reset_index(drop=True)


def load_events_for_uid(data_path, uid, eid):
    """
    Fast single-uid event loader used by draw_fig_a.

    Priority:
      1. <data_path>/by_user/{stream}/{uid}.parquet   — fast, per-user files
      2. <data_path>/<stream>*.csv                    — slow, full-dataset CSV

    Returns dict with keys: cgm, basal, bolus, meals.
    Each value is a DataFrame filtered to the single eid (or empty DataFrame).
    """
    eid_clean = _clean(eid)
    eid_filter = {eid_clean}
    try:
        eid_filter.add(str(int(float(eid_clean))))
        eid_filter.add(str(int(float(eid_clean))) + ".0")
    except Exception:
        pass

    by_user_root = os.path.join(data_path, "by_user")
    use_parquet  = os.path.isdir(by_user_root)

    streams = {
        "cgm":   ("cgm",        "cgm"),
        "basal": ("basal",      "basal_rate"),
        "bolus": ("bolus",      "bolus"),
        "meals": ("meals",      "carbs"),
    }

    result = {}
    for key, (stream_dir, val_col) in streams.items():
        df = None
        if use_parquet:
            df = _load_parquet_stream(by_user_root, stream_dir, uid, val_col, eid_filter)
        if df is None:
            # CSV fallback: scan full file and filter to this eid
            files = glob.glob(os.path.join(data_path, f"{stream_dir}*.csv"))
            if files:
                raw = pd.read_csv(files[0])
                raw = _norm_timestamp(raw)
                raw = _norm_eid_col(raw)
                if "_eid" in raw.columns:
                    raw = raw[raw["_eid"].isin(eid_filter)]
                if val_col not in raw.columns:
                    candidates = [c for c in raw.columns
                                  if any(x in c.lower()
                                         for x in [stream_dir, "rate", "dose",
                                                   "amount", "carb", "glucose"])]
                    if candidates:
                        raw = raw.rename(columns={candidates[0]: val_col})
                df = raw.sort_values("timestamp").reset_index(drop=True)
        # Basal unit conversion
        # Parquet/CSV stores basal as U per 5-min interval for pump users.
        # Convert to U/hr (× 12) so the y-axis reads in standard clinical units.
        if key == "basal" and df is not None and not df.empty and "basal_rate" in df.columns:
            if df["basal_rate"].median() < 0.5:   # U/5min scale (pump users)
                df = df.copy()
                df["basal_rate"] = df["basal_rate"] * 12.0

        result[key] = df if df is not None else pd.DataFrame()

    return result


# =============================================================================
# CACHING HELPERS
# =============================================================================

def _pool_path(cache_dir):  return os.path.join(cache_dir, "master_pool.pkl")
def _fig_sel(cache_dir, n): return os.path.join(cache_dir, f"{n}_selected.pkl")
def _fig_ev(cache_dir, n):  return os.path.join(cache_dir, f"{n}_events.pkl")

def pool_cached(cache_dir):   return os.path.exists(_pool_path(cache_dir))
def save_pool(cache_dir, df):
    df.to_pickle(_pool_path(cache_dir))
    print(f"  Pool cached: {_pool_path(cache_dir)}")

def load_pool(cache_dir):
    df = pd.read_pickle(_pool_path(cache_dir))
    print(f"  Pool loaded: {len(df):,} instances")
    return df

def save_fig(cache_dir, name, selected, events):
    with open(_fig_sel(cache_dir, name), "wb") as f: pickle.dump(selected, f)
    with open(_fig_ev(cache_dir,  name), "wb") as f: pickle.dump(events,   f)
    print(f"  Cached: {name}")

# =============================================================================
# SHARED PLOTTING PRIMITIVES
# =============================================================================

def _cgm_window(cgm_df_or_dict, eid):
    """
    Accepts either:
      - a dict keyed by eid (old style, used by Figs B/C)
      - a DataFrame directly (new style from load_events_for_uid)
    """
    if isinstance(cgm_df_or_dict, dict):
        sub = cgm_df_or_dict.get(_clean(eid))
    else:
        sub = cgm_df_or_dict if not cgm_df_or_dict.empty else None

    if sub is None or len(sub) < 48: return None
    a = sub["timestamp"].iloc[47]
    s = sub["timestamp"].iloc[0]
    e = a + pd.Timedelta(hours=4)
    t = [a + pd.Timedelta(minutes=m) for m in np.arange(5, 245, 5)[:48]]
    return sub, a, s, e, t


# ---------------------------------------------------------------------------
# Draw insulin/carb events on the right y-axis of an existing Axes
# ---------------------------------------------------------------------------

# Colour for basal in Fig A — purple to avoid clashing with AR1 green
COLOR_BASAL_PANEL = "#b07cc6"   # light lavender

# Keep the old _draw_events for Figs B & C (takes a dedicated axis, not twinx)
def _metric_badge(ax, obs, mp, lo, hi, color, pos="left", fontsize=8.5,
                  compact=False):
    """compact=True drops the repeated inline units and pads the labels to a
    fixed width, giving a narrow 4-line block instead of a wide one. Needed by
    the transposed Fig A, where 8-10 model rows leave each panel far smaller
    than the 3-column layout did; units move to the legend footnote."""
    sc90 = _slope(obs, mp, 18); sc4h = _slope4h(obs, mp)
    if compact:
        badge = (f"RMSE {_rmse(obs,mp):>6.1f}\n"
                 f"Cov  {_cov(obs,lo,hi):>6.2f}\n"
                 f"PIW  {float(np.mean(hi-lo)):>6.1f}\n"
                 f"IS   {_is(obs,lo,hi):>6.0f}")
    else:
        badge = (f"RMSE: {_rmse(obs,mp):.1f} mg/dL\n"
                 f"Coverage: {_cov(obs,lo,hi):.2f}\n"
                 f"Width: {float(np.mean(hi-lo)):.1f} mg/dL\n"
                 f"IS: {_is(obs,lo,hi):.0f}")
    x = 0.02 if pos == "left" else 0.98
    ha = "left" if pos == "left" else "right"
    ax.text(x, 0.97, badge, transform=ax.transAxes, ha=ha, va="top",
            fontsize=fontsize, family="monospace", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor=color,
                      boxstyle="round,pad=0.3", lw=1.0), zorder=5)

def _fmt_ax(ax, start_t, end_t, show_ylabel, show_xlabel,
            ylim=(30, 380), fs_tick=7, fs_label=9):
    ax.set_ylim(ylim); ax.set_xlim(start_t, end_t)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.get_xticklabels(), fontsize=fs_tick, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.spines[["top", "right"]].set_visible(False)
    if show_ylabel: ax.set_ylabel("Glucose (mg/dL)", fontsize=fs_label,
                                  fontweight="bold")
    if show_xlabel: ax.set_xlabel("Time of day", fontsize=fs_label)

def _place_event_labels(ax, items, ylo, yhi, start_t, end_t, pad, fontsize,
                        panel_w_pt, panel_h_pt, max_tiers=4):
    """Place the bolus/meal lollipop value labels without overlaps.

    items: list of (timestamp, tip_value, text, colour).

    The previous rule only nudged a label left/right when a bolus and a meal
    shared a 5-min bin, so clusters of boluses minutes apart ("0.1U" under
    "1.9U") still printed on top of each other -- and it got worse once the
    transposed grid narrowed each panel. This instead tests every label
    against every already-placed one in BOTH axes and pushes it up a tier
    until it is clear, which handles bolus-vs-bolus, meal-vs-meal and
    bolus-vs-meal uniformly. Labels near a panel edge are also flipped to
    inward alignment so they stop being clipped.
    """
    if not items:
        return
    x0 = mdates.date2num(pd.Timestamp(start_t))
    x1 = mdates.date2num(pd.Timestamp(end_t))
    xspan = max(x1 - x0, 1e-9)
    yspan = max(yhi - ylo, 1e-9)
    h = fontsize * 1.30 / panel_h_pt          # label height, axes fraction

    placed = []   # (centre_x, half_width, y) in axes fractions
    for t, tip, txt, color in sorted(items, key=lambda r: pd.Timestamp(r[0])):
        fx = (mdates.date2num(pd.Timestamp(t)) - x0) / xspan
        hw = len(txt) * fontsize * 0.62 / panel_w_pt / 2
        if fx - hw < 0.005:
            ha, cx = "left", fx + hw
        elif fx + hw > 0.995:
            ha, cx = "right", fx - hw
        else:
            ha, cx = "center", fx

        fy0 = (tip + pad - ylo) / yspan
        fy = fy0
        for tier in range(max_tiers):
            fy = fy0 + tier * h * 1.15
            if not any(abs(cx - pcx) < (hw + phw) and abs(fy - pfy) < h
                       for pcx, phw, pfy in placed):
                break
        placed.append((cx, hw, fy))

        # Faint leader so a lifted label still reads as belonging to its dot
        if fy > fy0 + 1e-6:
            ax.plot([fx, fx], [(tip - ylo) / yspan, fy], transform=ax.transAxes,
                    color=color, lw=0.5, alpha=0.35, zorder=3)
        ax.text(fx, fy, txt, transform=ax.transAxes, color=color,
                fontsize=fontsize, ha=ha, va="bottom", fontweight="bold",
                zorder=5)


def _base_traces(ax, sub_cgm, t_pred, obs, mp, lo, hi, color, anchor_t,
                 draw_history=True):
    ax.axhspan(GLUCOSE_LO, GLUCOSE_HI, color="#e8f5e9", alpha=0.25, zorder=0)
    ax.axvline(anchor_t, color="gray", lw=1.0, ls=":", alpha=0.6, zorder=1)
    if draw_history:
        ax.plot(sub_cgm["timestamp"].iloc[:48], sub_cgm["cgm"].iloc[:48],
                color=COLOR_HISTORY, lw=2.0, zorder=4)
        ax.plot(t_pred, obs, color=COLOR_TRUTH, lw=1.8, ls="--", alpha=0.9, zorder=3)
    ax.fill_between(t_pred, lo, hi, color=color, alpha=0.18, zorder=2)
    ax.plot(t_pred, mp, color=color, lw=2.0, zorder=3)

def _generic_model_legend(fig, y_anchor=0.005, fontsize=9, footnote_dy=0.0):
    """Compact legend for the transposed Fig A grid.

    Each model is one ROW there and is already keyed by its coloured row
    label, so per-model mean/PI entries are redundant. This also keeps the
    legend a FIXED size as baselines are added -- the old per-model legend
    grew two entries per model and had silently started dropping the ones it
    could no longer fit.
    """
    neutral = "#555555"
    handles = [
        Line2D([0],[0], color=COLOR_HISTORY, lw=2.0, label="CGM history"),
        Line2D([0],[0], color=COLOR_TRUTH, lw=1.8, ls="--", label="Ground truth"),
        Line2D([0],[0], color=neutral, lw=2.0, label="Predicted mean"),
        plt.Rectangle((0,0),1,1, fc=neutral, alpha=0.18,
                      label="Predicted 95% PI"),
        Line2D([0],[0], color=COLOR_BASAL_PANEL, lw=4, alpha=0.4, label="Basal (U/hr)"),
        Line2D([0],[0], color=COLOR_BOLUS, marker="o", lw=0, ms=6, label="Bolus (U)"),
        Line2D([0],[0], color=COLOR_CARBS, marker="s", lw=0, ms=6, label="Carbs (g)"),
        plt.Rectangle((0,0),1,1, fc="#e8f5e9", alpha=0.6,
                      label="Target range (70–180 mg/dL)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=fontsize,
               frameon=False, bbox_to_anchor=(0.5, y_anchor + footnote_dy))


# =============================================================================
# FIGURE A SELECTION
# =============================================================================

def select_fig_a(pool_df, meals_csv, bolus_csv):
    all_eids = set(pool_df["eid"].unique())

    carbs = {}
    if os.path.exists(meals_csv):
        print("  Reading meals.csv for carbs...")
        df = pd.read_csv(meals_csv)
        df.rename(columns={"MLDTC":"datestamp","meal_dose":"carbs",
                            "MLDOSE":"carbs","USUBJID":"id"}, inplace=True, errors="ignore")
        cc = next((c for c in ["carbs","meal_dose"] if c in df.columns), None)
        ec = next((c for c in ["ex","instance_id"]  if c in df.columns), None)
        if cc and ec:
            df["_eid"] = df[ec].astype(str).str.replace(r"\.0$", "", regex=True)
            df = df[df["_eid"].isin(all_eids) & (df[cc] > 0)]
            carbs = df.groupby("_eid")[cc].sum().to_dict()
        print(f"    carbs: {len(carbs):,} eids")

    bolus = {}
    if os.path.exists(bolus_csv):
        print("  Reading bolus.csv for bolus totals...")
        df = pd.read_csv(bolus_csv)
        df.rename(columns={"USUBJID":"id","datestamp":"timestamp"}, inplace=True, errors="ignore")
        dc = next((c for c in ["bolus","dose"] if c in df.columns), None)
        ec = next((c for c in ["ex","instance_id"] if c in df.columns), None)
        if dc and ec:
            df["_eid"] = df[ec].astype(str).str.replace(r"\.0$", "", regex=True)
            df = df[df["_eid"].isin(all_eids) & (df[dc] > 0)]
            bolus = df.groupby("_eid")[dc].sum().to_dict()
        print(f"    bolus: {len(bolus):,} eids")

    print("  Pre-filtering to eids with valid CGM data...")
    cgm_pq = os.path.join(os.path.dirname(meals_csv), "cgm.parquet")  # cgm.csv in older builds
    cgm_files = [cgm_pq] if os.path.exists(cgm_pq) else glob.glob(
        os.path.join(os.path.dirname(meals_csv), "cgm*.csv"))
    valid_cgm_eids = set()
    if cgm_files:
        cgm_df = (pd.read_parquet(cgm_files[0], columns=["ex"]) if cgm_files[0].endswith(".parquet")
                  else pd.read_csv(cgm_files[0]))
        ex_col = next((c for c in cgm_df.columns if any(
            x in c.lower() for x in ["ex","instance"])), None)
        if ex_col:
            cgm_df["_eid"] = cgm_df[ex_col].astype(str).str.replace(r"\.0$", "", regex=True)
            counts = cgm_df.groupby("_eid").size()
            valid_cgm_eids = set(counts[counts >= 48].index)
    print(f"    {len(valid_cgm_eids):,} eids with >=48 CGM readings")

    pool = pool_df.copy()
    pool["total_carbs"] = pool["eid"].map(lambda e: carbs.get(e, 0.0))
    pool["total_bolus"] = pool["eid"].map(lambda e: bolus.get(e, 0.0))
    median_g = pool["g_rmse"].median()

    # Automatic selection (used with --reselect; the paper's instances are
    # pinned in figure_selection.json).
    selected = {}
    for itype in ITYPE_ORDER:
        sub = pool[pool["itype"] == itype].copy()
        if sub.empty:
            print(f"  Warning: no {itype} instances in pool"); continue

        sub = sub[sub["eid"].isin(valid_cgm_eids)].copy()
        if sub.empty:
            print(f"  Warning: no {itype} instances with valid CGM data"); continue

        sub_q = sub[
            (sub["g_rmse"] < 2.5 * sub["l_rmse"]) &
            (sub["g_rmse"] < 3 * median_g) &
            (sub["obs_min"] >= 55) &
            (sub["obs_max"] <= 380)
        ].copy()
        if sub_q.empty:
            print(f"  Warning: relaxing quality filter for {itype}")
        else:
            sub = sub_q

        if itype in ("MEAL", "CARB_INSTANCE"):
            med = sub["total_carbs"].median()
            sub["dist"] = (sub["total_carbs"] - med).abs()
            anchor = f"median carbs={med:.0f}g"
        elif itype == "CORRECTION":
            med = sub["total_bolus"].median()
            sub["dist"] = (sub["total_bolus"] - med).abs()
            anchor = f"median bolus={med:.2f}U"
        else:
            med = sub["obs_std"].median()
            sub["dist"] = (sub["obs_std"] - med).abs()
            anchor = f"median obs_std={med:.1f} mg/dL"

        row = sub.loc[sub["dist"].idxmin()]
        selected[itype] = {"uid": row["uid"], "eid": row["eid"]}
        print(f"  [A] {itype:14s} uid={row['uid']:>6} eid={row['eid']:>10} "
              f"{anchor} G_RMSE={row['g_rmse']:.1f} L_RMSE={row['l_rmse']:.1f} "
              f"A_RMSE={row.get('a_rmse', float('nan')):.1f}")
    return selected


# =============================================================================
# FIGURE A DRAWING  — one ROW per model, one COLUMN per instance type
# =============================================================================

def draw_fig_a(selected, all_data, event_data_legacy, out_path, data_path=None,
               mkeys=None, col_labels=None, col_colors=None):
    """
    TRANSPOSED grid: n_models rows x n_instance_types columns.

    Models are the rows (a tall figure fits a page) and the four clinical
    instance types are the columns.

    Consequences of the transpose:
      - Shared glucose y-limits are now per COLUMN (per instance), since a
        column is one CGM window shown under different models.
      - Model identity moves from a column header to a coloured, rotated row
        label on the left; the legend therefore drops per-model entries.
      - The basal strip is drawn ONCE per column at the bottom instead of
        once per panel: basal is a property of the instance, not the model,
        so repeating it down every row was n_models identical copies.

    Retained from the previous version: value-scaled lollipops globally
    normalised across all instances, separate bolus/meal vertical bands,
    label offsetting on overlap.

    `mkeys` / `col_labels` / `col_colors` still describe MODELS (kept for
    call-site compatibility); they now drive rows rather than columns.
    """
    itypes_present = [t for t in ITYPE_ORDER if t in selected]
    n_cols = len(itypes_present)

    mkeys      = mkeys      or ["Global", "LSTM_MC", "AR1Pure"]
    row_labels = col_labels or [LABEL_GLOBAL, LABEL_LSTM, LABEL_AR1]
    row_colors = col_colors or [COLOR_GLOBAL, COLOR_LSTM, COLOR_AR1]
    n_rows = len(mkeys)
    mid_col = n_cols // 2   # column that carries the shared x-axis label

    # Geometry
    # Grid rows: [glucose, spacer] per model, then [gap, basal strip] once.
    #
    # Row count is NOT fixed -- baselines keep getting added (currently up to
    # 9: the 3 core models plus AR1-RLS, Transformer-MC, and the four
    # LSTM/Transformer x Ensemble/Conformal cells). So panel height is solved
    # for rather than hard-coded: it shrinks as rows are added so the figure
    # stays inside MAX_ASPECT and keeps fitting a portrait page at
    # \textwidth. Below PANEL_H_MIN the panels would stop being readable, so
    # past that point the figure is allowed to grow taller instead.
    COL_W        = 2.24                  # inches per instance-type column
    TOP_BAND     = 0.55                  # column headers (no suptitle)
    BOT_BAND     = 1.35                  # x tick labels + xlabel + legend
    PANEL_H_PREF = 1.14                  # preferred inches per glucose row
    PANEL_H_MIN  = 0.86                  # floor before legibility goes
    MAX_ASPECT   = 1.30                  # height/width of a portrait text block

    height_ratios, glc_rows = [], []
    for i in range(n_rows):
        glc_rows.append(len(height_ratios))
        height_ratios.append(1.0)
        if i < n_rows - 1:
            height_ratios.append(0.16)   # inter-model spacer
    height_ratios.append(0.06)           # gap above the basal strip
    strip_row = len(height_ratios)
    height_ratios.append(0.20)           # single shared basal strip
    ratio_sum = sum(height_ratios)

    fig_w = COL_W * n_cols + 1.05        # + left margin for row labels
    usable_h = MAX_ASPECT * fig_w - TOP_BAND - BOT_BAND
    PANEL_H  = min(PANEL_H_PREF, max(PANEL_H_MIN, usable_h / ratio_sum))
    fig_h    = PANEL_H * ratio_sum + TOP_BAND + BOT_BAND
    # Anything typeset along the vertical axis (the model row label and the
    # "Glucose (mg/dL)" y-label) is bounded by PANEL_H, which shrinks as rows
    # are added -- so size it against the row rather than fixing it.
    FS_ROW    = max(7.0, min(9.5, PANEL_H * 72 / 8.0))
    ROW_CHARS = max(8, int(PANEL_H * 72 / (0.62 * FS_ROW)))
    print(f"  Fig A grid: {n_rows} model rows x {n_cols} instance columns, "
          f"{fig_w:.1f} x {fig_h:.1f} in (panel {COL_W:.2f} x {PANEL_H:.2f} in, "
          f"row-label {FS_ROW:.1f}pt / {ROW_CHARS} chars)")

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        len(height_ratios), n_cols, figure=fig,
        height_ratios=height_ratios,
        hspace=0.0, wspace=0.14,
        left=1.12 / fig_w, right=1.0 - 0.12 / fig_w,
        top=1.0 - TOP_BAND / fig_h, bottom=BOT_BAND / fig_h
    )

    # Column headers: instance types
    header_y = 1.0 - (TOP_BAND - 0.26) / fig_h
    for c, itype in enumerate(itypes_present):
        d = fig.add_subplot(gs[0, c]); d.set_visible(False)
        pos = d.get_position()
        fig.text(pos.x0 + pos.width / 2, header_y, ITYPE_HEADERS[itype],
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color="#333333", linespacing=1.3, transform=fig.transFigure)

    # Pre-load all event data and find GLOBAL max bolus & meal values
    _ev_cache = {}
    for itype in itypes_present:
        uid = selected[itype]["uid"]
        eid = selected[itype]["eid"]
        if uid not in _ev_cache:
            if data_path is not None:
                try:
                    _ev_cache[uid] = load_events_for_uid(data_path, uid, eid)
                except Exception as exc:
                    print(f"  Warning: uid={uid}: {exc}")
                    _ev_cache[uid] = None
            else:
                _ev_cache[uid] = None

    global_bolus_max = 0.0
    global_meal_max  = 0.0
    for itype in itypes_present:
        uid   = selected[itype]["uid"]
        eid   = selected[itype]["eid"]
        eid_c = _clean(eid)
        ev = _ev_cache.get(uid)
        if ev is None:
            ev = {
                "bolus": event_data_legacy.get("bolus", {}).get(eid_c, pd.DataFrame()),
                "meals": event_data_legacy.get("meals", {}).get(eid_c, pd.DataFrame()),
            }
        bdf = ev.get("bolus", pd.DataFrame())
        mdf = ev.get("meals", pd.DataFrame())
        if not bdf.empty and "bolus" in bdf.columns:
            global_bolus_max = max(global_bolus_max, float(bdf["bolus"].max()))
        if not mdf.empty and "carbs" in mdf.columns:
            global_meal_max = max(global_meal_max, float(mdf["carbs"].max()))

    global_bolus_max = max(global_bolus_max, 1.0)
    global_meal_max  = max(global_meal_max,  1.0)

    # Per-column context (one CGM window, shared by every model row)
    cols = []
    for itype in itypes_present:
        uid   = selected[itype]["uid"]
        eid   = selected[itype]["eid"]
        eid_c = _clean(eid)

        uid_events = _ev_cache.get(uid)
        cgm_src = (uid_events.get("cgm", pd.DataFrame())
                   if uid_events is not None
                   else event_data_legacy.get("cgm", {}))

        win_ref = _cgm_window(cgm_src, eid)
        if win_ref is None:
            print(f"  Warning: no CGM window for {itype} (uid={uid} eid={eid})")
            cols.append(None)
            continue
        _, anchor_t_ref, start_t_ref, end_t_ref, _ = win_ref

        # Shared glucose y-limits across every model shown in this column
        col_vals = [win_ref[0]["cgm"].iloc[:48].values]
        for mkey in mkeys:
            bundle = all_data.get(mkey, {}).get(uid)
            if bundle is None: continue
            tids = [_clean(x) for x in bundle["test_ids"]]
            if eid_c not in tids: continue
            idx  = tids.index(eid_c)
            obs  = bundle["obs"][idx]
            samp = bundle["samples"][:, idx, :]
            col_vals.extend([obs, samp.mean(0),
                             np.percentile(samp, 2.5, axis=0),
                             np.percentile(samp, 97.5, axis=0)])
        all_col = np.concatenate([v for v in col_vals if len(v) > 0])
        col_ylo = max(30,  np.nanmin(all_col) - 20)
        col_yhi = min(420, np.nanmax(all_col) + 30)
        auto_ylo, auto_yhi = col_ylo, col_yhi
        o_ylo, o_yhi = YLIM_OVERRIDE.get(itype, (None, None))
        if o_ylo is not None: col_ylo = o_ylo
        if o_yhi is not None: col_yhi = o_yhi
        print(f"  [{itype}] y-lim auto=({auto_ylo:.1f}, {auto_yhi:.1f}) "
              f"data_range=({np.nanmin(all_col):.1f}, {np.nanmax(all_col):.1f}) "
              f"-> used=({col_ylo:.1f}, {col_yhi:.1f})")
        glc_range = col_yhi - col_ylo

        if uid_events is not None:
            panel_events = uid_events
        else:
            panel_events = {
                "basal": event_data_legacy.get("basal", {}).get(eid_c, pd.DataFrame()),
                "bolus": event_data_legacy.get("bolus", {}).get(eid_c, pd.DataFrame()),
                "meals": event_data_legacy.get("meals", {}).get(eid_c, pd.DataFrame()),
            }

        # Lollipop bands (in glucose axis units, from col_ylo)
        # Bolus band: 4% – 11% of glc_range   (shorter stems)
        # Meal  band: 13% – 21% of glc_range  (taller stems, distinct tier)
        # Within each band, height scales linearly with value / global_max.
        BOLUS_BASE = glc_range * 0.04
        BOLUS_TOP  = glc_range * 0.11
        MEAL_BASE  = glc_range * 0.13
        MEAL_TOP   = glc_range * 0.21
        LABEL_PAD  = glc_range * 0.025

        bolus_df = panel_events.get("bolus", pd.DataFrame())
        meals_df = panel_events.get("meals", pd.DataFrame())

        # Basal strip limits
        basal_df  = panel_events.get("basal", pd.DataFrame())
        basal_max = 0.0
        if not basal_df.empty and "basal_rate" in basal_df.columns:
            bdf_win = basal_df[(basal_df["timestamp"] >= start_t_ref) &
                               (basal_df["timestamp"] <= end_t_ref)]
            if not bdf_win.empty:
                basal_max = float(bdf_win["basal_rate"].max())

        cols.append(dict(
            itype=itype, uid=uid, eid=eid, eid_c=eid_c, cgm_src=cgm_src,
            anchor_t=anchor_t_ref, start_t=start_t_ref, end_t=end_t_ref,
            ylo=col_ylo, yhi=col_yhi,
            bolus_df=bolus_df, meals_df=meals_df, basal_df=basal_df,
            label_pad=LABEL_PAD,
            strip_ymax=max(basal_max * 1.5, 0.1),
            bolus_tip=(lambda v, ylo=col_ylo, b=BOLUS_BASE, t=BOLUS_TOP:
                       ylo + b + (t - b) * (v / global_bolus_max)),
            meal_tip=(lambda v, ylo=col_ylo, b=MEAL_BASE, t=MEAL_TOP:
                      ylo + b + (t - b) * (v / global_meal_max)),
        ))

    # Glucose panels: one row per model
    for r, (mkey, mcolor, mlabel) in enumerate(zip(mkeys, row_colors, row_labels)):
        row_axes = []
        for c, col in enumerate(cols):
            if col is None:
                continue
            ax = fig.add_subplot(gs[glc_rows[r], c])
            row_axes.append(ax)

            def _finish(a=ax, cc=c, cl=col):
                # No per-row y-label: repeating it 8-10x both wastes the
                # left margin and collides with the wrapped row labels once
                # rows get short. One shared label is drawn below instead.
                _fmt_ax(a, cl["start_t"], cl["end_t"], show_ylabel=False,
                        show_xlabel=False, ylim=(cl["ylo"], cl["yhi"]),
                        fs_tick=7)
                plt.setp(a.get_xticklabels(), visible=False)
                a.tick_params(axis="x", length=2)

            bundle = all_data.get(mkey, {}).get(col["uid"])
            if bundle is None:
                ax.text(0.5, 0.5, f"{mlabel}\nnot available",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="gray")
                _finish()
                continue

            tids = [_clean(x) for x in bundle["test_ids"]]
            if col["eid_c"] not in tids:
                ax.text(0.5, 0.5, "instance\nnot found", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8, color="gray")
                _finish()
                continue

            idx  = tids.index(col["eid_c"])
            obs  = bundle["obs"][idx]
            samp = bundle["samples"][:, idx, :]
            mp   = samp.mean(0)
            lo, hi = np.percentile(samp, [2.5, 97.5], axis=0)

            win = _cgm_window(col["cgm_src"], col["eid"])
            if win is None:
                _finish(); continue
            sub_cgm, anchor_t, start_t, end_t, t_pred = win

            _base_traces(ax, sub_cgm, t_pred, obs, mp, lo, hi, mcolor,
                         anchor_t, draw_history=True)
            _metric_badge(ax, obs, mp, lo, hi, mcolor, fontsize=7.0, compact=True)
            _finish()

            # Bolus + meal lollipops. Markers are drawn here; the value labels
            # are collected and placed together afterwards so they can be
            # de-collided against each other (see _place_event_labels).
            lollipops = []

            bolus_df = col["bolus_df"]
            if not bolus_df.empty and "bolus" in bolus_df.columns:
                b_win2 = bolus_df[(bolus_df["timestamp"] >= start_t) &
                                  (bolus_df["timestamp"] <= end_t) &
                                  (bolus_df["bolus"] > 0)]
                for _, row in b_win2.iterrows():
                    v   = float(row["bolus"])
                    t   = row["timestamp"]
                    tip = col["bolus_tip"](v)
                    ax.vlines(t, col["ylo"], tip,
                              color=COLOR_BOLUS, lw=1.2, alpha=0.9, zorder=3)
                    ax.scatter(t, tip, color=COLOR_BOLUS,
                               s=12, marker="o", zorder=4)
                    lollipops.append((t, tip, f"{v:.1f}U", COLOR_BOLUS))

            meals_df = col["meals_df"]
            if not meals_df.empty and "carbs" in meals_df.columns:
                m_win2 = meals_df[(meals_df["timestamp"] >= start_t) &
                                  (meals_df["timestamp"] <= end_t) &
                                  (meals_df["carbs"] > 0)]
                for _, row in m_win2.iterrows():
                    v   = float(row["carbs"])
                    t   = row["timestamp"]
                    tip = col["meal_tip"](v)
                    ax.vlines(t, col["ylo"], tip,
                              color=COLOR_CARBS, lw=1.2, alpha=0.9, zorder=3)
                    ax.scatter(t, tip, color=COLOR_CARBS,
                               s=12, marker="s", zorder=4)
                    lollipops.append((t, tip, f"{v:.0f}g", COLOR_CARBS))

            _place_event_labels(ax, lollipops, col["ylo"], col["yhi"],
                                start_t, end_t, pad=col["label_pad"],
                                fontsize=5.0, panel_w_pt=COL_W * 72,
                                panel_h_pt=PANEL_H * 72)

        # Row label = model name, coloured, in the left margin. Rotated text is
        # bounded by the ROW height, so names longer than a row can hold
        # ("Transformer-Conformal") are wrapped onto a second stacked line.
        if row_axes:
            txt = mlabel
            if len(txt) > ROW_CHARS:
                cut = max(txt.rfind("-", 0, len(txt) - 3),
                          txt.rfind(" ", 0, len(txt) - 3))
                if cut > 0:
                    txt = txt[:cut + 1].rstrip() + "\n" + txt[cut + 1:]
            pos = row_axes[0].get_position()
            fig.text(0.30 / fig_w, pos.y0 + pos.height / 2, txt,
                     ha="center", va="center", rotation=90,
                     fontsize=FS_ROW, fontweight="bold", color=mcolor,
                     linespacing=1.1, transform=fig.transFigure)

    # One shared y-axis label spanning every model row
    y_top = gs[glc_rows[0], 0].get_position(fig).y1
    y_bot = gs[glc_rows[-1], 0].get_position(fig).y0
    fig.text(0.74 / fig_w, (y_top + y_bot) / 2, "Glucose (mg/dL)",
             ha="center", va="center", rotation=90,
             fontsize=11, fontweight="bold", transform=fig.transFigure)

    # Basal strip: once per column, beneath the last model row
    for c, col in enumerate(cols):
        if col is None:
            continue
        ax_s = fig.add_subplot(gs[strip_row, c])
        ax_s.set_xlim(col["start_t"], col["end_t"])
        ax_s.set_ylim(0, col["strip_ymax"])

        basal_df = col["basal_df"]
        if not basal_df.empty and "basal_rate" in basal_df.columns:
            bdf2 = basal_df[(basal_df["timestamp"] >= col["start_t"]) &
                            (basal_df["timestamp"] <= col["end_t"])].copy()
            if not bdf2.empty:
                ax_s.step(bdf2["timestamp"], bdf2["basal_rate"],
                          where="post", color=COLOR_BASAL_PANEL,
                          lw=1.0, alpha=0.8, zorder=2)
                ax_s.fill_between(bdf2["timestamp"], 0, bdf2["basal_rate"],
                                  step="post", color=COLOR_BASAL_PANEL,
                                  alpha=0.30, zorder=1)

        ax_s.axvline(col["anchor_t"], color="gray", lw=0.8, ls=":", alpha=0.5)
        ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_s.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax_s.get_xticklabels(), fontsize=6.5, rotation=30, ha="right")
        if c == mid_col:
            ax_s.set_xlabel("Time of day", fontsize=9)

        ax_s.tick_params(axis="y", labelsize=5.5,
                         colors=COLOR_BASAL_PANEL, length=2, pad=1)
        ax_s.yaxis.set_major_locator(plt.MaxNLocator(2, integer=False))
        ax_s.spines[["top", "right"]].set_visible(False)
        ax_s.spines["left"].set_edgecolor(COLOR_BASAL_PANEL)
        ax_s.spines["left"].set_alpha(0.6)
        ax_s.spines["bottom"].set_visible(False)
        ax_s.tick_params(axis="x", length=2)
        if c == 0:
            ax_s.set_ylabel("U/hr", fontsize=6,
                            color=COLOR_BASAL_PANEL, labelpad=2)

    _generic_model_legend(fig, y_anchor=0.10 / fig_h, fontsize=8.0,
                          footnote_dy=0.22 / fig_h)
    _save(fig, out_path)


def _save(fig, out_path):
    for ext in (".pdf", ".png"):
        p = out_path.replace(".pdf", ext)
        fig.savefig(p, bbox_inches="tight", dpi=300 if ext==".pdf" else 180)
        print(f"  Saved: {p}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Representative-instance forecast grid (all models on the "
                    "same four test instances).")
    parser.add_argument("--prior",    type=float, default=0.30)
    parser.add_argument("--reselect", action="store_true",
                        help="Ignore figure_selection.json and rerun the automatic "
                             "instance selection (select_fig_a) on a freshly built "
                             "cohort pool.")
    parser.add_argument("--rebuild-pool", action="store_true",
                        help="Rebuild the cached per-instance pool (master_pool.pkl).")
    args = parser.parse_args()

    from eval_config import get_config
    import figure_selection
    from evaluate import imap_cache_path
    cfg          = get_config(prior_scale=args.prior)
    TARGET_USERS = cfg["TARGET_USERS"]
    DATA_DIR     = cfg["DATA_DIR"]
    batch_dirs   = list(cfg["BATCH_LIST"])
    BASE_OUT     = cfg["OUT_DIR"]
    MEALS_CSV    = os.path.join(DATA_DIR, "meals.csv")
    BOLUS_CSV    = os.path.join(DATA_DIR, "bolus.csv")

    VIS_DIR   = os.path.join(BASE_OUT, "main_vis_figure")
    CACHE_DIR = os.path.join(VIS_DIR,  "selection_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_a = os.path.join(VIS_DIR, "qualitative.pdf")

    # Per-instance metrics pool for the three core models; used by --reselect
    # and by extra_tables.py (instance percentiles).
    if args.rebuild_pool or not pool_cached(CACHE_DIR):
        imap_path = imap_cache_path(os.path.join(DATA_DIR, cfg["INST_CSV_NAME"]))
        if not os.path.exists(imap_path):
            raise FileNotFoundError(f"{imap_path} not found; run evaluate.py first.")
        with open(imap_path, "rb") as f:
            imap = pickle.load(f)
        print("Building cohort pool (lazy per-user loading)...")
        pool_df = build_pool_lazy(batch_dirs, TARGET_USERS, imap)
        save_pool(CACHE_DIR, pool_df)
    else:
        pool_df = load_pool(CACHE_DIR)

    if args.reselect:
        sel_a = select_fig_a(pool_df, MEALS_CSV, BOLUS_CSV)
    else:
        sel_a = figure_selection.resolve("qualitative", DATA_DIR, cfg["INST_CSV_NAME"])
    print("  Instances: " + ", ".join(f"{k}=user {v['uid']}" for k, v in sel_a.items()))
    ev_a = load_events(DATA_DIR, [v["eid"] for v in sel_a.values()])
    save_fig(CACHE_DIR, "a", sel_a, ev_a)

    _EXTRA_MODEL_KEYS = ["AR1RLS", "TransformerMC", "LSTMEnsemble", "LSTMConformal",
                         "TransformerConformal", "TransformerEnsemble"]
    uids = sorted({v["uid"] for v in sel_a.values()})
    all_data = load_bundles(batch_dirs, uids, models=["Global", "LSTM_MC", "AR1Pure"])
    extra_data = load_bundles(batch_dirs, uids, models=_EXTRA_MODEL_KEYS)
    extras_present = [mk for mk in _EXTRA_MODEL_KEYS if len(extra_data.get(mk, {})) > 0]
    all_data.update(extra_data)

    # Row order matches the main results table (evaluate.DISPLAY_ORDER).
    _DISPLAY_ORDER_KEYS = ["Global", "LSTM_MC", "LSTMEnsemble", "LSTMConformal",
                           "TransformerMC", "TransformerEnsemble", "TransformerConformal",
                           "AR1Pure", "AR1RLS"]
    _LABEL_COLOR = {
        "Global":        (LABEL_GLOBAL,          COLOR_GLOBAL),
        "LSTM_MC":       (LABEL_LSTM,             COLOR_LSTM),
        "LSTMEnsemble":  (LABEL_LSTM_ENSEMBLE,    COLOR_LSTM_ENSEMBLE),
        "LSTMConformal": (LABEL_LSTM_CONFORMAL,   COLOR_LSTM_CONFORMAL),
        "TransformerMC": (LABEL_TRANSFORMER,      COLOR_TRANSFORMER),
        "TransformerEnsemble":  (LABEL_TRF_ENSEMBLE,  COLOR_TRF_ENSEMBLE),
        "TransformerConformal": (LABEL_TRF_CONFORMAL, COLOR_TRF_CONFORMAL),
        "AR1Pure":       (LABEL_AR1,              COLOR_AR1),
        "AR1RLS":        (LABEL_AR1RLS,            COLOR_AR1RLS),
    }
    _present = {"Global", "LSTM_MC", "AR1Pure"} | set(extras_present)
    mkeys_a, col_labels_a, col_colors_a = [], [], []
    for mk in _DISPLAY_ORDER_KEYS:
        if mk in _present:
            lbl, color = _LABEL_COLOR[mk]
            mkeys_a.append(mk); col_labels_a.append(lbl); col_colors_a.append(color)
    print(f"  Columns: {', '.join(col_labels_a)}")

    draw_fig_a(sel_a, all_data, ev_a, out_a, data_path=DATA_DIR,
               mkeys=mkeys_a, col_labels=col_labels_a, col_colors=col_colors_a)
    print(f"Done. Output: {out_a}")


if __name__ == "__main__":
    main()
