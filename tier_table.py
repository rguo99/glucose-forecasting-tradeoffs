#!/usr/bin/env python3
"""tier_table.py
=============
Certification-tier heatmap (hypoglycemia, hyperglycemia and net-trend tiers
per model, horizon and instance context) and the matching LaTeX table.

Reads metrics_df.pkl from evaluate.py.

Usage:
    python tier_table.py --prior 0.30

Outputs (under <OUT_DIR>/paper_outputs/):
    tables/tab_tier_certification.tex
    figures/fig_tier_heatmap.pdf (+ .png)
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Display labels
LABEL_SAFE = 'Bayesian Loop'
LABEL_BASE = 'LSTM-MC'
LABEL_AR1  = 'Pure AR1'
LABEL_RLS  = 'Adaptive AR1-RLS'
LABEL_TRF  = 'Transformer-MC'
LABEL_ENS  = 'LSTM-Ensemble'
LABEL_CONF = 'LSTM-Conformal'
LABEL_TRF_CONF = 'Transformer-Conformal'
LABEL_TRF_ENS  = 'Transformer-Ensemble'
LABEL_MNODE = 'MNODE'  # not in the active presence-check loop below — out of scope for this revision

# These must match the horizon strings in metrics_df.pkl exactly
H_60  = '60 min'
H_240 = '240 min'

# Display strings shown in the figure
HZ_DISPLAY = {
    '60 min':  '1 hr',
    '240 min': '4 hrs',
    '30 min':  '30 min',
    '90 min':  '90 min',
    '120 min': '2 hrs',
    '180 min': '3 hrs',
}

CONTEXTS = ['Aggregate', 'Meal', 'Carb', 'Correction', 'Stable']
ITYPE_MAP = {
    'Aggregate':  None,
    'Meal':       'MEAL',
    'Carb':       'CARB_INSTANCE',
    'Correction': 'CORRECTION',
    'Stable':     'STABLE',
}

TIER1_LOWER = 0.80
TIER2_LOWER = 0.60

TIER_COLORS = {
    1: '#EAF3DE', 2: '#FAEEDA', 3: '#FCEBEB',
}
TIER_TEXT_COLORS = {
    1: '#27500A', 2: '#633806', 3: '#791F1F',
}
ROW_COLORS = {
    LABEL_SAFE: '#1f77b4',
    LABEL_BASE: '#d62728',
    LABEL_AR1:  '#2e7d32',
    LABEL_RLS:  '#9C27B0',
    LABEL_TRF:  '#FF9800',
    LABEL_ENS:  '#009688',
    LABEL_CONF: '#795548',
    LABEL_TRF_CONF: '#3F51B5',
    LABEL_TRF_ENS: '#C2185B',
    LABEL_MNODE: '#E91E63',
}
# Fixed prefix map for every model this script knows about; build_results()
# filters to whichever prefixes actually have data in the loaded df.
MODEL_PREFIX_ALL = {LABEL_SAFE: 'G', LABEL_BASE: 'L', LABEL_AR1: 'A',
                    LABEL_RLS: 'R', LABEL_TRF: 'T',
                    LABEL_ENS: 'E', LABEL_CONF: 'C', LABEL_TRF_CONF: 'F',
                    LABEL_TRF_ENS: 'N',
                    LABEL_MNODE: 'M'}


# =============================================================================
# TIER / DATA HELPERS
# =============================================================================

def assign_tier(alpha):
    if np.isnan(alpha): return None
    alpha = round(alpha, 4)
    if alpha >= TIER1_LOWER: return 1
    if alpha >= TIER2_LOWER: return 2
    return 3


def get_alpha(df, horizon_label, itype, col_prefix, col_suffix):
    col = f'{col_prefix}_{col_suffix}'
    if col not in df.columns: return np.nan
    sub = df[df['horizon'] == horizon_label]
    if itype is not None:
        sub = sub[sub['itype'] == itype]
    vals = sub[col].dropna()
    return float(vals.mean()) if len(vals) > 0 else np.nan


def build_results(df, model_prefix):
    rows = []

    def add(criterion, context, horizon_label, itype, col_suffix):
        for model, prefix in model_prefix.items():
            alpha = get_alpha(df, horizon_label, itype, prefix, col_suffix)
            rows.append({'criterion': criterion, 'context': context,
                         'horizon': horizon_label, 'model': model,
                         'alpha': alpha, 'tier': assign_tier(alpha)})

    for ctx in CONTEXTS:
        itype = ITYPE_MAP[ctx]
        # Hypoglycaemia — 60 min and 240 min
        add('Hypo sens',  ctx, H_60,  itype, 'P_Hypo_Sens_60min')
        add('Hypo sens',  ctx, H_240, itype, 'P_Hypo_Sens')
        add('Hypo spec',  ctx, H_60,  itype, 'P_Hypo_Spec_60min')
        add('Hypo spec',  ctx, H_240, itype, 'P_Hypo_Spec')
        # Hyperglycaemia — 60 min and 240 min
        add('Hyper sens', ctx, H_60,  itype, 'P_Hyper_Sens_60min')
        add('Hyper sens', ctx, H_240, itype, 'P_Hyper_Sens')
        add('Hyper spec', ctx, H_60,  itype, 'P_Hyper_Spec_60min')
        add('Hyper spec', ctx, H_240, itype, 'P_Hyper_Spec')
        # Net Trend — 60 min and 240 min
        add('Net Trend',  ctx, H_60,  itype, 'P_Net_Trend')
        add('Net Trend',  ctx, H_240, itype, 'P_Net_Trend')

    return pd.DataFrame(rows)


# =============================================================================
# LATEX TABLE
# =============================================================================

def write_latex_table(results_df, out_path, models_present=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_models = len(models_present) if models_present else 3
    tier_desc = (r'T1: $\alpha \geq 0.80$ (green). '
                r'T2: $0.60 \leq \alpha < 0.80$ (yellow). '
                r'T3: $\alpha < 0.60$ (red).')
    caption = (r'\textbf{Certification tier assignments across criteria, '
               f'contexts, and horizons for all {n_models} models.'
               r'} ' + tier_desc)
    lines = [r'% Generated by tier_table.py',
             r'\begin{table}[htbp]', r'\centering\small',
             f'\\caption{{{caption}}}', r'\label{tab:tier_certification}',
             r'\begin{tabular}{lllrrl}', r'\toprule',
             r'Criterion & Context & Horizon & Model & $\alpha$ & Tier \\',
             r'\midrule']
    for crit in ['Hypo sens', 'Hypo spec', 'Hyper sens', 'Hyper spec', 'Net Trend']:
        sub = results_df[results_df['criterion'] == crit]
        if sub.empty: continue
        lines += [r'\midrule', f'\\multicolumn{{6}}{{l}}{{\\textit{{{crit}}}}} \\\\']
        for ctx in CONTEXTS:
            for h in [H_60, H_240]:
                group = sub[(sub['context'] == ctx) & (sub['horizon'] == h)]
                if group.empty: continue
                for _, row in group.iterrows():
                    a = f"{row['alpha']:.3f}" if not np.isnan(row['alpha']) else '---'
                    t = f"T{int(row['tier'])}" if row['tier'] is not None else '---'
                    h_disp = HZ_DISPLAY.get(h, h)
                    lines.append(
                        f"  {crit} & {ctx} & {h_disp} & {row['model']} & {a} & {t} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  LaTeX table: {out_path}')


# =============================================================================
# PANEL DRAWING
# =============================================================================

def _draw_panel(ax, results_df, col_groups, row_defs,
                panel_label=None, show_legend=False,
                col_width=1.0, left_w=4.2,
                xlim_override=None, show_row_labels=True):
    """
    Draw one heatmap panel.
    All horizon display strings use HZ_DISPLAY for human-readable labels.

    show_row_labels=False suppresses the model/horizon labels at the left of the
    grid — used for the 2nd/3rd panels in the side-by-side layout, which share
    the row labels printed once on the leftmost panel.
    """
    ctx_short = {
        'Aggregate': 'All', 'Meal': 'Meal', 'Carb': 'Carb',
        'Correction': 'Corr', 'Stable': 'Stab',
    }

    # Flatten columns
    flat_cols = []
    for grp in col_groups:
        for ctx in grp['contexts']:
            for hz in grp['horizons']:
                flat_cols.append({
                    'criterion': grp['criterion'], 'context': ctx,
                    'horizon':   hz, 'color': grp['color'],
                    'label':     grp['label'],
                })

    n_cols = len(flat_cols)
    n_rows = len(row_defs)
    cw     = col_width
    ch     = 1.0
    pad    = 0.055
    header_h = 2.2
    legend_h = 2.4 if show_legend else 0.0

    # Sizes are tuned to a 1.0 in cell: the context labels ('Meal', 'Carb', …)
    # are the binding constraint at FS_CTX=28 (~0.9 in wide), and the criterion
    # headers must stay larger than the context labels they span.
    FS_PANEL  = 34
    FS_CRIT   = 30
    FS_CTX    = 28
    FS_HZ     = 20
    FS_TIER   = 28
    FS_ALPHA  = 22
    FS_MODEL  = 26
    FS_HORIZ  = 22
    FS_LEGEND = 18

    xlim_right = xlim_override if xlim_override is not None else n_cols * cw + 0.2
    ax.set_xlim(-left_w, xlim_right)
    ax.set_ylim(-legend_h - 0.15, n_rows * ch + header_h)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_facecolor('white')
    ax.set_clip_on(True)

    if panel_label:
        # Anchored to the grid's left edge, not the row-label margin, so every
        # panel's letter sits directly above its own heatmap.
        ax.text(0.05, n_rows * ch + header_h + 0.12,
                panel_label,
                ha='left', va='bottom',
                fontsize=FS_PANEL, fontweight='bold', color='#222222',
                clip_on=False)

    # Section spans (criterion groups)
    section_spans = {}
    for c, col in enumerate(flat_cols):
        k = col['label']
        if k not in section_spans:
            section_spans[k] = {'start': c, 'end': c, 'color': col['color']}
        else:
            section_spans[k]['end'] = c

    for crit_key, span in section_spans.items():
        c0, c1 = span['start'], span['end']
        mid_x  = (c0 + c1 + 1) / 2 * cw
        color  = span['color']
        ax.text(mid_x, n_rows * ch + 1.80, crit_key,
                ha='center', va='center', fontsize=FS_CRIT,
                fontweight='bold', color=color)
        ax.plot([c0 * cw + 0.08, (c1 + 1) * cw - 0.08],
                [n_rows * ch + 1.28, n_rows * ch + 1.28],
                color=color, lw=1.5, solid_capstyle='round', zorder=4, clip_on=True)
        if c0 > 0:
            ax.plot([c0 * cw, c0 * cw], [0, n_rows * ch],
                    color='#666666', lw=2.0, zorder=5, clip_on=True)

    # Context labels
    ctx_spans = {}
    for c, col in enumerate(flat_cols):
        key = (col['label'], col['context'])
        if key not in ctx_spans:
            ctx_spans[key] = {'start': c, 'end': c}
        else:
            ctx_spans[key]['end'] = c

    for (lbl, ctx), span in ctx_spans.items():
        c0, c1 = span['start'], span['end']
        mid_x  = (c0 + c1 + 1) / 2 * cw
        ax.text(mid_x, n_rows * ch + 0.82,
                ctx_short.get(ctx, ctx),
                ha='center', va='center', fontsize=FS_CTX, color='#444444')

    # Horizon sub-labels — use HZ_DISPLAY for human-readable strings
    hz_per_label_ctx = {}
    for col in flat_cols:
        key = (col['label'], col['context'])
        if key not in hz_per_label_ctx: hz_per_label_ctx[key] = set()
        hz_per_label_ctx[key].add(col['horizon'])

    for c, col in enumerate(flat_cols):
        key = (col['label'], col['context'])
        if len(hz_per_label_ctx[key]) > 1:
            hz_label = HZ_DISPLAY.get(col['horizon'], col['horizon'])
            ax.text((c + 0.5) * cw, n_rows * ch + 0.38,
                    hz_label,
                    ha='center', va='center', fontsize=FS_HZ, color='#444444')
            if (c > 0 and flat_cols[c-1]['context'] == col['context'] and
                    flat_cols[c-1]['label'] == col['label']):
                ax.plot([c * cw, c * cw], [0, n_rows * ch],
                        color='#E0E0E0', lw=0.5, zorder=3, clip_on=True)

    # Header baseline
    ax.plot([0, n_cols * cw], [n_rows * ch, n_rows * ch],
            color='#CCCCCC', lw=0.9, zorder=4, clip_on=True)

    # Cells
    for r, row in enumerate(row_defs):
        for c, col in enumerate(flat_cols):
            match_horizon = row['horizon']
            sub = results_df[
                (results_df['criterion'] == col['criterion']) &
                (results_df['context']   == col['context'])   &
                (results_df['horizon']   == match_horizon)    &
                (results_df['model']     == row['model'])
            ]
            alpha = sub.iloc[0]['alpha'] if not sub.empty else np.nan
            t     = sub.iloc[0]['tier']  if not sub.empty else None

            x0 = c * cw + pad
            y0 = (n_rows - r - 1) * ch + pad
            w  = cw - 2 * pad
            h  = ch - 2 * pad

            if t is None or (isinstance(alpha, float) and np.isnan(alpha)):
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0, y0), w, h, boxstyle='round,pad=0.03',
                    fc='#F3F3F3', ec='white', lw=0.3, zorder=2))
            else:
                ti = int(t)
                cy = (n_rows - r - 0.5) * ch
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0, y0), w, h, boxstyle='round,pad=0.03',
                    fc=TIER_COLORS[ti], ec='white', lw=0.3, zorder=2))
                ax.text((c + 0.5) * cw, cy + 0.18,
                        f'T{ti}', ha='center', va='center',
                        fontsize=FS_TIER, fontweight='bold',
                        color=TIER_TEXT_COLORS[ti], zorder=3)
                ax.text((c + 0.5) * cw, cy - 0.22,
                        f'{alpha:.2f}', ha='center', va='center',
                        fontsize=FS_ALPHA, color=TIER_TEXT_COLORS[ti], zorder=3)

    # Row labels — horizon shown using HZ_DISPLAY.
    # Model-block divider lines are always drawn; the text is optional.
    prev_model = None
    for r, row in enumerate(row_defs):
        cy = (n_rows - r - 0.5) * ch
        if row['model'] != prev_model:
            if prev_model is not None:
                y_div = (n_rows - r) * ch
                ax.plot([0, n_cols * cw], [y_div, y_div],
                        color='#555555', lw=1.8, zorder=6, clip_on=True)
            if show_row_labels:
                ax.text(-0.15, cy + 0.20, row['model'],
                        ha='right', va='center', fontsize=FS_MODEL,
                        fontweight='bold', color=row['color'])
            prev_model = row['model']
        if show_row_labels:
            hz_label = HZ_DISPLAY.get(row['horizon'], row['horizon'])
            ax.text(-0.15, cy - 0.22, hz_label,
                    ha='right', va='center', fontsize=FS_HORIZ, color='#444444')

    # Outer border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), n_cols * cw, n_rows * ch,
        boxstyle='square,pad=0', fill=False,
        ec='#CCCCCC', lw=0.9, zorder=7, clip_on=True))

    # Legend
    if show_legend:
        legend_items = [
            (1, 'T1: α ≥ 0.80'),
            (2, 'T2: 0.60 ≤ α < 0.80'),
            (3, 'T3: α < 0.60'),
        ]
        vis_w    = n_cols * cw
        item_w   = (vis_w + left_w) / 2
        box_w    = 0.35; box_h = 0.50; text_gap = 0.12
        top_gap  = 0.90; row_gap = 0.35
        block_w  = item_w * 2
        start_x  = -left_w + ((vis_w + left_w) - block_w) / 2
        row1_y   = -top_gap - box_h
        row2_y   = row1_y - row_gap - box_h
        positions = [
            (start_x,              row1_y),
            (start_x + item_w,     row1_y),
            (start_x + item_w * 2, row1_y),
        ]
        for i, (ti, lbl) in enumerate(legend_items):
            xi, yi = positions[i]
            patch = mpatches.FancyBboxPatch(
                (xi, yi), box_w, box_h, boxstyle='round,pad=0.02',
                fc=TIER_COLORS[ti], ec='#AAAAAA', lw=1.0, zorder=3,
                clip_on=False)
            ax.add_patch(patch)
            ax.text(xi + box_w + text_gap, yi + box_h / 2, lbl,
                    ha='left', va='center', fontsize=FS_LEGEND,
                    color='#333333', clip_on=False)


# =============================================================================
# FIGURE-LEVEL LEGEND
# =============================================================================

def _draw_legend_band(fig, fig_w, fig_h, bottom_in, height_in):
    """
    Tier legend on its own full-width axes at the bottom of the figure.
    Axes data units are inches, so swatch sizes are layout-independent.
    """
    ax = fig.add_axes([0.0, bottom_in / fig_h, 1.0, height_in / fig_h])
    ax.set_xlim(0, fig_w); ax.set_ylim(0, height_in)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_facecolor('white')

    legend_items = [
        (1, 'T1: α ≥ 0.80'),
        (2, 'T2: 0.60 ≤ α < 0.80'),
        (3, 'T3: α < 0.60'),
    ]
    # item_pitch must clear the widest label ('T2: 0.60 ≤ α < 0.80', ~3.9 in at
    # 26 pt) plus swatch and gap, so the three items never run together.
    box_w, box_h, text_gap = 0.44, 0.44, 0.16
    item_pitch = 5.4
    block_w    = item_pitch * len(legend_items)
    start_x    = fig_w / 2 - block_w / 2
    y0         = height_in / 2 - box_h / 2

    for i, (ti, lbl) in enumerate(legend_items):
        xi = start_x + i * item_pitch
        ax.add_patch(mpatches.FancyBboxPatch(
            (xi, y0), box_w, box_h, boxstyle='round,pad=0.02',
            fc=TIER_COLORS[ti], ec='#AAAAAA', lw=1.0, zorder=3))
        ax.text(xi + box_w + text_gap, height_in / 2, lbl,
                ha='left', va='center', fontsize=26, color='#333333')


# =============================================================================
# FIGURE LAYOUT  — A | B | C side by side, shared row labels, legend below
# =============================================================================

def plot_tier_heatmap(results_df, out_path, models_present=None):
    """
    Layout (v7):

        +--------------------+------------------+------------+
        | A  Hypoglycemia    | B Hyperglycemia  | C Net      |
        |    Safety          |   Safety         |   Trend    |
        | [model labels]     |                  |            |
        +--------------------+------------------+------------+
                              [legend]

    Three panels in one row: A (Hypo, 10 cols), B (Hyper, 10 cols),
    C (Net Trend, 5 cols). Model/horizon row labels are printed once, on
    panel A only — all three panels share the identical row order, so the
    labels apply across the row. Keeps the figure short enough to stay legible
    as the model count grows. Cell width identical across panels (square cells).
    """
    models_present = models_present or [LABEL_SAFE, LABEL_BASE, LABEL_AR1]
    n_models = len(models_present)

    CW        = 1.0
    CH        = 1.0
    # The left margin now only has to fit the row labels — the panel letter sits
    # above the grid — so size it from the longest model name rather than the
    # model count. ~0.215 in/char at FS_MODEL=26 pt, plus a gap.
    LEFT_W    = max(3.0, 0.215 * max(len(m) for m in models_present) + 0.40)
    RIGHT_PAD = 0.3
    # Panels B and C carry no row labels, so they only need a hair of margin
    # on the left for their panel letter and the section underline.
    INNER_W   = 0.45

    N_CTX     = len(CONTEXTS)   # 5
    N_AB_COLS = N_CTX * 2       # 10  (sens 5 + spec 5)
    N_C_COLS  = N_CTX           # 5

    N_ROWS    = 2 * n_models    # 2 horizons × n models

    HEADER_H  = 2.2
    LEGEND_H  = 1.7
    TOP_PAD   = 0.75            # room for the bold panel letters above the axes
    COL_GAP   = 1.00

    panel_h = HEADER_H + N_ROWS * CH

    w_a = LEFT_W  + N_AB_COLS * CW + RIGHT_PAD
    w_b = INNER_W + N_AB_COLS * CW + RIGHT_PAD
    w_c = INNER_W + N_C_COLS  * CW + RIGHT_PAD

    bottom_pad = 0.15

    FIG_W = w_a + COL_GAP + w_b + COL_GAP + w_c
    FIG_H = bottom_pad + LEGEND_H + panel_h + TOP_PAD

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('white')

    def _ax(left_in, bottom_in, width_in, height_in):
        return fig.add_axes([left_in / FIG_W, bottom_in / FIG_H,
                             width_in / FIG_W, height_in / FIG_H])

    panel_bottom = bottom_pad + LEGEND_H

    ax_a = _ax(0,                       panel_bottom, w_a, panel_h)
    ax_b = _ax(w_a + COL_GAP,           panel_bottom, w_b, panel_h)
    ax_c = _ax(w_a + COL_GAP + w_b + COL_GAP,
                                        panel_bottom, w_c, panel_h)

    # Row definitions — one (model, horizon) pair per row, in order
    def _rows():
        r = []
        for model in models_present:
            for hz in (H_60, H_240):
                r.append(dict(model=model, horizon=hz, color=ROW_COLORS[model]))
        return r
    safety_rows    = _rows()
    net_trend_rows = _rows()

    # Column groups
    hypo_col_groups = [
        dict(label='Hypo Sensitivity', criterion='Hypo sens', color='#a93226',
             contexts=CONTEXTS, horizons=[H_60]),
        dict(label='Hypo Specificity', criterion='Hypo spec', color='#c0392b',
             contexts=CONTEXTS, horizons=[H_60]),
    ]
    hyper_col_groups = [
        dict(label='Hyper Sensitivity', criterion='Hyper sens', color='#ca6f1e',
             contexts=CONTEXTS, horizons=[H_60]),
        dict(label='Hyper Specificity', criterion='Hyper spec', color='#d4ac0d',
             contexts=CONTEXTS, horizons=[H_60]),
    ]
    net_trend_col_groups = [
        dict(label='Net Trend Accuracy', criterion='Net Trend', color='#1a5276',
             contexts=CONTEXTS, horizons=[H_60]),
    ]

    # Draw panels — row labels on A only; B and C share A's row order
    _draw_panel(ax_a, results_df, hypo_col_groups, safety_rows,
                panel_label='A  Hypoglycemia Safety',
                show_legend=False, col_width=CW, left_w=LEFT_W,
                show_row_labels=True)

    _draw_panel(ax_b, results_df, hyper_col_groups, safety_rows,
                panel_label='B  Hyperglycemia Safety',
                show_legend=False, col_width=CW, left_w=INNER_W,
                show_row_labels=False)

    _draw_panel(ax_c, results_df, net_trend_col_groups, net_trend_rows,
                panel_label='C  Net Trend Accuracy',
                show_legend=False, col_width=CW, left_w=INNER_W,
                show_row_labels=False)

    _draw_legend_band(fig, FIG_W, FIG_H, bottom_pad, LEGEND_H)

    for ext in ('.pdf', '.png'):
        p = out_path.replace('.pdf', ext)
        fig.savefig(p, bbox_inches='tight', dpi=200)
        print(f'  Figure: {p}')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate tier certification heatmap v5')
    parser.add_argument('--prior',   type=float, default=0.30)
    parser.add_argument('--out-dir', type=str,   default=None)
    args = parser.parse_args()

    from eval_config import get_config
    cfg     = get_config(prior_scale=args.prior)
    OUT_DIR = args.out_dir or cfg['OUT_DIR']

    pkl_path = os.path.join(OUT_DIR, 'metrics_df.pkl')
    if not os.path.exists(pkl_path):
        print(f'  MISSING: metrics_df.pkl not found at {pkl_path}')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'TIER TABLE  —  prior={args.prior}')
    print(f'Layout: A+B stacked on left | C+legend on right')
    print(f'Horizons: {H_60} -> "{HZ_DISPLAY[H_60]}", {H_240} -> "{HZ_DISPLAY[H_240]}"')
    print(f'Models: {LABEL_SAFE} | {LABEL_BASE} | {LABEL_AR1}')
    print(f'{"="*60}\n')

    df = pd.read_pickle(pkl_path)
    print(f'  Loaded {len(df):,} rows')
    print(f'  Horizons in pkl: {sorted(df["horizon"].unique())}')
    print(f'  Instance types:  {sorted(df["itype"].unique())}')

    # Verify the horizons we need are present
    for h in [H_60, H_240]:
        if h not in df['horizon'].values:
            print(f'  MISSING: Required horizon "{h}" not found in pkl.')
            print(f'    Available: {sorted(df["horizon"].unique())}')
            sys.exit(1)

    a_cols = [c for c in df.columns if c.startswith('A_')]
    if not a_cols:
        print('  WARNING: No A_ columns found — Pure AR1 cells will be empty.')
    else:
        print(f'  AR1Pure columns found: {len(a_cols)}')

    # Only include the two additional baselines if their columns exist AND carry
    # real (non-NaN) data — keeps this script identical for other n configs.
    model_prefix = {LABEL_SAFE: 'G', LABEL_BASE: 'L', LABEL_AR1: 'A'}
    for label, prefix in [(LABEL_RLS, 'R'), (LABEL_TRF, 'T'),
                          (LABEL_ENS, 'E'), (LABEL_CONF, 'C'), (LABEL_TRF_CONF, 'F'),
                          (LABEL_TRF_ENS, 'N')]:
        col = f'{prefix}_RMSE'
        if col in df.columns and df[col].notna().any():
            model_prefix[label] = prefix
            print(f'  {label} columns found — including in tier table')
    # Reorder into the paper's canonical display order (see evaluate.py's
    # DISPLAY_ORDER): Bayesian Loop, LSTM family (MC->Ensemble->Conformal),
    # Transformer family (same UQ sequence), classical AR1 controls last.
    # Deliberately not performance-ordered -- the ranking flips at 180 min
    # and differs by metric, so any strength ordering would be unstable.
    # dict insertion order above is add-order, which is arbitrary to a reader.
    _DISPLAY = ['G', 'L', 'E', 'C', 'T', 'N', 'F', 'A', 'R']
    model_prefix = dict(
        sorted(model_prefix.items(),
               key=lambda kv: (_DISPLAY.index(kv[1])
                               if kv[1] in _DISPLAY else len(_DISPLAY)))
    )
    models_present = list(model_prefix.keys())
    print(f'  Display order: {", ".join(models_present)}')

    print('\n  Computing tier assignments...')
    results_df = build_results(df, model_prefix)

    # Diagnostic: show which cells have data
    print('\n  Summary (criterion / model / tier counts):')
    for crit in ['Hypo sens', 'Hypo spec', 'Hyper sens', 'Hyper spec', 'Net Trend']:
        sub = results_df[results_df['criterion'] == crit]
        for model in models_present:
            m_sub = sub[(sub['model'] == model) & sub['tier'].notna()]
            if m_sub.empty:
                print(f'    {crit:<12} {model:<14} (no data)')
                continue
            tc = m_sub['tier'].astype(int).value_counts().sort_index()
            print(f'    {crit:<12} {model:<14} '
                  + '  '.join(f'T{t}:{n}' for t, n in tc.items()))

    tab_dir = os.path.join(OUT_DIR, 'paper_outputs', 'tables')
    fig_dir = os.path.join(OUT_DIR, 'paper_outputs', 'figures')
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    write_latex_table(results_df,
                      os.path.join(tab_dir, 'tab_tier_certification.tex'),
                      models_present)
    plot_tier_heatmap(results_df,
                      os.path.join(fig_dir, 'fig_tier_heatmap.pdf'),
                      models_present)

    csv_path = os.path.join(OUT_DIR, 'tier_certification_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'  Results CSV: {csv_path}')

    print(f'\n{"="*60}')
    print(f'Done.')
    print(f'  Table:  {tab_dir}/tab_tier_certification.tex')
    print(f'  Figure: {fig_dir}/fig_tier_heatmap.pdf')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()