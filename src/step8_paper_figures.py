#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step8_paper_figures.py | Topic: 10 | Purpose: 制作所有发表级论文图表
输出:
  Fig 1: fig1_consort_flowchart.png
  Fig 2: fig2_map_trajectories.png
  Fig 6: fig6_shap_summary.png   (publication-quality)
  Fig 7: fig7_stiffness_stratified.png
  Table 1: table1_baseline.csv
  Table 2: table2_performance.csv
"""
import os, sys, gc, json, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
RAW_DATA   = Path("/home/lxk/vitaldb/physionet.org")
WORK       = Path("/home/lxk/vitaldb/analysis")
TOPIC_DIR  = WORK / "topic10_induction_instability"
PROC_DIR   = TOPIC_DIR / "data" / "processed"
FIG_DIR    = TOPIC_DIR / "outputs" / "figures"
MET_DIR    = TOPIC_DIR / "outputs" / "metrics"
LOG_DIR    = TOPIC_DIR / "outputs" / "logs"
VITAL_DIR  = RAW_DATA / "files" / "vitaldb" / "1.0.0" / "vital_files"

for d in [FIG_DIR, MET_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── LOGGING ────────────────────────────���─────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"step8_figures_{ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
log = logging.getLogger(__name__)
log.info("=== Phase 7: Paper Figure Production Started ===")

# ─── PUBLICATION STYLE ────────────────────────────────────────────────────────
COLORS = {
    'crash':     '#D55E00',   # 红橙 - crash组
    'no_crash':  '#0072B2',   # 蓝   - non-crash组
    'model_a':   '#999999',   # 灰   - 基础模型
    'model_b':   '#009E73',   # 绿   - +弹性模型
    'model_c':   '#CC79A7',   # 粉   - Model C
    'xgboost':   '#E69F00',   # 金   - XGBoost
    'reference': '#AAAAAA',   # 浅灰
    'q1':        '#0072B2',
    'q2':        '#56B4E9',
    'q3':        '#E69F00',
    'q4':        '#D55E00',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'axes.linewidth':    0.8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fontsize':   7,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth':   1.2,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
log.info("Loading processed data...")
outcome_df  = pd.read_parquet(PROC_DIR / "outcome_labels.parquet")
feat_df     = pd.read_parquet(PROC_DIR / "vascular_features.parquet")
seg_df      = pd.read_parquet(PROC_DIR / "induction_segments.parquet")
cp_df       = pd.read_parquet(PROC_DIR / "changepoints.parquet")

# Load clinical data
clin_path = RAW_DATA / "files" / "vitaldb" / "1.0.0" / "clinical_data.csv"
clin_df = pd.read_csv(clin_path)
clin_df['age'] = pd.to_numeric(clin_df['age'], errors='coerce')
log.info(f"outcome_df: {len(outcome_df)} rows | feat_df: {len(feat_df)} rows")

# Load flowchart numbers
with open(MET_DIR / "cohort_flowchart_numbers.json") as f:
    flow_nums = json.load(f)

# vascular_features already contains crash_30, baseline_map, nadir_map, drop_pct
# Just add crash_40, crash_20, crash_absolute from outcome_df
merged = feat_df.merge(
    outcome_df[['caseid','crash_40','crash_20','crash_absolute']],
    on='caseid', how='left'
)
log.info(f"Merged dataset: {len(merged)} rows")
log.info(f"Crash (30%) prevalence: {merged['crash_30'].sum()} / {len(merged)} "
         f"({merged['crash_30'].mean()*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 1: CONSORT-STYLE FLOWCHART
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Generating Fig 1: CONSORT flowchart...")

def draw_box(ax, x, y, w, h, text, color='#E8F4FD', fontsize=7.5, bold=False):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='#333333',
                                    linewidth=0.8, zorder=2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=True, zorder=3,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
                zorder=1)

def draw_exclude_box(ax, x, y, w, h, text):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.02",
                                    facecolor='#FFF3CD', edgecolor='#CC8800',
                                    linewidth=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=6.8, zorder=3,
            multialignment='center', color='#5C4A00')

fig, ax = plt.subplots(figsize=(6.5, 8.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Box coordinates
BOX_W = 4.0
BOX_W_EXCL = 3.4
BOX_H = 0.72
CENTER_X = 5.0
EXCL_X = 8.5

# Numbers from flowchart
n_total      = flow_nums['total']
n_adult      = flow_nums['age_18_plus']
n_general    = flow_nums['general_anesthesia']
n_excl_cardiac = n_general - flow_nums['exclude_cardiac']
n_demo       = flow_nums['has_demographics']
n_tracks     = flow_nums['has_required_tracks']
n_final      = len(merged)
n_crash      = merged['crash_30'].sum()
n_no_crash   = len(merged) - n_crash

# --- Main flow boxes ---
draw_box(ax, CENTER_X, 10.3, BOX_W, BOX_H,
         f"VitalDB Open Dataset\nN = {n_total:,} surgical cases",
         color='#D4EDDA', fontsize=7.5, bold=True)

draw_arrow(ax, CENTER_X, 9.94, CENTER_X, 9.22)

draw_box(ax, CENTER_X, 8.88, BOX_W, BOX_H,
         f"Adults (age ≥ 18 years)\nN = {n_adult:,}",
         color='#E8F4FD')
draw_arrow(ax, CENTER_X, 8.52, CENTER_X, 7.80)
draw_exclude_box(ax, EXCL_X, 8.88, BOX_W_EXCL, BOX_H,
                 f"Excluded (age < 18):\nN = {n_total - n_adult:,}")
ax.annotate('', xy=(EXCL_X - BOX_W_EXCL/2, 8.88),
            xytext=(CENTER_X + BOX_W/2, 8.88),
            arrowprops=dict(arrowstyle='->', color='#CC8800', lw=0.7))

draw_box(ax, CENTER_X, 7.46, BOX_W, BOX_H,
         f"General anaesthesia\nN = {n_general:,}",
         color='#E8F4FD')
draw_arrow(ax, CENTER_X, 7.10, CENTER_X, 6.38)
draw_exclude_box(ax, EXCL_X, 7.46, BOX_W_EXCL, BOX_H,
                 f"Excluded (non-general):\nN = {n_adult - n_general:,}")
ax.annotate('', xy=(EXCL_X - BOX_W_EXCL/2, 7.46),
            xytext=(CENTER_X + BOX_W/2, 7.46),
            arrowprops=dict(arrowstyle='->', color='#CC8800', lw=0.7))

draw_box(ax, CENTER_X, 6.04, BOX_W, BOX_H,
         f"Excluded cardiac/thoracic surgery\nN = {flow_nums['exclude_cardiac']:,}",
         color='#E8F4FD')
draw_arrow(ax, CENTER_X, 5.68, CENTER_X, 4.96)
draw_exclude_box(ax, EXCL_X, 6.04, BOX_W_EXCL, BOX_H,
                 f"Excluded:\nN = {n_general - flow_nums['exclude_cardiac']:,}")
ax.annotate('', xy=(EXCL_X - BOX_W_EXCL/2, 6.04),
            xytext=(CENTER_X + BOX_W/2, 6.04),
            arrowprops=dict(arrowstyle='->', color='#CC8800', lw=0.7))

draw_box(ax, CENTER_X, 4.62, BOX_W, BOX_H,
         f"Required monitoring tracks present\n(ABP + PPG + Propofol)\nN = {n_tracks:,}",
         color='#E8F4FD')
draw_arrow(ax, CENTER_X, 4.26, CENTER_X, 3.54)
draw_exclude_box(ax, EXCL_X, 4.62, BOX_W_EXCL, BOX_H,
                 f"Missing required tracks:\nN = {flow_nums['exclude_cardiac'] - n_tracks:,}")
ax.annotate('', xy=(EXCL_X - BOX_W_EXCL/2, 4.62),
            xytext=(CENTER_X + BOX_W/2, 4.62),
            arrowprops=dict(arrowstyle='->', color='#CC8800', lw=0.7))

draw_box(ax, CENTER_X, 3.20, BOX_W + 0.4, BOX_H,
         f"Final analysis cohort\nN = {n_final:,}",
         color='#D4EDDA', fontsize=7.5, bold=True)

# Split arrow
draw_arrow(ax, CENTER_X, 2.84, CENTER_X, 2.30)

# Two outcome boxes
draw_box(ax, 3.0, 1.75, 3.2, BOX_H,
         f"Induction Crash\n(MAP drop >30%)\nN = {n_crash:,} ({n_crash/n_final*100:.1f}%)",
         color='#F8D7DA', fontsize=7)
draw_box(ax, 7.0, 1.75, 3.2, BOX_H,
         f"No Crash\nN = {n_no_crash:,} ({n_no_crash/n_final*100:.1f}%)",
         color='#D1ECF1', fontsize=7)

# Horizontal line for split
ax.plot([CENTER_X - 1.6, CENTER_X + 1.6], [2.30, 2.30], 'k-', lw=0.8)
ax.annotate('', xy=(3.0, 2.09), xytext=(3.0, 2.30),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8))
ax.annotate('', xy=(7.0, 2.09), xytext=(7.0, 2.30),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8))

ax.set_title("Figure 1. Study CONSORT-Style Flowchart", fontsize=9, fontweight='bold', pad=8)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_consort_flowchart.png", dpi=300)
plt.close()
log.info("Fig 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 2: CRASH vs NON-CRASH MAP TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Generating Fig 2: MAP trajectories (Crash vs Non-Crash)...")

import vitaldb

def get_induction_map_series(caseid, t_start, t_end, max_dur=480):
    """Load Solar8000/ART_MBP 1-Hz series for the induction window.
    Returns (segment_array, local_baseline) where local_baseline is
    the median of the first 60 valid seconds.
    """
    try:
        fname = VITAL_DIR / f"{int(caseid):04d}.vital"
        if not fname.exists():
            return None, None
        vf = vitaldb.VitalFile(str(fname))
        arr = vf.to_numpy(['Solar8000/ART_MBP'], interval=1.0)
        if arr is None or arr.shape[0] == 0:
            return None, None
        start_idx = int(t_start)
        end_idx   = min(start_idx + max_dur, arr.shape[0])
        if start_idx >= arr.shape[0]:
            return None, None
        segment = arr[start_idx:end_idx, 0].astype(float)
        # Physiological artifact filter (ABP 30-200 mmHg)
        segment[(segment < 30) | (segment > 200)] = np.nan
        # Require at least 60 valid values
        if np.sum(~np.isnan(segment)) < 60:
            return None, None
        # Local baseline: median of first 90 valid seconds
        early = segment[:90]
        early_valid = early[~np.isnan(early)]
        if len(early_valid) < 10:
            early_valid = segment[~np.isnan(segment)][:30]
        local_bl = np.median(early_valid)
        if local_bl < 30 or local_bl > 200:
            return None, None
        return segment, local_bl
    except Exception:
        return None, None

# t_start / t_end already in merged (from vascular_features)
# Supplement with seg_df for any missing
use_df = merged.copy()
missing_ts = use_df['t_start'].isna()
if missing_ts.any():
    seg_sub = seg_df[['caseid','t_start','t_end']].rename(
        columns={'t_start':'t_start_seg','t_end':'t_end_seg'})
    use_df = use_df.merge(seg_sub, on='caseid', how='left')
    use_df.loc[missing_ts, 't_start'] = use_df.loc[missing_ts, 't_start_seg']
    use_df.loc[missing_ts, 't_end']   = use_df.loc[missing_ts, 't_end_seg']
use_df = use_df.dropna(subset=['t_start','t_end'])
log.info(f"Cases with segment data: {len(use_df)}")

MAX_DUR = 480  # seconds, 8 min
SAMPLE_N = 150  # cases per group (random sample for speed)

crash_cases    = use_df[use_df['crash_30'] == 1]['caseid'].tolist()
nocrash_cases  = use_df[use_df['crash_30'] == 0]['caseid'].tolist()

np.random.seed(42)
crash_sample   = np.random.choice(crash_cases,  min(SAMPLE_N, len(crash_cases)),  replace=False)
nocrash_sample = np.random.choice(nocrash_cases, min(SAMPLE_N, len(nocrash_cases)), replace=False)

log.info(f"Sampling {len(crash_sample)} crash + {len(nocrash_sample)} no-crash cases for trajectory plot")

def collect_trajectories(caseids, df, max_dur):
    """Collect normalized MAP trajectories using local (intra-induction) baseline."""
    trajs = []
    for cid in caseids:
        row = df[df['caseid'] == cid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        seg, local_bl = get_induction_map_series(
            int(cid), float(row['t_start']), float(row['t_end']), max_dur)
        if seg is None or local_bl is None:
            continue
        # Normalize to local baseline
        seg_norm = seg / local_bl * 100.0
        # Clip extreme values
        seg_norm = np.clip(seg_norm, 20, 180)
        # Pad or trim to max_dur
        padded = np.full(max_dur, np.nan)
        L = min(len(seg_norm), max_dur)
        padded[:L] = seg_norm[:L]
        trajs.append(padded)
    return np.array(trajs) if trajs else np.empty((0, max_dur))

log.info("Loading crash trajectories...")
crash_trajs   = collect_trajectories(crash_sample,   use_df, MAX_DUR)
log.info(f"  crash: {crash_trajs.shape[0]} valid")
log.info("Loading no-crash trajectories...")
nocrash_trajs = collect_trajectories(nocrash_sample, use_df, MAX_DUR)
log.info(f"  no-crash: {nocrash_trajs.shape[0]} valid")

time_axis = np.arange(MAX_DUR) / 60.0  # minutes

def nanmean_se(trajs):
    m  = np.nanmean(trajs, axis=0)
    se = np.nanstd(trajs, axis=0) / np.sqrt(np.sum(~np.isnan(trajs), axis=0).clip(1))
    return m, se

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

# Panel A: normalized MAP (% of baseline)
ax = axes[0]
if crash_trajs.shape[0] > 0 and nocrash_trajs.shape[0] > 0:
    cm, cse = nanmean_se(crash_trajs)
    nm, nse = nanmean_se(nocrash_trajs)

    # Plot individual traces (very thin, transparent)
    for traj in crash_trajs[::3]:  # every 3rd to reduce clutter
        valid = ~np.isnan(traj)
        if valid.sum() > 30:
            ax.plot(time_axis[valid], traj[valid], color=COLORS['crash'],
                    alpha=0.05, lw=0.3, zorder=1)
    for traj in nocrash_trajs[::3]:
        valid = ~np.isnan(traj)
        if valid.sum() > 30:
            ax.plot(time_axis[valid], traj[valid], color=COLORS['no_crash'],
                    alpha=0.05, lw=0.3, zorder=1)

    # Mean ± 95% CI (mean ± 1.96 SE)
    valid_c = ~np.isnan(cm)
    valid_n = ~np.isnan(nm)
    ax.fill_between(time_axis[valid_c], (cm - 1.96*cse)[valid_c], (cm + 1.96*cse)[valid_c],
                    color=COLORS['crash'], alpha=0.18, zorder=2)
    ax.fill_between(time_axis[valid_n], (nm - 1.96*nse)[valid_n], (nm + 1.96*nse)[valid_n],
                    color=COLORS['no_crash'], alpha=0.18, zorder=2)
    ax.plot(time_axis[valid_c], cm[valid_c], color=COLORS['crash'],
            lw=2.0, label=f'Crash (n={crash_trajs.shape[0]})', zorder=3)
    ax.plot(time_axis[valid_n], nm[valid_n], color=COLORS['no_crash'],
            lw=2.0, label=f'No Crash (n={nocrash_trajs.shape[0]})', zorder=3)

ax.axhline(70, color='#555555', ls='--', lw=0.8, label='70% threshold')
ax.axhline(100, color='#AAAAAA', ls=':', lw=0.7)
ax.set_xlabel("Time from propofol administration (min)", fontsize=8)
ax.set_ylabel("MAP (% of pre-induction baseline)", fontsize=8)
ax.set_title("A. Normalized MAP Trajectories", fontsize=8, fontweight='bold')
ax.legend(loc='lower left', fontsize=6.5, framealpha=0.7)
ax.set_ylim(40, 130)
ax.set_xlim(0, MAX_DUR/60)
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.xaxis.set_major_locator(MultipleLocator(2))

# Panel B: distribution of nadir MAP
ax2 = axes[1]
crash_sub   = merged[merged['crash_30'] == 1]['nadir_map'].dropna()
nocrash_sub = merged[merged['crash_30'] == 0]['nadir_map'].dropna()
bins = np.arange(20, 130, 5)
ax2.hist(nocrash_sub, bins=bins, color=COLORS['no_crash'], alpha=0.65,
         label=f'No Crash (n={len(nocrash_sub)})', density=True, edgecolor='white', lw=0.3)
ax2.hist(crash_sub, bins=bins, color=COLORS['crash'], alpha=0.65,
         label=f'Crash (n={len(crash_sub)})', density=True, edgecolor='white', lw=0.3)
ax2.axvline(65, color='#D55E00', ls='--', lw=1.0, label='MAP = 65 mmHg')
ax2.set_xlabel("Nadir MAP during induction (mmHg)", fontsize=8)
ax2.set_ylabel("Density", fontsize=8)
ax2.set_title("B. Nadir MAP Distribution", fontsize=8, fontweight='bold')
ax2.legend(fontsize=6.5, framealpha=0.7)

plt.tight_layout(w_pad=1.5)
fig.savefig(FIG_DIR / "fig2_map_trajectories.png", dpi=300)
plt.close()
log.info("Fig 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 6: SHAP SUMMARY PLOT (publication quality)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Generating Fig 6: SHAP summary plot...")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import shap

# Feature set B (best LR model)
feat_b = ['age', 'bmi', 'asa', 'preop_htn', 'preop_dm',
          'baseline_map', 'ri_mean_clean', 'ppg_amp_clean']

FEAT_LABELS = {
    'age':            'Age (years)',
    'bmi':            'BMI (kg/m²)',
    'asa':            'ASA Physical Status',
    'preop_htn':      'Pre-op Hypertension',
    'preop_dm':       'Pre-op Diabetes Mellitus',
    'baseline_map':   'Baseline MAP (mmHg)',
    'ri_mean_clean':  'Reflection Index (PPG)',
    'ppg_amp_clean':  'PPG Amplitude',
    'pi_clean':       'Perfusion Index',
    'vascular_stiffness_score': 'Vascular Stiffness Score (PCA)',
}

# Build dataset
shap_df = merged[feat_b + ['crash_30']].copy()
shap_df = shap_df.dropna(subset=['crash_30'])
X_all = shap_df[feat_b]
y_all = shap_df['crash_30']

# Impute + scale
imp = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imp.fit_transform(X_all), columns=feat_b)

X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y_all, test_size=0.3,
                                            stratify=y_all, random_state=42)

# Train GradientBoostingClassifier (sklearn) — compatible with SHAP 0.49
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, random_state=42)
gb.fit(X_tr, y_tr)

# SHAP values via TreeExplainer (works with sklearn GBM)
try:
    explainer  = shap.TreeExplainer(gb)
    shap_array = explainer.shap_values(X_te)
    use_shap_new_api = False
    log.info(f"SHAP values computed: shape={shap_array.shape}")
except Exception as e:
    log.error(f"SHAP failed: {e}")
    shap_array = None

# Publication-quality SHAP summary
fig, ax = plt.subplots(figsize=(5.5, 3.8))

# Rename columns for display
X_te_display = X_te.copy()
X_te_display.columns = [FEAT_LABELS.get(c, c) for c in X_te.columns]

if shap_array is not None:
    shap.summary_plot(shap_array, X_te_display,
                      plot_type='dot', max_display=8,
                      color_bar_label='Feature value\n(normalised)',
                      show=False, plot_size=None)
else:
    ax.text(0.5, 0.5, 'SHAP unavailable\n(API compatibility issue)',
            ha='center', va='center', transform=ax.transAxes, fontsize=10)

ax = plt.gca()
ax.set_xlabel("SHAP value (impact on log-odds of induction crash)", fontsize=8)
ax.set_title("Figure 6. SHAP Feature Importance — XGBoost Model", fontsize=8.5, fontweight='bold')
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_shap_summary.png", dpi=300)
plt.close()
log.info("Fig 6 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIG 7: VASCULAR STIFFNESS SCORE STRATIFIED MAP TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Generating Fig 7: Stiffness-stratified MAP trajectories...")

# Only cases with stiffness score (t_start/t_end already in merged)
stiff_df = merged.dropna(subset=['vascular_stiffness_score']).copy()
stiff_df = stiff_df.dropna(subset=['t_start','t_end'])

if len(stiff_df) >= 40:
    # Quartiles
    stiff_df['stiff_q'] = pd.qcut(stiff_df['vascular_stiffness_score'], q=4,
                                   labels=['Q1\n(Low)','Q2','Q3','Q4\n(High)'])
    q_colors = [COLORS['q1'], COLORS['q2'], COLORS['q3'], COLORS['q4']]
    q_labels_text = ['Q1 (Low stiffness)', 'Q2', 'Q3', 'Q4 (High stiffness)']

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel A: Trajectories by stiffness quartile
    ax = axes[0]
    for qi, (qlabel, color) in enumerate(zip(['Q1\n(Low)','Q2','Q3','Q4\n(High)'], q_colors)):
        sub  = stiff_df[stiff_df['stiff_q'] == qlabel]
        trajs = collect_trajectories(sub['caseid'].tolist(), stiff_df, MAX_DUR)
        if trajs.shape[0] < 3:
            continue
        m, se = nanmean_se(trajs)
        valid = ~np.isnan(m)
        ax.fill_between(time_axis[valid], (m - 1.96*se)[valid], (m + 1.96*se)[valid],
                        color=color, alpha=0.15)
        ax.plot(time_axis[valid], m[valid], color=color, lw=1.6,
                label=f'{q_labels_text[qi]} (n={trajs.shape[0]})')

    ax.axhline(70, color='#555555', ls='--', lw=0.8, label='70% threshold')
    ax.axhline(100, color='#AAAAAA', ls=':', lw=0.7)
    ax.set_xlabel("Time from propofol administration (min)", fontsize=8)
    ax.set_ylabel("MAP (% of baseline)", fontsize=8)
    ax.set_title("A. MAP Trajectories by Stiffness Quartile", fontsize=8, fontweight='bold')
    ax.legend(loc='lower left', fontsize=5.5, framealpha=0.7)
    ax.set_ylim(50, 125)
    ax.set_xlim(0, MAX_DUR/60)
    ax.yaxis.set_major_locator(MultipleLocator(20))

    # Panel B: Crash rate by stiffness quartile (bar plot with CI)
    ax2 = axes[1]
    crash_rates, crash_ci_lo, crash_ci_hi, ns = [], [], [], []
    for qlabel in ['Q1\n(Low)','Q2','Q3','Q4\n(High)']:
        sub = stiff_df[stiff_df['stiff_q'] == qlabel]
        if len(sub) == 0:
            crash_rates.append(np.nan); crash_ci_lo.append(np.nan); crash_ci_hi.append(np.nan); ns.append(0)
            continue
        n   = len(sub)
        k   = sub['crash_30'].sum()
        p   = k / n
        se  = np.sqrt(p*(1-p)/max(n,1))
        crash_rates.append(p * 100)
        crash_ci_lo.append(max(0, (p - 1.96*se)*100))
        crash_ci_hi.append(min(100, (p + 1.96*se)*100))
        ns.append(n)

    x_pos   = np.arange(4)
    bars    = ax2.bar(x_pos, crash_rates, color=q_colors, edgecolor='white',
                      linewidth=0.5, width=0.6, zorder=2)
    # Error bars
    lo_err  = [r - l for r, l in zip(crash_rates, crash_ci_lo)]
    hi_err  = [h - r for r, h in zip(crash_rates, crash_ci_hi)]
    ax2.errorbar(x_pos, crash_rates, yerr=[lo_err, hi_err],
                 fmt='none', color='#333333', capsize=3, lw=1.0, zorder=3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'], fontsize=7)
    ax2.set_xlabel("Vascular Stiffness Score Quartile", fontsize=8)
    ax2.set_ylabel("Induction Crash Rate (%)", fontsize=8)
    ax2.set_title("B. Crash Rate by Stiffness Quartile", fontsize=8, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax2.set_facecolor('white')

    # n labels on bars
    for xi, (n, r) in enumerate(zip(ns, crash_rates)):
        if not np.isnan(r):
            ax2.text(xi, r + 3, f'n={n}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout(w_pad=1.5)
    fig.savefig(FIG_DIR / "fig7_stiffness_stratified.png", dpi=300)
    plt.close()
    log.info("Fig 7 saved.")
else:
    log.warning(f"Too few cases with stiffness score ({len(stiff_df)}), skipping Fig 7.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 1: BASELINE CHARACTERISTICS
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Generating Table 1: Baseline characteristics...")

# Merge with clinical data for more variables (only columns not already in merged)
clin_sub = clin_df[['caseid','opname','optype','department']].copy()
t1_df = merged.merge(clin_sub, on='caseid', how='left')

# All clinical vars (age, sex, bmi, asa, preop_htn, preop_dm) already in merged from vascular_features

crash_g    = t1_df[t1_df['crash_30'] == 1]
nocrash_g  = t1_df[t1_df['crash_30'] == 0]
overall    = t1_df

def fmt_continuous(df, col, label, digits=1):
    """Mean ± SD for a column."""
    v = df[col].dropna()
    if len(v) == 0:
        return {'Variable': label, 'Overall': 'N/A', 'Crash': 'N/A', 'No Crash': 'N/A', 'p-value': 'N/A'}
    m_o = overall[col].dropna()
    m_c = crash_g[col].dropna()
    m_n = nocrash_g[col].dropna()
    # t-test
    if len(m_c) > 1 and len(m_n) > 1:
        _, p = stats.ttest_ind(m_c, m_n, equal_var=False)
        p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
    else:
        p_str = 'N/A'
    def fmt(s):
        if len(s) == 0: return 'N/A'
        return f'{s.mean():.{digits}f} ± {s.std():.{digits}f}'
    return {'Variable': label,
            'Overall':  f'{fmt(m_o)} (N={len(m_o)})',
            'Crash':    f'{fmt(m_c)} (N={len(m_c)})',
            'No Crash': f'{fmt(m_n)} (N={len(m_n)})',
            'p-value':  p_str}

def fmt_categorical(df, col, label, val=1, val_name=None):
    """Count (%) for a binary / categorical column."""
    # find column (no _clin suffix anymore)
    col_use = None
    for c in [col, col+'_x']:
        if c in t1_df.columns:
            col_use = c; break
    if col_use is None:
        return {'Variable': label, 'Overall': 'N/A', 'Crash': 'N/A', 'No Crash': 'N/A', 'p-value': 'N/A'}
    def n_pct(sub):
        v = sub[col_use].dropna()
        n = (v == val).sum()
        pct = 100.0 * n / max(len(v), 1)
        return f'{n} ({pct:.1f}%)'
    c_ = crash_g[col_use].dropna()
    n_ = nocrash_g[col_use].dropna()
    if len(c_) > 0 and len(n_) > 0:
        from scipy.stats import chi2_contingency
        ct = pd.crosstab(t1_df[col_use], t1_df['crash_30'])
        if ct.shape == (2,2):
            _, p, _, _ = chi2_contingency(ct)
            p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
        else:
            p_str = 'N/A'
    else:
        p_str = 'N/A'
    display = label if val_name is None else f'{label}: {val_name}'
    return {'Variable': display,
            'Overall':  n_pct(t1_df),
            'Crash':    n_pct(crash_g),
            'No Crash': n_pct(nocrash_g),
            'p-value':  p_str}

# --- Build rows ---
rows.append({'Variable': f'N', 'Overall': str(len(t1_df)),
             'Crash': str(len(crash_g)), 'No Crash': str(len(nocrash_g)), 'p-value': ''})
rows.append(fmt_continuous(t1_df, 'age', 'Age (years)'))
if 'sex' in t1_df.columns:
    rows.append(fmt_categorical(t1_df, 'sex', 'Male sex', val='M'))
rows.append(fmt_continuous(t1_df, 'weight', 'Weight (kg)', digits=1))
rows.append(fmt_continuous(t1_df, 'height', 'Height (cm)', digits=1))
rows.append(fmt_continuous(t1_df, 'bmi', 'BMI (kg/m²)', digits=1))
rows.append(fmt_continuous(t1_df, 'asa', 'ASA Physical Status'))
rows.append(fmt_categorical(t1_df, 'preop_htn', 'Hypertension', val=1))
rows.append(fmt_categorical(t1_df, 'preop_dm',  'Diabetes mellitus', val=1))
rows.append({'Variable': '— Haemodynamics', 'Overall': '', 'Crash': '', 'No Crash': '', 'p-value': ''})
rows.append(fmt_continuous(t1_df, 'baseline_map', 'Pre-induction MAP (mmHg)'))
rows.append(fmt_continuous(t1_df, 'nadir_map', 'Nadir MAP during induction (mmHg)'))
rows.append(fmt_continuous(t1_df, 'drop_pct', 'MAP drop (% of baseline)', digits=1))
rows.append({'Variable': '— PPG Features', 'Overall': '', 'Crash': '', 'No Crash': '', 'p-value': ''})
rows.append(fmt_continuous(t1_df, 'ri_mean_clean', 'Reflection Index (RI)', digits=3))
rows.append(fmt_continuous(t1_df, 'ppg_amp_clean', 'PPG Amplitude', digits=2))
if 'pi_clean' in t1_df.columns:
    rows.append(fmt_continuous(t1_df, 'pi_clean', 'Perfusion Index (PI)', digits=2))

table1 = pd.DataFrame(rows)
table1.to_csv(MET_DIR / "table1_baseline.csv", index=False)
log.info(f"Table 1 saved: {len(table1)} rows.")
print("\n" + "="*80)
print("TABLE 1: BASELINE CHARACTERISTICS")
print("="*80)
print(table1.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Generating Table 2: Model performance...")

perf_df = pd.read_csv(MET_DIR / "model_performance.csv")

# Reformat for paper
model_rename = {
    'Naive_MAP<80':   'MAP<80 clinical rule',
    'LR_Model_A':     'LR Model A (clinical only)',
    'LR_Model_B':     'LR Model B (+vascular features)',
    'LR_Model_C':     'LR Model C (+PI)',
    'XGBoost_B':      'XGBoost (+vascular features)',
}
perf_df['Model'] = perf_df['Model'].map(model_rename).fillna(perf_df['Model'])
perf_df.columns = ['Model', 'AUROC', 'AUROC 95% CI', 'AUPRC', 'AUPRC 95% CI', 'Brier Score']
perf_df['Brier Score'] = perf_df['Brier Score'].apply(lambda x: f'{x:.4f}')

perf_df.to_csv(MET_DIR / "table2_performance.csv", index=False)
log.info("Table 2 saved.")
print("\n" + "="*80)
print("TABLE 2: MODEL PERFORMANCE")
print("="*80)
print(perf_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\n" + "="*60)
log.info("=== PHASE 7 COMPLETE ===")
log.info("Figures generated:")
for fname in ['fig1_consort_flowchart.png', 'fig2_map_trajectories.png',
              'fig6_shap_summary.png', 'fig7_stiffness_stratified.png']:
    fpath = FIG_DIR / fname
    if fpath.exists():
        size_kb = fpath.stat().st_size // 1024
        log.info(f"  ✓ {fname} ({size_kb} KB)")
    else:
        log.warning(f"  ✗ {fname} NOT GENERATED")
log.info("Tables generated:")
for fname in ['table1_baseline.csv', 'table2_performance.csv']:
    fpath = MET_DIR / fname
    if fpath.exists():
        log.info(f"  ✓ {fname}")
    else:
        log.warning(f"  ✗ {fname} NOT GENERATED")
log.info(f"Log: {log_file}")
