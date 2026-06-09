#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step8c_fix_fig6_shap.py | Topic: 10 | Purpose: Regenerate Fig 6 SHAP plot
Fix: use sklearn GradientBoostingClassifier + shap.TreeExplainer (bypasses XGBoost 3.x bug)
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
TOPIC_DIR = Path(__file__).resolve().parents[1]
PROC_DIR  = TOPIC_DIR / "data" / "processed"
FIG_DIR   = TOPIC_DIR / "outputs" / "figures"
RAW_DATA  = Path("vitaldb_data")  # local VitalDB physionet.org root; override as needed
assert not str(FIG_DIR.resolve()).startswith(str(RAW_DATA.resolve()))

# ── Load data ──────────────────────────────────────────────────────────────────
feat_df    = pd.read_parquet(PROC_DIR / "vascular_features.parquet")
outcome_df = pd.read_parquet(PROC_DIR / "outcome_labels.parquet")
merged = feat_df.merge(
    outcome_df[['caseid', 'crash_40', 'crash_20', 'crash_absolute']],
    on='caseid', how='left'
)
print(f"Dataset: {len(merged)} rows, crash_30={int(merged['crash_30'].sum())} ({merged['crash_30'].mean()*100:.1f}%)")

# ── Features ───────────────────────────────────────────────────────────────────
feat_b = ['age', 'bmi', 'asa', 'preop_htn', 'preop_dm',
          'baseline_map', 'ri_mean_clean', 'ppg_amp_clean']

FEAT_LABELS = {
    'age':           'Age (years)',
    'bmi':           'BMI (kg/m²)',
    'asa':           'ASA Physical Status',
    'preop_htn':     'Pre-op Hypertension',
    'preop_dm':      'Pre-op Diabetes Mellitus',
    'baseline_map':  'Baseline MAP (mmHg)',
    'ri_mean_clean': 'Reflection Index (PPG)',
    'ppg_amp_clean': 'PPG Amplitude',
}

df = merged[feat_b + ['crash_30']].dropna(subset=['crash_30']).copy()
X_all = df[feat_b]
y_all = df['crash_30']

imp   = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imp.fit_transform(X_all), columns=feat_b)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_imp, y_all, test_size=0.3, stratify=y_all, random_state=42
)
print(f"Train={len(X_tr)}, Test={len(X_te)}")

# ── Train GradientBoostingClassifier (sklearn — compatible with shap 0.49) ─────
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb.fit(X_tr, y_tr)
print(f"GBM trained. Test AUROC (approx): ", end='')
from sklearn.metrics import roc_auc_score
print(f"{roc_auc_score(y_te, gb.predict_proba(X_te)[:,1]):.3f}")

# ── SHAP via TreeExplainer ─────────────────────────────────────────────────────
explainer  = shap.TreeExplainer(gb)
shap_raw   = explainer.shap_values(X_te)

# sklearn binary GBM: shap_values() may return a list [neg_class, pos_class]
# or a single 2D array depending on shap version. Normalise to pos-class array.
if isinstance(shap_raw, list):
    shap_array = shap_raw[1]          # positive class (crash=1)
    print(f"SHAP values (list→pos class): shape={shap_array.shape}")
else:
    shap_array = shap_raw
    print(f"SHAP values (array): shape={shap_array.shape}")

# ── Publication-quality beeswarm plot ─────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        8,
    'axes.labelsize':   9,
    'axes.titlesize':   9,
    'axes.linewidth':   0.8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# Rename columns for display labels
X_te_display = X_te.rename(columns=FEAT_LABELS)

fig = plt.figure(figsize=(5.5, 3.8))
shap.summary_plot(
    shap_array, X_te_display,
    plot_type='dot',
    max_display=8,
    color_bar_label='Feature value\n(normalised)',
    show=False,
    plot_size=None,
)
ax = plt.gca()
ax.set_xlabel("SHAP value  (impact on log-odds of induction crash)", fontsize=8)
ax.set_title(
    "Figure 6.  Feature Importance — SHAP Analysis (Gradient Boosting Model)",
    fontsize=8.5, fontweight='bold', pad=6
)
ax.axvline(0, color='#888888', lw=0.6, ls='--', zorder=0)

plt.tight_layout()
out_path = FIG_DIR / "fig6_shap_summary.png"
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFig 6 saved → {out_path}")

# ── Also print feature importance table ───────────────────────────────────────
mean_abs = np.abs(shap_array).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feat_b,
    'display_name': [FEAT_LABELS[f] for f in feat_b],
    'mean_abs_shap': mean_abs
}).sort_values('mean_abs_shap', ascending=False)

print("\nSHAP Feature Importance (mean |SHAP value|):")
print(importance_df[['display_name', 'mean_abs_shap']].to_string(index=False))

importance_df.to_csv(
    TOPIC_DIR / "outputs" / "metrics" / "shap_importance.csv", index=False
)
print("SHAP importance CSV saved.")
