#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step8b_tables_shap.py | Topic: 10
Purpose: Generate Table 1, Table 2, and Fig 6 (SHAP) — standalone
"""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency

RAW_DATA  = Path("/home/lxk/vitaldb/physionet.org")
PROC_DIR  = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability/data/processed")
FIG_DIR   = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability/outputs/figures")
MET_DIR   = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability/outputs/metrics")

# ── Load data ──────────────────────────────────────────────────────────────────
feat   = pd.read_parquet(PROC_DIR / "vascular_features.parquet")
oc     = pd.read_parquet(PROC_DIR / "outcome_labels.parquet")
merged = feat.merge(oc[['caseid','crash_40','crash_20','crash_absolute']], on='caseid', how='left')

clin_path = RAW_DATA / "files" / "vitaldb" / "1.0.0" / "clinical_data.csv"
clin_df   = pd.read_csv(clin_path)
clin_df['age']    = pd.to_numeric(clin_df['age'],    errors='coerce')
clin_df['weight'] = pd.to_numeric(clin_df['weight'], errors='coerce')
clin_df['height'] = pd.to_numeric(clin_df['height'], errors='coerce')
clin_df['bmi']    = pd.to_numeric(clin_df['bmi'],    errors='coerce')
clin_df['asa']    = pd.to_numeric(clin_df['asa'],    errors='coerce')

# Merge only extra clinical columns not already in merged
extra_cols = [c for c in ['opname','optype','department'] if c in clin_df.columns]
t1_df = merged.merge(clin_df[['caseid'] + extra_cols], on='caseid', how='left')

crash_g   = t1_df[t1_df['crash_30'] == 1]
nocrash_g = t1_df[t1_df['crash_30'] == 0]
print(f"Total: {len(t1_df)}  Crash: {len(crash_g)}  No-crash: {len(nocrash_g)}")

# ── Helper functions ───────────────────────────────────────────────────────────
def fmt_cont(col, label, digits=1):
    o = t1_df[col].dropna()
    c = crash_g[col].dropna()
    n = nocrash_g[col].dropna()
    if len(c) < 2 or len(n) < 2:
        p_str = 'N/A'
    else:
        _, p = stats.ttest_ind(c, n, equal_var=False)
        p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
    def s(v): return f'{v.mean():.{digits}f} ± {v.std():.{digits}f}' if len(v) else 'N/A'
    return {'Variable': label,
            'Overall':  f'{s(o)} (N={len(o)})',
            'Crash':    f'{s(c)} (N={len(c)})',
            'No Crash': f'{s(n)} (N={len(n)})',
            'p-value':  p_str}

def fmt_cat(col, label, val=1):
    if col not in t1_df.columns:
        return {'Variable': label, 'Overall': 'N/A', 'Crash': 'N/A', 'No Crash': 'N/A', 'p-value': 'N/A'}
    def np_(sub):
        v = sub[col].dropna()
        k = (v == val).sum()
        return f'{k} ({100*k/max(len(v),1):.1f}%)'
    c_ = crash_g[col].dropna()
    n_ = nocrash_g[col].dropna()
    try:
        ct = pd.crosstab(t1_df[col], t1_df['crash_30'])
        if ct.shape == (2, 2):
            _, p, _, _ = chi2_contingency(ct)
            p_str = f'{p:.3f}' if p >= 0.001 else '<0.001'
        else:
            p_str = 'N/A'
    except Exception:
        p_str = 'N/A'
    return {'Variable': label,
            'Overall':  np_(t1_df),
            'Crash':    np_(crash_g),
            'No Crash': np_(nocrash_g),
            'p-value':  p_str}

# ── TABLE 1 ────────────────────────────────────────────────────────────────────
rows = []
rows.append({'Variable': 'N', 'Overall': str(len(t1_df)),
             'Crash': str(len(crash_g)), 'No Crash': str(len(nocrash_g)), 'p-value': ''})
rows.append(fmt_cont('age',    'Age (years)'))
rows.append(fmt_cat('sex',     'Male sex', val='M'))
rows.append(fmt_cont('weight', 'Weight (kg)'))
rows.append(fmt_cont('height', 'Height (cm)'))
rows.append(fmt_cont('bmi',    'BMI (kg/m²)'))
rows.append(fmt_cont('asa',    'ASA Physical Status', digits=1))
rows.append(fmt_cat('preop_htn', 'Hypertension'))
rows.append(fmt_cat('preop_dm',  'Diabetes mellitus'))
rows.append({'Variable': '— Haemodynamics', 'Overall': '', 'Crash': '', 'No Crash': '', 'p-value': ''})

# Clean haemodynamic values before reporting
t1_df['baseline_map_clean'] = t1_df['baseline_map'].where(
    (t1_df['baseline_map'] > 40) & (t1_df['baseline_map'] < 200))
t1_df['nadir_map_clean'] = t1_df['nadir_map'].where(
    (t1_df['nadir_map'] > 20) & (t1_df['nadir_map'] < 200))
t1_df['drop_pct_clean'] = t1_df['drop_pct'].where(
    (t1_df['drop_pct'] > 0) & (t1_df['drop_pct'] < 100))
crash_g   = t1_df[t1_df['crash_30'] == 1]
nocrash_g = t1_df[t1_df['crash_30'] == 0]

rows.append(fmt_cont('baseline_map_clean', 'Pre-induction MAP (mmHg)†'))
rows.append(fmt_cont('nadir_map_clean',    'Nadir MAP during induction (mmHg)†'))
rows.append(fmt_cont('drop_pct_clean',     'MAP drop (% of baseline)†', digits=1))
rows.append({'Variable': '— PPG Features', 'Overall': '', 'Crash': '', 'No Crash': '', 'p-value': ''})
rows.append(fmt_cont('ri_mean_clean',  'Reflection Index (RI)', digits=3))
rows.append(fmt_cont('ppg_amp_clean',  'PPG Amplitude', digits=2))
rows.append(fmt_cont('pi_clean',       'Perfusion Index (PI)', digits=2))

table1 = pd.DataFrame(rows)
table1.to_csv(MET_DIR / "table1_baseline.csv", index=False)
print("\nTABLE 1:")
print(table1.to_string(index=False))

# ── TABLE 2 ────────────────────────────────────────────────────────────────────
perf = pd.read_csv(MET_DIR / "model_performance.csv")
rename = {
    'Naive_MAP<80':  'MAP<80 clinical rule',
    'LR_Model_A':    'LR Model A (clinical only)',
    'LR_Model_B':    'LR Model B (+vascular features)',
    'LR_Model_C':    'LR Model C (+PI)',
    'XGBoost_B':     'XGBoost (+vascular features)',
}
perf['Model'] = perf['Model'].map(rename).fillna(perf['Model'])
perf.columns  = ['Model','AUROC','AUROC 95% CI','AUPRC','AUPRC 95% CI','Brier Score']
perf['Brier Score'] = perf['Brier Score'].apply(lambda x: f'{float(x):.4f}')
perf.to_csv(MET_DIR / "table2_performance.csv", index=False)
print("\nTABLE 2:")
print(perf.to_string(index=False))

# ── FIG 6: SHAP (GradientBoostingClassifier) ──────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import shap

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'axes.labelsize':9,'figure.dpi':300,'savefig.dpi':300,
                     'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})

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

feat_b = list(FEAT_LABELS.keys())
df_s   = merged[feat_b + ['crash_30']].dropna(subset=['crash_30'])
X      = df_s[feat_b]; y = df_s['crash_30']
imp    = SimpleImputer(strategy='median')
X_imp  = pd.DataFrame(imp.fit_transform(X), columns=feat_b)
X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.3, stratify=y, random_state=42)

gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                 learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr, y_tr)
print(f"\nGBM trained. Test AUROC check...")

explainer  = shap.TreeExplainer(gb)
shap_vals  = explainer.shap_values(X_te)
print(f"SHAP values shape: {shap_vals.shape}")

X_te_disp = X_te.rename(columns=FEAT_LABELS)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
shap.summary_plot(shap_vals, X_te_disp, plot_type='dot', max_display=8,
                  color_bar_label='Feature value\n(normalised)',
                  show=False, plot_size=None)
ax = plt.gca()
ax.set_xlabel("SHAP value (impact on log-odds of induction crash)", fontsize=8)
ax.set_title("Figure 6. SHAP Feature Importance — Gradient Boosting Model",
             fontsize=8.5, fontweight='bold')
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_shap_summary.png", dpi=300)
plt.close()
print("Fig 6 saved.")

print("\n=== DONE ===")
print(f"Table 1: {MET_DIR}/table1_baseline.csv")
print(f"Table 2: {MET_DIR}/table2_performance.csv")
print(f"Fig 6:   {FIG_DIR}/fig6_shap_summary.png")
