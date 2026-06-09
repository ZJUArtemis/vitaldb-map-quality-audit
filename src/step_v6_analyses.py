#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step_v6_analyses.py | Topic: 10
Purpose: Paired bootstrap CIs for delta-AUROC; optimism-corrected AUROC;
         missingness summary table; PPG amp by group.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)

PROC = (Path(__file__).resolve().parents[1] / "data" / "processed")
MET  = (Path(__file__).resolve().parents[1] / "outputs" / "metrics")

# ── Load data ─────────────────────────────────────────────────────────────────
feat = pd.read_parquet(PROC / "vascular_features.parquet")
outcome = pd.read_parquet(PROC / "outcome_labels.parquet")
feat = feat.merge(outcome[['caseid','crash_absolute']], on='caseid', how='left')
arc = feat[feat['drop_pct'].notna() & np.isfinite(feat['drop_pct']) & (feat['drop_pct'] > -500)].copy()

CLIN = ['age','bmi','asa','preop_htn','preop_dm']
y_all = arc['crash_30'].values
X_tr_idx, X_te_idx = train_test_split(arc.index.values, test_size=0.30, stratify=y_all, random_state=42)
tr = arc.loc[X_tr_idx]; te = arc.loc[X_te_idx]
y_tr = tr['crash_30'].values; y_te = te['crash_30'].values
print(f"ARC: N={len(arc)}, Train={len(tr)}, Test={len(te)}, Events(test)={y_te.sum()}")

def build_pipe():
    return Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('sc',StandardScaler()),
                     ('lr',LogisticRegression(max_iter=1000,random_state=42))])

def fit_proba(feats, tr_df, te_df, y_tr):
    p = build_pipe()
    p.fit(tr_df[feats].values, y_tr)
    return p.predict_proba(te_df[feats].values)[:,1]

# ── Fit all models on ARC ─────────────────────────────────────────────────────
proba = {}
proba['M0']  = fit_proba(CLIN,                                        tr, te, y_tr)
proba['M1']  = fit_proba(CLIN+['baseline_map'],                       tr, te, y_tr)
proba['M2']  = fit_proba(CLIN+['ri_mean_clean'],                      tr, te, y_tr)
proba['M3']  = fit_proba(CLIN+['baseline_map','ri_mean_clean'],       tr, te, y_tr)
proba['M4']  = fit_proba(CLIN+['baseline_map','ri_mean_clean','ppg_amp_clean'], tr, te, y_tr)

arc['map_valid_20'] = (arc['baseline_map'] >= 20).astype(float)
tr = arc.loc[X_tr_idx]; te = arc.loc[X_te_idx]  # refresh after adding column
proba['Mv']  = fit_proba(CLIN+['map_valid_20'],                       tr, te, y_tr)
proba['M1v'] = fit_proba(CLIN+['map_valid_20','baseline_map'],        tr, te, y_tr)

# ── Paired bootstrap CI for ΔAUROC ───────────────────────────────────────────
def paired_delta_ci(y_true, p_a, p_b, n=2000, seed=0):
    """Paired bootstrap 95% CI for AUC(A) - AUC(B)."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        da = roc_auc_score(y_true[idx], p_a[idx])
        db = roc_auc_score(y_true[idx], p_b[idx])
        deltas.append(da - db)
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas,2.5)), float(np.percentile(deltas,97.5))

print("\n=== PAIRED BOOTSTRAP DELTA-AUROC (2000 iterations) ===")
pairs = [
    ('M1 - M0', 'M1', 'M0'),
    ('M3 - M1', 'M3', 'M1'),
    ('M4 - M3', 'M4', 'M3'),
    ('Mv - M0', 'Mv', 'M0'),
    ('M1v - Mv', 'M1v', 'Mv'),
    ('M1 - M1v', 'M1', 'M1v'),
]
delta_results = {}
for label, a, b in pairs:
    mean_d, lo, hi = paired_delta_ci(y_te, proba[a], proba[b])
    point = roc_auc_score(y_te, proba[a]) - roc_auc_score(y_te, proba[b])
    print(f"  {label}: point={point:+.3f}, boot mean={mean_d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    delta_results[label] = (point, lo, hi)

# ── Complete-case paired bootstrap ───────────────────────────────────────────
print("\n=== COMPLETE-CASE PAIRED BOOTSTRAP ===")
cc = arc[arc['ri_mean_clean'].notna() & arc['ppg_amp_clean'].notna()].copy()
y_cc = cc['crash_30'].values
X_cc_tr, X_cc_te = train_test_split(cc.index.values, test_size=0.30, stratify=y_cc, random_state=42)
tr_cc = cc.loc[X_cc_tr]; te_cc = cc.loc[X_cc_te]
y_tr_cc = tr_cc['crash_30'].values; y_te_cc = te_cc['crash_30'].values
print(f"CC: N={len(cc)}, Test={len(te_cc)}, Events={y_te_cc.sum()}")

p_cc_m1 = fit_proba(CLIN+['baseline_map'],                      tr_cc, te_cc, y_tr_cc)
p_cc_m3 = fit_proba(CLIN+['baseline_map','ri_mean_clean'],      tr_cc, te_cc, y_tr_cc)
p_cc_m4 = fit_proba(CLIN+['baseline_map','ri_mean_clean','ppg_amp_clean'], tr_cc, te_cc, y_tr_cc)

mean_d, lo, hi = paired_delta_ci(y_te_cc, p_cc_m3, p_cc_m1)
point = roc_auc_score(y_te_cc, p_cc_m3) - roc_auc_score(y_te_cc, p_cc_m1)
print(f"  CC: M3 - M1: point={point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")

mean_d, lo, hi = paired_delta_ci(y_te_cc, p_cc_m4, p_cc_m3)
point = roc_auc_score(y_te_cc, p_cc_m4) - roc_auc_score(y_te_cc, p_cc_m3)
print(f"  CC: M4 - M3: point={point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")

# ── Optimism-corrected AUROC (on full ARC, not test set) ──────────────────────
print("\n=== OPTIMISM-CORRECTED AUROC (full ARC, 200 bootstrap samples) ===")
X_full = arc.copy()
y_full = X_full['crash_30'].values

def optimism_corrected(feats, df, y, n_boot=200, seed=42):
    """Harrell optimism correction: apparent - mean(auc_boot - auc_orig)."""
    rng = np.random.default_rng(seed)
    # Apparent AUROC on full data
    p_app = build_pipe()
    X_arr = df[feats].values
    p_app.fit(X_arr, y)
    auc_app = roc_auc_score(y, p_app.predict_proba(X_arr)[:,1])
    # Bootstrap optimism
    optimisms = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        X_b = X_arr[idx]; y_b = y[idx]
        if len(np.unique(y_b)) < 2: continue
        p_b = build_pipe()
        p_b.fit(X_b, y_b)
        auc_b  = roc_auc_score(y_b, p_b.predict_proba(X_b)[:,1])
        auc_bo = roc_auc_score(y, p_b.predict_proba(X_arr)[:,1])
        optimisms.append(auc_b - auc_bo)
    opt = float(np.mean(optimisms))
    return auc_app, auc_app - opt, opt

for label, feats in [
    ('M1', CLIN+['baseline_map']),
    ('M4', CLIN+['baseline_map','ri_mean_clean','ppg_amp_clean']),
]:
    auc_app, auc_corr, opt = optimism_corrected(feats, X_full, y_full)
    print(f"  {label}: apparent={auc_app:.3f}, optimism={opt:+.3f}, corrected={auc_corr:.3f}")
    print(f"    Note: test-set AUROC is NOT directly comparable; apparent is on full ARC training data.")

# ── Missingness summary table ─────────────────────────────────────────────────
print("\n=== MISSINGNESS SUMMARY TABLE ===")
full_c = feat.copy()
arc_c = feat[feat['drop_pct'].notna() & np.isfinite(feat['drop_pct']) & (feat['drop_pct'] > -500)].copy()
cc_c  = arc_c[arc_c['ri_mean_clean'].notna() & arc_c['ppg_amp_clean'].notna()].copy()

rows = []
for vname, col in [('baseline MAP','baseline_map'),('RI','ri_mean_clean'),('PPG amplitude','ppg_amp_clean'),
                   ('ASA','asa'),('Age','age'),('BMI','bmi')]:
    r = {'Variable': vname}
    for name, df in [('Full cohort (N=909)',full_c),('ARC (N=528)',arc_c),('Complete-case (N=391)',cc_c)]:
        n_m = df[col].isna().sum()
        r[name] = f"{n_m} ({n_m/len(df)*100:.1f}%)" if n_m>0 else "0"
    rows.append(r)
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))
tbl.to_csv(MET / "missingness_by_set.csv", index=False)

print("\n=== Done ===")
