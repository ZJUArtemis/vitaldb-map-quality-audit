#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step12_repeated_cv.py | Topic: 10
Purpose: Robustness of the nested-model decomposition to the data split.
         Replaces the single 70/30 split (seed=42) with REPEATED stratified
         Monte-Carlo cross-validation (200 random 70/30 splits) and reports the
         distribution (mean, SD, 2.5-97.5 percentile) of held-out test AUROC for
         M0-M4, the validity-indicator model, and the key incremental gains
         (M0->M1, M1->M3, M3->M4). Mirrors step9's pipeline exactly. Also repeats
         the independent NIBP reference cohort (M0/M1/M3).
"""
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

WORK = Path(__file__).resolve().parents[1]
PROC = WORK / "data" / "processed"
MET  = WORK / "outputs" / "metrics"
MET.mkdir(parents=True, exist_ok=True)

N_REPEAT = 200
TEST_FRAC = 0.30

def make_pipe():
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler()),
                     ('lr',  LogisticRegression(penalty='l2', C=1.0,
                                                max_iter=1000, random_state=42))])

def repeated_cv(df, specs, n_repeat=N_REPEAT):
    """Return dict: model name -> array of held-out AUROC over n_repeat splits."""
    y = df['_y'].values.astype(float)
    out = {name: np.full(n_repeat, np.nan) for name, _ in specs}
    for r in range(n_repeat):
        itr, ite = train_test_split(np.arange(len(df)), test_size=TEST_FRAC,
                                    stratify=y, random_state=r)
        for name, feats in specs:
            X = df[feats].values.astype(float)
            pipe = make_pipe(); pipe.fit(X[itr], y[itr])
            p = pipe.predict_proba(X[ite])[:, 1]
            try:
                out[name][r] = roc_auc_score(y[ite], p)
            except ValueError:
                pass
    return out

def summ(a):
    a = a[~np.isnan(a)]
    return dict(mean=round(float(np.mean(a)), 3), sd=round(float(np.std(a, ddof=1)), 3),
                p2_5=round(float(np.percentile(a, 2.5)), 3),
                p97_5=round(float(np.percentile(a, 97.5)), 3), n=int(len(a)))

results = {"n_repeat": N_REPEAT, "test_frac": TEST_FRAC}

# ── ARC cohort (= step9 feat_clean) ────────────────────────────────────────────
feat = pd.read_parquet(PROC / "vascular_features.parquet")
arc = feat[~np.isinf(feat['drop_pct']) & (feat['drop_pct'] > -500)].copy()
arc['_y'] = arc['crash_30'].astype(float)
arc['map_valid'] = (arc['baseline_map'] >= 20).astype(float)
CLIN = ['age', 'bmi', 'asa', 'preop_htn', 'preop_dm']
arc_specs = [
    ("M0", CLIN),
    ("M1", CLIN + ['baseline_map']),
    ("M2", CLIN + ['ri_mean_clean']),
    ("M3", CLIN + ['baseline_map', 'ri_mean_clean']),
    ("M4", CLIN + ['baseline_map', 'ri_mean_clean', 'ppg_amp_clean']),
    ("Mv", CLIN + ['map_valid']),
]
print(f"ARC n={len(arc)}, events={int(arc['_y'].sum())}")
arc_auc = repeated_cv(arc, arc_specs)
results['ARC'] = {k: summ(v) for k, v in arc_auc.items()}
results['ARC_delta'] = {
    "M0_to_M1": summ(arc_auc['M1'] - arc_auc['M0']),
    "M1_to_M3": summ(arc_auc['M3'] - arc_auc['M1']),
    "M3_to_M4": summ(arc_auc['M4'] - arc_auc['M3']),
    "M0_to_Mv": summ(arc_auc['Mv'] - arc_auc['M0']),
}

# ── NIBP-corrected cohort (= step11 modelling cohort) ──────────────────────────
nibp = pd.read_parquet(PROC / "outcome_labels_nibp.parquet")
v2 = pd.read_parquet(PROC / "vascular_features_v2.parquet")
keep = ['caseid'] + [c for c in ['age', 'bmi', 'asa', 'preop_htn', 'preop_dm',
                                 'ri_mean_clean', 'ppg_amp_clean'] if c in v2.columns]
nb = nibp.merge(v2[keep], on='caseid', how='inner')
nb = nb[nb['crash_30'].notna()].copy()
nb['baseline_map'] = nb['nibp_baseline']
nb['_y'] = nb['crash_30'].astype(float)
nibp_specs = [("M0", CLIN), ("M1", CLIN + ['baseline_map']),
              ("M3", CLIN + ['baseline_map', 'ri_mean_clean'])]
print(f"NIBP n={len(nb)}, events={int(nb['_y'].sum())}")
nb_auc = repeated_cv(nb, nibp_specs)
results['NIBP'] = {k: summ(v) for k, v in nb_auc.items()}
results['NIBP_delta'] = {
    "M0_to_M1": summ(nb_auc['M1'] - nb_auc['M0']),
    "M1_to_M3": summ(nb_auc['M3'] - nb_auc['M1']),
}

out = MET / "repeated_cv_robustness.json"
out.write_text(json.dumps(results, indent=2))

# ── Console summary ────────────────────────────────────────────────────────────
def line(tag, s):
    print(f"  {tag:<10} {s['mean']:.3f} +/- {s['sd']:.3f}  "
          f"[{s['p2_5']:.3f}, {s['p97_5']:.3f}]")
print(f"\n=== ARC repeated MC-CV ({N_REPEAT} splits) ===")
for k in ['M0', 'M1', 'M2', 'M3', 'M4', 'Mv']:
    line(k, results['ARC'][k])
print("  -- incremental --")
for k, v in results['ARC_delta'].items():
    line(k, v)
print(f"\n=== NIBP repeated MC-CV ({N_REPEAT} splits) ===")
for k in ['M0', 'M1', 'M3']:
    line(k, results['NIBP'][k])
for k, v in results['NIBP_delta'].items():
    line(k, v)
print(f"\nSaved: {out}")
