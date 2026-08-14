#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step13_fig3_arc_nested_roc.py | Topic: 10
Purpose: Regenerate main-text Figure 3 = ARC nested logistic-regression ROC
         (M0-M4) on the canonical seed-42 70/30 split (test n=159), matching
         the manuscript caption (M1=0.946, M3=0.952). Deterministic; median
         imputation for RI / amplitude so all 528 ARC cases are retained.
         Writes the current ARC nested-model ROC figure.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

np.random.seed(42)
PROC = (Path(__file__).resolve().parents[1] / "data/processed")
FIG  = (Path(__file__).resolve().parents[1] / "outputs/figures")

feat = pd.read_parquet(PROC / "vascular_features.parquet")
outcome = pd.read_parquet(PROC / "outcome_labels.parquet")
feat = feat.merge(outcome[['caseid', 'crash_absolute']], on='caseid', how='left')

arc = feat[feat['drop_pct'].notna() & np.isfinite(feat['drop_pct']) & (feat['drop_pct'] > -500)].copy()
CLIN = ['age', 'bmi', 'asa', 'preop_htn', 'preop_dm']
y_all = arc['crash_30'].values
tr_idx, te_idx = train_test_split(arc.index.values, test_size=0.30,
                                  stratify=y_all, random_state=42)
tr, te = arc.loc[tr_idx], arc.loc[te_idx]
y_tr, y_te = tr['crash_30'].values, te['crash_30'].values
print(f"ARC: N={len(arc)}, Train={len(tr)}, Test={len(te)}, Test events={int(y_te.sum())}")

def pipe():
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('lr', LogisticRegression(max_iter=1000, random_state=42))])

MODELS = {
    'M0: Clinical only':            CLIN,
    'M1: + Baseline MAP':           CLIN + ['baseline_map'],
    'M2: + RI only':                CLIN + ['ri_mean_clean'],
    'M3: + MAP + RI':               CLIN + ['baseline_map', 'ri_mean_clean'],
    'M4: + MAP + RI + Amp':         CLIN + ['baseline_map', 'ri_mean_clean', 'ppg_amp_clean'],
}
COLORS = {'M0: Clinical only':'#888888','M1: + Baseline MAP':'#4C9BD5',
          'M2: + RI only':'#E8A33D','M3: + MAP + RI':'#2E8B72',
          'M4: + MAP + RI + Amp':'#C8502A'}

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':300,'savefig.dpi':300})
fig, ax = plt.subplots(figsize=(5.5, 4.6))
aucs = {}
for name, feats in MODELS.items():
    p = pipe(); p.fit(tr[feats].values, y_tr)
    prob = p.predict_proba(te[feats].values)[:, 1]
    a = roc_auc_score(y_te, prob); aucs[name] = a
    fpr, tpr, _ = roc_curve(y_te, prob)
    lw = 2.4 if name.startswith(('M1','M3')) else 1.6
    ax.plot(fpr, tpr, color=COLORS[name], lw=lw, label=f"{name}  (AUROC={a:.3f})")
    print(f"  {name}: AUROC={a:.3f}")
ax.plot([0,1],[0,1],'--',color='#bbbbbb',lw=1)
ax.set_xlabel('1 − Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title('Nested logistic-regression models (ARC, test $n$=159)',
             fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=7.8, frameon=True)
ax.set_xlim(-0.01,1.01); ax.set_ylim(-0.01,1.02)
fig.tight_layout()
out = FIG / "fig3_arc_nested_roc.png"
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(str(out)[:-4] + ".pdf", format='pdf', bbox_inches='tight')
print(f"\nSaved: {out}  (+ .pdf vector)")
print("Caption check -> M1={:.3f}  M3={:.3f}  M0={:.3f}".format(
    aucs['M1: + Baseline MAP'], aucs['M3: + MAP + RI'], aucs['M0: Clinical only']))
