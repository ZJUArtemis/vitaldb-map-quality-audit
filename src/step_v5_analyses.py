#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step_v5_analyses.py | Topic: 10
Purpose: Compute CIs for validity-indicator, complete-case, full-cohort analyses;
         validity threshold sensitivity (10/20/30 mmHg);
         baseline/nadir MAP distribution figure for manuscript v5.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
rng = np.random.default_rng(42)

PROC = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability/data/processed")
FIG  = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability/outputs/figures")
MET  = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability/outputs/metrics")

# ── Load data ─────────────────────────────────────────────────────────────────
feat = pd.read_parquet(PROC / "vascular_features.parquet")
outcome = pd.read_parquet(PROC / "outcome_labels.parquet")
feat = feat.merge(outcome[['caseid', 'crash_absolute']], on='caseid', how='left')

# ── ARC definition ────────────────────────────────────────────────────────────
arc = feat[feat['drop_pct'].notna() & np.isfinite(feat['drop_pct']) & (feat['drop_pct'] > -500)].copy()
print(f"ARC: N={len(arc)}, events={arc['crash_30'].sum()}, rate={arc['crash_30'].mean():.3f}")

arc['map_valid_10']  = (arc['baseline_map'] >= 10).astype(float)
arc['map_valid_20']  = (arc['baseline_map'] >= 20).astype(float)
arc['map_valid_30']  = (arc['baseline_map'] >= 30).astype(float)
arc['map_valid_15']  = (arc['baseline_map'] >= 15).astype(float)

CLIN = ['age', 'bmi', 'asa', 'preop_htn', 'preop_dm']
y_all = arc['crash_30'].values

# ── Stratified 70/30 split ────────────────────────────────────────────────────
X_idx = arc.index.values
X_tr_idx, X_te_idx = train_test_split(X_idx, test_size=0.30, stratify=y_all, random_state=42)
tr = arc.loc[X_tr_idx]
te = arc.loc[X_te_idx]
y_tr = tr['crash_30'].values
y_te = te['crash_30'].values
print(f"Train: N={len(tr)}, Test: N={len(te)}, Test events={y_te.sum()} ({y_te.mean():.3f})")

# ── Bootstrap CI helper ───────────────────────────────────────────────────────
def bootstrap_auroc(y_true, y_score, n=1000, seed=0):
    rng_b = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng_b.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

# ── Model builder ─────────────────────────────────────────────────────────────
def build_pipe():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('lr',  LogisticRegression(max_iter=1000, random_state=42))
    ])

def run_model(feats, tr_df, te_df, y_tr, y_te, label, compute_ci=True):
    X_tr = tr_df[feats].values
    X_te = te_df[feats].values
    pipe = build_pipe()
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    if compute_ci:
        lo, hi = bootstrap_auroc(y_te, proba)
    else:
        lo, hi = np.nan, np.nan
    print(f"  {label}: AUROC={auc:.3f} [{lo:.3f}–{hi:.3f}]  (test N={len(y_te)}, events={y_te.sum()})")
    return auc, lo, hi, proba

print("\n=== PRIMARY ARC ANALYSIS (reproduced) ===")
auc_m0, lo0, hi0, _ = run_model(CLIN, tr, te, y_tr, y_te, "M0 (clinical)")
auc_m1, lo1, hi1, proba_m1 = run_model(CLIN+['baseline_map'], tr, te, y_tr, y_te, "M1 (clin+MAP)")

print("\n=== VALIDITY-INDICATOR ANALYSIS ===")
# Mv: clinical + validity indicator (MAP>=20)
auc_mv, lo_mv, hi_mv, _ = run_model(CLIN+['map_valid_20'], tr, te, y_tr, y_te, "Mv (clin+valid_20)")
# M1v: clinical + validity + MAP value
auc_m1v, lo_m1v, hi_m1v, _ = run_model(CLIN+['map_valid_20','baseline_map'], tr, te, y_tr, y_te, "M1v (clin+valid+MAP)")

# Quantify fraction explained
gain_mv  = auc_mv  - auc_m0
gain_m1  = auc_m1  - auc_m0
frac_explained = gain_mv / gain_m1 if gain_m1 > 0 else np.nan
print(f"\n  M0 gain explained by Mv: {frac_explained:.1%} ({gain_mv:.3f}/{gain_m1:.3f})")
print(f"  M1v vs Mv gap: {auc_m1v - auc_mv:.3f}  (M1v gives additional {auc_m1v - auc_mv:.3f} AUROC beyond binary indicator)")
print(f"  M1v vs M1 gap: {auc_m1 - auc_m1v:.3f}  (M1 vs M1v)")

print("\n=== VALIDITY THRESHOLD SENSITIVITY ===")
for thr, col in [(10,'map_valid_10'), (15,'map_valid_15'), (20,'map_valid_20'), (30,'map_valid_30')]:
    run_model(CLIN+[col], tr, te, y_tr, y_te, f"Mv threshold={thr}mmHg", compute_ci=True)

print("\n=== COMPLETE-CASE ANALYSIS ===")
cc = arc[arc['baseline_map'].notna() & arc['ri_mean_clean'].notna() & arc['ppg_amp_clean'].notna()].copy()
y_cc = cc['crash_30'].values
print(f"Complete-case: N={len(cc)}, events={y_cc.sum()}, rate={y_cc.mean():.3f}")

X_cc_idx = cc.index.values
X_cc_tr, X_cc_te = train_test_split(X_cc_idx, test_size=0.30, stratify=y_cc, random_state=42)
tr_cc = cc.loc[X_cc_tr]; te_cc = cc.loc[X_cc_te]
y_tr_cc = tr_cc['crash_30'].values; y_te_cc = te_cc['crash_30'].values
print(f"  CC test: N={len(te_cc)}, events={y_te_cc.sum()} ({y_te_cc.mean():.3f})")

run_model(CLIN, tr_cc, te_cc, y_tr_cc, y_te_cc, "CC M0 (clinical)", compute_ci=True)
run_model(CLIN+['baseline_map'], tr_cc, te_cc, y_tr_cc, y_te_cc, "CC M1 (clin+MAP)", compute_ci=True)
run_model(CLIN+['baseline_map','ri_mean_clean'], tr_cc, te_cc, y_tr_cc, y_te_cc, "CC M3 (clin+MAP+RI)", compute_ci=True)
run_model(CLIN+['baseline_map','ri_mean_clean','ppg_amp_clean'], tr_cc, te_cc, y_tr_cc, y_te_cc, "CC M4", compute_ci=True)

print("\n=== FULL-COHORT (noisy-label) ANALYSIS ===")
full = feat.copy()
full['crash_30_noisy'] = full['crash_30'].fillna(0).astype(float)
y_full = full['crash_30_noisy'].values
print(f"Full cohort: N={len(full)}, events={y_full.sum()}, rate={y_full.mean():.3f}")

X_full_idx = full.index.values
X_full_tr, X_full_te = train_test_split(X_full_idx, test_size=0.30, stratify=y_full, random_state=42)
tr_full = full.loc[X_full_tr]; te_full = full.loc[X_full_te]
y_tr_full = tr_full['crash_30_noisy'].values; y_te_full = te_full['crash_30_noisy'].values
print(f"  Full test: N={len(te_full)}, events={y_te_full.sum()} ({y_te_full.mean():.3f})")

run_model(CLIN+['baseline_map'], tr_full, te_full, y_tr_full, y_te_full, "Full M1", compute_ci=True)
run_model(CLIN+['baseline_map','ri_mean_clean'], tr_full, te_full, y_tr_full, y_te_full, "Full M3", compute_ci=True)

# ── FIGURE: Baseline/nadir MAP distribution ────────────────────────────────────
print("\n=== Generating MAP distribution figure ===")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: Baseline MAP distribution
bm = arc['baseline_map'].dropna().values
# Bins: fine resolution in 0-30 (near-zero region), broader for 30-250
bins_base = list(np.arange(-20, 5, 2)) + list(np.arange(5, 30, 5)) + list(np.arange(30, 200, 20))

ax = axes[0]
ax.hist(bm, bins=60, range=(-20, 200), color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.85)
ax.axvline(20, color='firebrick', linestyle='--', linewidth=1.5, label='20 mmHg threshold')
ax.axvline(5,  color='darkorange', linestyle=':', linewidth=1.2, label='5 mmHg (near-zero)')
ax.set_xlabel('Baseline MAP (mmHg)', fontsize=11)
ax.set_ylabel('Number of cases', fontsize=11)
ax.set_title('A  Baseline MAP distribution\n(ARC, $n$=528)', fontsize=11)
ax.legend(fontsize=9)
n_lt5  = (bm < 5).sum()
n_5_20 = ((bm >= 5) & (bm < 20)).sum()
n_ge20 = (bm >= 20).sum()
ax.text(0.62, 0.85, f'<5 mmHg: {n_lt5} ({n_lt5/len(bm)*100:.0f}%)\n5–19: {n_5_20} ({n_5_20/len(bm)*100:.0f}%)\n≥20: {n_ge20} ({n_ge20/len(bm)*100:.0f}%)',
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Panel B: Nadir MAP distribution
nm_vals = arc['nadir_map'].dropna().values
ax2 = axes[1]
ax2.hist(nm_vals, bins=60, range=(-20, 200), color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.85)
ax2.axvline(20, color='firebrick', linestyle='--', linewidth=1.5, label='20 mmHg threshold')
ax2.axvline(5,  color='darkorange', linestyle=':', linewidth=1.2, label='5 mmHg (near-zero)')
ax2.axvline(65, color='green', linestyle='-.', linewidth=1.2, label='65 mmHg (abs. threshold)')
ax2.set_xlabel('Nadir MAP (mmHg)', fontsize=11)
ax2.set_ylabel('Number of cases', fontsize=11)
ax2.set_title('B  Nadir MAP distribution\n(ARC, $n$=528)', fontsize=11)
ax2.legend(fontsize=9)
n2_lt5  = (nm_vals < 5).sum()
n2_5_20 = ((nm_vals >= 5) & (nm_vals < 20)).sum()
n2_ge20 = (nm_vals >= 20).sum()
ax2.text(0.62, 0.85, f'<5 mmHg: {n2_lt5} ({n2_lt5/len(nm_vals)*100:.0f}%)\n5–19: {n2_5_20} ({n2_5_20/len(nm_vals)*100:.0f}%)\n≥20: {n2_ge20} ({n2_ge20/len(nm_vals)*100:.0f}%)',
         transform=ax2.transAxes, fontsize=8.5, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Figure 2: Distribution of Baseline and Nadir MAP Values in the ARC\n'
             '(Solar8000/ART\_MBP; no physiological filter applied in original pipeline)',
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "fig2_map_distribution.png", dpi=300, bbox_inches='tight')
print(f"Saved: {FIG}/fig2_map_distribution.png")
plt.close()

print("\n=== Done ===")
