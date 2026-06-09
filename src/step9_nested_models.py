#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step9_nested_models.py | Topic: 10
Purpose: Decompose incremental contributions of baseline MAP vs RI vs PPG amplitude
         using nested logistic regression models. Also produces clean sensitivity
         analysis excluding inf/corrupt drop_pct cases.
Response to reviewer Issue 1, 2, 3.
"""
import warnings; warnings.filterwarnings('ignore')
import logging, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK     = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability")
PROC_DIR = WORK / "data" / "processed"
FIG_DIR  = WORK / "outputs" / "figures"
MET_DIR  = WORK / "outputs" / "metrics"
RAW      = Path("/home/lxk/vitaldb/physionet.org")
for d in [FIG_DIR, MET_DIR]: d.mkdir(parents=True, exist_ok=True)
assert not str(FIG_DIR.resolve()).startswith(str(RAW.resolve()))

# ── Logging ────────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(WORK / "outputs" / "logs" / f"{ts}_step9_nested_models.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)

# ── Bootstrap AUROC ────────────────────────────────────────────────────────────
def bootstrap_auroc(y_true, y_prob, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except ValueError:
            pass
    return np.median(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def bootstrap_auprc(y_true, y_prob, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            vals.append(average_precision_score(y_true[idx], y_prob[idx]))
        except ValueError:
            pass
    return np.median(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

# ── Load data ──────────────────────────────────────────────────────────────────
feat = pd.read_parquet(PROC_DIR / "vascular_features.parquet")
log.info(f"Full dataset: {len(feat)} cases")

# ── Data quality audit: inf drop_pct ──────────────────────────────────────────
n_inf   = np.isinf(feat['drop_pct']).sum()
n_neg   = (feat['drop_pct'] < -500).sum()
log.info(f"\n=== Data Quality Audit ===")
log.info(f"  inf drop_pct cases: {n_inf} (all labelled crash_30=1)")
log.info(f"  extreme negative drop_pct (<-500): {n_neg}")
log.info(f"  crash_30=1 (full cohort): {feat['crash_30'].sum()} / {len(feat)}")

# Sensitivity: exclude inf cases (likely baseline_map recording artifacts)
feat_clean = feat[~np.isinf(feat['drop_pct']) & (feat['drop_pct'] > -500)].copy()
log.info(f"  Clean cohort (excl. inf/extreme): {len(feat_clean)} cases, "
         f"crash_30={feat_clean['crash_30'].sum()} ({feat_clean['crash_30'].mean()*100:.1f}%)")

# Missing data summary
log.info(f"\n=== Missing Data (full cohort, N=909) ===")
for col in ['baseline_map', 'ri_mean_clean', 'ppg_amp_clean', 'pi_clean']:
    n_miss = feat[col].isna().sum()
    log.info(f"  {col}: {n_miss}/{len(feat)} missing ({n_miss/len(feat)*100:.1f}%)")

# ── Build imputer/LR pipeline ──────────────────────────────────────────────────
def make_pipe():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('lr',  LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42))
    ])

def eval_model(name, y_test, y_prob):
    auroc, alo, ahi = bootstrap_auroc(y_test, y_prob)
    auprc, plo, phi = bootstrap_auprc(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    log.info(f"  {name:<35} AUROC={auroc:.3f} [{alo:.3f}-{ahi:.3f}]  "
             f"AUPRC={auprc:.3f} [{plo:.3f}-{phi:.3f}]  Brier={brier:.4f}")
    return dict(name=name, auroc=auroc, auroc_lo=alo, auroc_hi=ahi,
                auprc=auprc, auprc_lo=plo, auprc_hi=phi, brier=brier)

def run_nested(df, label='Full cohort'):
    """Fit nested models and return results dict."""
    y = df['crash_30'].values.astype(float)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3,
                                      stratify=y, random_state=42)
    results = []

    model_specs = [
        ("M0: Clinical only",           ['age','bmi','asa','preop_htn','preop_dm']),
        ("M1: Clinical + MAP",          ['age','bmi','asa','preop_htn','preop_dm','baseline_map']),
        ("M2: Clinical + RI",           ['age','bmi','asa','preop_htn','preop_dm','ri_mean_clean']),
        ("M3: Clinical + MAP + RI",     ['age','bmi','asa','preop_htn','preop_dm','baseline_map','ri_mean_clean']),
        ("M4: Clinical + MAP + RI + Amp",['age','bmi','asa','preop_htn','preop_dm','baseline_map','ri_mean_clean','ppg_amp_clean']),
    ]

    probs = {}
    for mname, feats in model_specs:
        X  = df[feats].values.astype(float)
        Xtr, Xte = X[idx_tr], X[idx_te]
        ytr, yte = y[idx_tr], y[idx_te]
        pipe = make_pipe()
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        probs[mname] = (yte, prob)
        r = eval_model(mname, yte, prob)
        results.append(r)

    return results, probs, idx_te, y

# ══════════════════════════════════════════════════════════════════════════════
log.info("\n\n=== NESTED MODEL COMPARISON — Full Cohort (N=909, imputed) ===")
res_full, probs_full, idx_te_full, y_full = run_nested(feat, 'Full cohort')

log.info("\n\n=== NESTED MODEL COMPARISON — Clean Cohort (excl. inf drop_pct) ===")
res_clean, probs_clean, idx_te_clean, y_clean = run_nested(feat_clean, 'Clean cohort')

# ── Incremental AUROC table ────────────────────────────────────────────────────
log.info("\n=== INCREMENTAL AUROC (Full Cohort) ===")
log.info(f"  {'Step':<45} {'ΔAUROC':>8}")
aucs = [r['auroc'] for r in res_full]
steps = [
    ("M0→M1  Adding baseline MAP", aucs[1]-aucs[0]),
    ("M0→M2  Adding RI only",       aucs[2]-aucs[0]),
    ("M1→M3  Adding RI | MAP already in", aucs[3]-aucs[1]),
    ("M3→M4  Adding PPG amplitude | MAP+RI", aucs[4]-aucs[3]),
]
for desc, delta in steps:
    log.info(f"  {desc:<45} {delta:>+8.3f}")

# ── Save nested model results ──────────────────────────────────────────────────
nested_rows = []
for r in res_full:
    nested_rows.append({
        'dataset': 'full_imputed',
        'model': r['name'],
        'auroc': f"{r['auroc']:.3f}",
        'auroc_95ci': f"[{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]",
        'auprc': f"{r['auprc']:.3f}",
        'auprc_95ci': f"[{r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}]",
        'brier': f"{r['brier']:.4f}",
    })
for r in res_clean:
    nested_rows.append({
        'dataset': 'clean_excl_inf',
        'model': r['name'],
        'auroc': f"{r['auroc']:.3f}",
        'auroc_95ci': f"[{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]",
        'auprc': f"{r['auprc']:.3f}",
        'auprc_95ci': f"[{r['auprc_lo']:.3f}-{r['auprc_hi']:.3f}]",
        'brier': f"{r['brier']:.4f}",
    })
nested_df = pd.DataFrame(nested_rows)
nested_df.to_csv(MET_DIR / "nested_model_comparison.csv", index=False)
log.info(f"\nSaved: {MET_DIR}/nested_model_comparison.csv")

# ── Figure: nested ROC curves (full cohort) ───────────────────────────────────
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':300,'savefig.dpi':300})
COLORS_N = {
    "M0: Clinical only":            '#999999',
    "M1: Clinical + MAP":           '#56B4E9',
    "M2: Clinical + RI":            '#E69F00',
    "M3: Clinical + MAP + RI":      '#009E73',
    "M4: Clinical + MAP + RI + Amp":'#D55E00',
}
LABELS_N = {
    "M0: Clinical only":             "M0: Clinical only",
    "M1: Clinical + MAP":            "M1: + Baseline MAP",
    "M2: Clinical + RI":             "M2: + RI only",
    "M3: Clinical + MAP + RI":       "M3: + MAP + RI",
    "M4: Clinical + MAP + RI + Amp": "M4: + MAP + RI + Amp (= Model B)",
}

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for r in res_full:
    yte, prob = probs_full[r['name']]
    fpr, tpr, _ = roc_curve(yte, prob)
    ax.plot(fpr, tpr, color=COLORS_N[r['name']], lw=1.8,
            label=f"{LABELS_N[r['name']]}  (AUC={r['auroc']:.3f})")
ax.plot([0,1],[0,1],'--',color='#aaaaaa',lw=0.8)
ax.set_xlabel("1 − Specificity", fontsize=9)
ax.set_ylabel("Sensitivity", fontsize=9)
ax.set_title("Figure S2.  Nested Model Comparison — ROC Curves\n"
             "(Incremental contribution of MAP vs RI)", fontsize=9, fontweight='bold')
ax.legend(fontsize=6.5, loc='lower right')
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_nested_model_roc.png", dpi=300, bbox_inches='tight')
plt.close()
log.info(f"Saved: fig_nested_model_roc.png")

# ── Figure: AUROC ladder plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 3.5))
model_labels = [LABELS_N[r['name']] for r in res_full]
aucs_f  = [r['auroc']    for r in res_full]
aucs_lo = [r['auroc_lo'] for r in res_full]
aucs_hi = [r['auroc_hi'] for r in res_full]
colors  = [COLORS_N[r['name']] for r in res_full]
y_pos   = np.arange(len(res_full))
for i, (a, lo, hi, c) in enumerate(zip(aucs_f, aucs_lo, aucs_hi, colors)):
    ax.barh(i, a, color=c, alpha=0.85, height=0.5)
    ax.errorbar(a, i, xerr=[[a-lo],[hi-a]], fmt='none', color='#333333', capsize=3, lw=1.2)
    ax.text(a+0.003, i, f"{a:.3f}", va='center', fontsize=7.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_labels, fontsize=7.5)
ax.set_xlabel("AUROC (held-out test set)", fontsize=8)
ax.set_xlim(0.45, 1.02)
ax.axvline(0.5, color='#aaaaaa', lw=0.8, ls='--')
ax.set_title("Incremental AUROC: Nested Model Comparison", fontsize=8.5, fontweight='bold')
ax.grid(axis='x', alpha=0.2)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_nested_auroc_ladder.png", dpi=300, bbox_inches='tight')
plt.close()
log.info(f"Saved: fig_nested_auroc_ladder.png")

# ── Threshold sensitivity (clean cohort, excl inf) ────────────────────────────
log.info("\n=== THRESHOLD SENSITIVITY — Clean Cohort ===")
clean_thresh_rows = []
for thresh in [20, 25, 30, 35, 40]:
    col = f'crash_{thresh}'
    if col not in feat_clean.columns:
        feat_clean[col] = (feat_clean['drop_pct'] > thresh).astype(int)
    y = feat_clean[col].values.astype(float)
    n_crash = int(y.sum())
    log.info(f"  >{thresh}%: {n_crash} crashes ({n_crash/len(feat_clean)*100:.1f}%)")
    if n_crash < 10:
        log.warning(f"  Too few events at >{thresh}%, skipping model")
        continue
    idx_tr, idx_te = train_test_split(np.arange(len(feat_clean)), test_size=0.3,
                                      stratify=y, random_state=42)
    feats_B = ['age','bmi','asa','preop_htn','preop_dm','baseline_map','ri_mean_clean','ppg_amp_clean']
    X = feat_clean[feats_B].values.astype(float)
    pipe = make_pipe()
    pipe.fit(X[idx_tr], y[idx_tr])
    prob = pipe.predict_proba(X[idx_te])[:, 1]
    y_te = y[idx_te]
    auroc, lo, hi = bootstrap_auroc(y_te, prob)
    auprc = average_precision_score(y_te, prob)
    brier = brier_score_loss(y_te, prob)
    log.info(f"  Model B AUROC={auroc:.3f} [{lo:.3f}-{hi:.3f}]  AUPRC={auprc:.3f}")
    clean_thresh_rows.append({'threshold':thresh,'n_crash':n_crash,
                               'prevalence':n_crash/len(feat_clean),
                               'auroc':auroc,'auroc_lo':lo,'auroc_hi':hi,
                               'auprc':auprc,'brier':brier})

clean_thresh_df = pd.DataFrame(clean_thresh_rows)
clean_thresh_df.to_csv(MET_DIR / "threshold_sensitivity_clean.csv", index=False)
log.info(f"Saved: threshold_sensitivity_clean.csv")

log.info("\n=== Step 9 Complete ===")
log.info(f"Outputs: {MET_DIR}/nested_model_comparison.csv")
log.info(f"         {FIG_DIR}/fig_nested_model_roc.png")
log.info(f"         {FIG_DIR}/fig_nested_auroc_ladder.png")
log.info(f"         {MET_DIR}/threshold_sensitivity_clean.csv")
