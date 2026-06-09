#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step11_nibp_corrected.py | Topic: 10
Purpose: NIBP-based CORRECTED reference arm for the data-quality audit.
         Solar8000/ART_MBP (and SNUADC/ART) have no valid PRE-induction baseline
         because the arterial line is placed during/after induction in most TIVA
         cases. The non-invasive oscillometric cuff (Solar8000/NIBP_MBP) IS valid
         pre-induction. This script:
           1. Derives a physiologically valid pre-induction baseline MAP from NIBP.
           2. Reports the baseline distribution + corrected crash-30 event rate.
           3. Refits M0/M1/M3 on the NIBP-evaluable cohort (same pipeline as step9).
         Output: reference numbers for the manuscript's corrected arm.
"""
import warnings; warnings.filterwarnings('ignore')
import logging, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import vitaldb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK     = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability")
PROC_DIR = WORK / "data" / "processed"
MET_DIR  = WORK / "outputs" / "metrics"
RAW      = Path("/home/lxk/vitaldb/physionet.org")
for d in [PROC_DIR, MET_DIR]: d.mkdir(parents=True, exist_ok=True)
assert not str(PROC_DIR.resolve()).startswith(str(RAW.resolve()))

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = WORK / "outputs" / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_dir / f"{ts}_step11_nibp_corrected.log"),
              logging.StreamHandler()])
log = logging.getLogger(__name__)
np.random.seed(42)

# ── Parameters ─────────────────────────────────────────────────────────────────
MAP_MIN, MAP_MAX = 20, 200      # physiological plausibility filter (same as step10)
BASELINE_SECS    = 300          # 5-min pre-induction window
INDUCTION_SECS   = 600          # 10-min post-induction window
ABS_THRESH       = 65

def load_nibp_1hz(caseid):
    try:
        d = vitaldb.load_case(int(caseid), ['Solar8000/NIBP_MBP'], interval=1)
        if d is None or len(d) == 0:
            return None
        arr = np.array(d).flatten().astype(float)
        arr[(arr < MAP_MIN) | (arr > MAP_MAX)] = np.nan
        return arr
    except Exception as e:
        log.warning(f"Case {caseid}: NIBP load error - {e}")
        return None

def win_stat(arr, a, b, fn):
    a, b = max(0, int(a)), min(int(b), len(arr))
    if b <= a: return np.nan, 0
    w = arr[a:b]; v = w[~np.isnan(w)]
    if len(v) < 1: return np.nan, 0
    return float(fn(v)), len(v)

# ── Cohort ─────────────────────────────────────────────────────────────────────
seg = pd.read_parquet(PROC_DIR / "induction_segments.parquet")
log.info(f"Loaded {len(seg)} induction segments")

rows = []
for _, r in tqdm(seg.iterrows(), total=len(seg), desc="NIBP outcomes"):
    cid, t0, t1 = int(r['caseid']), r['t_start'], r['t_end']
    arr = load_nibp_1hz(cid)
    if arr is None:
        rows.append(dict(caseid=cid, nibp_baseline=np.nan, nibp_nadir=np.nan,
                         n_base=0, n_nadir=0)); continue
    base, nb = win_stat(arr, t0 - BASELINE_SECS, t0, np.median)
    nadir, nn = win_stat(arr, t0, min(t0 + INDUCTION_SECS, t1), np.min)
    rows.append(dict(caseid=cid, nibp_baseline=base, nibp_nadir=nadir,
                     n_base=nb, n_nadir=nn))

nibp = pd.DataFrame(rows)
nibp['drop_pct'] = np.where(
    nibp['nibp_baseline'].notna() & nibp['nibp_nadir'].notna() & (nibp['nibp_baseline'] > 0),
    (nibp['nibp_baseline'] - nibp['nibp_nadir']) / nibp['nibp_baseline'] * 100, np.nan)
nibp['crash_30'] = np.where(nibp['drop_pct'].notna(), (nibp['drop_pct'] > 30).astype(float), np.nan)
nibp['crash_abs'] = np.where(nibp['nibp_nadir'].notna(), (nibp['nibp_nadir'] < ABS_THRESH).astype(float), np.nan)

# ── Distribution report ────────────────────────────────────────────────────────
bm = nibp['nibp_baseline'].dropna()
log.info("\n=== NIBP pre-induction baseline distribution ===")
log.info(f"  cases total:               {len(nibp)}")
log.info(f"  valid NIBP baseline:       {bm.notna().sum()} ({bm.notna().sum()/len(nibp)*100:.1f}%)")
log.info(f"  median [IQR]:              {bm.median():.1f} [{bm.quantile(.25):.1f}-{bm.quantile(.75):.1f}] mmHg")
log.info(f"  in 60-120 mmHg (plausible):{((bm>=60)&(bm<=120)).sum()} ({((bm>=60)&(bm<=120)).mean()*100:.1f}%)")
log.info(f"  >=20 mmHg:                 {(bm>=20).sum()} ({(bm>=20).mean()*100:.1f}%)")
log.info(f"  <5 mmHg (artefact):        {(bm<5).sum()} ({(bm<5).mean()*100:.1f}%)")

ev = nibp['crash_30'].notna()
log.info("\n=== Corrected outcome (NIBP relative drop) ===")
log.info(f"  evaluable (baseline&nadir valid): {ev.sum()}")
log.info(f"  crash_30 = {int(nibp.loc[ev,'crash_30'].sum())}/{int(ev.sum())} "
         f"({nibp.loc[ev,'crash_30'].mean()*100:.1f}%)")
ea = nibp['crash_abs'].notna()
log.info(f"  crash_absolute (nadir<65) = {int(nibp.loc[ea,'crash_abs'].sum())}/{int(ea.sum())} "
         f"({nibp.loc[ea,'crash_abs'].mean()*100:.1f}%)")

# ── Merge clinical + RI features, refit M0/M1/M3 ────────────────────────────────
feat = pd.read_parquet(PROC_DIR / "vascular_features_v2.parquet")
clin = ['age','bmi','asa','preop_htn','preop_dm','ri_mean_clean','ppg_amp_clean']
keep = ['caseid'] + [c for c in clin if c in feat.columns]
df = nibp.merge(feat[keep], on='caseid', how='inner')
df = df[df['crash_30'].notna()].copy()
df['baseline_map'] = df['nibp_baseline']
log.info(f"\n=== Modeling cohort (NIBP-evaluable, features merged): N={len(df)}, "
         f"events={int(df['crash_30'].sum())} ({df['crash_30'].mean()*100:.1f}%) ===")

def make_pipe():
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('lr', LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42))])

def boot_auroc(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        try: a.append(roc_auc_score(y[i], p[i]))
        except ValueError: pass
    return np.median(a), np.percentile(a, 2.5), np.percentile(a, 97.5)

specs = [
    ("M0: Clinical only",       ['age','bmi','asa','preop_htn','preop_dm']),
    ("M1: Clinical + MAP",      ['age','bmi','asa','preop_htn','preop_dm','baseline_map']),
    ("M3: Clinical + MAP + RI", ['age','bmi','asa','preop_htn','preop_dm','baseline_map','ri_mean_clean']),
]
y = df['crash_30'].values.astype(float)
itr, ite = train_test_split(np.arange(len(df)), test_size=0.3, stratify=y, random_state=42)
log.info(f"  split: train={len(itr)} ({int(y[itr].sum())} ev), test={len(ite)} ({int(y[ite].sum())} ev)")
results = {}
log.info("\n=== Nested models on NIBP-corrected cohort (held-out test) ===")
for name, fs in specs:
    X = df[fs].values.astype(float)
    pipe = make_pipe(); pipe.fit(X[itr], y[itr])
    p = pipe.predict_proba(X[ite])[:, 1]
    au, lo, hi = boot_auroc(y[ite], p)
    br = brier_score_loss(y[ite], p)
    results[name] = dict(auroc=au, lo=lo, hi=hi, brier=br)
    log.info(f"  {name:<28} AUROC={au:.3f} [{lo:.3f}-{hi:.3f}]  Brier={br:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
summary = dict(
    n_total=int(len(nibp)),
    valid_nibp_baseline=int(bm.notna().sum()),
    valid_nibp_baseline_pct=round(bm.notna().sum()/len(nibp)*100, 1),
    baseline_median=round(float(bm.median()), 1),
    baseline_iqr=[round(float(bm.quantile(.25)), 1), round(float(bm.quantile(.75)), 1)],
    baseline_plausible_60_120_pct=round(((bm>=60)&(bm<=120)).mean()*100, 1),
    evaluable=int(ev.sum()),
    crash30_n=int(nibp.loc[ev,'crash_30'].sum()),
    crash30_pct=round(nibp.loc[ev,'crash_30'].mean()*100, 1),
    crash_abs_pct=round(nibp.loc[ea,'crash_abs'].mean()*100, 1),
    model_cohort_n=int(len(df)),
    model_events=int(df['crash_30'].sum()),
    test_n=int(len(ite)), test_events=int(y[ite].sum()),
    models={k: {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in results.items()},
)
out = MET_DIR / "nibp_corrected_reference.json"
out.write_text(json.dumps(summary, indent=2))
nibp.to_parquet(PROC_DIR / "outcome_labels_nibp.parquet", index=False)
log.info(f"\nSaved: {out}")
log.info(f"Saved: {PROC_DIR}/outcome_labels_nibp.parquet")
log.info("\n=== Step 11 Complete ===")
