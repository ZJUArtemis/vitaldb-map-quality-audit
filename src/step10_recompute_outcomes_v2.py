#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step10_recompute_outcomes_v2.py | Topic: 10
Purpose: Recompute outcome labels using SNUADC/ART (raw ABP waveform at 500Hz,
         loaded at 1s intervals) instead of Solar8000/ART_MBP (patient monitor
         display, which records near-zero values before art-line connection).

Produces:
  - outcome_labels_v2.parquet   (with valid MAP-based outcomes)
  - vascular_features_v2.parquet (features merged with new outcomes)
  - data_quality_table.csv      (cohort flow table for manuscript)
"""
import warnings; warnings.filterwarnings('ignore')
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import vitaldb
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK     = Path(__file__).resolve().parents[1]
PROC_DIR = WORK / "data" / "processed"
MET_DIR  = WORK / "outputs" / "metrics"
RAW      = Path("vitaldb_data")  # local VitalDB physionet.org root; override as needed
for d in [PROC_DIR, MET_DIR]: d.mkdir(parents=True, exist_ok=True)
assert not str(PROC_DIR.resolve()).startswith(str(RAW.resolve()))

# ── Logging ────────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = WORK / "outputs" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"{ts}_step10_recompute.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
MAP_MIN        = 20    # physiological lower bound (mmHg)
MAP_MAX        = 200   # physiological upper bound (mmHg)
BASELINE_SECS  = 300   # 5 min before induction
MIN_VALID_SECS = 30    # minimum valid seconds required in window
INDUCTION_SECS = 600   # 10 min post-induction for nadir
ABSOLUTE_THRESH = 65   # mmHg for crash_absolute

np.random.seed(42)

def load_map_1hz(caseid):
    """Load SNUADC/ART at 1-second intervals, filter to physiological range."""
    try:
        data = vitaldb.load_case(caseid, ['SNUADC/ART'], interval=1)
        if data is None or len(data) == 0:
            return None
        arr = np.array(data).flatten().astype(float)
        # Mask non-physiological values
        arr[(arr < MAP_MIN) | (arr > MAP_MAX)] = np.nan
        return arr
    except Exception as e:
        log.warning(f"Case {caseid}: SNUADC/ART load error - {e}")
        return None

def compute_baseline_map(arr, t_start):
    """Compute baseline MAP as median of valid values in 5-min pre-induction window."""
    bs_start = max(0, int(t_start) - BASELINE_SECS)
    bs_end   = int(t_start)
    if bs_end <= bs_start or bs_end > len(arr):
        return np.nan, 0
    window = arr[bs_start:bs_end]
    valid  = window[~np.isnan(window)]
    if len(valid) < MIN_VALID_SECS:
        return np.nan, len(valid)
    return float(np.median(valid)), len(valid)

def compute_nadir_map(arr, t_start, t_end):
    """Compute nadir MAP as minimum valid value during induction window."""
    ind_start = int(t_start)
    ind_end   = min(int(t_start) + INDUCTION_SECS, int(t_end), len(arr))
    if ind_end <= ind_start:
        return np.nan, 0
    window = arr[ind_start:ind_end]
    valid  = window[~np.isnan(window)]
    if len(valid) < MIN_VALID_SECS:
        return np.nan, len(valid)
    return float(np.min(valid)), len(valid)

# ── Load induction segments ─────────────────────────────────────────────────────
seg_df = pd.read_parquet(PROC_DIR / "induction_segments.parquet")
log.info(f"Loaded {len(seg_df)} induction segments")

# ── Recompute outcomes ──────────────────────────────────────────────────────────
log.info("Recomputing outcomes from SNUADC/ART waveform...")
outcomes = []
skip_no_art   = 0
skip_no_bsln  = 0
skip_no_nadir = 0

for _, row in tqdm(seg_df.iterrows(), total=len(seg_df), desc="Outcome v2"):
    caseid  = int(row['caseid'])
    t_start = row['t_start']
    t_end   = row['t_end']

    arr = load_map_1hz(caseid)
    if arr is None:
        skip_no_art += 1
        continue

    baseline_map, n_baseline_valid = compute_baseline_map(arr, t_start)
    nadir_map,    n_nadir_valid    = compute_nadir_map(arr, t_start, t_end)

    # Relative drop (only if both are available)
    if not np.isnan(baseline_map) and not np.isnan(nadir_map) and baseline_map > 0:
        drop_pct = (baseline_map - nadir_map) / baseline_map * 100
    else:
        drop_pct = np.nan

    crash_30       = int(drop_pct > 30)  if not np.isnan(drop_pct) else np.nan
    crash_40       = int(drop_pct > 40)  if not np.isnan(drop_pct) else np.nan
    crash_20       = int(drop_pct > 20)  if not np.isnan(drop_pct) else np.nan
    crash_absolute = int(nadir_map < ABSOLUTE_THRESH) if not np.isnan(nadir_map) else np.nan

    if np.isnan(baseline_map):
        skip_no_bsln += 1
    if np.isnan(nadir_map):
        skip_no_nadir += 1

    outcomes.append({
        'caseid':            caseid,
        'baseline_map':      baseline_map,
        'nadir_map':         nadir_map,
        'drop_pct':          drop_pct,
        'crash_30':          crash_30,
        'crash_40':          crash_40,
        'crash_20':          crash_20,
        'crash_absolute':    crash_absolute,
        'n_baseline_valid':  n_baseline_valid,
        'n_nadir_valid':     n_nadir_valid,
    })

outcome_v2 = pd.DataFrame(outcomes)
log.info(f"\n=== Outcome v2 summary ({len(outcome_v2)} cases with SNUADC/ART) ===")
log.info(f"  No SNUADC/ART track:          {skip_no_art}")
log.info(f"  No valid baseline window:      {skip_no_bsln} ({skip_no_bsln/len(outcome_v2)*100:.1f}%)")
log.info(f"  No valid nadir window:         {skip_no_nadir} ({skip_no_nadir/len(outcome_v2)*100:.1f}%)")

for col in ['crash_30', 'crash_40', 'crash_20', 'crash_absolute']:
    valid_rows = outcome_v2[col].notna()
    n_valid = valid_rows.sum()
    n_crash = outcome_v2.loc[valid_rows, col].sum()
    log.info(f"  {col}: {int(n_crash)}/{int(n_valid)} ({n_crash/n_valid*100:.1f}%) [excludes NaN]")

# Save
out_path = PROC_DIR / "outcome_labels_v2.parquet"
outcome_v2.to_parquet(out_path, index=False)
log.info(f"\nSaved: {out_path}")

# ── Merge with vascular features ───────────────────────────────────────────────
feat = pd.read_parquet(PROC_DIR / "vascular_features.parquet")
# Drop old outcome columns, keep only vascular features
drop_cols = [c for c in ['baseline_map','nadir_map','drop_pct','crash_30','crash_40',
                          'crash_20','crash_absolute'] if c in feat.columns]
feat_core = feat.drop(columns=drop_cols)
feat_v2 = feat_core.merge(
    outcome_v2[['caseid','baseline_map','nadir_map','drop_pct',
                'crash_30','crash_40','crash_20','crash_absolute',
                'n_baseline_valid','n_nadir_valid']],
    on='caseid', how='inner'
)
feat_v2_path = PROC_DIR / "vascular_features_v2.parquet"
feat_v2.to_parquet(feat_v2_path, index=False)
log.info(f"Saved merged features v2: {feat_v2_path} (N={len(feat_v2)})")

# ── Cohort accounting table ────────────────────────────────────────────────────
log.info("\n=== Data Quality Flow Table ===")
n_orig   = len(seg_df)           # induction segments
n_art    = len(outcome_v2)       # with SNUADC/ART data
n_bsln   = outcome_v2['baseline_map'].notna().sum()   # valid baseline
n_nadir  = outcome_v2['nadir_map'].notna().sum()       # valid nadir
n_both   = (outcome_v2['baseline_map'].notna() & outcome_v2['nadir_map'].notna()).sum()
n_crash30 = outcome_v2['crash_30'].notna().sum()

log.info(f"  Induction segments: {n_orig}")
log.info(f"  With SNUADC/ART:    {n_art}")
log.info(f"  Valid baseline MAP: {n_bsln}")
log.info(f"  Valid nadir MAP:    {n_nadir}")
log.info(f"  Both valid (relative-drop cohort): {n_both}")
log.info(f"  crash_30 evaluable: {n_crash30}")

# Save accounting table
rows = [
    {'step': '1. Induction segments (from step1-2)',            'n': n_orig,   'note': 'propofol+ABP+PPG, ≥10 min induction'},
    {'step': '2. With SNUADC/ART waveform data',               'n': n_art,    'note': 'continuous arterial BP waveform present'},
    {'step': '3. Valid pre-induction baseline MAP',             'n': n_bsln,   'note': f'≥{MIN_VALID_SECS}s valid 20-200 mmHg in 5-min baseline window'},
    {'step': '4. Valid induction nadir MAP',                    'n': n_nadir,  'note': f'≥{MIN_VALID_SECS}s valid 20-200 mmHg in 10-min induction window'},
    {'step': '5. Relative-drop cohort (baseline AND nadir valid)', 'n': n_both, 'note': 'crash_30/20/40 evaluable'},
]
if outcome_v2['crash_30'].notna().any():
    n_crash = int(outcome_v2['crash_30'].dropna().sum())
    rows.append({'step': '   crash_30=1', 'n': n_crash, 'note': 'relative MAP drop ≥30% within 10 min'})

tbl = pd.DataFrame(rows)
tbl_path = MET_DIR / "data_quality_flow_table.csv"
tbl.to_csv(tbl_path, index=False)
log.info(f"\nSaved: {tbl_path}")

# Also report absolute cohort
n_abs_valid = outcome_v2['crash_absolute'].notna().sum()
n_abs_crash = int(outcome_v2['crash_absolute'].dropna().sum())
log.info(f"\n  Absolute-threshold cohort (nadir MAP valid): {n_abs_valid}")
log.info(f"  crash_absolute=1 (nadir <65 mmHg): {n_abs_crash} ({n_abs_crash/n_abs_valid*100:.1f}% if valid>0)")

# ── baseline_map stats for reporting ──────────────────────────────────────────
bm_valid = outcome_v2['baseline_map'].dropna()
log.info(f"\n=== New baseline_map (from SNUADC/ART, valid >=30s) ===")
log.info(f"  N valid: {len(bm_valid)}")
if len(bm_valid) > 0:
    pcts = np.percentile(bm_valid, [5,10,25,50,75,90,95])
    log.info(f"  5th pct={pcts[0]:.1f}, 25th={pcts[2]:.1f}, median={pcts[3]:.1f}, 75th={pcts[4]:.1f}, 95th={pcts[6]:.1f}")

log.info("\n=== Step 10 Complete ===")
