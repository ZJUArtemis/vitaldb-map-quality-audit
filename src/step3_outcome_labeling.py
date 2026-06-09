#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step3_outcome_labeling.py | Topic: 10 | Purpose: 定义并标注 Induction Crash 结局
"""
import os, sys, json, logging
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import vitaldb
from tqdm import tqdm

# === PATH SAFETY ===
RAW_DATA = Path("/home/lxk/vitaldb/physionet.org")
WORK = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability")

def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = WORK / "outputs" / "logs"
    lf = log_dir / f"{ts}_step3_outcome_labeling.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()]
    )
    return lf

def main():
    log_file = init_log()
    logging.info("=== Topic 10 Phase 2 Step 1: Outcome Labeling ===")

    # Load induction segments
    seg_path = WORK / "data" / "processed" / "induction_segments.parquet"
    seg_df = pd.read_parquet(seg_path)
    logging.info(f"Loaded {len(seg_df)} segmented cases")

    outcomes = []

    for _, row in tqdm(seg_df.iterrows(), total=len(seg_df), desc="Labeling outcomes"):
        caseid = row['caseid']
        t_start = row['t_start']
        t_end = row['t_end']

        try:
            # Load MAP during induction
            map_array = vitaldb.load_case(caseid, ["Solar8000/ART_MBP"], interval=1)
            if map_array is None or len(map_array) == 0:
                continue
            map_data = pd.Series(map_array[:, 0], index=np.arange(len(map_array)))

            # Baseline MAP (5 min before t_start)
            baseline_start = int(max(0, t_start - 300))
            baseline_end = int(t_start)
            baseline_window = map_data.iloc[baseline_start:baseline_end]
            if len(baseline_window) == 0:
                continue
            baseline_map = baseline_window.median()

            # Induction MAP
            induction_start = int(t_start)
            induction_end = int(t_end)
            induction_map = map_data.iloc[induction_start:induction_end]
            if len(induction_map) == 0:
                continue
            nadir_map = induction_map.min()

            # Calculate drop percentage
            drop_pct = (baseline_map - nadir_map) / baseline_map * 100

            # Multiple outcome definitions
            crash_30 = 1 if drop_pct > 30 else 0
            crash_40 = 1 if drop_pct > 40 else 0
            crash_20 = 1 if drop_pct > 20 else 0
            crash_absolute = 1 if nadir_map < 65 else 0

            outcomes.append({
                'caseid': caseid,
                'baseline_map': baseline_map,
                'nadir_map': nadir_map,
                'drop_pct': drop_pct,
                'crash_30': crash_30,
                'crash_40': crash_40,
                'crash_20': crash_20,
                'crash_absolute': crash_absolute
            })

        except Exception as e:
            logging.warning(f"Case {caseid}: labeling error - {e}")
            continue

    # Save outcomes
    outcome_df = pd.DataFrame(outcomes)
    out_path = safe_path(WORK / "data" / "processed" / "outcome_labels.parquet")
    outcome_df.to_parquet(out_path, index=False)
    logging.info(f"Saved outcome labels: {out_path}")

    # Summary statistics
    for col in ['crash_30', 'crash_40', 'crash_20', 'crash_absolute']:
        rate = outcome_df[col].mean() * 100
        logging.info(f"{col}: {outcome_df[col].sum()}/{len(outcome_df)} ({rate:.1f}%)")

    logging.info("=== DONE ===")

if __name__ == "__main__":
    main()