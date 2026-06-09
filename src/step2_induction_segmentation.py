#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step2_induction_segmentation.py | Topic: 10 | Purpose: 自动分割麻醉诱导期 (T_start, T_end)
"""
import os, sys, json, logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import vitaldb

# === PATH SAFETY ===
RAW_DATA = Path("vitaldb_data")  # local VitalDB physionet.org root; override as needed
WORK = Path(__file__).resolve().parents[1]

def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)

# === LOGGING ===
def init_log():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = WORK / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lf = log_dir / f"{ts}_step2_induction_segmentation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()]
    )
    return lf

def main():
    log_file = init_log()
    logging.info("=== Topic 10 Phase 1 Step 2: Induction Segmentation ===")
    logging.info(f"Log file: {log_file}")

    # Load eligible caseids
    eligible_path = WORK / "outputs" / "metrics" / "eligible_caseids.csv"
    eligible_df = pd.read_csv(eligible_path)
    caseids = eligible_df['caseid'].tolist()
    logging.info(f"Number of eligible cases: {len(caseids)}")

    records = []
    durations = []

    for caseid in tqdm(caseids, desc="Segmenting induction"):
        try:
            # Load propofol infusion rate (1 Hz default)
            prop_data = vitaldb.load_case(caseid, ["Orchestra/PPF20_RATE"], interval=1)
            if prop_data is None or len(prop_data) == 0:
                continue
            prop_rate = pd.Series(prop_data[:, 0], index=np.arange(len(prop_data)))
            # Find first time > 0
            t_start_idx = np.where(prop_rate > 0)[0]
            if len(t_start_idx) == 0:
                continue
            t_start = t_start_idx[0]

            # Load MAP (use Solar8000/ART_MBP)
            map_data = vitaldb.load_case(caseid, ["Solar8000/ART_MBP"], interval=1)
            if map_data is None or len(map_data) == 0:
                continue
            map_series = pd.Series(map_data[:, 0], index=np.arange(len(map_data)))

            # Define T_end as earliest of:
            #   a) T_start + 20 minutes (hard limit)
            #   b) MAP recovers and stays within 5% of baseline for 30 sec after a dip
            hard_limit = t_start + 20 * 60

            # Baseline MAP: median of 30 sec before t_start (if available)
            baseline_start = max(0, t_start - 30)
            baseline_map = map_series.iloc[baseline_start:t_start].median()
            # Detect first point after t_start where MAP stays above 0.95*baseline for >=30 sec
            post_series = map_series.iloc[t_start:]
            recovered = hard_limit
            if not np.isnan(baseline_map):
                above_thr = post_series >= 0.95 * baseline_map
                # find runs of True of length >=30 sec
                run_len = 0
                for idx in range(len(above_thr)):
                    if above_thr.iloc[idx]:
                        run_len += 1
                        if run_len >= 30:
                            recovered = t_start + idx - run_len + 1
                            break
                    else:
                        run_len = 0
            t_end = min(hard_limit, recovered)

            duration = t_end - t_start
            if duration < 60 or duration > 20 * 60:
                # skip unrealistic durations (will be reviewed later)
                continue

            records.append({"caseid": caseid, "t_start": float(t_start), "t_end": float(t_end), "duration_sec": float(duration)})
            durations.append(duration)
        except Exception as e:
            logging.warning(f"Case {caseid}: segmentation error - {e}")
            continue

    # Save segmentation table
    seg_df = pd.DataFrame.from_records(records)
    out_parquet = safe_path(WORK / "data" / "processed" / "induction_segments.parquet")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_parquet(out_parquet, index=False)
    logging.info(f"Saved induction segments: {out_parquet}")

    # Plot duration histogram
    plt.figure(figsize=(4,3))
    plt.hist(durations, bins=20, color="#0072B2", edgecolor="black")
    plt.title("Induction Duration (seconds)")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    fig_path = safe_path(WORK / "outputs" / "figures" / "induction_duration_histogram.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    logging.info(f"Saved duration histogram: {fig_path}")

    # Summary stats
    if durations:
        logging.info(f"Mean duration: {np.mean(durations):.1f}s, median: {np.median(durations):.1f}s")
        logging.info(f"Number of successfully segmented cases: {len(seg_df)}")
    else:
        logging.warning("No valid segmentation found.")

    logging.info("=== DONE ===")

if __name__ == "__main__":
    main()
