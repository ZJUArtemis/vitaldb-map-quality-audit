#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step4_changepoint_detection.py | Topic: 10 | Purpose: 变点检测 (CUSUM + Bayesian)
"""
import os, sys, json, logging
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import vitaldb
import ruptures as rpt

# === PATH SAFETY ===
RAW_DATA = Path("vitaldb_data")  # local VitalDB physionet.org root; override as needed
WORK = Path(__file__).resolve().parents[1]

def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = WORK / "outputs" / "logs"
    lf = log_dir / f"{ts}_step4_changepoint_detection.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()]
    )
    return lf

def main():
    log_file = init_log()
    logging.info("=== Topic 10 Phase 3: Change-Point Detection ===")

    # Load induction segments
    seg_path = WORK / "data" / "processed" / "induction_segments.parquet"
    seg_df = pd.read_parquet(seg_path)
    logging.info(f"Loaded {len(seg_df)} segmented cases")

    # Load outcome labels
    outcome_path = WORK / "data" / "processed" / "outcome_labels.parquet"
    outcome_df = pd.read_parquet(outcome_path)
    logging.info(f"Loaded {len(outcome_df)} outcome labels")

    # Merge
    merged = seg_df.merge(outcome_df, on='caseid', how='inner')
    logging.info(f"Merged dataset: {len(merged)} cases")

    changepoint_records = []
    example_cases = []  # For visualization

    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Detecting changepoints"):
        caseid = row['caseid']
        t_start = int(row['t_start'])
        t_end = int(row['t_end'])
        crash_30 = row['crash_30']

        try:
            # Load MAP during induction
            map_array = vitaldb.load_case(caseid, ["Solar8000/ART_MBP"], interval=1)
            if map_array is None or len(map_array) == 0:
                continue
            map_data = pd.Series(map_array[:, 0], index=np.arange(len(map_array)))

            # Extract induction window
            induction_map = map_data.iloc[t_start:t_end]
            if len(induction_map) < 60:  # Too short
                continue

            # Remove NaN
            induction_map_clean = induction_map.dropna()
            if len(induction_map_clean) < 60:
                continue

            # CUSUM change-point detection
            # Use Binseg with l2 model (O(n log n), much faster than PELT+rbf)
            signal = induction_map_clean.values.reshape(-1, 1)
            try:
                algo = rpt.Binseg(model="l2", min_size=20, jump=5).fit(signal)
                changepoints = algo.predict(n_bkps=2)
            except Exception as e:
                logging.warning(f"Case {caseid}: Binseg failed - {e}")
                continue

            # Extract first significant changepoint
            if len(changepoints) > 1:
                cp_idx = changepoints[0]
                cp_time_relative = induction_map_clean.index[cp_idx] - t_start

                # Calculate metrics
                map_before_cp = induction_map_clean.iloc[:cp_idx].mean()
                map_after_cp = induction_map_clean.iloc[cp_idx:].mean()
                drop_magnitude = map_before_cp - map_after_cp

                # Time to nadir
                nadir_idx = induction_map_clean.idxmin()
                time_to_nadir = nadir_idx - t_start
                nadir_map = induction_map_clean.min()

                # Drop rate (mmHg/min)
                time_window = (nadir_idx - (t_start + cp_time_relative)) / 60.0  # minutes
                if time_window > 0:
                    drop_rate = drop_magnitude / time_window
                else:
                    drop_rate = 0.0

                changepoint_records.append({
                    'caseid': caseid,
                    'cp_time_relative': float(cp_time_relative),
                    'map_before_cp': float(map_before_cp),
                    'map_after_cp': float(map_after_cp),
                    'drop_magnitude': float(drop_magnitude),
                    'drop_rate': float(drop_rate),
                    'nadir_map': float(nadir_map),
                    'time_to_nadir': float(time_to_nadir),
                    'crash_30': int(crash_30)
                })

                # Collect examples for visualization (5 crash + 5 non-crash)
                if len(example_cases) < 10:
                    if (crash_30 == 1 and sum(1 for e in example_cases if e['crash_30'] == 1) < 5) or \
                       (crash_30 == 0 and sum(1 for e in example_cases if e['crash_30'] == 0) < 5):
                        example_cases.append({
                            'caseid': caseid,
                            'map_series': induction_map_clean,
                            'cp_idx': cp_idx,
                            'cp_time': cp_time_relative,
                            't_start': t_start,
                            'crash_30': crash_30
                        })

        except Exception as e:
            logging.warning(f"Case {caseid}: changepoint detection error - {e}")
            continue

    # Save changepoint data
    cp_df = pd.DataFrame(changepoint_records)
    out_path = safe_path(WORK / "data" / "processed" / "changepoints.parquet")
    cp_df.to_parquet(out_path, index=False)
    logging.info(f"Saved changepoint data: {out_path}")

    # Summary statistics
    logging.info(f"\n=== Changepoint Detection Summary ===")
    logging.info(f"Total cases with changepoints: {len(cp_df)}")

    crash_cp = cp_df[cp_df['crash_30'] == 1]
    no_crash_cp = cp_df[cp_df['crash_30'] == 0]

    logging.info(f"\nCrash group (n={len(crash_cp)}):")
    logging.info(f"  Mean lag to changepoint: {crash_cp['cp_time_relative'].mean():.1f}s")
    logging.info(f"  Mean drop magnitude: {crash_cp['drop_magnitude'].mean():.1f} mmHg")
    logging.info(f"  Mean drop rate: {crash_cp['drop_rate'].mean():.1f} mmHg/min")

    logging.info(f"\nNo-crash group (n={len(no_crash_cp)}):")
    logging.info(f"  Mean lag to changepoint: {no_crash_cp['cp_time_relative'].mean():.1f}s")
    logging.info(f"  Mean drop magnitude: {no_crash_cp['drop_magnitude'].mean():.1f} mmHg")
    logging.info(f"  Mean drop rate: {no_crash_cp['drop_rate'].mean():.1f} mmHg/min")

    # Visualization: Example cases
    if len(example_cases) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()

        for i, example in enumerate(example_cases[:6]):
            ax = axes[i]
            map_series = example['map_series']
            cp_idx = example['cp_idx']
            t_start = example['t_start']
            crash = example['crash_30']

            # Plot MAP
            time_axis = (map_series.index - t_start) / 60.0  # minutes
            ax.plot(time_axis, map_series.values, 'k-', linewidth=1)

            # Mark changepoint
            cp_time_min = (map_series.index[cp_idx] - t_start) / 60.0
            ax.axvline(cp_time_min, color='red', linestyle='--', linewidth=1.5, label='Changepoint')

            ax.set_xlabel('Time (min)')
            ax.set_ylabel('MAP (mmHg)')
            title_suffix = 'Crash' if crash == 1 else 'No Crash'
            ax.set_title(f"Case {example['caseid']} ({title_suffix})", fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7)

        # Hide unused subplots
        for i in range(len(example_cases), 6):
            axes[i].axis('off')

        plt.tight_layout()
        fig_path = safe_path(WORK / "outputs" / "figures" / "changepoint_examples.png")
        plt.savefig(fig_path, dpi=300)
        logging.info(f"Saved changepoint examples: {fig_path}")
        plt.close()

    logging.info("=== DONE ===")

if __name__ == "__main__":
    main()
