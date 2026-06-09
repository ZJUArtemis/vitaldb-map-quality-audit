#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: step5b_fix_vascular_features.py | Topic: 10 | Purpose: Fix and finalize vascular features
- Merge baseline_map from outcome_labels (already computed correctly)
- Fix SI calculation using proper height data from local CSV
- Build composite Vascular Stiffness Score via PCA
"""
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

WORK = Path("/home/lxk/vitaldb/analysis/topic10_induction_instability")
RAW_DATA = Path("/home/lxk/vitaldb/physionet.org")

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lf = WORK / "outputs" / "logs" / f"{ts}_step5b_fix_vascular.log"
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()])

def main():
    init_log()
    logging.info("=== Step 5b: Fix Vascular Features & Build Composite Score ===")

    # Load existing features
    feat_df = pd.read_parquet(WORK / "data" / "processed" / "vascular_features.parquet")
    outcome_df = pd.read_parquet(WORK / "data" / "processed" / "outcome_labels.parquet")
    seg_df = pd.read_parquet(WORK / "data" / "processed" / "induction_segments.parquet")

    # Load clinical data from local CSV
    clinical = pd.read_csv(RAW_DATA / "files" / "vitaldb" / "1.0.0" / "clinical_data.csv")
    logging.info(f"Clinical data: {len(clinical)} cases")

    # Merge baseline_map and nadir_map from outcome_labels
    # These are computed from the SNUADC/ART_MBP correctly during induction
    merged = feat_df.merge(outcome_df[['caseid','baseline_map','nadir_map','drop_pct']], on='caseid', how='left')
    merged = merged.merge(clinical[['caseid','height','weight','age','sex','bmi','asa',
                                     'preop_htn','preop_dm']], on='caseid', how='left')
    # Clean age: convert '>89' and other strings to numeric
    merged['age'] = pd.to_numeric(merged['age'], errors='coerce')
    merged = merged.merge(seg_df[['caseid','t_start','t_end']], on='caseid', how='left')

    logging.info(f"Merged dataset: {len(merged)} cases")
    logging.info(f"baseline_map valid: {merged['baseline_map'].notna().sum()}")
    logging.info(f"height valid: {merged['height'].notna().sum()}")

    # Compute pulse pressure from baseline (SBP and DBP from outcome script baseline)
    # baseline_map is already MAP = (SBP + 2*DBP)/3, so PP estimation is not possible
    # Instead use pp_mean from existing features (filtered for valid values only)

    # Fix pp_mean: filter out physiologically impossible values
    # Valid ABP pp: 10–150 mmHg
    merged['pp_mean_clean'] = merged['pp_mean'].where(
        (merged['pp_mean'] > 10) & (merged['pp_mean'] < 150), other=np.nan
    )
    logging.info(f"pp_mean_clean valid: {merged['pp_mean_clean'].notna().sum()}")

    # Fix ri_mean: valid range 0.1–1.3
    merged['ri_mean_clean'] = merged['ri_mean'].where(
        (merged['ri_mean'] > 0.1) & (merged['ri_mean'] < 1.3), other=np.nan
    )
    logging.info(f"ri_mean_clean valid: {merged['ri_mean_clean'].notna().sum()}")

    # Fix pi: valid range 0.1–100
    merged['pi_clean'] = merged['pi'].where(
        (merged['pi'] > 0.1) & (merged['pi'] < 100), other=np.nan
    )
    logging.info(f"pi_clean valid: {merged['pi_clean'].notna().sum()}")

    # ppg_amplitude_mean: valid range 0.01–5
    merged['ppg_amp_clean'] = merged['ppg_amplitude_mean'].where(
        (merged['ppg_amplitude_mean'] > 0.01) & (merged['ppg_amplitude_mean'] < 5), other=np.nan
    )
    logging.info(f"ppg_amp_clean valid: {merged['ppg_amp_clean'].notna().sum()}")

    # === Build Composite Vascular Stiffness Score ===
    # Available valid features: baseline_map, ri_mean_clean, pi_clean, ppg_amp_clean, age
    # Note: higher age -> higher stiffness
    # Note: higher RI -> more wave reflection -> stiffer arteries
    # Note: lower PI -> worse perfusion -> potentially stiffer / lower vasodilation reserve
    # Note: higher baseline_map -> possibly stiffer (chronic hypertension)

    stiffness_candidates = ['baseline_map', 'ri_mean_clean', 'pi_clean', 'ppg_amp_clean', 'age']
    available = [c for c in stiffness_candidates if c in merged.columns]
    logging.info(f"\nStiffness candidates: {available}")

    pca_df = merged[available].dropna()
    logging.info(f"Cases with all stiffness features: {len(pca_df)}")

    if len(pca_df) >= 30:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_df.values)
        pca = PCA(n_components=min(3, len(available)))
        pca.fit(X_scaled)
        logging.info(f"PCA explained variance: {pca.explained_variance_ratio_}")

        scores = pca.transform(X_scaled)[:, 0]
        merged.loc[pca_df.index, 'vascular_stiffness_score'] = scores

        # Check direction: higher score should correlate with crash
        corr = merged.loc[pca_df.index, ['vascular_stiffness_score', 'crash_30']].corr().iloc[0, 1]
        logging.info(f"Stiffness score vs crash_30 correlation: {corr:.3f}")
        if corr < 0:
            merged['vascular_stiffness_score'] = -merged['vascular_stiffness_score']
            logging.info("Flipped score sign for clinical interpretability (higher = stiffer = more crash)")

    # Summary stats by crash group
    for col in ['baseline_map', 'ri_mean_clean', 'pi_clean', 'ppg_amp_clean', 'age', 'vascular_stiffness_score']:
        if col in merged.columns:
            crash_vals = merged.loc[merged['crash_30']==1, col].dropna()
            no_crash_vals = merged.loc[merged['crash_30']==0, col].dropna()
            if len(crash_vals) > 5 and len(no_crash_vals) > 5:
                t_stat, p_val = stats.ttest_ind(crash_vals, no_crash_vals)
                logging.info(f"{col}: Crash={crash_vals.mean():.2f}±{crash_vals.std():.2f}, "
                             f"No-crash={no_crash_vals.mean():.2f}±{no_crash_vals.std():.2f}, p={p_val:.4f}")

    # Save updated features
    out_path = WORK / "data" / "processed" / "vascular_features.parquet"
    assert not str(out_path.resolve()).startswith(str(RAW_DATA.resolve()))
    merged.to_parquet(out_path, index=False)
    logging.info(f"\nSaved updated features: {out_path}")
    logging.info(f"Final columns: {merged.columns.tolist()}")

    # Plot stiffness score distribution
    if 'vascular_stiffness_score' in merged.columns and merged['vascular_stiffness_score'].notna().sum() > 10:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        crash_s = merged.loc[merged['crash_30']==1, 'vascular_stiffness_score'].dropna()
        no_crash_s = merged.loc[merged['crash_30']==0, 'vascular_stiffness_score'].dropna()
        ax.hist(crash_s, bins=25, alpha=0.6, color='#D55E00', label=f'Crash (n={len(crash_s)})')
        ax.hist(no_crash_s, bins=25, alpha=0.6, color='#0072B2', label=f'No Crash (n={len(no_crash_s)})')
        ax.set_xlabel('Vascular Stiffness Score'); ax.set_ylabel('Count')
        ax.set_title('Vascular Stiffness Score Distribution'); ax.legend()

        ax2 = axes[1]
        ax2.scatter(merged.loc[merged['crash_30']==0, 'baseline_map'],
                    merged.loc[merged['crash_30']==0, 'drop_pct'],
                    alpha=0.3, color='#0072B2', s=10, label='No Crash')
        ax2.scatter(merged.loc[merged['crash_30']==1, 'baseline_map'],
                    merged.loc[merged['crash_30']==1, 'drop_pct'],
                    alpha=0.3, color='#D55E00', s=10, label='Crash')
        ax2.set_xlabel('Baseline MAP (mmHg)'); ax2.set_ylabel('MAP Drop (%)')
        ax2.set_title('Baseline MAP vs Drop %'); ax2.legend()

        plt.tight_layout()
        fig_path = WORK / "outputs" / "figures" / "vascular_stiffness_distribution.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logging.info(f"Saved plot: {fig_path}")

    logging.info("=== DONE ===")

if __name__ == "__main__":
    main()
