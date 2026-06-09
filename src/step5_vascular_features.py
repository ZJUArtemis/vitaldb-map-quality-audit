#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step5_vascular_features.py | Topic: 10 | Purpose: 血管弹性特征提取 (ABP + PPG)
"""
import os, sys, logging, gc
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import vitaldb
import neurokit2 as nk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === PATH SAFETY ===
RAW_DATA = Path("vitaldb_data")  # local VitalDB physionet.org root; override as needed
WORK = Path(__file__).resolve().parents[1]
VITAL_FILES_DIR = RAW_DATA / "files" / "vitaldb" / "1.0.0" / "vital_files"

def load_local_case(caseid, tracks, interval):
    """Load tracks from local .vital file (no network required)."""
    fname = VITAL_FILES_DIR / f"{caseid:04d}.vital"
    if not fname.exists():
        return None
    vf = vitaldb.VitalFile(str(fname))
    return vf, vf.to_numpy(tracks, interval=interval)

def load_all_channels_local(caseid):
    """Parse vital file once and return all needed arrays."""
    fname = VITAL_FILES_DIR / f"{caseid:04d}.vital"
    if not fname.exists():
        return None, None, None
    vf = vitaldb.VitalFile(str(fname))
    abp_arr = vf.to_numpy(["Solar8000/ART_SBP", "Solar8000/ART_DBP", "Solar8000/ART_MBP"], interval=1)
    ppg_arr = vf.to_numpy(["SNUADC/PLETH"], interval=0.01)
    return abp_arr, ppg_arr

def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lf = WORK / "outputs" / "logs" / f"{ts}_step5_vascular_features.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()]
    )

def extract_abp_features(abp_signal, fs=500):
    """Extract vascular stiffness features from ABP waveform."""
    try:
        abp_clean = abp_signal[~np.isnan(abp_signal)]
        if len(abp_clean) < fs * 10:  # Need at least 10 seconds
            return None

        # Artifact rejection: keep only physiologically valid values
        abp_clean = abp_clean[(abp_clean >= 20) & (abp_clean <= 300)]
        if len(abp_clean) < fs * 10:
            return None

        # Find systolic peaks using neurokit2
        peaks_dict = nk.signal_findpeaks(abp_clean, height_min=40)
        peaks = peaks_dict["Peaks"]
        if len(peaks) < 5:
            return None

        # Find diastolic troughs between peaks
        troughs = []
        for i in range(len(peaks) - 1):
            seg = abp_clean[peaks[i]:peaks[i+1]]
            troughs.append(peaks[i] + np.argmin(seg))
        troughs = np.array(troughs)

        if len(troughs) < 4:
            return None

        # Beat-by-beat features
        sbp_vals, dbp_vals, pp_vals, dpdt_vals = [], [], [], []

        for i in range(min(len(peaks), len(troughs))):
            sbp = abp_clean[peaks[i]]
            dbp = abp_clean[troughs[i]]
            pp = sbp - dbp

            if pp < 5 or pp > 150:  # Artifact check
                continue

            sbp_vals.append(sbp)
            dbp_vals.append(dbp)
            pp_vals.append(pp)

            # dP/dt_max: max rate of pressure rise in systolic upstroke
            if i < len(troughs) and troughs[i] < peaks[i]:
                upstroke = abp_clean[troughs[i]:peaks[i]]
                if len(upstroke) > 1:
                    dpdt = np.max(np.diff(upstroke)) * fs
                    dpdt_vals.append(dpdt)

        if len(sbp_vals) < 3:
            return None

        sbp_arr = np.array(sbp_vals)
        dbp_arr = np.array(dbp_vals)
        pp_arr = np.array(pp_vals)

        # Augmentation Index (AIx) estimation
        # Using second derivative of ABP to find inflection point
        aix_vals = []
        for i in range(min(len(peaks) - 1, len(troughs))):
            if i >= len(troughs):
                break
            beat_start = troughs[i]
            beat_end = peaks[i]
            if beat_end <= beat_start + 5:
                continue
            beat = abp_clean[beat_start:beat_end]
            if len(beat) < 10:
                continue
            # Find inflection point via second derivative
            d2 = np.diff(np.diff(beat))
            if len(d2) < 3:
                continue
            inflection_candidates = np.where(np.diff(np.sign(d2)))[0]
            if len(inflection_candidates) > 0:
                infl_idx = inflection_candidates[0] + 2
                if infl_idx < len(beat):
                    p1 = beat[0]   # foot
                    p2 = beat[infl_idx]  # inflection (reflected wave)
                    pp_beat = beat[-1] - beat[0]
                    if pp_beat > 5:
                        aix = (p2 - p1) / pp_beat * 100
                        if -50 < aix < 100:  # Physiological range
                            aix_vals.append(aix)

        return {
            'sbp_mean': float(np.mean(sbp_arr)),
            'dbp_mean': float(np.mean(dbp_arr)),
            'pp_mean': float(np.mean(pp_arr)),
            'pp_cv': float(np.std(pp_arr) / np.mean(pp_arr)) if np.mean(pp_arr) > 0 else np.nan,
            'dpdt_max_mean': float(np.mean(dpdt_vals)) if dpdt_vals else np.nan,
            'aix_mean': float(np.mean(aix_vals)) if aix_vals else np.nan,
            'n_beats': len(sbp_vals)
        }
    except Exception as e:
        return None


def extract_ppg_features(ppg_signal, height_cm, fs=100):
    """Extract vascular stiffness features from PPG waveform."""
    try:
        ppg_clean = ppg_signal[~np.isnan(ppg_signal)]
        if len(ppg_clean) < fs * 10:
            return None

        # Normalize PPG
        ppg_norm = (ppg_clean - np.min(ppg_clean)) / (np.max(ppg_clean) - np.min(ppg_clean) + 1e-8)

        # Find systolic peaks
        peaks_dict = nk.signal_findpeaks(ppg_norm, height_min=0.3)
        peaks = peaks_dict["Peaks"]
        if len(peaks) < 5:
            return None

        # Find troughs
        troughs = []
        for i in range(len(peaks) - 1):
            seg = ppg_norm[peaks[i]:peaks[i+1]]
            troughs.append(peaks[i] + np.argmin(seg))
        troughs = np.array(troughs)

        if len(troughs) < 4:
            return None

        # Amplitude and Perfusion Index
        amp_vals = []
        ri_vals = []
        si_vals = []

        for i in range(min(len(peaks) - 1, len(troughs))):
            sys_amp = ppg_norm[peaks[i]]
            dia_val = ppg_norm[troughs[i]]
            amp = sys_amp - dia_val
            if amp < 0.05:
                continue
            amp_vals.append(amp)

            # Reflection Index (RI): diastolic peak / systolic peak
            # Find diastolic peak between systolic peak and next trough
            if i + 1 < len(troughs):
                seg = ppg_norm[peaks[i]:troughs[i+1]]
                if len(seg) > 5:
                    # Diastolic peak is secondary peak after systolic
                    mid = len(seg) // 2
                    if mid < len(seg):
                        dia_peak = np.max(seg[mid:])
                        ri = dia_peak / sys_amp if sys_amp > 0 else np.nan
                        if 0 < ri < 1.5:
                            ri_vals.append(ri)

            # Stiffness Index (SI) = height / time between systolic and diastolic peaks
            if i + 1 < len(troughs) and height_cm > 0:
                seg = ppg_norm[peaks[i]:troughs[i+1]]
                if len(seg) > 5:
                    mid = len(seg) // 2
                    if mid < len(seg):
                        dia_peak_idx = mid + np.argmax(seg[mid:])
                        dt = dia_peak_idx / fs  # seconds
                        if dt > 0.1:
                            si = (height_cm / 100.0) / dt  # m/s
                            if 1 < si < 20:  # Physiological range
                                si_vals.append(si)

        if len(amp_vals) < 3:
            return None

        # Perfusion Index: AC/DC ratio
        dc_component = np.mean(ppg_clean)
        ac_component = np.mean(amp_vals) * (np.max(ppg_clean) - np.min(ppg_clean))
        pi = (ac_component / (dc_component + 1e-8)) * 100 if dc_component > 0 else np.nan

        return {
            'ppg_amplitude_mean': float(np.mean(amp_vals)),
            'ppg_amplitude_cv': float(np.std(amp_vals) / np.mean(amp_vals)) if np.mean(amp_vals) > 0 else np.nan,
            'ri_mean': float(np.mean(ri_vals)) if ri_vals else np.nan,
            'si_mean': float(np.mean(si_vals)) if si_vals else np.nan,
            'pi': float(pi) if not np.isnan(pi) else np.nan,
            'n_ppg_beats': len(amp_vals)
        }
    except Exception as e:
        return None


def main():
    init_log()
    logging.info("=== Topic 10 Phase 4: Vascular Stiffness Feature Extraction ===")

    # Load induction segments
    seg_df = pd.read_parquet(WORK / "data" / "processed" / "induction_segments.parquet")
    outcome_df = pd.read_parquet(WORK / "data" / "processed" / "outcome_labels.parquet")
    merged = seg_df.merge(outcome_df, on='caseid', how='inner')
    logging.info(f"Processing {len(merged)} cases")

    # Load clinical data for height
    clinical = vitaldb.load_clinical_data(['height'])
    height_map = dict(zip(clinical['caseid'], clinical['height']))

    records = []
    abp_fail, ppg_fail = 0, 0

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Extracting features"):
        caseid = int(row['caseid'])
        t_start = int(row['t_start'])
        baseline_start = max(0, t_start - 300)  # 5 min before induction
        height_cm = height_map.get(caseid, np.nan)

        rec = {'caseid': caseid, 'crash_30': int(row['crash_30'])}

        try:
            # Parse vital file ONCE, extract both ABP (1Hz numeric) and PPG (100Hz)
            abp_array, ppg_array = load_all_channels_local(caseid)

            # ABP numeric features (SBP, DBP, MBP at 1Hz)
            if abp_array is not None and len(abp_array) > 0:
                abp_seg = abp_array[baseline_start:t_start, :]  # rows = seconds
                if len(abp_seg) >= 30:
                    sbp = abp_seg[:, 0]
                    dbp = abp_seg[:, 1]
                    mbp = abp_seg[:, 2]
                    # Remove NaN
                    sbp = sbp[~np.isnan(sbp)]
                    dbp = dbp[~np.isnan(dbp)]
                    mbp = mbp[~np.isnan(mbp)]
                    if len(sbp) >= 10:
                        pp = sbp - dbp[:len(sbp)]
                        rec['sbp_mean'] = float(np.mean(sbp))
                        rec['dbp_mean'] = float(np.mean(dbp))
                        rec['mbp_mean'] = float(np.mean(mbp))
                        rec['pp_mean'] = float(np.mean(pp))
                        rec['pp_cv'] = float(np.std(pp) / np.mean(pp)) if np.mean(pp) > 0 else np.nan
                        rec['sbp_cv'] = float(np.std(sbp) / np.mean(sbp)) if np.mean(sbp) > 0 else np.nan
                        rec['mbp_cv'] = float(np.std(mbp) / np.mean(mbp)) if np.mean(mbp) > 0 else np.nan
                    else:
                        abp_fail += 1
                else:
                    abp_fail += 1
            else:
                abp_fail += 1

            # PPG waveform features (100 Hz)
            if ppg_array is not None and len(ppg_array) > 0:
                start_idx = int(baseline_start / 0.01)
                end_idx = int(t_start / 0.01)
                if end_idx > start_idx and end_idx <= len(ppg_array):
                    ppg_seg = ppg_array[start_idx:end_idx, 0]
                    ppg_feats = extract_ppg_features(ppg_seg, height_cm, fs=100)
                    if ppg_feats:
                        rec.update(ppg_feats)
                    else:
                        ppg_fail += 1
                else:
                    ppg_fail += 1
            else:
                ppg_fail += 1

        except Exception as e:
            logging.warning(f"Case {caseid}: {e}")
            abp_fail += 1
            ppg_fail += 1

        records.append(rec)
        gc.collect()

    feat_df = pd.DataFrame(records)
    out_path = safe_path(WORK / "data" / "processed" / "vascular_features.parquet")
    feat_df.to_parquet(out_path, index=False)
    logging.info(f"Saved vascular features: {out_path}")
    logging.info(f"ABP extraction failures: {abp_fail}/{len(merged)}")
    logging.info(f"PPG extraction failures: {ppg_fail}/{len(merged)}")

    # Feature availability summary
    abp_cols = ['sbp_mean', 'dbp_mean', 'pp_mean', 'pp_cv', 'dpdt_max_mean', 'aix_mean']
    ppg_cols = ['ppg_amplitude_mean', 'ri_mean', 'si_mean', 'pi']
    for col in abp_cols + ppg_cols:
        if col in feat_df.columns:
            n_valid = feat_df[col].notna().sum()
            logging.info(f"  {col}: {n_valid}/{len(feat_df)} valid ({n_valid/len(feat_df)*100:.1f}%)")

    # === Step 4.3: Composite Vascular Stiffness Score (PCA) ===
    stiffness_cols = [c for c in ['aix_mean', 'si_mean', 'ri_mean', 'dpdt_max_mean', 'pp_mean']
                      if c in feat_df.columns]
    logging.info(f"\nBuilding composite stiffness score from: {stiffness_cols}")

    pca_df = feat_df[stiffness_cols].dropna()
    logging.info(f"Cases with complete stiffness features: {len(pca_df)}")

    if len(pca_df) >= 50:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_df.values)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X_scaled).flatten()
        logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]:.2%}")

        feat_df.loc[pca_df.index, 'vascular_stiffness_score'] = scores
        feat_df.to_parquet(out_path, index=False)

        # Plot stiffness score distribution by crash group
        fig, ax = plt.subplots(figsize=(6, 4))
        crash_scores = feat_df.loc[feat_df['crash_30'] == 1, 'vascular_stiffness_score'].dropna()
        no_crash_scores = feat_df.loc[feat_df['crash_30'] == 0, 'vascular_stiffness_score'].dropna()
        ax.hist(crash_scores, bins=30, alpha=0.6, color='#D55E00', label=f'Crash (n={len(crash_scores)})')
        ax.hist(no_crash_scores, bins=30, alpha=0.6, color='#0072B2', label=f'No Crash (n={len(no_crash_scores)})')
        ax.set_xlabel('Vascular Stiffness Score (PCA)')
        ax.set_ylabel('Count')
        ax.set_title('Vascular Stiffness Score Distribution')
        ax.legend()
        plt.tight_layout()
        fig_path = safe_path(WORK / "outputs" / "figures" / "vascular_stiffness_distribution.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logging.info(f"Saved stiffness distribution plot: {fig_path}")

        logging.info(f"\nCrash group stiffness score: mean={crash_scores.mean():.3f}, std={crash_scores.std():.3f}")
        logging.info(f"No-crash group stiffness score: mean={no_crash_scores.mean():.3f}, std={no_crash_scores.std():.3f}")
    else:
        logging.warning(f"Too few cases ({len(pca_df)}) for PCA — skipping composite score")

    logging.info("=== DONE ===")

if __name__ == "__main__":
    main()
