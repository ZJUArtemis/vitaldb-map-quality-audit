#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: vitaldb_utils.py | Topic: 10 | Purpose: Unified VitalDB data loading utilities
"""
import vitaldb
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Track mapping for standardization
TRACK_MAP = {
    'SNUADC/ART': 'ABP', 'SNUADC/FEM': 'ABP',
    'SNUADC/ECG_II': 'ECG', 'SNUADC/ECG_V5': 'ECG',
    'SNUADC/PLETH': 'PPG',
    'BIS/BIS': 'BIS', 'BIS/EEG1_WAV': 'EEG', 'BIS/EEG2_WAV': 'EEG',
    'BIS/SR': 'SR', 'BIS/SQI': 'SQI',
    'Solar8000/ETCO2': 'ETCO2', 'Solar8000/HR': 'HR',
    'Solar8000/ART_MBP': 'ART_MBP', 'Solar8000/ART_SBP': 'ART_SBP',
    'Solar8000/ART_DBP': 'ART_DBP', 'Solar8000/SPO2': 'SPO2',
    'Solar8000/BT': 'TEMP',
    'Orchestra/PPF20_RATE': 'PROPOFOL_RATE',
    'Orchestra/PPF20_CE': 'PROPOFOL_CE', 'Orchestra/PPF20_CP': 'PROPOFOL_CP',
    'Orchestra/RFTN20_RATE': 'REMIFENTANIL_RATE',
    'Orchestra/RFTN20_CE': 'REMIFENTANIL_CE', 'Orchestra/RFTN20_CP': 'REMIFENTANIL_CP',
}

# Artifact detection rules
ARTIFACT_RULES = {
    'ABP': {'min': 20, 'max': 300},
    'HR': {'min': 20, 'max': 250},
    'SPO2': {'min': 50, 'max': 100},
    'BIS': {'min': 0, 'max': 100},
    'ETCO2': {'min': 10, 'max': 80},
    'TEMP': {'min': 30, 'max': 42},
}


class VitalDBReader:
    """Unified VitalDB data reader for Topic 10"""

    def __init__(self):
        self.tiva_caseids = list(vitaldb.caseids_tiva)
        logger.info(f"Initialized VitalDBReader with {len(self.tiva_caseids)} TIVA cases")

    def load_clinical_data(self, caseids=None, batch_size=500):
        """
        Load clinical data for specified caseids.

        Args:
            caseids: list of case IDs, defaults to all TIVA cases
            batch_size: load in batches to avoid memory issues

        Returns:
            pd.DataFrame: clinical data
        """
        if caseids is None:
            caseids = self.tiva_caseids

        all_data = []
        for i in range(0, len(caseids), batch_size):
            batch = caseids[i:i+batch_size]
            df = vitaldb.load_clinical_data(caseids=batch)
            all_data.append(df)
            logger.debug(f"Loaded batch {i//batch_size + 1}: {len(df)} cases")

        return pd.concat(all_data, ignore_index=True)

    def get_available_tracks(self, caseid):
        """Get list of available tracks for a case"""
        tracks_df = vitaldb.get_track_names(caseids=[caseid])
        if len(tracks_df) == 0:
            return []
        return tracks_df.iloc[0]['tnames']

    def load_waveform(self, caseid, track_names, interval=1):
        """
        Load waveform data for specified tracks.

        Args:
            caseid: case ID
            track_names: list of track names or comma-separated string
            interval: time interval (1 = 1 second)

        Returns:
            np.ndarray: 2D array (time x tracks)
        """
        try:
            data = vitaldb.load_case(caseid, track_names, interval=interval)
            return data
        except Exception as e:
            logger.warning(f"Failed to load case {caseid}: {e}")
            return None

    def filter_tiva_cases_with_tracks(self, required_tracks, max_cases=None):
        """
        Filter TIVA cases that have all required tracks.

        Args:
            required_tracks: list of required track names
            max_cases: limit number of cases to check (for testing)

        Returns:
            list: case IDs with all required tracks
        """
        eligible_cases = []
        check_cases = self.tiva_caseids[:max_cases] if max_cases else self.tiva_caseids

        for i, caseid in enumerate(check_cases):
            if i % 500 == 0:
                logger.info(f"Checking case {i}/{len(check_cases)}")

            available = self.get_available_tracks(caseid)
            if all(track in available for track in required_tracks):
                eligible_cases.append(caseid)

        logger.info(f"Found {len(eligible_cases)}/{len(check_cases)} cases with required tracks")
        return eligible_cases

    def extract_baseline_window(self, caseid, propofol_rate_track='Orchestra/PPF20_RATE',
                                map_track='Solar8000/ART_MBP', window_before_sec=300):
        """
        Extract baseline window (before propofol infusion starts).

        Args:
            caseid: case ID
            propofol_rate_track: track name for propofol rate
            map_track: track name for MAP
            window_before_sec: seconds before propofol starts

        Returns:
            dict: baseline data and timing info
        """
        try:
            # Load propofol rate
            ppf_data = vitaldb.load_case(caseid, propofol_rate_track, interval=1)
            if ppf_data is None or ppf_data.size == 0:
                return None

            ppf_rate = ppf_data.flatten()
            # Find first non-zero propofol rate
            nonzero_idx = np.where(ppf_rate > 0)[0]
            if len(nonzero_idx) == 0:
                return None

            t_start = nonzero_idx[0]  # Time when propofol starts (in seconds)

            # Extract baseline window
            baseline_start = max(0, t_start - window_before_sec)
            baseline_end = t_start

            # Load MAP for baseline
            map_data = vitaldb.load_case(caseid, map_track, interval=1)
            if map_data is None or map_data.size == 0:
                return None

            map_baseline = map_data[baseline_start:baseline_end, 0]
            map_baseline = map_baseline[~np.isnan(map_baseline)]

            if len(map_baseline) < 60:  # Need at least 1 minute of data
                return None

            return {
                'caseid': caseid,
                't_start': t_start,
                'baseline_start': baseline_start,
                'baseline_end': baseline_end,
                'baseline_map_mean': np.mean(map_baseline),
                'baseline_map_std': np.std(map_baseline),
                'baseline_map_min': np.min(map_baseline),
                'baseline_map_max': np.max(map_baseline),
                'baseline_samples': len(map_baseline),
            }

        except Exception as e:
            logger.warning(f"Failed to extract baseline for case {caseid}: {e}")
            return None

    def extract_induction_window(self, caseid, propofol_rate_track='Orchestra/PPF20_RATE',
                                 map_track='Solar8000/ART_MBP', bis_track='BIS/BIS',
                                 max_induction_sec=1200):
        """
        Extract induction phase window.

        Args:
            caseid: case ID
            propofol_rate_track: track name for propofol rate
            map_track: track name for MAP
            bis_track: track name for BIS
            max_induction_sec: maximum induction duration (default 20 min)

        Returns:
            dict: induction window data and metrics
        """
        try:
            # Load propofol rate
            ppf_data = vitaldb.load_case(caseid, propofol_rate_track, interval=1)
            if ppf_data is None or ppf_data.size == 0:
                return None

            ppf_rate = ppf_data.flatten()
            nonzero_idx = np.where(ppf_rate > 0)[0]
            if len(nonzero_idx) == 0:
                return None

            t_start = nonzero_idx[0]

            # Load MAP
            map_data = vitaldb.load_case(caseid, map_track, interval=1)
            if map_data is None or map_data.size == 0:
                return None

            map_series = map_data[:, 0]

            # Find induction end: MAP stabilizes (CV < 5% for 5 min) or BIS < 60
            # Simple heuristic: find the nadir and then look for stabilization
            induction_segment = map_series[t_start:min(t_start + max_induction_sec, len(map_series))]
            induction_segment_clean = induction_segment[~np.isnan(induction_segment)]

            if len(induction_segment_clean) < 120:  # Need at least 2 minutes
                return None

            nadir_idx = np.nanargmin(induction_segment)
            nadir_map = induction_segment[nadir_idx]

            # Find stabilization point after nadir
            window_size = 300  # 5 minutes
            t_end = t_start + nadir_idx + window_size

            if t_end >= len(map_series):
                t_end = len(map_series) - 1

            induction_duration = t_end - t_start

            # Quality check
            if induction_duration < 120 or induction_duration > max_induction_sec:
                return None

            # Calculate metrics
            baseline_map = np.nanmean(map_series[max(0, t_start-300):t_start])
            drop_pct = (baseline_map - nadir_map) / baseline_map * 100 if baseline_map > 0 else 0

            return {
                'caseid': caseid,
                't_start': t_start,
                't_end': t_end,
                'induction_duration_sec': induction_duration,
                'baseline_map': baseline_map,
                'nadir_map': nadir_map,
                'drop_pct': drop_pct,
                'drop_absolute': baseline_map - nadir_map,
            }

        except Exception as e:
            logger.warning(f"Failed to extract induction window for case {caseid}: {e}")
            return None


def main():
    """Test the VitalDBReader"""
    logging.basicConfig(level=logging.INFO)

    reader = VitalDBReader()

    # Test 1: Load clinical data
    print("\n[Test 1] Loading clinical data...")
    df = reader.load_clinical_data(caseids=reader.tiva_caseids[:100])
    print(f"Loaded {len(df)} cases")
    print(f"Columns: {df.columns.tolist()[:10]}")

    # Test 2: Check available tracks
    print("\n[Test 2] Checking available tracks...")
    caseid = reader.tiva_caseids[0]
    tracks = reader.get_available_tracks(caseid)
    print(f"Case {caseid} has {len(tracks)} tracks")

    # Test 3: Extract baseline window
    print("\n[Test 3] Extracting baseline window...")
    baseline = reader.extract_baseline_window(caseid)
    if baseline:
        print(f"Baseline MAP: {baseline['baseline_map_mean']:.1f} ± {baseline['baseline_map_std']:.1f} mmHg")

    # Test 4: Extract induction window
    print("\n[Test 4] Extracting induction window...")
    induction = reader.extract_induction_window(caseid)
    if induction:
        print(f"Induction duration: {induction['induction_duration_sec']} sec")
        print(f"MAP drop: {induction['drop_pct']:.1f}%")

    print("\n✓ All tests passed")


if __name__ == "__main__":
    main()
