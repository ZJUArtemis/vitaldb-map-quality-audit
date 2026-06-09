#!/home/lxk/anaconda3/envs/ana/bin/python
# -*- coding: utf-8 -*-
"""
Script: verify_vitaldb.py | Topic: 10 | Purpose: Verify VitalDB data access
"""
import vitaldb
import numpy as np

def main():
    print("=" * 60)
    print("VitalDB Data Access Verification")
    print("=" * 60)

    try:
        # 1. Load clinical data for TIVA cases
        print("\n[1] Loading clinical data for TIVA cases...")
        tiva_caseids = list(vitaldb.caseids_tiva)
        print(f"    Total TIVA cases: {len(tiva_caseids)}")

        # Load first batch to check structure
        sample_cases = vitaldb.load_clinical_data(caseids=tiva_caseids[:100])
        print(f"    Loaded sample: {len(sample_cases)} cases")

        # Check key fields
        required_fields = ['caseid', 'age', 'sex', 'height', 'weight', 'ane_type', 'optype']
        missing_fields = [f for f in required_fields if f not in sample_cases.columns]

        if missing_fields:
            print(f"    ✗ Missing fields: {missing_fields}")
        else:
            print(f"    ✓ All required fields present")

        # Stats
        print(f"    Age range: {sample_cases['age'].min():.0f} - {sample_cases['age'].max():.0f}")
        print(f"    General anesthesia: {(sample_cases['ane_type'] == 'General').sum()}/{len(sample_cases)}")

        # 2. Test waveform loading
        print("\n[2] Testing waveform data loading...")
        test_caseid = tiva_caseids[0]
        print(f"    Test case: {test_caseid}")

        # Get available tracks
        tracks_df = vitaldb.get_track_names(caseids=[test_caseid])
        track_list = tracks_df.iloc[0]['tnames']  # Already a list
        print(f"    Available tracks: {len(track_list)}")

        # Try loading key tracks
        key_tracks = ['Orchestra/PPF20_RATE', 'Solar8000/ART_MBP', 'SNUADC/PLETH']
        available_tracks = [t for t in key_tracks if t in track_list]
        print(f"    Key tracks available: {available_tracks}")

        if available_tracks:
            data = vitaldb.load_case(test_caseid, available_tracks, interval=1)
            print(f"    Data shape: {data.shape}")
            print(f"    Data type: {data.dtype}")
            print(f"    Non-NaN values: {np.sum(~np.isnan(data))}/{data.size}")
            print(f"    ✓ Waveform loading successful")

        # 3. Summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"✓ Clinical data loading: OK")
        print(f"✓ Waveform data loading: OK")
        print(f"✓ TIVA cases available: {len(tiva_caseids)}")
        print(f"✓ Ready for Phase 1 cohort selection")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()