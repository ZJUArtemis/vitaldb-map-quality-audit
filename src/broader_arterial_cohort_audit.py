#!/usr/bin/env python3
"""PPG-free arterial-source sensitivity within the same clinical filters.

The original 926-case source cohort required PPG because RI was an intended
predictor.  This sensitivity removes only that signal requirement.  It retains
the executable clinical filters used by cohort construction and therefore is
not presented as a prevalence estimate for all VitalDB cases.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb


HERE = Path(__file__).resolve()
PROJECT = Path(os.environ.get(
    "TOPIC10_PROJECT_ROOT",
    HERE.parents[1] if (HERE.parents[1] / "outputs").exists() else HERE.parents[3],
))
OUT = Path(__file__).resolve().parent
TRACKS = PROJECT / "outputs/metrics/track_availability.csv"
PRIMARY = PROJECT / "outputs/metrics/revision_v5_fixed_window_case_results.csv"
DEFAULT_VITAL_DIR = PROJECT.parents[1] / "physionet.org/files/vitaldb/1.0.0/vital_files"
if not DEFAULT_VITAL_DIR.exists():
    DEFAULT_VITAL_DIR = PROJECT / "vitaldb_data"
VITAL_DIR = Path(os.environ.get("VITALDB_VITAL_DIR", DEFAULT_VITAL_DIR))
CACHE = OUT / "broader_arterial_v9_cache"


def find_track(vf, name):
    return next((track for track in vf.trks.values() if track.dtname == name), None)


def onset(caseid: int):
    path = VITAL_DIR / f"{caseid:04d}.vital"
    try:
        vf = vitaldb.VitalFile(str(path), track_names=["Orchestra/PPF20_RATE"])
        track = find_track(vf, "Orchestra/PPF20_RATE")
        if track is None:
            return None
        positive = [float(r["dt"]) for r in track.recs if np.isfinite(float(r["val"])) and float(r["val"]) > 0]
        return min(positive) if positive else None
    except Exception:
        return None


def load_native_module():
    path = OUT / "native_coverage_audit.py"
    spec = importlib.util.spec_from_file_location("native_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_new(caseid: int, force: bool = False):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{caseid:04d}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    t0 = onset(caseid)
    if t0 is None:
        row = {"caseid": caseid, "onset_estimable": False, "native_primary_valid": False, "native_event30": None}
    else:
        native = load_native_module()
        # Use the same native-record implementation and its cache.
        row = native.process(caseid, t0, force)
        row["onset_estimable"] = True
    cache.write_text(json.dumps(row, indent=2, allow_nan=False))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tracks = pd.read_csv(TRACKS)
    broad_ids = sorted(tracks.loc[tracks.has_abp & tracks.has_propofol, "caseid"].astype(int).tolist())
    primary = pd.read_csv(PRIMARY)[["caseid", "t_start_absolute"]]
    primary_ids = set(primary.caseid.astype(int))
    added = [caseid for caseid in broad_ids if caseid not in primary_ids]

    # Existing primary cases use the already completed native validation.
    native_existing = pd.read_csv(OUT / "native_numeric_coverage_case_results_v9.csv")
    existing_rows = native_existing[native_existing.caseid.isin(primary_ids)].to_dict("records")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        added_rows = pool.starmap(process_new, [(caseid, args.force) for caseid in added])
    all_rows = pd.DataFrame(existing_rows + added_rows).drop_duplicates("caseid").sort_values("caseid")
    valid = all_rows["native_primary_valid"].fillna(False).astype(bool)
    labels = all_rows.loc[valid, "native_event30"].dropna().astype(int)
    summary = {
        "scope": "same executable adult/general/non-cardiothoracic/demographic filters; PPG requirement removed",
        "not_all_vitaldb": True,
        "primary_ri_inclusive_n": int(len(primary_ids)),
        "ppg_free_arterial_source_n": int(len(broad_ids)),
        "additional_cases_n": int(len(added)),
        "onset_estimable_n": int(all_rows.get("onset_estimable", pd.Series(True, index=all_rows.index)).fillna(True).sum()),
        "native_primary_valid_n": int(valid.sum()),
        "events": int(labels.sum()),
        "non_events": int(len(labels) - labels.sum()),
    }
    all_rows.to_csv(OUT / "broader_arterial_cohort_case_results_v9.csv", index=False)
    (OUT / "broader_arterial_cohort_summary_v9.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
