#!/usr/bin/env python3
"""Site-combined invasive-waveform (ART/FEM) sensitivity audit.

This script reuses the fixed propofol-defined windows and the operational raw
waveform criteria of the locked fixed-window audit.  It scans every one of the 926 source
cases for SNUADC/FEM directly from its local .vital file; it does not use the
remote track-name API or treat a loading error as absent data.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import vitaldb
from project_paths import METRICS_DIR, VITAL_DIR

import fixed_window_validity_audit as fixed_audit

OUT = METRICS_DIR
VITAL = VITAL_DIR
CASE_RESULTS = METRICS_DIR / "revision_v5_fixed_window_case_results.csv"
CACHE = METRICS_DIR / "fem_waveform_v7_cache"
# Optional locally generated FEM track-availability metadata (not distributed).
FEM_METADATA = METRICS_DIR / "fem_track_availability_metadata.csv"


def process_case(caseid: int, onset: float, force: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{caseid:04d}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    out = {"caseid": caseid, "t_start_absolute": onset, "fem_track_present": False}
    path = VITAL / f"{caseid:04d}.vital"
    if not path.exists():
        out["error"] = "vital file missing"
        cache.write_text(json.dumps(out, indent=2))
        return out
    try:
        vf = vitaldb.VitalFile(str(path), track_names=["SNUADC/FEM"])
        track = fixed_audit.find_track(vf, "SNUADC/FEM")
        out["fem_track_present"] = track is not None
        if track is None:
            out["error"] = None
        else:
            rel = float(onset) - float(vf.dtstart)
            raw = fixed_audit.raw_window_array(track, float(vf.dtstart), rel - 300, 300)
            mean, amplitude, observed = fixed_audit.raw_second_metrics(raw, float(track.srate))
            out["fem_srate_hz"] = float(track.srate)
            out["fem_baseline_any_observed"] = bool(observed.any())
            out["fem_baseline_observed_fraction"] = float(observed.mean())
            for threshold in (3, 5, 10):
                pulsatile = observed & np.isfinite(mean) & (mean >= 20) & (mean <= 200) & np.isfinite(amplitude) & (amplitude >= threshold)
                out[f"fem_baseline_any_pulsatile_amp{threshold}"] = bool(pulsatile.any())
                out[f"fem_baseline_pulsatile_fraction_amp{threshold}"] = float(pulsatile.mean())
            out["error"] = None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    cache.write_text(json.dumps(out, indent=2, allow_nan=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    source = pd.read_csv(CASE_RESULTS)
    source = source[["caseid", "t_start_absolute"]].dropna().astype({"caseid": int})
    rows = []
    # VitalFile can retain sizeable buffers.  Recycling workers after each file
    # prevents memory accumulation during this one-time full-cohort audit.
    jobs = [(int(r.caseid), float(r.t_start_absolute), args.force) for r in source.itertuples(index=False)]
    with mp.get_context("spawn").Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.starmap(process_case, jobs), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"processed {i}/{len(jobs)}", flush=True)
    data = pd.DataFrame(rows).sort_values("caseid")
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "fem_waveform_v7_case_results.csv", index=False)
    present = data.fem_track_present.fillna(False)
    metadata_candidates = None
    if FEM_METADATA.is_file():
        metadata = pd.read_csv(FEM_METADATA)
        if "has_SNUADC_FEM" in metadata:
            metadata_candidates = int(metadata["has_SNUADC_FEM"].fillna(False).sum())
    summary = {
        "source_cases": int(len(data)),
        "processing_errors": int(data.error.notna().sum()),
        "fem_metadata_candidates_n": metadata_candidates,
        "fem_local_raw_track_present_n": int(present.sum()),
    }
    for threshold in (3, 5, 10):
        frac = data.get(f"fem_baseline_pulsatile_fraction_amp{threshold}", pd.Series(0.0, index=data.index)).fillna(0)
        anyp = data.get(f"fem_baseline_any_pulsatile_amp{threshold}", pd.Series(False, index=data.index)).fillna(False)
        summary[f"fem_any_pulsatile_amp{threshold}_n"] = int(anyp.sum())
        summary[f"fem_pulsatile_ge80_amp{threshold}_n"] = int((frac >= 0.80).sum())
    (OUT / "fem_waveform_v7_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
