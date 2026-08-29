#!/usr/bin/env python3
"""Independent cadence accounting and task-window-length sensitivity for v8."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

from project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
PROJECT = ROOT / "analysis/topic10_induction_instability"
VITAL_DIR = ROOT / "physionet.org/files/vitaldb/1.0.0/vital_files"
SOURCE = PROJECT / "outputs/metrics/revision_v5_fixed_window_case_results.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
CACHE = OUT / "cadence_window_v8_cache"

BASELINES = (60, 120, 180, 300)
OUTCOMES = (300, 600)
LOWER, UPPER = 20.0, 200.0


def find_track(vf, dtname):
    return next((track for track in vf.trks.values() if track.dtname == dtname), None)


def fixed_window(arr, start, length):
    out = np.full(length, np.nan, dtype=float)
    lo, hi = max(0, start), min(len(arr), start + length)
    if hi > lo:
        dst = lo - start
        out[dst:dst + hi - lo] = arr[lo:hi]
    return out


def longest_run(mask, cadence):
    idx = np.flatnonzero(mask)
    if not len(idx) or not np.isfinite(cadence):
        return 0.0
    starts = np.r_[0, np.flatnonzero(np.diff(idx) > cadence * 1.5) + 1]
    ends = np.r_[starts[1:] - 1, len(idx) - 1]
    return float(np.max(idx[ends] - idx[starts] + cadence))


def process(caseid, onset, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{caseid:04d}.json"
    if cached.exists() and not force:
        return json.loads(cached.read_text())
    row = {"caseid": int(caseid), "art_mbp_present": False,
           "cadence_estimable": False, "cadence_s": None, "error": None}
    try:
        vf = vitaldb.VitalFile(str(VITAL_DIR / f"{caseid:04d}.vital"),
                               track_names=["Solar8000/ART_MBP"])
        track = find_track(vf, "Solar8000/ART_MBP")
        row["art_mbp_present"] = track is not None
        if track is None:
            cached.write_text(json.dumps(row, indent=2))
            return row
        stamps = np.asarray([float(rec["dt"]) for rec in track.recs], dtype=float)
        gaps = np.diff(stamps)
        gaps = gaps[np.isfinite(gaps) & (gaps > 0.25) & (gaps <= 10)]
        if len(gaps):
            cadence = float(np.median(gaps))
            row["cadence_estimable"] = True
            row["cadence_s"] = cadence
        else:
            cached.write_text(json.dumps(row, indent=2))
            return row
        arr = np.asarray(vf.to_numpy(["Solar8000/ART_MBP"], interval=1), dtype=float).reshape(-1)
        t0 = int(round(float(onset) - float(vf.dtstart)))
        for baseline_s in BASELINES:
            base = fixed_window(arr, t0 - baseline_s, baseline_s)
            observed_b = np.isfinite(base)
            valid_b = observed_b & (base >= LOWER) & (base <= UPPER)
            bcov = min(1.0, float(observed_b.sum() / max(1.0, baseline_s / cadence)))
            binrange = float(valid_b.sum() / observed_b.sum()) if observed_b.any() else 0.0
            brun = longest_run(valid_b, cadence)
            for outcome_s in OUTCOMES:
                out = fixed_window(arr, t0, outcome_s)
                observed_o = np.isfinite(out)
                valid_o = observed_o & (out >= LOWER) & (out <= UPPER)
                ocov = min(1.0, float(observed_o.sum() / max(1.0, outcome_s / cadence)))
                oinrange = float(valid_o.sum() / observed_o.sum()) if observed_o.any() else 0.0
                usable = bcov >= .8 and ocov >= .8 and binrange >= .8 and oinrange >= .8 and brun >= min(60, baseline_s)
                key = f"b{baseline_s}_o{outcome_s}"
                row[key + "_usable"] = bool(usable)
                if valid_b.any() and valid_o.any():
                    baseline = float(np.median(base[valid_b]))
                    nadir = float(np.min(out[valid_o]))
                    drop = (baseline - nadir) / baseline * 100 if baseline > 0 else np.nan
                    row[key + "_event"] = int(drop > 30) if np.isfinite(drop) else None
                else:
                    row[key + "_event"] = None
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    cached.write_text(json.dumps(row, indent=2, allow_nan=False))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    src = pd.read_csv(SOURCE)[["caseid", "t_start_absolute"]].dropna()
    jobs = [(int(r.caseid), float(r.t_start_absolute), args.force) for r in src.itertuples(index=False)]
    rows = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.starmap(process, jobs), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"processed {i}/{len(jobs)}", flush=True)
    data = pd.DataFrame(rows).sort_values("caseid")
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "cadence_window_v8_case_results.csv", index=False)
    cadence = data.loc[data.cadence_estimable.fillna(False), "cadence_s"].astype(float)
    summary = {
        "source_cases": int(len(data)),
        "processing_errors": int(data.error.notna().sum()),
        "art_mbp_present_n": int(data.art_mbp_present.fillna(False).sum()),
        "cadence_estimable_n": int(data.cadence_estimable.fillna(False).sum()),
        "cadence_not_estimable_n": int((~data.cadence_estimable.fillna(False)).sum()),
        "nominal_fallback_n": 0,
        "cadence_median_s": float(cadence.median()),
        "cadence_min_s": float(cadence.min()),
        "cadence_max_s": float(cadence.max()),
    }
    sensitivity = []
    for baseline_s in BASELINES:
        for outcome_s in OUTCOMES:
            key = f"b{baseline_s}_o{outcome_s}"
            mask = data[key + "_usable"].eq(True)
            labels = data.loc[mask, key + "_event"].dropna().astype(int)
            sensitivity.append({"baseline_s": baseline_s, "outcome_s": outcome_s,
                                "retained_n": int(mask.sum()), "events": int(labels.sum()),
                                "non_events": int(len(labels) - labels.sum())})
    pd.DataFrame(sensitivity).to_csv(OUT / "window_length_sensitivity_v8.csv", index=False)
    summary["window_length_sensitivity"] = sensitivity
    (OUT / "cadence_window_v8_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
