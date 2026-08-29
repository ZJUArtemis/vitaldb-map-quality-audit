#!/usr/bin/env python3
"""Validate materialised numeric coverage against native ART_MBP records.

The primary fixed-window audit previously counted finite values after 1-s
materialisation while estimating expected counts from native timestamps.  This
independent audit calculates coverage, plausibility and continuity directly
from native records, compares classifications case by case, and adds symmetric
outcome-window continuity sensitivities.
"""

from __future__ import annotations

import argparse
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
DEFAULT_VITAL_DIR = PROJECT.parents[1] / "physionet.org/files/vitaldb/1.0.0/vital_files"
if not DEFAULT_VITAL_DIR.exists():
    DEFAULT_VITAL_DIR = PROJECT / "vitaldb_data"
VITAL_DIR = Path(os.environ.get("VITALDB_VITAL_DIR", DEFAULT_VITAL_DIR))
SOURCE = PROJECT / "outputs/metrics/revision_v5_fixed_window_case_results.csv"
OUT = Path(__file__).resolve().parent
CACHE = OUT / "native_numeric_coverage_v9_cache"
BASELINE_SECONDS = 300
OUTCOME_SECONDS = 600
LOWER = 20.0
UPPER = 200.0


def find_track(vf, name):
    return next((track for track in vf.trks.values() if track.dtname == name), None)


def longest_run(timestamps: np.ndarray, cadence: float) -> float:
    if timestamps.size == 0 or not np.isfinite(cadence):
        return 0.0
    timestamps = np.sort(timestamps)
    breaks = np.flatnonzero(np.diff(timestamps) > cadence * 1.5)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(timestamps) - 1]
    return float(np.max(timestamps[ends] - timestamps[starts] + cadence))


def maximum_gap(timestamps: np.ndarray, start: float, end: float) -> float:
    if timestamps.size == 0:
        return float(end - start)
    ordered = np.sort(timestamps)
    gaps = np.diff(np.r_[start, ordered, end])
    return float(np.max(gaps))


def process(caseid: int, onset: float, force: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{caseid:04d}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    row = {
        "caseid": int(caseid), "track_present": False, "cadence_s": None,
        "cadence_estimable": False, "error": None,
    }
    try:
        path = VITAL_DIR / f"{caseid:04d}.vital"
        vf = vitaldb.VitalFile(str(path), track_names=["Solar8000/ART_MBP"])
        track = find_track(vf, "Solar8000/ART_MBP")
        row["track_present"] = track is not None
        if track is None:
            cache.write_text(json.dumps(row, indent=2))
            return row
        timestamps = np.asarray([float(rec["dt"]) for rec in track.recs], dtype=float)
        values = np.asarray([float(rec["val"]) for rec in track.recs], dtype=float)
        finite_t = np.isfinite(timestamps)
        timestamps, values = timestamps[finite_t], values[finite_t]
        order = np.argsort(timestamps)
        timestamps, values = timestamps[order], values[order]
        gaps = np.diff(timestamps)
        cadence_gaps = gaps[np.isfinite(gaps) & (gaps > 0.25) & (gaps <= 10.0)]
        if cadence_gaps.size == 0:
            cache.write_text(json.dumps(row, indent=2))
            return row
        cadence = float(np.median(cadence_gaps))
        row.update(cadence_s=cadence, cadence_estimable=True)
        for prefix, start, end in (
            ("baseline", onset - BASELINE_SECONDS, onset),
            ("outcome", onset, onset + OUTCOME_SECONDS),
        ):
            within = (timestamps >= start) & (timestamps < end)
            window_t, window_v = timestamps[within], values[within]
            observed = np.isfinite(window_v)
            valid = observed & (window_v >= LOWER) & (window_v <= UPPER)
            duration = end - start
            expected = max(1.0, duration / cadence)
            row[f"native_{prefix}_observed"] = min(1.0, float(observed.sum() / expected))
            row[f"native_{prefix}_inrange"] = float(valid.sum() / observed.sum()) if observed.any() else 0.0
            row[f"native_{prefix}_longest_run_s"] = longest_run(window_t[valid], cadence)
            row[f"native_{prefix}_max_valid_gap_s"] = maximum_gap(window_t[valid], start, end)
            if valid.any():
                row[f"native_{prefix}_median"] = float(np.median(window_v[valid]))
                row[f"native_{prefix}_minimum"] = float(np.min(window_v[valid]))
            else:
                row[f"native_{prefix}_median"] = None
                row[f"native_{prefix}_minimum"] = None
        b = row.get("native_baseline_median")
        o = row.get("native_outcome_minimum")
        drop = (b - o) / b * 100.0 if b is not None and o is not None and b > 0 else None
        row["native_drop_pct"] = drop
        row["native_event30"] = int(drop > 30) if drop is not None else None
        row["native_primary_valid"] = bool(
            row["native_baseline_observed"] >= 0.8
            and row["native_outcome_observed"] >= 0.8
            and row["native_baseline_inrange"] >= 0.8
            and row["native_outcome_inrange"] >= 0.8
            and row["native_baseline_longest_run_s"] >= 60
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    cache.write_text(json.dumps(row, indent=2, allow_nan=False))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = pd.read_csv(SOURCE)
    jobs = [(int(r.caseid), float(r.t_start_absolute), args.force) for r in source.itertuples(index=False)]
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool:
        rows = []
        for i, row in enumerate(pool.starmap(process, jobs), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"native records {i}/{len(jobs)}", flush=True)
    native = pd.DataFrame(rows).sort_values("caseid")
    data = source.merge(native, on="caseid", how="left", validate="one_to_one")
    materialised = (
        data["lb20_observed_baseline"].ge(0.8)
        & data["lb20_observed_outcome"].ge(0.8)
        & data["lb20_inrange_baseline"].ge(0.8)
        & data["lb20_inrange_outcome"].ge(0.8)
        & data["lb20_baseline_longest_run_s"].ge(60)
    )
    native_valid = data["native_primary_valid"].fillna(False).astype(bool)
    data["materialised_primary_valid"] = materialised
    data["classification_agreement"] = materialised.eq(native_valid)

    comparisons = {}
    for window in ("baseline", "outcome"):
        old = data[f"lb20_observed_{window}"].to_numpy(float)
        new = data[f"native_{window}_observed"].fillna(0).to_numpy(float)
        difference = new - old
        comparisons[window] = {
            "median_native_minus_materialised": float(np.median(difference)),
            "median_absolute_difference": float(np.median(np.abs(difference))),
            "q95_absolute_difference": float(np.percentile(np.abs(difference), 95)),
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
            "pearson_correlation": float(np.corrcoef(old, new)[0, 1]),
        }

    outcome_sensitivity = []
    base_native = native_valid
    for seconds in (0, 30, 60, 120):
        mask = base_native & data["native_outcome_longest_run_s"].ge(seconds)
        labels = data.loc[mask, "native_event30"].dropna().astype(int)
        outcome_sensitivity.append({
            "outcome_continuous_plausible_seconds": seconds,
            "retained_n": int(mask.sum()),
            "events": int(labels.sum()),
            "non_events": int(len(labels) - labels.sum()),
        })

    discordant = data.loc[~data["classification_agreement"], [
        "caseid", "materialised_primary_valid", "native_primary_valid",
        "lb20_observed_baseline", "native_baseline_observed",
        "lb20_observed_outcome", "native_outcome_observed",
        "lb20_inrange_baseline", "native_baseline_inrange",
        "lb20_inrange_outcome", "native_outcome_inrange",
    ]]
    summary = {
        "source_cases": int(len(data)),
        "processing_errors": int(data.get("error_y", pd.Series(index=data.index, dtype=object)).notna().sum()),
        "native_track_present": int(data["track_present"].fillna(False).sum()),
        "native_cadence_estimable": int(data["cadence_estimable"].fillna(False).sum()),
        "materialised_primary_valid_n": int(materialised.sum()),
        "native_primary_valid_n": int(native_valid.sum()),
        "classification_agreement_n": int(data["classification_agreement"].sum()),
        "classification_agreement_fraction": float(data["classification_agreement"].mean()),
        "discordant_n": int(len(discordant)),
        "coverage_comparison": comparisons,
        "outcome_continuity_sensitivity": outcome_sensitivity,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "native_numeric_coverage_case_results_v9.csv", index=False)
    discordant.to_csv(OUT / "native_numeric_coverage_discordant_v9.csv", index=False)
    pd.DataFrame(outcome_sensitivity).to_csv(OUT / "outcome_continuity_sensitivity_v9.csv", index=False)
    (OUT / "native_numeric_coverage_summary_v9.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
