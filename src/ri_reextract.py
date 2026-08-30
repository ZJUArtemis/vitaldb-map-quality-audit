#!/usr/bin/env python3
"""Gap-aware, same-pulse Reflection Index re-extraction for semantic release.

This script corrects three defects in the historical feature code:

1. finite samples separated by missing intervals are never concatenated;
2. systolic and diastolic/reflected fiducials must belong to the same pulse;
3. both amplitudes are measured from that pulse's onset (foot):

       RI = (A_diastolic - A_foot) / (A_systolic - A_foot).

The primary fiducials are obtained with pyPPG 1.0.41.  A beat is retained only
when the strict temporal order onset < systolic peak < dicrotic notch <
diastolic peak < next onset is satisfied.  A case-level RI is the median of at
least 10 retained beats.  The script writes a separate v14 table and never
overwrites the historical feature table.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import vitaldb
from dotmap import DotMap
from pyPPG import PPG
from pyPPG.fiducials import FpCollection
from pyPPG.preproc import Preprocess

from project_paths import PROCESSED_DIR, RI_OUTPUT_DIR, VITAL_DIR

FS = 100
BASELINE_SECONDS = 300
MIN_SEGMENT_SECONDS = 15
MIN_CASE_BEATS = 10
MIN_BEAT_SECONDS = 0.33
MAX_BEAT_SECONDS = 1.50
MAX_RI = 1.30
MIN_TEMPLATE_CORRELATION = 0.80
MAX_EXTREME_PLATEAU_FRACTION = 0.12

OUT = RI_OUTPUT_DIR
INPUT = PROCESSED_DIR / "vascular_features.parquet"


def finite_runs(x: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Return half-open finite runs without bridging any missing sample."""
    valid = np.isfinite(x)
    if not valid.any():
        return []
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_samples]


def analyse_segment(raw: np.ndarray, segment_start: int) -> tuple[list[dict], dict]:
    """Extract strictly ordered, foot-referenced RI beats from one finite run."""
    raw = np.asarray(raw, dtype=float)
    diagnostics = {
        "detected_pulses": 0,
        "complete_fiducials": 0,
        "ordered_fiducials": 0,
        "candidate_ri_beats": 0,
        "valid_ri_beats": 0,
    }
    if raw.size < FS * MIN_SEGMENT_SECONDS or not np.isfinite(raw).all():
        return [], diagnostics

    robust_range = float(np.percentile(raw, 95) - np.percentile(raw, 5))
    if not np.isfinite(robust_range) or robust_range <= 1e-8:
        return [], diagnostics

    signal = DotMap(
        v=raw,
        fs=FS,
        filtering=True,
        correction=pd.DataFrame(),
        start_sig=0,
        end_sig=len(raw),
        name="v14-segment",
    )
    prep = Preprocess(
        fL=0.5000001,
        fH=12,
        order=4,
        sm_wins={"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10},
    )
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(signal)
    ppg = PPG(signal, check_ppg_len=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp = FpCollection(ppg).get_fiducials(ppg)

    diagnostics["detected_pulses"] = int(len(fp))
    needed = fp[["on", "sp", "dn", "dp", "off"]].dropna()
    diagnostics["complete_fiducials"] = int(len(needed))

    candidates: list[dict] = []
    filtered_range = float(np.percentile(signal.ppg, 95) - np.percentile(signal.ppg, 5))
    min_sys_amplitude = max(1e-10, 0.05 * filtered_range)
    for pulse_index, row in needed.iterrows():
        on, sp, dn, dp, off = (int(row[c]) for c in ("on", "sp", "dn", "dp", "off"))
        if not (0 <= on < sp < dn < dp < off < len(signal.ppg)):
            continue
        diagnostics["ordered_fiducials"] += 1

        duration = (off - on) / FS
        if not (MIN_BEAT_SECONDS <= duration <= MAX_BEAT_SECONDS):
            continue
        # Enforce the pyPPG definition that the diastolic peak precedes 80% of
        # the pulse interval.  This excludes next-beat systolic peaks.
        if dp >= on + 0.80 * (off - on):
            continue

        foot = float(signal.ppg[on])
        systolic_amplitude = float(signal.ppg[sp] - foot)
        diastolic_amplitude = float(signal.ppg[dp] - foot)
        if systolic_amplitude <= min_sys_amplitude or diastolic_amplitude <= 0:
            continue
        ri = diastolic_amplitude / systolic_amplitude
        if not np.isfinite(ri) or not (0 < ri < MAX_RI):
            continue

        raw_beat = raw[on:off + 1]
        raw_span = float(np.max(raw_beat) - np.min(raw_beat))
        if raw_span <= 1e-8:
            continue
        tolerance = 0.01 * raw_span
        extreme_fraction = float(
            max(
                np.mean(raw_beat <= np.min(raw_beat) + tolerance),
                np.mean(raw_beat >= np.max(raw_beat) - tolerance),
            )
        )
        if extreme_fraction > MAX_EXTREME_PLATEAU_FRACTION:
            continue

        waveform = (signal.ppg[on:off + 1] - foot) / systolic_amplitude
        phase = np.linspace(0.0, 1.0, len(waveform))
        normalised_waveform = np.interp(np.linspace(0.0, 1.0, 100), phase, waveform)

        candidates.append(
            {
                "pulse_index": int(pulse_index),
                "on": int(segment_start + on),
                "sp": int(segment_start + sp),
                "dn": int(segment_start + dn),
                "dp": int(segment_start + dp),
                "off": int(segment_start + off),
                "duration_s": float(duration),
                "systolic_amplitude": systolic_amplitude,
                "diastolic_amplitude": diastolic_amplitude,
                "ri": float(ri),
                "extreme_plateau_fraction": extreme_fraction,
                "_normalised_waveform": normalised_waveform,
            }
        )

    diagnostics["candidate_ri_beats"] = int(len(candidates))
    beats: list[dict] = []
    if len(candidates) >= MIN_CASE_BEATS:
        template = np.median(
            np.vstack([candidate["_normalised_waveform"] for candidate in candidates]),
            axis=0,
        )
        template_sd = float(np.std(template))
        for candidate in candidates:
            waveform = candidate.pop("_normalised_waveform")
            if template_sd <= 1e-8 or float(np.std(waveform)) <= 1e-8:
                continue
            correlation = float(np.corrcoef(template, waveform)[0, 1])
            if not np.isfinite(correlation) or correlation < MIN_TEMPLATE_CORRELATION:
                continue
            candidate["template_correlation"] = correlation
            beats.append(candidate)

    diagnostics["valid_ri_beats"] = int(len(beats))
    return beats, diagnostics


def process_case(case: dict) -> tuple[dict, list[dict]]:
    caseid = int(case["caseid"])
    t_start = int(case["t_start"])
    result = {
        "caseid": caseid,
        "t_start": t_start,
        "ri_mean_historical": case.get("ri_mean", np.nan),
        "ppg_baseline_expected_samples": int(BASELINE_SECONDS * FS),
        "ppg_baseline_finite_samples": 0,
        "ppg_continuous_segments_n": 0,
        "ppg_continuous_seconds": 0.0,
        "detected_pulses": 0,
        "complete_fiducials": 0,
        "ordered_fiducials": 0,
        "candidate_ri_beats": 0,
        "valid_ri_beats": 0,
        "ri_median_v14": np.nan,
        "ri_q1_v14": np.nan,
        "ri_q3_v14": np.nan,
        "status": "not_processed",
        "error": "",
    }
    all_beats: list[dict] = []
    try:
        path = VITAL_DIR / f"{caseid:04d}.vital"
        if not path.exists():
            result["status"] = "missing_vital_file"
            return result, all_beats
        vf = vitaldb.VitalFile(str(path))
        arr = vf.to_numpy(["SNUADC/PLETH"], interval=1 / FS)
        if arr is None or arr.size == 0:
            result["status"] = "missing_ppg_track"
            return result, all_beats
        ppg = np.asarray(arr[:, 0], dtype=float)
        end = min(len(ppg), max(0, t_start * FS))
        start = max(0, end - BASELINE_SECONDS * FS)
        baseline = ppg[start:end]
        if baseline.size < FS * MIN_SEGMENT_SECONDS:
            result["status"] = "baseline_too_short"
            return result, all_beats

        result["ppg_baseline_expected_samples"] = int(baseline.size)
        result["ppg_baseline_finite_samples"] = int(np.isfinite(baseline).sum())
        runs = finite_runs(baseline, FS * MIN_SEGMENT_SECONDS)
        result["ppg_continuous_segments_n"] = int(len(runs))
        result["ppg_continuous_seconds"] = float(sum(b - a for a, b in runs) / FS)
        if not runs:
            result["status"] = "no_continuous_finite_segment"
            return result, all_beats

        for segment_index, (a, b) in enumerate(runs):
            beats, diag = analyse_segment(baseline[a:b], start + a)
            for key in (
                "detected_pulses", "complete_fiducials", "ordered_fiducials",
                "candidate_ri_beats", "valid_ri_beats",
            ):
                result[key] += int(diag[key])
            for beat in beats:
                beat.update({"caseid": caseid, "segment_index": int(segment_index)})
                all_beats.append(beat)

        if len(all_beats) < MIN_CASE_BEATS:
            result["status"] = "insufficient_valid_beats"
            return result, all_beats
        values = np.asarray([b["ri"] for b in all_beats], dtype=float)
        result["ri_median_v14"] = float(np.median(values))
        result["ri_q1_v14"] = float(np.percentile(values, 25))
        result["ri_q3_v14"] = float(np.percentile(values, 75))
        result["status"] = "valid"
        return result, all_beats
    except Exception as exc:  # retain a machine-readable failure audit
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result, all_beats


def summarise(cases: pd.DataFrame, beats: pd.DataFrame) -> dict:
    valid = cases.loc[cases["status"].eq("valid"), "ri_median_v14"].dropna()
    historical = cases["ri_mean_historical"].dropna()
    paired = cases[["ri_mean_historical", "ri_median_v14"]].dropna()
    return {
        "version": "v14",
        "definition": "median beat-level (A_dp-A_on)/(A_sp-A_on), same pulse only",
        "fiducial_implementation": "pyPPG 1.0.41 with strict temporal-order and morphology QC",
        "gap_handling": "finite contiguous segments >=15 s; missing intervals never concatenated",
        "case_minimum_valid_beats": MIN_CASE_BEATS,
        "minimum_template_correlation": MIN_TEMPLATE_CORRELATION,
        "maximum_extreme_plateau_fraction": MAX_EXTREME_PLATEAU_FRACTION,
        "cases_total": int(len(cases)),
        "cases_valid": int(len(valid)),
        "cases_invalid_or_missing": int(len(cases) - len(valid)),
        "status_counts": {str(k): int(v) for k, v in cases["status"].value_counts().items()},
        "valid_beats_total": int(len(beats)),
        "new_ri": {
            "mean": float(valid.mean()) if len(valid) else None,
            "sd": float(valid.std()) if len(valid) else None,
            "median": float(valid.median()) if len(valid) else None,
            "q1": float(valid.quantile(0.25)) if len(valid) else None,
            "q3": float(valid.quantile(0.75)) if len(valid) else None,
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
        },
        "historical_ri": {
            "n": int(len(historical)),
            "median": float(historical.median()) if len(historical) else None,
            "q1": float(historical.quantile(0.25)) if len(historical) else None,
            "q3": float(historical.quantile(0.75)) if len(historical) else None,
        },
        "paired_old_new": {
            "n": int(len(paired)),
            "spearman": float(paired.corr(method="spearman").iloc[0, 1]) if len(paired) > 2 else None,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=None, help="Development-only case limit")
    args = parser.parse_args()

    source = pd.read_parquet(INPUT)
    needed = source[["caseid", "t_start", "ri_mean"]].copy()
    if args.limit is not None:
        needed = needed.head(args.limit)

    records: list[dict] = []
    beats: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_case, row._asdict()): int(row.caseid) for row in needed.itertuples(index=False)}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                record, case_beats = future.result()
            except Exception:
                cid = futures[future]
                record = {"caseid": cid, "status": "worker_error", "error": traceback.format_exc()}
                case_beats = []
            records.append(record)
            beats.extend(case_beats)
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)

    case_df = pd.DataFrame(records).sort_values("caseid").reset_index(drop=True)
    beat_df = pd.DataFrame(beats).sort_values(["caseid", "on"]).reset_index(drop=True) if beats else pd.DataFrame()
    case_df.to_csv(OUT / "ri_case_features_v14.csv", index=False)
    case_df.to_parquet(OUT / "ri_case_features_v14.parquet", index=False)
    beat_df.to_parquet(OUT / "ri_beat_fiducials_v14.parquet", index=False)
    summary = summarise(case_df, beat_df)
    (OUT / "ri_reextraction_summary_v14.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
