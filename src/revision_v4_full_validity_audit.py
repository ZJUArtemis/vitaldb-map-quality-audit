#!/usr/bin/env python3
"""Full-cohort, task-specific ART measurement-validity audit.

The analysis implements two safeguards for window-level measurement support:

1. coverage is calculated against the *complete intended* baseline/outcome
   windows, so a recording that begins less than 300 s before induction is not
   treated as having complete baseline coverage; and
2. raw SNUADC/ART availability and pulsatility are evaluated in every one of
   the 910 source induction segments, rather than in a small technical sample.

Numeric cadence is estimated from native record timestamps rather than
finite-value gaps after 1-s materialisation. Raw-waveform pulsatility is
evaluated at amplitude thresholds of 3, 5, and 10 mmHg.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb


WORK = Path(__file__).resolve().parents[1]
PROC = WORK / "data" / "processed"
MET = WORK / "outputs" / "metrics"
CACHE = PROC / "revision_v4_case_cache"
VITAL_DIR = Path(os.environ.get("VITALDB_VITAL_DIR", WORK / "vitaldb_data"))

BASELINE_SECONDS = 300
OUTCOME_SECONDS = 600
LOWER_BOUNDS = (20, 30, 40)
UPPER_BOUND = 200
OBSERVED_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)
INRANGE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)
RUN_SECONDS = (0, 30, 60, 120)
RAW_SECOND_COVERAGE = 0.80
RAW_PULSE_AMPLITUDES = (3.0, 5.0, 10.0)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def native_numeric_cadence(track) -> float:
    """Estimate native cadence from record timestamps, independent of missing values."""
    if track is None or len(track.recs) < 2:
        return 2.0  # Official Solar 8000M numeric acquisition interval.
    timestamps = np.asarray([float(rec["dt"]) for rec in track.recs], dtype=float)
    gaps = np.diff(timestamps)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0.25) & (gaps <= 10.0)]
    return float(np.median(gaps)) if len(gaps) else 2.0


def longest_run_seconds(mask: np.ndarray, cadence: float) -> float:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return 0.0
    starts = np.r_[0, np.flatnonzero(np.diff(indices) > cadence * 1.5) + 1]
    ends = np.r_[starts[1:] - 1, len(indices) - 1]
    durations = indices[ends] - indices[starts] + cadence
    return float(np.max(durations))


def numeric_metrics(
    arr: np.ndarray, t0: int, outcome_duration: int, lower: int, cadence: float
) -> dict:
    baseline = arr[max(0, t0 - BASELINE_SECONDS):max(0, t0)]
    outcome = arr[max(0, t0):min(len(arr), t0 + outcome_duration)]

    observed_base = np.isfinite(baseline)
    observed_out = np.isfinite(outcome)
    valid_base = observed_base & (baseline >= lower) & (baseline <= UPPER_BOUND)
    valid_out = observed_out & (outcome >= lower) & (outcome <= UPPER_BOUND)

    expected_base = max(1.0, BASELINE_SECONDS / cadence)
    expected_out = max(1.0, outcome_duration / cadence)
    base_observed_coverage = min(1.0, float(observed_base.sum() / expected_base))
    out_observed_coverage = min(1.0, float(observed_out.sum() / expected_out))
    base_inrange = float(valid_base.sum() / observed_base.sum()) if observed_base.any() else 0.0
    out_inrange = float(valid_out.sum() / observed_out.sum()) if observed_out.any() else 0.0
    longest = longest_run_seconds(valid_base, cadence)

    valid_base_values = baseline[valid_base]
    valid_out_values = outcome[valid_out]
    baseline_map = float(np.median(valid_base_values)) if len(valid_base_values) else np.nan
    nadir_map = float(np.min(valid_out_values)) if len(valid_out_values) else np.nan
    drop_pct = (
        float((baseline_map - nadir_map) / baseline_map * 100)
        if np.isfinite(baseline_map) and baseline_map > 0 and np.isfinite(nadir_map)
        else np.nan
    )
    return {
        "observed_baseline": base_observed_coverage,
        "observed_outcome": out_observed_coverage,
        "inrange_baseline": base_inrange,
        "inrange_outcome": out_inrange,
        "baseline_longest_run_s": longest,
        "baseline_map": baseline_map,
        "nadir_map": nadir_map,
        "drop_pct": drop_pct,
        "event30": int(drop_pct > 30) if np.isfinite(drop_pct) else np.nan,
        "cadence_s": cadence,
    }


def raw_window_array(track, dtstart: float, window_start_s: float, window_seconds: int) -> np.ndarray:
    """Materialise only the requested raw-waveform window."""
    srate = float(track.srate)
    nsamp = int(round(window_seconds * srate))
    ret = np.full(nsamp, np.nan, dtype=np.float32)
    absolute_start = dtstart + window_start_s
    absolute_end = absolute_start + window_seconds

    for rec in track.recs:
        rec_start = float(rec["dt"])
        values = np.asarray(rec["val"])
        rec_end = rec_start + len(values) / srate
        if rec_end <= absolute_start or rec_start >= absolute_end:
            continue
        target_start = int(round((rec_start - absolute_start) * srate))
        source_start = 0
        target_end = target_start + len(values)
        source_end = len(values)
        if target_start < 0:
            source_start = -target_start
            target_start = 0
        if target_end > nsamp:
            source_end -= target_end - nsamp
            target_end = nsamp
        if source_end > source_start and target_end > target_start:
            ret[target_start:target_end] = values[source_start:source_end]

    if track.fmt > 2:
        ret *= float(track.gain)
        ret += float(track.offset)
    ret[np.isinf(ret) | (ret > 4e9)] = np.nan
    return ret


def raw_second_metrics(samples: np.ndarray, srate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nsec = len(samples) // int(round(srate))
    if nsec == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty.astype(bool)
    x = samples[: nsec * int(round(srate))].reshape(nsec, int(round(srate)))
    finite = np.isfinite(x)
    observed_fraction = finite.mean(axis=1)
    observed = observed_fraction >= RAW_SECOND_COVERAGE
    mean = np.full(nsec, np.nan, dtype=float)
    amplitude = np.full(nsec, np.nan, dtype=float)
    any_finite = finite.any(axis=1)
    if any_finite.any():
        with np.errstate(invalid="ignore"):
            mean[any_finite] = np.nanmean(x[any_finite], axis=1)
            amplitude[any_finite] = (
                np.nanpercentile(x[any_finite], 95, axis=1)
                - np.nanpercentile(x[any_finite], 5, axis=1)
            )
    return mean, amplitude, observed


def find_track(vf, dtname: str):
    for track in vf.trks.values():
        if track.dtname == dtname:
            return track
    return None


def process_case(caseid: int, t_start: float, t_end: float, force: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{int(caseid):04d}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    result = {"caseid": int(caseid), "t_start": float(t_start), "t_end": float(t_end)}
    path = VITAL_DIR / f"{int(caseid):04d}.vital"
    if not path.exists():
        result.update({"file_present": False, "error": "vital file missing"})
        cache_path.write_text(json.dumps(result, indent=2))
        return result

    try:
        vf = vitaldb.VitalFile(
            str(path), track_names=["SNUADC/ART", "Solar8000/ART_MBP"]
        )
        result["file_present"] = True
        t0 = int(round(float(t_start)))
        outcome_duration = int(max(1, min(OUTCOME_SECONDS, float(t_end) - float(t_start))))

        numeric_track = find_track(vf, "Solar8000/ART_MBP")
        native_cadence = native_numeric_cadence(numeric_track)
        numeric = np.asarray(
            vf.to_numpy(["Solar8000/ART_MBP"], interval=1), dtype=float
        ).reshape(-1)
        result["numeric_track_present"] = numeric_track is not None
        result["numeric_native_cadence_s"] = native_cadence
        for lower in LOWER_BOUNDS:
            metrics = numeric_metrics(numeric, t0, outcome_duration, lower, native_cadence)
            for key, value in metrics.items():
                result[f"lb{lower}_{key}"] = value

        raw_track = find_track(vf, "SNUADC/ART")
        result["raw_track_present"] = raw_track is not None
        if raw_track is not None:
            total_seconds = BASELINE_SECONDS + outcome_duration
            raw = raw_window_array(
                raw_track,
                float(vf.dtstart),
                float(t_start) - BASELINE_SECONDS,
                total_seconds,
            )
            mean, amplitude, observed = raw_second_metrics(raw, float(raw_track.srate))
            base_slice = slice(0, BASELINE_SECONDS)
            out_slice = slice(BASELINE_SECONDS, BASELINE_SECONDS + outcome_duration)
            result.update(
                {
                    "raw_srate_hz": float(raw_track.srate),
                    "raw_baseline_observed_fraction": float(observed[base_slice].mean()),
                    "raw_outcome_observed_fraction": float(observed[out_slice].mean()),
                    "raw_baseline_any_observed": bool(observed[base_slice].any()),
                }
            )
            pulsatile_by_amp = {}
            for amp_threshold in RAW_PULSE_AMPLITUDES:
                amp_label = int(amp_threshold)
                pulsatile = (
                    observed
                    & np.isfinite(mean)
                    & (mean >= 20)
                    & (mean <= UPPER_BOUND)
                    & np.isfinite(amplitude)
                    & (amplitude >= amp_threshold)
                )
                pulsatile_by_amp[amp_label] = pulsatile
                result[f"raw_baseline_pulsatile_fraction_amp{amp_label}"] = float(
                    pulsatile[base_slice].mean()
                )
                result[f"raw_outcome_pulsatile_fraction_amp{amp_label}"] = float(
                    pulsatile[out_slice].mean()
                )
                result[f"raw_baseline_any_pulsatile_amp{amp_label}"] = bool(
                    pulsatile[base_slice].any()
                )
            pulsatile_valid = pulsatile_by_amp[5]
            result["raw_baseline_pulsatile_fraction"] = result[
                "raw_baseline_pulsatile_fraction_amp5"
            ]
            result["raw_outcome_pulsatile_fraction"] = result[
                "raw_outcome_pulsatile_fraction_amp5"
            ]
            result["raw_baseline_any_pulsatile"] = result[
                "raw_baseline_any_pulsatile_amp5"
            ]

            numeric_window = np.full(total_seconds, np.nan, dtype=float)
            nb = numeric[max(0, t0 - BASELINE_SECONDS):t0]
            baseline_offset = BASELINE_SECONDS - len(nb)
            if len(nb):
                numeric_window[baseline_offset:BASELINE_SECONDS] = nb
            no = numeric[t0:min(len(numeric), t0 + outcome_duration)]
            if len(no):
                numeric_window[BASELINE_SECONDS:BASELINE_SECONDS + len(no)] = no
            overlap = (
                pulsatile_valid
                & np.isfinite(numeric_window)
                & (numeric_window >= 20)
                & (numeric_window <= UPPER_BOUND)
            )
            result["numeric_raw_overlap_seconds"] = int(overlap.sum())
            if overlap.sum() >= 30:
                result["numeric_raw_corr"] = float(np.corrcoef(numeric_window[overlap], mean[overlap])[0, 1])
                result["numeric_raw_mae_mmHg"] = float(np.mean(np.abs(numeric_window[overlap] - mean[overlap])))
            else:
                result["numeric_raw_corr"] = np.nan
                result["numeric_raw_mae_mmHg"] = np.nan
        else:
            for key in (
                "raw_srate_hz",
                "raw_baseline_observed_fraction",
                "raw_outcome_observed_fraction",
                "raw_baseline_pulsatile_fraction",
                "raw_outcome_pulsatile_fraction",
                "numeric_raw_overlap_seconds",
                "numeric_raw_corr",
                "numeric_raw_mae_mmHg",
            ):
                result[key] = np.nan
            result["raw_baseline_any_observed"] = False
            result["raw_baseline_any_pulsatile"] = False
            for amp_threshold in RAW_PULSE_AMPLITUDES:
                amp_label = int(amp_threshold)
                result[f"raw_baseline_pulsatile_fraction_amp{amp_label}"] = np.nan
                result[f"raw_outcome_pulsatile_fraction_amp{amp_label}"] = np.nan
                result[f"raw_baseline_any_pulsatile_amp{amp_label}"] = False
        result["error"] = None
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    serialisable = {key: _json_safe(value) for key, value in result.items()}
    cache_path.write_text(json.dumps(serialisable, indent=2, allow_nan=False))
    return serialisable


def grid_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lower in LOWER_BOUNDS:
        p = f"lb{lower}_"
        for obs_threshold in OBSERVED_THRESHOLDS:
            for inrange_threshold in INRANGE_THRESHOLDS:
                numerical = (
                    (data[p + "observed_baseline"] >= obs_threshold)
                    & (data[p + "observed_outcome"] >= obs_threshold)
                    & (data[p + "inrange_baseline"] >= inrange_threshold)
                    & (data[p + "inrange_outcome"] >= inrange_threshold)
                )
                for run in RUN_SECONDS:
                    retained = numerical if run == 0 else numerical & (data[p + "baseline_longest_run_s"] >= run)
                    labels = data.loc[retained, p + "event30"].dropna().astype(int)
                    rows.append(
                        {
                            "lower_bound_mmHg": lower,
                            "observed_coverage_threshold": obs_threshold,
                            "inrange_fraction_threshold": inrange_threshold,
                            "continuity_seconds": run,
                            "retained_n": int(retained.sum()),
                            "labelled_n": int(len(labels)),
                            "events": int(labels.sum()),
                            "non_events": int((1 - labels).sum()),
                            "event_rate": float(labels.mean()) if len(labels) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def build_summary(data: pd.DataFrame, grid: pd.DataFrame) -> dict:
    primary = grid[
        (grid.lower_bound_mmHg == 20)
        & (grid.observed_coverage_threshold == 0.80)
        & (grid.inrange_fraction_threshold == 0.80)
        & (grid.continuity_seconds == 60)
    ].iloc[0]
    lb20_grid = grid[
        (grid.lower_bound_mmHg == 20) & (grid.continuity_seconds == 60)
    ]
    overlap = data[data.numeric_raw_overlap_seconds >= 30]
    summary = {
        "source_segments": int(len(data)),
        "successfully_processed": int(data.error.isna().sum()),
        "processing_errors": int(data.error.notna().sum()),
        "primary_criteria": {
            "range_mmHg": "20-200",
            "observed_coverage_threshold": 0.80,
            "inrange_fraction_threshold": 0.80,
            "continuity_seconds": 60,
            "retained_n": int(primary.retained_n),
            "events": int(primary.events),
            "non_events": int(primary.non_events),
        },
        "lb20_operational_grid_run60_retained_range": [
            int(lb20_grid.retained_n.min()), int(lb20_grid.retained_n.max())
        ],
        "raw_track_present_n": int(data.raw_track_present.fillna(False).sum()),
        "raw_baseline_any_observed_n": int(data.raw_baseline_any_observed.fillna(False).sum()),
        "raw_baseline_any_pulsatile_n": int(data.raw_baseline_any_pulsatile.fillna(False).sum()),
        "raw_baseline_observed_ge50_n": int((data.raw_baseline_observed_fraction >= 0.50).sum()),
        "raw_baseline_observed_ge80_n": int((data.raw_baseline_observed_fraction >= 0.80).sum()),
        "raw_baseline_pulsatile_ge50_n": int((data.raw_baseline_pulsatile_fraction >= 0.50).sum()),
        "raw_baseline_pulsatile_ge80_n": int((data.raw_baseline_pulsatile_fraction >= 0.80).sum()),
        "median_raw_baseline_observed_fraction_all": float(data.raw_baseline_observed_fraction.fillna(0).median()),
        "median_raw_baseline_pulsatile_fraction_all": float(data.raw_baseline_pulsatile_fraction.fillna(0).median()),
        "numeric_raw_overlap_ge30_n": int(len(overlap)),
        "median_numeric_raw_corr": float(overlap.numeric_raw_corr.median()) if len(overlap) else None,
        "median_numeric_raw_mae_mmHg": float(overlap.numeric_raw_mae_mmHg.median()) if len(overlap) else None,
        "raw_amplitude_sensitivity": {},
        "numeric_native_cadence_s": {
            "median": float(data.numeric_native_cadence_s.median()),
            "minimum": float(data.numeric_native_cadence_s.min()),
            "maximum": float(data.numeric_native_cadence_s.max()),
            "within_0_05_s_of_2_s_n": int(
                np.isclose(data.numeric_native_cadence_s, 2.0, atol=0.05).sum()
            ),
        },
    }
    for amp_threshold in RAW_PULSE_AMPLITUDES:
        amp_label = int(amp_threshold)
        fraction_col = f"raw_baseline_pulsatile_fraction_amp{amp_label}"
        any_col = f"raw_baseline_any_pulsatile_amp{amp_label}"
        summary["raw_amplitude_sensitivity"][str(amp_label)] = {
            "any_pulsatile_n": int(data[any_col].fillna(False).sum()),
            "pulsatile_ge80_n": int((data[fraction_col] >= 0.80).sum()),
            "median_pulsatile_fraction_all": float(data[fraction_col].fillna(0).median()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Development only: process first N cases")
    parser.add_argument("--force", action="store_true", help="Ignore per-case cache")
    parser.add_argument("--tag", default="", help="Optional output suffix, e.g. _test")
    args = parser.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    segments = pd.read_parquet(PROC / "induction_segments.parquet")[["caseid", "t_start", "t_end"]]
    segments = segments.sort_values("caseid").reset_index(drop=True)
    if args.limit:
        segments = segments.head(args.limit)
    print(f"Processing {len(segments)} cases with {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_case, int(r.caseid), float(r.t_start), float(r.t_end), args.force): int(r.caseid)
            for r in segments.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            caseid = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"caseid": caseid, "error": f"worker failure: {exc}"})
            if completed % 25 == 0 or completed == len(futures):
                print(f"Completed {completed}/{len(futures)}", flush=True)

    data = pd.DataFrame(rows).sort_values("caseid").reset_index(drop=True)
    suffix = args.tag
    data.to_csv(MET / f"revision_v4_full_validity_case_results{suffix}.csv", index=False)
    data.to_parquet(PROC / f"revision_v4_full_validity_case_results{suffix}.parquet", index=False)
    grid = grid_summary(data)
    grid.to_csv(MET / f"revision_v4_threshold_grid{suffix}.csv", index=False)
    summary = build_summary(data, grid)
    (MET / f"revision_v4_full_validity_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
