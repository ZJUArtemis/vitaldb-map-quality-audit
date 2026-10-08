#!/usr/bin/env python3
"""Quantify distinct NIBP value states in the reference windows.

Solar8000 numeric tracks repeat the most recently reported cuff value at the
monitor cadence; consequently, finite numeric records are not cuff inflations.
This audit aligns the concurrently recorded SBP/DBP/MBP triplet to two-second
bins and counts changes in that triplet. The result is reported conservatively
as distinct observed NIBP value states, not validated cuff inflations because
VitalDB has no explicit cuff-event timestamp.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import vitaldb

from project_paths import PROCESSED_DIR, RI_OUTPUT_DIR, VITAL_DIR

OUT = RI_OUTPUT_DIR
TRACKS = (
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/NIBP_MBP",
)


def find_track(vf: vitaldb.VitalFile, name: str):
    return next((track for track in vf.trks.values() if track.dtname == name), None)


def aligned_states(vf: vitaldb.VitalFile) -> pd.DataFrame:
    columns = {}
    for name in TRACKS:
        track = find_track(vf, name)
        if track is None:
            return pd.DataFrame()
        short = name.rsplit("_", 1)[-1]
        records = pd.DataFrame(
            {
                "time": [float(record["dt"]) - float(vf.dtstart) for record in track.recs],
                short: [float(record["val"]) for record in track.recs],
            }
        )
        records = records[np.isfinite(records["time"]) & np.isfinite(records[short])]
        records["bin"] = np.rint(records["time"] / 2.0).astype(int) * 2
        columns[short] = records.groupby("bin", sort=True)[short].last()
    states = pd.concat(columns.values(), axis=1).sort_index()
    states.columns = list(columns)
    states = states.dropna()
    plausible = (
        states["SBP"].between(40, 300)
        & states["DBP"].between(10, 200)
        & states["MBP"].between(20, 200)
        & (states["SBP"] >= states["MBP"])
        & (states["MBP"] >= states["DBP"])
    )
    return states[plausible]


def count_states(states: pd.DataFrame, start: float, end: float) -> int:
    window = states[(states.index >= start) & (states.index < end)]
    if window.empty:
        return 0
    return int(window.ne(window.shift()).any(axis=1).sum())


def one_case(task: tuple[int, float, float]) -> dict:
    caseid, t0, t1 = int(task[0]), float(task[1]), float(task[2])
    result = {"caseid": caseid, "baseline_nibp_states": 0, "outcome_nibp_states": 0}
    path = VITAL_DIR / f"{caseid:04d}.vital"
    if not path.exists():
        return result
    try:
        vf = vitaldb.VitalFile(str(path), track_names=list(TRACKS))
        states = aligned_states(vf)
        result["baseline_nibp_states"] = count_states(states, t0 - 300.0, t0)
        result["outcome_nibp_states"] = count_states(states, t0, min(t0 + 600.0, t1))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def count_summary(values: pd.Series) -> dict:
    values = values.fillna(0).astype(int)
    return {
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "zero_n": int((values == 0).sum()),
        "one_n": int((values == 1).sum()),
        "two_n": int((values == 2).sum()),
        "three_or_more_n": int((values >= 3).sum()),
    }


def main() -> None:
    segments = pd.read_parquet(PROCESSED_DIR / "induction_segments.parquet")
    tasks = [
        (int(row.caseid), float(row.t_start), float(row.t_end))
        for row in segments[["caseid", "t_start", "t_end"]].itertuples(index=False)
    ]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        rows = list(pool.map(one_case, tasks, chunksize=8))
    result = pd.DataFrame(rows).sort_values("caseid")
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "nibp_value_states_v14.csv", index=False)

    evaluable = pd.read_parquet(PROCESSED_DIR / "outcome_labels_nibp.parquet")
    merged = evaluable.merge(result, on="caseid", how="left")
    both = merged[merged["nibp_baseline"].notna() & merged["nibp_nadir"].notna()].copy()
    summary = {
        "definition": (
            "distinct plausible SBP/DBP/MBP value states in two-second monitor bins; "
            "proxy for cuff updates, not validated cuff-inflation timestamps"
        ),
        "source_segments_n": int(len(result)),
        "nibp_evaluable_n": int(len(both)),
        "baseline_states_among_evaluable": count_summary(both["baseline_nibp_states"]),
        "outcome_states_among_evaluable": count_summary(both["outcome_nibp_states"]),
        "evaluable_with_at_least_two_baseline_states_n": int((both["baseline_nibp_states"] >= 2).sum()),
        "evaluable_with_at_least_two_outcome_states_n": int((both["outcome_nibp_states"] >= 2).sum()),
    }
    (OUT / "nibp_value_state_summary_v14.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
