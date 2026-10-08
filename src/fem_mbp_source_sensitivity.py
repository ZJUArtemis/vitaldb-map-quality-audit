#!/usr/bin/env python3
"""Audit whether Solar8000/FEM_MBP changes the fixed-window conclusions.

This source-completeness sensitivity has two deliberately separate parts:

1. official VitalDB track metadata are used to compare the frozen primary
   track-level selection with an otherwise identical selection that also
   accepts Solar8000/FEM_MBP; and
2. within the frozen 926-case cohort, FEM_MBP is used only when ART_MBP was
   unavailable in the primary numeric audit.  The pointwise-only count, the
   primary support count, and every cell of the 192-definition grid are then
   recomputed without refitting a model.

The script requires the case-level output of fixed_window_validity_audit.py.
It writes detailed local results under the ignored outputs/ directory; no
precomputed summary is distributed in this minimal code release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

import fixed_window_validity_audit as fixed_audit
from project_paths import METRICS_DIR, VITAL_DIR


ORIGINAL_ARTERIAL_TRACKS = {
    "SNUADC/ART", "SNUADC/FEM", "Solar8000/ART_MBP"
}
EXPANDED_ARTERIAL_TRACKS = ORIGINAL_ARTERIAL_TRACKS | {"Solar8000/FEM_MBP"}
PPG_TRACK = "SNUADC/PLETH"
PROPOFOL_TRACKS = {"Orchestra/PPF20_RATE", "Orchestra/PPF20_CE"}


def load_track_metadata(path: Path | None) -> pd.DataFrame:
    """Read a supplied official metadata snapshot or query the public API."""
    if path is not None:
        data = pd.read_csv(path)
        required = {"caseid", "tname"}
        if not required <= set(data.columns):
            raise ValueError(f"Track metadata must contain {sorted(required)}")
        return data[["caseid", "tname"]].copy()
    caseids = sorted(int(caseid) for caseid in vitaldb.caseids_ppf)
    rows = vitaldb.get_track_names(caseids=caseids)
    return rows.explode("tnames").rename(columns={"tnames": "tname"})[
        ["caseid", "tname"]
    ]


def load_clinical(path: Path | None, caseids: list[int]) -> pd.DataFrame:
    """Read a supplied clinical table or query the public VitalDB API."""
    data = pd.read_csv(path, encoding="utf-8-sig") if path else vitaldb.load_clinical_data(caseids=caseids)
    data = data.copy()
    # The downloadable PhysioNet table masks ages above 89 as ``>89``;
    # these are adults and correspond to numeric ages in the API table.
    age_text = data["age"].astype(str).replace({">89": "90"})
    data["age_numeric"] = pd.to_numeric(age_text, errors="coerce")
    return data


def cohort_sets(
    clinical: pd.DataFrame, tracks: pd.DataFrame, propofol_caseids: set[int]
) -> tuple[set[int], set[int]]:
    """Reconstruct original and FEM_MBP-expanded track-level cohorts."""
    eligible = clinical[
        clinical["caseid"].astype(int).isin(propofol_caseids)
    ].copy()
    eligible = eligible[eligible["age_numeric"] >= 18]
    eligible = eligible[eligible["ane_type"].str.contains("General", case=False, na=False)]
    eligible = eligible[
        ~eligible["department"].str.contains("Cardiac|Thoracic", case=False, na=False)
    ]
    eligible = eligible.dropna(subset=["age_numeric", "sex", "height", "weight"])
    names = (
        tracks[tracks["caseid"].isin(eligible["caseid"])]
        .groupby("caseid")["tname"]
        .agg(set)
        .to_dict()
    )

    original: set[int] = set()
    expanded: set[int] = set()
    for caseid in eligible["caseid"].astype(int):
        available = names.get(caseid, set())
        common = PPG_TRACK in available and bool(PROPOFOL_TRACKS & available)
        if common and bool(ORIGINAL_ARTERIAL_TRACKS & available):
            original.add(caseid)
        if common and bool(EXPANDED_ARTERIAL_TRACKS & available):
            expanded.add(caseid)
    return original, expanded


def find_numeric_track(vital_file, name: str):
    """Return an exact named track, using the primary audit's name handling."""
    return fixed_audit.find_track(vital_file, name)


def fem_metrics(caseid: int, onset_absolute: float) -> dict | None:
    """Calculate fixed-window FEM_MBP metrics for one local source file."""
    path = VITAL_DIR / f"{caseid:04d}.vital"
    if not path.is_file():
        return None
    vital_file = vitaldb.VitalFile(str(path), track_names=["Solar8000/FEM_MBP"])
    track = find_numeric_track(vital_file, "Solar8000/FEM_MBP")
    if track is None:
        return None
    cadence = fixed_audit.native_numeric_cadence(track)
    values = np.asarray(
        vital_file.to_numpy(["Solar8000/FEM_MBP"], interval=1), dtype=float
    ).reshape(-1)
    onset_relative = float(onset_absolute) - float(vital_file.dtstart)
    t0 = int(round(onset_relative))
    result = {
        "caseid": caseid,
        "numeric_source": "Solar8000/FEM_MBP",
        "t_start_absolute": float(onset_absolute),
    }
    for lower in fixed_audit.LOWER_BOUNDS:
        metrics = fixed_audit.numeric_metrics(
            values, t0, fixed_audit.OUTCOME_SECONDS, lower, cadence
        )
        result.update({f"lb{lower}_{key}": value for key, value in metrics.items()})
    return result


def pointwise_mask(data: pd.DataFrame) -> pd.Series:
    """At least one plausible 20--200 mmHg value in each fixed window."""
    return data["lb20_baseline_map"].notna() & data["lb20_nadir_map"].notna()


def primary_mask(data: pd.DataFrame) -> pd.Series:
    """Primary window-support definition used in the manuscript."""
    return (
        (data["lb20_observed_baseline"] >= 0.80)
        & (data["lb20_observed_outcome"] >= 0.80)
        & (data["lb20_inrange_baseline"] >= 0.80)
        & (data["lb20_inrange_outcome"] >= 0.80)
        & (data["lb20_baseline_longest_run_s"] >= 60)
        & data["lb20_event30"].notna()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track-metadata",
        type=Path,
        help="Optional official https://api.vitaldb.net/trks CSV snapshot",
    )
    parser.add_argument(
        "--clinical-data",
        type=Path,
        help="Optional public VitalDB clinical_data.csv snapshot",
    )
    parser.add_argument(
        "--case-results",
        type=Path,
        default=METRICS_DIR / "revision_v5_fixed_window_case_results.csv",
        help="Output produced by fixed_window_validity_audit.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=METRICS_DIR / "fem_mbp_source_sensitivity",
    )
    args = parser.parse_args()

    tracks = load_track_metadata(args.track_metadata)
    propofol_caseids = set(int(caseid) for caseid in vitaldb.caseids_ppf)
    clinical = load_clinical(args.clinical_data, sorted(propofol_caseids))
    original_metadata, expanded_metadata = cohort_sets(
        clinical, tracks, propofol_caseids
    )

    data = pd.read_csv(args.case_results).sort_values("caseid").reset_index(drop=True)
    frozen_ids = set(data["caseid"].astype(int))
    if frozen_ids != original_metadata:
        only_frozen = sorted(frozen_ids - original_metadata)
        only_metadata = sorted(original_metadata - frozen_ids)
        raise RuntimeError(
            "Frozen case results do not match metadata reconstruction: "
            f"frozen-only={only_frozen[:10]}, metadata-only={only_metadata[:10]}"
        )

    updated = data.copy()
    replacement_rows: list[dict] = []
    missing_art = ~updated["numeric_track_present"].fillna(False)
    fem_mbp_metadata_ids = set(
        tracks.loc[tracks["tname"].eq("Solar8000/FEM_MBP"), "caseid"].astype(int)
    )
    replacement_candidates = missing_art & updated["caseid"].astype(int).isin(
        fem_mbp_metadata_ids
    )
    for row in updated.loc[
        replacement_candidates, ["caseid", "t_start_absolute"]
    ].itertuples(index=False):
        metrics = fem_metrics(int(row.caseid), float(row.t_start_absolute))
        if metrics is None:
            continue
        replacement_rows.append(metrics)
        index = updated.index[updated["caseid"].astype(int) == int(row.caseid)][0]
        updated.loc[index, "numeric_track_present"] = True
        updated.loc[index, "numeric_source"] = "Solar8000/FEM_MBP"
        for key, value in metrics.items():
            if key.startswith("lb"):
                updated.loc[index, key] = value

    original_grid = fixed_audit.grid_summary(data)
    expanded_grid = fixed_audit.grid_summary(updated)
    grid_compare = original_grid.merge(
        expanded_grid,
        on=[
            "lower_bound_mmHg", "observed_coverage_threshold",
            "inrange_fraction_threshold", "continuity_seconds",
        ],
        suffixes=("_original", "_expanded"),
    )
    changed_grid = grid_compare[
        (grid_compare["retained_n_original"] != grid_compare["retained_n_expanded"])
        | (grid_compare["events_original"] != grid_compare["events_expanded"])
        | (grid_compare["non_events_original"] != grid_compare["non_events_expanded"])
    ]

    replacement = pd.DataFrame(replacement_rows)
    summary = {
        "purpose": "Solar8000/FEM_MBP source-completeness sensitivity",
        "primary_analysis_changed": False,
        "models_refitted": False,
        "metadata_original_cohort_n": len(original_metadata),
        "metadata_expanded_cohort_n": len(expanded_metadata),
        "metadata_added_case_n": len(expanded_metadata - original_metadata),
        "metadata_added_caseids": sorted(expanded_metadata - original_metadata),
        "frozen_case_results_n": len(data),
        "art_mbp_unavailable_n": int(missing_art.sum()),
        "art_mbp_unavailable_with_fem_mbp_metadata_n": int(
            replacement_candidates.sum()
        ),
        "fem_mbp_local_replacement_n": len(replacement),
        "fem_mbp_local_replacement_caseids": (
            replacement["caseid"].astype(int).tolist() if len(replacement) else []
        ),
        "fem_mbp_replacements_with_pointwise_label_n": (
            int(pointwise_mask(replacement).sum()) if len(replacement) else 0
        ),
        "fem_mbp_replacements_meeting_primary_support_n": (
            int(primary_mask(replacement).sum()) if len(replacement) else 0
        ),
        "pointwise_original_n": int(pointwise_mask(data).sum()),
        "pointwise_expanded_n": int(pointwise_mask(updated).sum()),
        "primary_original_n": int(primary_mask(data).sum()),
        "primary_expanded_n": int(primary_mask(updated).sum()),
        "grid_definitions_n": len(original_grid),
        "grid_changed_cells_n": len(changed_grid),
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    replacement.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_cases.csv"), index=False)
    grid_compare.to_csv(args.output_prefix.with_name(args.output_prefix.name + "_grid.csv"), index=False)
    args.output_prefix.with_name(args.output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
