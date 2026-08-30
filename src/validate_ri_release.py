#!/usr/bin/env python3
"""Deterministic integrity checks for the corrected v14 RI extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import RI_OUTPUT_DIR

HERE = Path(__file__).resolve().parent
OUT = RI_OUTPUT_DIR


def load_extractor():
    path = HERE / "ri_reextract.py"
    spec = importlib.util.spec_from_file_location("ri_reextract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    extractor = load_extractor()
    # A missing interval must produce two runs, never one stitched run.
    synthetic = np.r_[np.ones(20), np.full(7, np.nan), np.ones(30)]
    assert extractor.finite_runs(synthetic, 10) == [(0, 20), (27, 57)]

    case_path = OUT / "ri_case_features_v14.parquet"
    beat_path = OUT / "ri_beat_fiducials_v14.parquet"
    if not case_path.is_file() or not beat_path.is_file():
        raise FileNotFoundError(
            "Locally generated RI tables not found under "
            f"{OUT}. Run ri_reextract.py first; this code-only release does "
            "not distribute frozen aggregate outputs."
        )

    cases = pd.read_parquet(case_path)
    beats = pd.read_parquet(beat_path)
    assert len(cases) == 909
    assert cases["caseid"].is_unique
    assert len(beats) > 0
    assert (
        (beats["on"] < beats["sp"])
        & (beats["sp"] < beats["dn"])
        & (beats["dn"] < beats["dp"])
        & (beats["dp"] < beats["off"])
    ).all()
    assert (beats["template_correlation"] >= extractor.MIN_TEMPLATE_CORRELATION).all()
    assert (beats["extreme_plateau_fraction"] <= extractor.MAX_EXTREME_PLATEAU_FRACTION).all()
    calculated = beats["diastolic_amplitude"] / beats["systolic_amplitude"]
    assert np.allclose(calculated, beats["ri"], rtol=0, atol=1e-12)

    valid = cases[cases["status"].eq("valid")].copy()
    assert (valid["valid_ri_beats"] >= extractor.MIN_CASE_BEATS).all()
    medians = beats.groupby("caseid")["ri"].median()
    joined = valid.set_index("caseid").join(medians.rename("beat_median"))
    assert np.allclose(joined["ri_median_v14"], joined["beat_median"], rtol=0, atol=1e-12)

    # These broad bounds are safety checks, not physiological reference ranges.
    assert cases["ri_mean_historical"].dropna().median() > 0.90
    assert 0.05 < valid["ri_median_v14"].median() < 0.95
    print("V14 RI extraction integrity validation: PASS")


if __name__ == "__main__":
    main()
