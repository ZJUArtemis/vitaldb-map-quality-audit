#!/usr/bin/env python3
"""Regenerate canonical held-out NIBP model metrics and paired delta CIs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import METRICS_DIR, PROCESSED_DIR


def make_pipe() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)),
    ])


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, seed: int, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def bootstrap_delta_ci(
    y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray, seed: int, n: int = 2000
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[idx]).size == 2:
            values.append(
                roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx])
            )
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main() -> None:
    nibp = pd.read_parquet(PROCESSED_DIR / "outcome_labels_nibp.parquet")
    feat = pd.read_parquet(PROCESSED_DIR / "vascular_features_v2.parquet")
    keep = [
        "caseid", "age", "bmi", "asa", "preop_htn", "preop_dm",
        "ri_mean_clean", "ppg_amp_clean",
    ]
    df = nibp.merge(feat[[c for c in keep if c in feat.columns]], on="caseid", how="inner")
    df = df[df.crash_30.notna()].copy()
    df["baseline_map"] = df["nibp_baseline"]
    y = df.crash_30.to_numpy(dtype=float)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, stratify=y, random_state=42
    )
    specs = {
        "M0": ["age", "bmi", "asa", "preop_htn", "preop_dm"],
        "M1": ["age", "bmi", "asa", "preop_htn", "preop_dm", "baseline_map"],
        "M3": [
            "age", "bmi", "asa", "preop_htn", "preop_dm", "baseline_map",
            "ri_mean_clean",
        ],
    }
    predictions = {}
    models = {}
    for model_index, (name, features) in enumerate(specs.items()):
        model = make_pipe()
        model.fit(df.iloc[train_idx][features].to_numpy(dtype=float), y[train_idx])
        p = model.predict_proba(df.iloc[test_idx][features].to_numpy(dtype=float))[:, 1]
        predictions[name] = p
        auc = float(roc_auc_score(y[test_idx], p))
        models[name] = {
            "auroc": auc,
            "ci95": bootstrap_auc_ci(y[test_idx], p, seed=430 + model_index),
            "brier": float(brier_score_loss(y[test_idx], p)),
        }

    deltas = {}
    for name, new_name, ref_name, seed in (
        ("M1_minus_M0", "M1", "M0", 441),
        ("M3_minus_M1", "M3", "M1", 442),
    ):
        estimate = models[new_name]["auroc"] - models[ref_name]["auroc"]
        deltas[name] = {
            "estimate": float(estimate),
            "ci95": bootstrap_delta_ci(
                y[test_idx], predictions[new_name], predictions[ref_name], seed=seed
            ),
        }

    output = {
        "source_segments_n": int(len(nibp)),
        "range_plausible_baseline_n": int(nibp.nibp_baseline.notna().sum()),
        "range_plausible_baseline_median_mmHg": float(nibp.nibp_baseline.median()),
        "range_plausible_baseline_iqr_mmHg": [
            float(nibp.nibp_baseline.quantile(0.25)),
            float(nibp.nibp_baseline.quantile(0.75)),
        ],
        "relative_drop_evaluable_n": int(nibp.crash_30.notna().sum()),
        "relative_drop_events_n": int(nibp.crash_30.fillna(0).sum()),
        "model_cohort_n": int(len(df)),
        "model_events_n": int(y.sum()),
        "test_n": int(len(test_idx)),
        "test_events_n": int(y[test_idx].sum()),
        "models": models,
        "paired_delta_auroc": deltas,
    }
    target = METRICS_DIR / "nibp_model_metrics_v10.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
