#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step12_repeated_split_stability.py | Topic: 10
Purpose: Repeated 70/30 stratified split stability analysis for the arterial
         ARC and NIBP-corrected reference models.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from project_paths import METRICS_DIR, PROCESSED_DIR, PROJECT_ROOT

WORK = PROJECT_ROOT
PROC_DIR = PROCESSED_DIR
MET_DIR = METRICS_DIR
MET_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 200

CLIN = ["age", "bmi", "asa", "preop_htn", "preop_dm"]
SPECS = {
    "M0": CLIN,
    "M1": CLIN + ["baseline_map"],
    "M2": CLIN + ["ri_mean_clean"],
    "M3": CLIN + ["baseline_map", "ri_mean_clean"],
    "M4": CLIN + ["baseline_map", "ri_mean_clean", "ppg_amp_clean"],
}


def make_pipe(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                                  random_state=seed)),
    ])


def run_repeated(df, dataset, specs):
    rows = []
    y = df["crash_30"].astype(float).to_numpy()
    for seed in range(N_SPLITS):
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.3, stratify=y, random_state=seed
        )
        aucs = {}
        for model, cols in specs.items():
            x = df[cols].astype(float).to_numpy()
            pipe = make_pipe(seed)
            pipe.fit(x[train_idx], y[train_idx])
            pred = pipe.predict_proba(x[test_idx])[:, 1]
            aucs[model] = roc_auc_score(y[test_idx], pred)
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "model": model,
                "auroc": aucs[model],
            })

        deltas = {}
        if "M1" in aucs and "M0" in aucs:
            deltas["delta_M1_M0"] = aucs["M1"] - aucs["M0"]
        if "M2" in aucs and "M0" in aucs:
            deltas["delta_M2_M0"] = aucs["M2"] - aucs["M0"]
        if "M3" in aucs and "M1" in aucs:
            deltas["delta_M3_M1"] = aucs["M3"] - aucs["M1"]
        if "M4" in aucs and "M3" in aucs:
            deltas["delta_M4_M3"] = aucs["M4"] - aucs["M3"]
        for name, val in deltas.items():
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "model": name,
                "auroc": val,
            })
    return pd.DataFrame(rows)


def summarize(res):
    out = []
    for (dataset, model), g in res.groupby(["dataset", "model"]):
        v = g["auroc"].astype(float)
        out.append({
            "dataset": dataset,
            "model": model,
            "n_splits": len(v),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "median": v.median(),
            "q025": v.quantile(0.025),
            "q975": v.quantile(0.975),
        })
    return pd.DataFrame(out).sort_values(["dataset", "model"])


def main():
    feat = pd.read_parquet(PROC_DIR / "vascular_features.parquet")
    arc = feat[~np.isinf(feat["drop_pct"]) & (feat["drop_pct"] > -500)].copy()

    nibp = pd.read_parquet(PROC_DIR / "outcome_labels_nibp.parquet")
    feat2 = pd.read_parquet(PROC_DIR / "vascular_features_v2.parquet")
    keep = ["caseid"] + list(dict.fromkeys(SPECS["M4"]))
    nibp_df = nibp[nibp["crash_30"].notna()].merge(
        feat2[keep], on="caseid", how="inner"
    ).copy()
    nibp_df["baseline_map"] = nibp_df["nibp_baseline"]

    res = pd.concat([
        run_repeated(arc, "arterial_arc", SPECS),
        run_repeated(nibp_df, "nibp_corrected", {
            "M0": SPECS["M0"],
            "M1": SPECS["M1"],
            "M3": SPECS["M3"],
            "M4": SPECS["M4"],
        }),
    ], ignore_index=True)
    summary = summarize(res)

    res.to_csv(MET_DIR / "repeated_split_stability.csv", index=False)
    summary.to_csv(MET_DIR / "repeated_split_stability_summary.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
