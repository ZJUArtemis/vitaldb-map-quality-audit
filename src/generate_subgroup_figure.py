#!/usr/bin/env python3
"""Regenerate the exploratory ARC subgroup forest with canonical seed-0 CIs."""

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve()
DEFAULT_ROOT = HERE.parents[1] if (HERE.parents[1] / "outputs").exists() else HERE.parents[3]
ROOT = Path(os.environ.get("TOPIC10_PROJECT_ROOT", DEFAULT_ROOT))
OUT = HERE.parent
CLIN = ["age", "bmi", "asa", "preop_htn", "preop_dm"]


def ci(y, p, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1000):
        idx = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[idx]).size > 1:
            values.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(values, [2.5, 97.5])


def main():
    feat = pd.read_parquet(ROOT / "data/processed/vascular_features.parquet")
    arc = feat[feat.drop_pct.notna() & np.isfinite(feat.drop_pct) & (feat.drop_pct > -500)].copy()
    train_idx, test_idx = train_test_split(
        arc.index.to_numpy(), test_size=0.30, stratify=arc.crash_30, random_state=42
    )
    train, test = arc.loc[train_idx], arc.loc[test_idx].copy()
    features = CLIN + ["baseline_map"]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(train[features], train.crash_30)
    test["_probability"] = model.predict_proba(test[features])[:, 1]
    test["_label"] = test.crash_30.astype(int)
    groups = [
        ("Overall", np.ones(len(test), bool)),
        ("Age < 65", test.age < 65), ("Age $\\geq$ 65", test.age >= 65),
        ("Male", test.sex == "M"), ("Female", test.sex == "F"),
        ("BMI < 25", test.bmi < 25), ("BMI $\\geq$ 25", test.bmi >= 25),
        ("ASA I--II", test.asa <= 2), ("ASA III--IV", test.asa >= 3),
        ("Hypertension", test.preop_htn == 1), ("No hypertension", test.preop_htn == 0),
        ("Diabetes", test.preop_dm == 1), ("No diabetes", test.preop_dm == 0),
    ]
    rows = []
    for name, mask in groups:
        mask = np.asarray(mask)
        y = test.loc[mask, "_label"].to_numpy()
        p = test.loc[mask, "_probability"].to_numpy()
        if np.unique(y).size < 2 or y.sum() < 3 or (len(y) - y.sum()) < 3:
            rows.append((name, np.nan, np.nan, np.nan, len(y), int(y.sum())))
        else:
            low, high = ci(y, p)
            rows.append((name, roc_auc_score(y, p), low, high, len(y), int(y.sum())))

    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    positions = np.arange(len(rows))[::-1]
    ax.axvline(rows[0][1], color="#C8502A", ls="--", lw=1)
    for position, (name, auc, low, high, n, events) in zip(positions, rows):
        if np.isnan(auc):
            ax.text(0.515, position, f"{name} (n={n}, events={events}: too few)", va="center", fontsize=7.5)
            continue
        colour = "#C8502A" if name == "Overall" else "#1f6fb2"
        marker = "D" if name == "Overall" else "o"
        ax.plot([low, high], [position, position], color=colour, lw=1.6)
        ax.plot(auc, position, marker, color=colour, ms=7 if name == "Overall" else 5)
        ax.text(1.015, position, f"{auc:.3f} [{low:.3f}--{high:.3f}] n={n}", va="center", fontsize=7.3)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows], fontsize=8.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("M1 AUROC (95% CI)")
    ax.set_title("Subgroup analysis -- M1 (clinical + MAP), ARC test set ($n$=159)\nExploratory; subgroup cells are small", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "suppfig8_subgroup_forest_v9.pdf")
    fig.savefig(OUT / "suppfig8_subgroup_forest_v9.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
