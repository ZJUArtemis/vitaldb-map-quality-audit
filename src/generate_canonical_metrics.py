#!/usr/bin/env python3
"""Generate the frozen v9 legacy-model publication metrics.

This script eliminates the two historical sources of disagreement between the
main manuscript and Supplementary Tables S1--S2:

* the reported AUROC is always the point estimate on the fixed held-out set;
  bootstrap samples are used only for confidence limits;
* threshold sensitivity uses M1 (clinical covariates + baseline MAP), exactly
  as stated in the table caption, with the same seed-42 split convention as
  the primary >30% ARC consequence analysis.

The task-window audit is the primary scientific analysis.  These legacy-model
results are retained only as a consequence/noisy-label demonstration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve()
DEFAULT_ROOT = HERE.parents[1] if (HERE.parents[1] / "outputs").exists() else HERE.parents[3]
ROOT = Path(os.environ.get("TOPIC10_PROJECT_ROOT", DEFAULT_ROOT))
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
CLINICAL = ["age", "bmi", "asa", "preop_htn", "preop_dm"]
MODEL_SPECS = {
    "M0: Clinical only": CLINICAL,
    "M1: Clinical + MAP": CLINICAL + ["baseline_map"],
    "M2: Clinical + RI": CLINICAL + ["ri_mean_clean"],
    "M3: Clinical + MAP + RI": CLINICAL + ["baseline_map", "ri_mean_clean"],
    "M4: Clinical + MAP + RI + Amp": CLINICAL + ["baseline_map", "ri_mean_clean", "ppg_amp_clean"],
}


def pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def bootstrap_interval(y: np.ndarray, score: np.ndarray, metric, *, seed: int = 0, n: int = 1000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[idx]).size < 2:
            continue
        values.append(metric(y[idx], score[idx]))
    return np.percentile(np.asarray(values), [2.5, 97.5]).tolist()


def evaluate(df: pd.DataFrame, label: pd.Series, features: list[str]):
    indices = df.index.to_numpy()
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, stratify=label.to_numpy(), random_state=42
    )
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    y_train = label.loc[train_idx].to_numpy(dtype=int)
    y_test = label.loc[test_idx].to_numpy(dtype=int)
    model = pipeline().fit(train[features].to_numpy(dtype=float), y_train)
    probability = model.predict_proba(test[features].to_numpy(dtype=float))[:, 1]
    auroc = float(roc_auc_score(y_test, probability))
    auprc = float(average_precision_score(y_test, probability))
    return {
        "n_total": int(len(df)),
        "n_events": int(label.sum()),
        "n_test": int(len(test)),
        "n_test_events": int(y_test.sum()),
        "auroc": auroc,
        "auroc_ci_low": bootstrap_interval(y_test, probability, roc_auc_score)[0],
        "auroc_ci_high": bootstrap_interval(y_test, probability, roc_auc_score)[1],
        "auprc": auprc,
        "auprc_ci_low": bootstrap_interval(y_test, probability, average_precision_score)[0],
        "auprc_ci_high": bootstrap_interval(y_test, probability, average_precision_score)[1],
        "brier": float(brier_score_loss(y_test, probability)),
        "y_test": y_test,
        "probability": probability,
    }


def serialisable(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in {"y_test", "probability"}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(PROC / "vascular_features.parquet")
    arc = features[
        features["drop_pct"].notna()
        & np.isfinite(features["drop_pct"])
        & (features["drop_pct"] > -500)
    ].copy()

    # Original-pipeline reference: preserve the original NaN->0 convention.
    noisy_label = features["crash_30"].fillna(0).astype(int)
    noisy_results = {}
    noisy_rows = []
    for model_name, cols in MODEL_SPECS.items():
        result = evaluate(features, noisy_label, cols)
        noisy_results[model_name] = serialisable(result)
        noisy_rows.append({"model": model_name, **serialisable(result)})
    pd.DataFrame(noisy_rows).to_csv(OUT / "canonical_noisy_label_models_v9.csv", index=False)

    # M1 threshold sensitivity.  Each threshold defines a different label and
    # therefore receives its own stratified seed-42 split.  At >30%, this is
    # exactly the primary M1 ARC split and must reproduce its point AUROC.
    threshold_results = {}
    threshold_rows = []
    for threshold in (20, 25, 30, 35, 40):
        label = (arc["drop_pct"] > threshold).astype(int)
        result = evaluate(arc, label, MODEL_SPECS["M1: Clinical + MAP"])
        threshold_results[str(threshold)] = serialisable(result)
        threshold_rows.append({"threshold_percent": threshold, **serialisable(result)})
    pd.DataFrame(threshold_rows).to_csv(OUT / "canonical_arc_threshold_m1_v9.csv", index=False)

    primary_arc = evaluate(arc, arc["crash_30"].astype(int), MODEL_SPECS["M1: Clinical + MAP"])
    assert abs(primary_arc["auroc"] - threshold_results["30"]["auroc"]) < 1e-12

    # Full-cohort baseline-waveform availability: absent raw tracks contribute
    # zero support because the denominator is all 926 fixed task windows.
    case_path = ROOT / "outputs" / "metrics" / "revision_v5_fixed_window_case_results.csv"
    cases = pd.read_csv(case_path)
    observed = cases["raw_baseline_observed_fraction"].fillna(0).to_numpy(float)
    observed_summary = {
        "n": int(len(observed)),
        "median": float(np.median(observed)),
        "q1": float(np.percentile(observed, 25)),
        "q3": float(np.percentile(observed, 75)),
    }

    frozen = {
        "analysis_role": "legacy consequence and noisy-label reference; not clinical model validation",
        "split": "70/30 stratified, random_state=42",
        "reported_metric": "held-out point estimate; bootstrap percentile interval only",
        "bootstrap": {"iterations": 1000, "seed": 0},
        "noisy_label_models": noisy_results,
        "arc_threshold_m1": threshold_results,
        "primary_arc_m1": serialisable(primary_arc),
        "raw_baseline_observed_fraction_all_926": observed_summary,
    }
    (OUT / "canonical_publication_metrics_v9.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )

    # Regenerate the two supplementary figures from the same predictions.
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    palette = ["#666666", "#0072B2", "#E69F00", "#009E73", "#D55E00"]
    for (name, result), colour in zip(
        [(n, evaluate(features, noisy_label, c)) for n, c in MODEL_SPECS.items()], palette
    ):
        fpr, tpr, _ = roc_curve(result["y_test"], result["probability"])
        ax.plot(fpr, tpr, lw=1.6, color=colour, label=f"{name.split(':')[0]} (AUROC={result['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title="Original-Pipeline Noisy-Label Reference")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "suppfig2_noisy_label_roc_v9.pdf")
    fig.savefig(OUT / "suppfig2_noisy_label_roc_v9.png", dpi=300)
    plt.close(fig)

    table = pd.DataFrame(threshold_rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = table["threshold_percent"].to_numpy()
    y = table["auroc"].to_numpy()
    lo = table["auroc_ci_low"].to_numpy()
    hi = table["auroc_ci_high"].to_numpy()
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=4, color="#0072B2")
    ax.set(xlabel="Relative MAP-decrease threshold (%)", ylabel="M1 held-out AUROC",
           title="ARC Threshold Sensitivity (Clinical + MAP)")
    ax.set_xticks(x)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "suppfig1_threshold_m1_v9.pdf")
    fig.savefig(OUT / "suppfig1_threshold_m1_v9.png", dpi=300)
    plt.close(fig)

    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
