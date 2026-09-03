#!/usr/bin/env python3
"""Regenerate every publication model affected by corrected v14 PPG RI."""

from __future__ import annotations

import json
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

from project_paths import FIGURES_DIR, PROCESSED_DIR, RI_OUTPUT_DIR

OUT = RI_OUTPUT_DIR
PROC = PROCESSED_DIR
FIG = FIGURES_DIR
SUPPFIG = FIG / "supplementary"
CLIN = ["age", "bmi", "asa", "preop_htn", "preop_dm"]
SPECS = {
    "M0": CLIN,
    "M1": CLIN + ["baseline_map"],
    "M2": CLIN + ["ri_mean_clean"],
    "M3": CLIN + ["baseline_map", "ri_mean_clean"],
    "M4": CLIN + ["baseline_map", "ri_mean_clean", "ppg_amp_clean"],
}


def pipe(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def bootstrap_ci(y: np.ndarray, p: np.ndarray, seed: int = 0, n: int = 1000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def delta_ci(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray, seed: int = 0, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx]))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def fit_split(df: pd.DataFrame, specs: dict[str, list[str]], *, seed: int = 42) -> dict:
    y = df["crash_30"].astype(int).to_numpy()
    train, test = train_test_split(
        np.arange(len(df)), test_size=0.30, stratify=y, random_state=seed
    )
    models, predictions = {}, {}
    for offset, (name, cols) in enumerate(specs.items()):
        model = pipe(seed)
        model.fit(df.iloc[train][cols].to_numpy(float), y[train])
        p = model.predict_proba(df.iloc[test][cols].to_numpy(float))[:, 1]
        predictions[name] = p
        models[name] = {
            "auroc": float(roc_auc_score(y[test], p)),
            "auroc_ci95": bootstrap_ci(y[test], p, seed=100 + offset),
            "auprc": float(average_precision_score(y[test], p)),
            "brier": float(brier_score_loss(y[test], p)),
        }
    deltas = {}
    for offset, (name, new, ref) in enumerate(
        (("M1_minus_M0", "M1", "M0"), ("M2_minus_M0", "M2", "M0"),
         ("M3_minus_M1", "M3", "M1"), ("M4_minus_M3", "M4", "M3"))
    ):
        if new in predictions and ref in predictions:
            deltas[name] = {
                "estimate": float(models[new]["auroc"] - models[ref]["auroc"]),
                "ci95": delta_ci(y[test], predictions[new], predictions[ref], seed=200 + offset),
            }
    return {
        "n": int(len(df)),
        "events": int(y.sum()),
        "test_n": int(len(test)),
        "test_events": int(y[test].sum()),
        "models": models,
        "deltas": deltas,
        "y_test": y[test],
        "predictions": predictions,
    }


def strip_arrays(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in {"y_test", "predictions"}}


def repeated(df: pd.DataFrame, specs: dict[str, list[str]], n: int = 200) -> pd.DataFrame:
    y = df["crash_30"].astype(int).to_numpy()
    rows = []
    for seed in range(n):
        train, test = train_test_split(
            np.arange(len(df)), test_size=0.30, stratify=y, random_state=seed
        )
        aucs = {}
        for name, cols in specs.items():
            model = pipe(seed)
            model.fit(df.iloc[train][cols].to_numpy(float), y[train])
            p = model.predict_proba(df.iloc[test][cols].to_numpy(float))[:, 1]
            aucs[name] = float(roc_auc_score(y[test], p))
            rows.append({"seed": seed, "metric": name, "value": aucs[name]})
        for name, new, ref in (
            ("delta_M1_M0", "M1", "M0"),
            ("delta_M2_M0", "M2", "M0"),
            ("delta_M3_M1", "M3", "M1"),
            ("delta_M4_M3", "M4", "M3"),
        ):
            if new in aucs and ref in aucs:
                rows.append({"seed": seed, "metric": name, "value": aucs[new] - aucs[ref]})
    return pd.DataFrame(rows)


def repeated_summary(values: pd.DataFrame) -> list[dict]:
    rows = []
    for metric, group in values.groupby("metric"):
        v = group["value"]
        rows.append({
            "metric": metric,
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "median": float(v.median()),
            "q025": float(v.quantile(0.025)),
            "q975": float(v.quantile(0.975)),
        })
    return sorted(rows, key=lambda x: x["metric"])


def optimism_corrected(
    df: pd.DataFrame,
    cols: list[str],
    *,
    n_boot: int = 200,
    seed: int = 42,
) -> dict:
    """Harrell bootstrap optimism correction on the full analysis cohort."""
    x = df[cols].to_numpy(float)
    y = df["crash_30"].astype(int).to_numpy()
    apparent_model = pipe(seed)
    apparent_model.fit(x, y)
    apparent = float(roc_auc_score(y, apparent_model.predict_proba(x)[:, 1]))
    rng = np.random.default_rng(seed)
    optimism = []
    for bootstrap_index in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if np.unique(y[idx]).size < 2:
            continue
        model = pipe(seed + bootstrap_index + 1)
        model.fit(x[idx], y[idx])
        bootstrap_auc = roc_auc_score(y[idx], model.predict_proba(x[idx])[:, 1])
        original_auc = roc_auc_score(y, model.predict_proba(x)[:, 1])
        optimism.append(float(bootstrap_auc - original_auc))
    mean_optimism = float(np.mean(optimism))
    return {
        "apparent_auroc": apparent,
        "mean_optimism": mean_optimism,
        "optimism_corrected_auroc": float(apparent - mean_optimism),
        "bootstrap_samples": int(len(optimism)),
    }


def feature_table(base_name: str, ri: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_parquet(PROC / base_name)
    corrected = ri[["caseid", "ri_median_v14", "valid_ri_beats"]].copy()
    corrected = corrected.rename(columns={"ri_median_v14": "ri_mean_clean"})
    base = base.drop(columns=["ri_mean_clean"], errors="ignore").merge(corrected, on="caseid", how="left")
    base["ri_mean_v14"] = base["ri_mean_clean"]
    return base


def plot_roc(result: dict, target: Path) -> None:
    colours = {"M0": "#666666", "M1": "#0072B2", "M2": "#E69F00", "M3": "#009E73", "M4": "#D55E00"}
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    y = result["y_test"]
    for name, pred in result["predictions"].items():
        fpr, tpr, _ = roc_curve(y, pred)
        ax.plot(fpr, tpr, lw=1.7, color=colours[name], label=f"{name} (AUROC={result['models'][name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(target.with_suffix(".pdf"))
    fig.savefig(target.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_ladder(result: dict, target: Path) -> None:
    names = list(SPECS)
    y = [result["models"][name]["auroc"] for name in names]
    lo = [result["models"][name]["auroc_ci95"][0] for name in names]
    hi = [result["models"][name]["auroc_ci95"][1] for name in names]
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    x = np.arange(len(names))
    ax.errorbar(x, y, yerr=np.vstack([np.asarray(y)-lo, np.asarray(hi)-y]), marker="o", capsize=4, color="#0072B2")
    ax.set_xticks(x, names)
    ax.set(ylabel="Held-out AUROC", ylim=(0.35, 1.02))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(target.with_suffix(".pdf"))
    fig.savefig(target.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_ri_audit(ri: pd.DataFrame, target: Path) -> None:
    old = ri["ri_mean_historical"].dropna()
    new = ri["ri_median_v14"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].hist(old, bins=35, color="#999999", alpha=0.85)
    axes[0].set(xlabel="Historical RI", ylabel="Cases", title="A  Historical implementation")
    axes[1].hist(new, bins=35, color="#0072B2", alpha=0.85)
    axes[1].set(xlabel="Corrected same-pulse RI", ylabel="Cases", title="B  Corrected implementation")
    fig.tight_layout()
    fig.savefig(target.with_suffix(".pdf"))
    fig.savefig(target.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    SUPPFIG.mkdir(parents=True, exist_ok=True)
    ri = pd.read_parquet(OUT / "ri_case_features_v14.parquet")
    feat = feature_table("vascular_features.parquet", ri)
    feat2 = feature_table("vascular_features_v2.parquet", ri)
    feat.to_parquet(OUT / "vascular_features_v14.parquet", index=False)
    feat2.to_parquet(OUT / "vascular_features_v2_v14.parquet", index=False)

    arc = feat[feat["drop_pct"].notna() & np.isfinite(feat["drop_pct"]) & (feat["drop_pct"] > -500)].copy()
    arc_result = fit_split(arc, SPECS)
    observed_features = arc[
        arc["ri_mean_clean"].notna() & arc["ppg_amp_clean"].notna()
    ].copy()
    cc_specs = {k: SPECS[k] for k in ("M0", "M1", "M3", "M4")}
    observed_feature_result = fit_split(observed_features, cc_specs)

    noisy = feat.copy()
    noisy["crash_30"] = noisy["crash_30"].fillna(0).astype(int)
    noisy_result = fit_split(noisy, SPECS)

    nibp = pd.read_parquet(PROC / "outcome_labels_nibp.parquet")
    keep = ["caseid"] + list(dict.fromkeys(SPECS["M4"]))
    nibp_df = nibp[nibp["crash_30"].notna()].merge(feat2[keep], on="caseid", how="inner")
    nibp_df["baseline_map"] = nibp_df["nibp_baseline"]
    nibp_specs = {k: SPECS[k] for k in ("M0", "M1", "M3", "M4")}
    nibp_result = fit_split(nibp_df, nibp_specs)

    arc_repeated = repeated(arc, SPECS)
    nibp_repeated = repeated(nibp_df, nibp_specs)
    arc_repeated.assign(dataset="arterial_arc").to_csv(OUT / "ri_repeated_splits_arc_v14.csv", index=False)
    nibp_repeated.assign(dataset="nibp_reference").to_csv(OUT / "ri_repeated_splits_nibp_v14.csv", index=False)

    valid_ri = arc["ri_mean_clean"].dropna()
    event_ri = arc.loc[arc["crash_30"].eq(1), "ri_mean_clean"].dropna()
    nonevent_ri = arc.loc[arc["crash_30"].eq(0), "ri_mean_clean"].dropna()
    summary = {
        "version": "v14",
        "ri_definition": "same-pulse foot-referenced median RI",
        "arc": strip_arrays(arc_result),
        "ri_and_amplitude_observed": strip_arrays(observed_feature_result),
        "noisy_label_reference": strip_arrays(noisy_result),
        "nibp_reference": strip_arrays(nibp_result),
        "repeated_splits": {
            "arc": repeated_summary(arc_repeated),
            "nibp_reference": repeated_summary(nibp_repeated),
        },
        "optimism_corrected_arc": {
            name: optimism_corrected(arc, SPECS[name])
            for name in ("M1", "M3", "M4")
        },
        "arc_ri_distribution": {
            "available_n": int(len(valid_ri)),
            "missing_n": int(len(arc) - len(valid_ri)),
            "mean": float(valid_ri.mean()),
            "sd": float(valid_ri.std()),
            "median": float(valid_ri.median()),
            "q1": float(valid_ri.quantile(0.25)),
            "q3": float(valid_ri.quantile(0.75)),
            "event_n": int(len(event_ri)),
            "event_mean": float(event_ri.mean()),
            "event_sd": float(event_ri.std()),
            "nonevent_n": int(len(nonevent_ri)),
            "nonevent_mean": float(nonevent_ri.mean()),
            "nonevent_sd": float(nonevent_ri.std()),
        },
    }
    (OUT / "ri_model_results_v14.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_roc(arc_result, FIG / "Fig3_ROC_calibration")
    plot_ladder(arc_result, SUPPFIG / "SuppFig3_nested_auroc_ladder")
    plot_roc(noisy_result, SUPPFIG / "SuppFig2_nested_model_roc")
    plot_ri_audit(ri, SUPPFIG / "SuppFig_RI_algorithm_audit_v14")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
