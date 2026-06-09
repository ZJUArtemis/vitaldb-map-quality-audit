#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: step7_sensitivity_analysis.py | Topic: 10 | Purpose: Phase 6 敏感性分析
- Step 6.1: 不同Crash阈值 (20%/25%/30%/35%/40%) AUROC稳定性
- Step 6.2: 亚组分析 (年龄/性别/BMI/ASA/高血压) + Forest Plot
- Step 6.3: Age × baseline_map 交互作用分析
"""
import json, logging, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

RAW_DATA = Path("vitaldb_data")  # local VitalDB physionet.org root; override as needed
WORK = Path(__file__).resolve().parents[1]

def safe_path(p):
    assert not str(Path(p).resolve()).startswith(str(RAW_DATA.resolve()))
    return Path(p)

def init_log():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lf = WORK / "outputs" / "logs" / f"{ts}_step7_sensitivity.log"
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()])

def bootstrap_auroc(y_true, y_score, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(roc_auc_score(y_true[idx], y_score[idx]))
        except:
            pass
    scores = np.array(scores)
    return float(np.median(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def fit_and_predict(feat_df, feats, y, idx_train, idx_test):
    """Fit LR pipeline and return test predictions."""
    df_sub = feat_df[feats].copy()
    if 'asa' in df_sub.columns:
        df_sub['asa'] = pd.to_numeric(df_sub['asa'], errors='coerce')
    X = df_sub.values.astype(float)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42))
    ])
    pipe.fit(X[idx_train], y[idx_train])
    return pipe.predict_proba(X[idx_test])[:, 1]

def subgroup_auroc(y_true, y_prob, mask):
    """Compute AUROC + CI for a subgroup defined by mask."""
    yt = y_true[mask]
    yp = y_prob[mask]
    if len(yt) < 20 or yt.sum() < 3 or (yt == 0).sum() < 3:
        return None, None, None, len(yt)
    med, lo, hi = bootstrap_auroc(yt, yp, n=500)
    return med, lo, hi, len(yt)

def main():
    init_log()
    np.random.seed(42)
    logging.info("=== Topic 10 Phase 6: Sensitivity Analysis ===")

    # ── Load Data ──────────────────────────────────────────
    feat = pd.read_parquet(WORK / "data" / "processed" / "vascular_features.parquet")
    outcome = pd.read_parquet(WORK / "data" / "processed" / "outcome_labels.parquet")

    # Clean
    feat['asa'] = pd.to_numeric(feat['asa'], errors='coerce')
    feat['age'] = pd.to_numeric(feat['age'], errors='coerce')

    # Core features for Model B
    FEATS_B = ['age', 'bmi', 'asa', 'preop_htn', 'preop_dm', 'baseline_map', 'ri_mean_clean', 'ppg_amp_clean']

    logging.info(f"Dataset: {len(feat)} cases")

    # ══════════════════════════════════════════════════════
    # Step 6.1: Threshold Sensitivity Analysis
    # ══════════════════════════════════════════════════════
    logging.info("\n=== Step 6.1: Crash Threshold Sensitivity ===")

    thresholds = [20, 25, 30, 35, 40]
    threshold_results = []

    for thresh in thresholds:
        col = f'crash_{thresh}'
        if col not in feat.columns:
            # Compute from drop_pct
            if 'drop_pct' in feat.columns:
                feat[col] = (feat['drop_pct'] > thresh).astype(int)
            else:
                logging.warning(f"Cannot compute {col}, skipping")
                continue

        y = feat[col].values.astype(float)
        n_crash = int(y.sum())
        prevalence = float(y.mean())

        idx_train, idx_test = train_test_split(
            np.arange(len(feat)), test_size=0.3, stratify=y, random_state=42)

        # Fit Model B
        y_prob = fit_and_predict(feat, FEATS_B, y, idx_train, idx_test)
        y_test = y[idx_test]

        auroc, lo, hi = bootstrap_auroc(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        threshold_results.append({
            'threshold': thresh,
            'n_crash': n_crash,
            'prevalence': prevalence,
            'auroc': auroc, 'auroc_lo': lo, 'auroc_hi': hi,
            'auprc': auprc, 'brier': brier
        })
        logging.info(f"Threshold {thresh}%: n_crash={n_crash} ({prevalence*100:.1f}%), "
                     f"AUROC={auroc:.3f} [{lo:.3f}-{hi:.3f}], AUPRC={auprc:.3f}")

    thresh_df = pd.DataFrame(threshold_results)
    thresh_path = safe_path(WORK / "outputs" / "metrics" / "threshold_sensitivity.csv")
    thresh_df.to_csv(thresh_path, index=False)
    logging.info(f"Saved threshold sensitivity: {thresh_path}")

    # Plot: AUROC across thresholds
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(thresh_df['threshold'], thresh_df['auroc'],
                yerr=[thresh_df['auroc'] - thresh_df['auroc_lo'],
                      thresh_df['auroc_hi'] - thresh_df['auroc']],
                fmt='o-', color='#0072B2', linewidth=2, capsize=5, markersize=7)
    ax.axhline(0.7, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='AUROC=0.70')
    ax.set_xlabel('MAP Drop Threshold (%)')
    ax.set_ylabel('AUROC (95% CI)')
    ax.set_title('Model Performance Across Crash Definitions')
    ax.set_ylim(0.5, 1.0)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(safe_path(WORK / "outputs" / "figures" / "supp_threshold_sensitivity.png"), dpi=300)
    plt.close()
    logging.info("Saved threshold sensitivity plot")

    # ══════════════════════════════════════════════════════
    # Step 6.2: Subgroup Analysis + Forest Plot
    # ══════════════════════════════════════════════════════
    logging.info("\n=== Step 6.2: Subgroup Analysis ===")

    # Use crash_30 as primary outcome
    y30 = feat['crash_30'].values.astype(float)
    idx_train30, idx_test30 = train_test_split(
        np.arange(len(feat)), test_size=0.3, stratify=y30, random_state=42)

    # Get test set predictions from full model
    y_prob_full = fit_and_predict(feat, FEATS_B, y30, idx_train30, idx_test30)
    y_test30 = y30[idx_test30]
    feat_test = feat.iloc[idx_test30].copy().reset_index(drop=True)

    # Define subgroups (on test set)
    subgroups = {
        'Overall': np.ones(len(feat_test), dtype=bool),
        'Age < 65': feat_test['age'] < 65,
        'Age ≥ 65': feat_test['age'] >= 65,
        'Male': feat_test['sex'] == 'M',
        'Female': feat_test['sex'] == 'F',
        'BMI < 25': feat_test['bmi'] < 25,
        'BMI 25-30': (feat_test['bmi'] >= 25) & (feat_test['bmi'] < 30),
        'BMI ≥ 30': feat_test['bmi'] >= 30,
        'ASA I-II': feat_test['asa'].isin([1, 2]),
        'ASA III-IV': feat_test['asa'].isin([3, 4]),
        'Hypertension': feat_test['preop_htn'] == 1,
        'No Hypertension': feat_test['preop_htn'] == 0,
        'Diabetes': feat_test['preop_dm'] == 1,
        'No Diabetes': feat_test['preop_dm'] == 0,
    }

    subgroup_results = []
    for name, mask in subgroups.items():
        mask_arr = mask.values if hasattr(mask, 'values') else np.array(mask)
        auroc, lo, hi, n = subgroup_auroc(y_test30, y_prob_full, mask_arr)
        n_crash = int(y_test30[mask_arr].sum()) if mask_arr.sum() > 0 else 0
        subgroup_results.append({
            'subgroup': name, 'n': n, 'n_crash': n_crash,
            'auroc': auroc, 'auroc_lo': lo, 'auroc_hi': hi
        })
        if auroc is not None:
            logging.info(f"  {name:<20} n={n:3d} crash={n_crash:2d} AUROC={auroc:.3f} [{lo:.3f}-{hi:.3f}]")
        else:
            logging.info(f"  {name:<20} n={n:3d} — insufficient events for AUROC")

    sg_df = pd.DataFrame(subgroup_results)
    sg_path = safe_path(WORK / "outputs" / "metrics" / "subgroup_analysis.csv")
    sg_df.to_csv(sg_path, index=False)
    logging.info(f"Saved subgroup analysis: {sg_path}")

    # Forest Plot
    plot_df = sg_df[sg_df['auroc'].notna()].copy()
    fig, ax = plt.subplots(figsize=(8, len(plot_df) * 0.55 + 1.5))

    overall_auroc = plot_df[plot_df['subgroup'] == 'Overall']['auroc'].values[0]
    colors = ['#D55E00' if sg == 'Overall' else '#0072B2' for sg in plot_df['subgroup']]

    y_pos = np.arange(len(plot_df))[::-1]
    for i, (_, row) in enumerate(plot_df.iterrows()):
        yp = y_pos[i]
        ax.plot([row['auroc_lo'], row['auroc_hi']], [yp, yp],
                color=colors[i], linewidth=2, solid_capstyle='round')
        ax.plot(row['auroc'], yp, 'D', color=colors[i], markersize=7, zorder=5)
        ax.text(1.01, yp, f"{row['auroc']:.3f} [{row['auroc_lo']:.3f}-{row['auroc_hi']:.3f}]  n={row['n']}",
                va='center', fontsize=7.5, transform=ax.get_yaxis_transform())

    ax.axvline(overall_auroc, color='#D55E00', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['subgroup'], fontsize=8.5)
    ax.set_xlabel('AUROC (95% CI)', fontsize=9)
    ax.set_title('Subgroup Analysis — Model B (LR)\nPrimary Outcome: MAP Drop > 30%', fontsize=10)
    ax.set_xlim(0.5, 1.05)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(safe_path(WORK / "outputs" / "figures" / "fig_forest_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved forest plot")

    # ══════════════════════════════════════════════════════
    # Step 6.3: Age × baseline_map Interaction Analysis
    # ══════════════════════════════════════════════════════
    logging.info("\n=== Step 6.3: Age × baseline_map Interaction ===")

    # Full dataset logistic regression with interaction term
    df_int = feat[['age', 'bmi', 'asa', 'preop_htn', 'preop_dm',
                   'baseline_map', 'ri_mean_clean', 'ppg_amp_clean', 'crash_30']].copy()
    df_int['asa'] = pd.to_numeric(df_int['asa'], errors='coerce')
    df_int['age_z'] = (df_int['age'] - df_int['age'].mean()) / df_int['age'].std()
    df_int['map_z'] = (df_int['baseline_map'] - df_int['baseline_map'].mean()) / df_int['baseline_map'].std()
    df_int['age_x_map'] = df_int['age_z'] * df_int['map_z']

    feats_int = ['age_z', 'bmi', 'asa', 'preop_htn', 'preop_dm', 'map_z', 'ri_mean_clean', 'ppg_amp_clean', 'age_x_map']
    df_int_clean = df_int[feats_int + ['crash_30']].dropna()
    logging.info(f"Interaction analysis: {len(df_int_clean)} complete cases")

    imp = SimpleImputer(strategy='median')
    X_int = imp.fit_transform(df_int_clean[feats_int].values.astype(float))
    y_int = df_int_clean['crash_30'].values.astype(float)

    lr_int = LogisticRegression(penalty=None, max_iter=2000, random_state=42)
    try:
        lr_int.fit(X_int, y_int)
        coef_df = pd.DataFrame({'feature': feats_int, 'coef': lr_int.coef_[0]})

        # Bootstrap CI for interaction term
        rng = np.random.RandomState(42)
        int_coefs = []
        int_idx = feats_int.index('age_x_map')
        for _ in range(500):
            idx_b = rng.choice(len(X_int), len(X_int), replace=True)
            try:
                lr_b = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
                lr_b.fit(X_int[idx_b], y_int[idx_b])
                int_coefs.append(lr_b.coef_[0][int_idx])
            except:
                pass
        int_coefs = np.array(int_coefs)
        int_lo, int_hi = np.percentile(int_coefs, [2.5, 97.5])
        int_coef = lr_int.coef_[0][int_idx]
        int_p = 2 * (1 - stats.norm.cdf(abs(int_coef / (int_coefs.std() + 1e-8))))

        logging.info(f"Interaction (Age×MAP) coefficient: {int_coef:.4f} [{int_lo:.4f}-{int_hi:.4f}], p≈{int_p:.4f}")
        if int_p < 0.05:
            logging.info("  → SIGNIFICANT interaction: age and baseline MAP synergistically predict crash")
        else:
            logging.info("  → No significant interaction (p≥0.05)")

        logging.info("\nAll coefficients:")
        for _, r in coef_df.iterrows():
            logging.info(f"  {r['feature']:<20}: {r['coef']:.4f}")

        # Visualization: stratified MAP curves by age group
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        age_groups = {'Young (<55)': feat['age'] < 55, 'Middle (55-70)': (feat['age'] >= 55) & (feat['age'] < 70), 'Old (≥70)': feat['age'] >= 70}
        colors_age = {'Young (<55)': '#009E73', 'Middle (55-70)': '#E69F00', 'Old (≥70)': '#D55E00'}
        for gname, gmask in age_groups.items():
            sub = feat[gmask & feat['baseline_map'].notna() & feat['drop_pct'].notna()]
            if len(sub) < 10: continue
            ax.scatter(sub['baseline_map'], sub['drop_pct'],
                      alpha=0.3, s=8, color=colors_age[gname], label=f'{gname} (n={len(sub)})')
            # Regression line
            from numpy.polynomial import polynomial as P
            mask_v = sub['baseline_map'].notna() & sub['drop_pct'].notna()
            if mask_v.sum() > 5:
                x_sorted = np.sort(sub.loc[mask_v, 'baseline_map'].values)
                c = np.polyfit(sub.loc[mask_v, 'baseline_map'].values,
                               sub.loc[mask_v, 'drop_pct'].values, 1)
                ax.plot(x_sorted, np.polyval(c, x_sorted), color=colors_age[gname], linewidth=2)
        ax.set_xlabel('Baseline MAP (mmHg)'); ax.set_ylabel('MAP Drop (%)')
        ax.set_title('Baseline MAP vs Drop % by Age Group')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax2 = axes[1]
        for gname, gmask in age_groups.items():
            sub = feat[gmask & feat['crash_30'].notna()]
            if len(sub) < 10: continue
            map_bins = pd.qcut(sub['baseline_map'].fillna(sub['baseline_map'].median()), q=5, duplicates='drop')
            crash_by_bin = sub.groupby(map_bins)['crash_30'].mean()
            bin_mids = [iv.mid for iv in crash_by_bin.index.categories]
            ax2.plot(bin_mids, crash_by_bin.values, 'o-', color=colors_age[gname],
                    label=gname, linewidth=2, markersize=5)
        ax2.set_xlabel('Baseline MAP (mmHg)'); ax2.set_ylabel('Crash Rate')
        ax2.set_title('Crash Rate vs Baseline MAP by Age Group')
        ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(safe_path(WORK / "outputs" / "figures" / "fig_interaction_age_map.png"), dpi=300)
        plt.close()
        logging.info("Saved interaction plot")

        coef_df['int_ci'] = [f"{int_lo:.4f}-{int_hi:.4f}" if f == 'age_x_map' else '' for f in feats_int]
        coef_df.to_csv(safe_path(WORK / "outputs" / "metrics" / "interaction_coefficients.csv"), index=False)

    except Exception as e:
        logging.warning(f"Interaction analysis failed: {e}")

    # ══════════════════════════════════════════════════════
    # Step 6.4: Alternative outcome — crash_absolute
    # ══════════════════════════════════════════════════════
    logging.info("\n=== Step 6.4: Alternative Outcome (MAP < 65 mmHg) ===")
    # crash_absolute is in outcome_labels, merge it in
    if 'crash_absolute' not in feat.columns:
        feat = feat.merge(outcome[['caseid','crash_absolute']], on='caseid', how='left')
    y_abs = feat['crash_absolute'].values.astype(float)
    logging.info(f"crash_absolute prevalence: {y_abs.mean()*100:.1f}%")
    try:
        idx_tr_abs, idx_te_abs = train_test_split(
            np.arange(len(feat)), test_size=0.3, stratify=y_abs, random_state=42)
        y_prob_abs = fit_and_predict(feat, FEATS_B, y_abs, idx_tr_abs, idx_te_abs)
        auroc_abs, lo_abs, hi_abs = bootstrap_auroc(y_abs[idx_te_abs], y_prob_abs)
        logging.info(f"Model B on crash_absolute: AUROC={auroc_abs:.3f} [{lo_abs:.3f}-{hi_abs:.3f}]")
    except Exception as e:
        logging.warning(f"crash_absolute analysis: {e}")

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    logging.info("\n=== Phase 6 Summary ===")
    logging.info("Threshold Sensitivity (Model B AUROC across crash definitions):")
    for _, r in thresh_df.iterrows():
        logging.info(f"  MAP drop >{r['threshold']}%: AUROC={r['auroc']:.3f} [{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}]")
    logging.info("\nOutputs saved:")
    logging.info("  metrics/threshold_sensitivity.csv")
    logging.info("  metrics/subgroup_analysis.csv")
    logging.info("  metrics/interaction_coefficients.csv")
    logging.info("  figures/supp_threshold_sensitivity.png")
    logging.info("  figures/fig_forest_plot.png")
    logging.info("  figures/fig_interaction_age_map.png")
    logging.info("=== DONE ===")

if __name__ == "__main__":
    main()
