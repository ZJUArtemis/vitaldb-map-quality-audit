# Arterial Pressure Data Quality in VitalDB — Model-Decomposition Audit

Reproducibility code for the manuscript:

> **Arterial Pressure Data Quality in VitalDB Constrains Retrospective Prediction
> of Induction-Associated MAP Decrease: A Model-Decomposition Audit**
> Ge Gao, Department of Anesthesiology, The First Affiliated Hospital,
> Zhejiang University School of Medicine, Hangzhou, China.

This repository contains the complete analysis pipeline for a data-quality audit
of pre-induction arterial blood-pressure recordings in [VitalDB](https://vitaldb.net),
together with the nested model-decomposition and sensitivity analyses reported in
the paper.

## Key finding

The `Solar8000/ART_MBP` track records monitor-derived mean arterial pressure
(MAP) only **after** arterial-catheter connection and zeroing. In total
intravenous anaesthesia (TIVA) cases the catheter is frequently placed during or
after induction, so the pre-induction baseline window contains pre-connection
artefact rather than physiological MAP. We show that the apparent dominance of
baseline MAP in predicting induction-associated MAP decrease is **substantially a
data-validity artefact**, and that a physiologically valid pre-induction baseline
can instead be recovered from the non-invasive oscillometric cuff
(`Solar8000/NIBP_MBP`) — under which the MAP-associated AUROC gain collapses from
+0.421 to +0.029.

## Repository layout

```
src/                         Analysis pipeline (run in order)
  verify_vitaldb.py          Environment / VitalDB access check
  vitaldb_utils.py           Shared loading & track-mapping helpers
  step1_cohort_selection.py  Inclusion/exclusion → eligible cases
  step2_induction_segmentation.py
  step3_outcome_labeling.py  Original (artefact-prone) ΔMAP% labels
  step4_changepoint_detection.py
  step5_vascular_features.py / step5b_fix_vascular_features.py
  step6_risk_modeling.py
  step7_sensitivity_analysis.py
  step8_paper_figures.py / step8b_tables_shap.py / step8c_fix_fig6_shap.py
  step9_nested_models.py     Nested logistic regression (M0–M4) + validity indicator
  step10_recompute_outcomes_v2.py   SNUADC/ART recomputation
  step11_nibp_corrected.py   NIBP-based corrected reference arm
  step_v5_analyses.py / step_v6_analyses.py   Revision analyses
results/
  metrics/                   Aggregate result tables backing the paper (CSV/JSON)
  figures/                   Publication figures (PNG)
```

> **Patient-level derived data are not redistributed.** Per the VitalDB data-use
> terms, the intermediate per-case feature/outcome tables (`*.parquet`) are
> excluded (`.gitignore`). They are regenerated deterministically from VitalDB by
> running the pipeline below. The `results/` directory contains only
> aggregate, non-identifiable summaries and figures.

## Data access

VitalDB is openly available at <https://vitaldb.net>
(Lee et al., *Sci. Data* 2022, doi:10.1038/s41597-022-01411-5). No additional
ethics approval was required for this secondary analysis of fully anonymised data.
The `vitaldb` Python package streams cases on demand; no bulk download is needed.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10. `torch` is used only for seeding/utility helpers; a CPU
build is sufficient (no GPU required for any analysis in this paper).

## Reproducing the analysis

Run the scripts in `src/` in numerical order. **Note:** the scripts currently use
absolute output paths under the original author workspace
(`/home/lxk/vitaldb/analysis/...`); adjust the `WORK` / output-path constants near
the top of each script to your local directory before running. Each script writes
intermediate `.parquet` files (regenerated locally, not shipped) and the aggregate
tables/figures found under `results/`.

A typical end-to-end run:

```bash
python src/step1_cohort_selection.py
python src/step2_induction_segmentation.py
python src/step3_outcome_labeling.py
python src/step5_vascular_features.py
python src/step9_nested_models.py          # main nested-model results
python src/step11_nibp_corrected.py        # NIBP corrected reference arm
```

## Citation

If you use this code, please cite the manuscript above (full citation to be added
on acceptance) and the VitalDB dataset.

## License

MIT — see [LICENSE](LICENSE).
