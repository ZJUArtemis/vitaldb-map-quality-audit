# VitalDB Task-Window Validity Audit

This repository contains the analysis code for auditing arterial-pressure data validity in VitalDB for a fixed-window post-induction hypotension prediction task. The study examines whether invasive arterial-pressure monitor tracks provide sufficient temporal coverage and physiological plausibility to support pre-induction baseline and post-induction outcome measurements.

## Scope

The package keeps the primary fixed-window audit, its source/coverage
sensitivities, the prerequisite cohort/feature workflow, corrected RI,
corrected-RI models, and published supplementary analyses. Superseded models,
unreported SHAP/MLP/XGBoost work, an unused data loader and unreported figures
were removed. Executed statements in retained analysis modules are preserved.
The historical feature extraction is required for legacy amplitude and for
comparing historical RI with corrected RI; it must not be used as the final
RI definition. `regenerate_ri_models.py` supplies the corrected-RI final model
results and replaces historical RI-based results from prerequisite scripts.

## Restored feature-table prerequisite

The original public v21 release consumes `vascular_features_v2.parquet` but
omits its producer. The archived `step10_recompute_outcomes_v2.py` was
recovered from submission package v6 and included here. Its computation and
feature-row selection are preserved; only path configuration and explanatory
documentation were updated. This restores the missing producer instead of
substituting a different feature table. Its historical SNUADC/ART sampled
outcomes are not the primary validity audit or the final NIBP labels; the
NIBP stage replaces them with separately recorded cuff-based labels.
Raw clinical data and waveforms must be obtained from VitalDB separately.
No patient-level tables or waveforms are distributed.

## Installation and paths

Run commands from the extracted code-package root in a Python environment
compatible with `requirements.txt`:

```sh
python -m pip install -r requirements.txt
export TOPIC10_PROJECT_ROOT=.
export VITALDB_VITAL_DIR=./vitaldb_data
```

Place the raw case files in `vitaldb_data/` with four-digit names such as
`0001.vital`. For the legacy clinical-feature step, either retain the original
PhysioNet-style layout or set its root explicitly:

```sh
export VITALDB_PHYSIONET_ROOT=./physionet.org
```

The legacy CSV is expected at
`physionet.org/files/vitaldb/1.0.0/clinical_data.csv`. The default layout and all
examples are relative to this package. Environment variables may point to the
user's own data locations. Generated products go under `data/processed/` and
`outputs/`; raw inputs must not be overwritten.

## Execution order

### 1. Cohort selection and primary audit

```sh
python src/step1_cohort_selection.py
python src/fixed_window_validity_audit.py --workers 4
```

The cohort script produces `outputs/metrics/eligible_caseids.csv` and track
metadata. The primary audit reconstructs the first positive propofol-rate
anchor and uses fixed 300-s baseline and 600-s outcome windows, independently
of the legacy ART-derived segmentation.

### 2. Primary-audit sensitivities

```sh
python src/native_coverage_audit.py --workers 4
python src/broader_arterial_cohort_audit.py --workers 4
python src/fem_waveform_audit.py --workers 4
python src/fem_mbp_source_sensitivity.py
python src/cadence_window_sensitivity_audit.py
```

Run the native audit before the broader cohort sensitivity. These scripts
provide direct-record validation, symmetric outcome continuity, PPG-free
cohort checks, FEM waveform/numeric-source checks, and window-length analyses.

### 3. Legacy feature prerequisites

```sh
python src/step2_induction_segmentation.py
python src/step3_outcome_labeling.py
python src/step5_vascular_features.py
python src/step5b_fix_vascular_features.py
python src/step10_recompute_outcomes_v2.py
```

These produce the historical feature table consumed by corrected-RI
re-extraction. Keep the historical RI/amplitude columns for implementation
comparisons; the final model stage replaces the RI column with corrected RI.
The restored step 10 produces `vascular_features_v2.parquet` using the
original historical selection/merge. Do not use its arterial outcomes as
physiological prediction estimates. The NIBP step regenerates its own labels.

### 4. Corrected RI and NIBP outcome

```sh
python src/ri_reextract.py
python src/validate_ri_release.py
python src/render_ri_waveform_qc.py
python src/step11_nibp_corrected.py
python src/nibp_sampling_audit.py
```

RI is gap-aware, same-pulse and foot-referenced. The NIBP outcome uses the
fixed `[t0, t0+600 s)` window independent of the legacy ART-derived end.
`step11_nibp_corrected.py` creates the NIBP labels and some historical-RI
intermediate model outputs; those RI results are superseded by the next stage.

### 5. Final results and supplementary figures

```sh
python src/regenerate_ri_models.py
python src/generate_canonical_metrics.py
python src/arc_consequence_analysis.py
python src/generate_subgroup_figure.py
```

Authoritative corrected-RI ARC/NIBP models, complete-case sensitivity,
noisy-label reference, repeated splits and optimism correction come from
`outputs/ri_v14/ri_model_results_v14.json`, produced by
`regenerate_ri_models.py`. It also produces the corrected-RI ROC/ladder and RI
implementation-audit figures. Use `generate_canonical_metrics.py` for the
M1 threshold-sensitivity estimates and primary-audit cross-check; its older
RI model fields do not supersede the corrected-RI results.
`arc_consequence_analysis.py` provides the MAP-validity-indicator decomposition
and MAP-distribution figure; RI-related historical intermediate results in
that script do not supersede corrected-RI results. `generate_subgroup_figure.py`
provides the M1 (clinical plus MAP) exploratory subgroup figure.

## Preserved settings and validation limits

Cleanup did not change the retained executed analysis statements, thresholds,
random seeds, model formulas, feature-table filenames or clinical cohort
filters. All scripts passed syntax parsing; local import/file dependencies
were checked; hard-coded author/system paths were absent. These checks do not
constitute a full raw-data reproduction of the published numerical results.
## Citation

If you use this code in your research, please cite:

```bibtex
@article{gao2026taskwindow,
  title={Task-Window Validity of Pre-Induction Arterial Pressure for Hypotension Modelling in VitalDB},
  author={Gao, Ge},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

## License

MIT; see `LICENSE`.
