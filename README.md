# Task-Window Validity Audit — Minimal Code Release

This repository contains the Python source code used for the VitalDB
propofol-anchored arterial-pressure task-window validity audit. The primary
contribution is an empirical task-feasibility analysis, not a new preprocessing
algorithm and not a deployable prediction model.

## What is included

- The complete staged Python analysis workflow in `src/`;
- portable project-path and VitalDB loading utilities;
- this README and the MIT license;
- one complete dependency specification in `requirements.txt`.

This is a deliberately minimal **code-only release**. It does not distribute
raw VitalDB waveforms, cohort metadata, patient-level physiological data,
feature tables, outcome tables, generated metrics, figures, model weights,
logs, caches, archives, or other analysis outputs. The manuscript's numerical
results must therefore be regenerated locally after the user obtains the
public VitalDB source data.

## Data prerequisite and paths

VitalDB raw `.vital` files must be obtained independently from the public
VitalDB release. Do not place raw data under version control. Configure a
local data directory and project root before running scripts:

```sh
export VITALDB_VITAL_DIR=./vitaldb_data
export TOPIC10_PROJECT_ROOT=.
```

`VITALDB_VITAL_DIR` defaults to `vitaldb_data/` beneath the repository. The
scripts resolve project paths through `src/project_paths.py` and do not depend
on an author-specific absolute path. Raw data are read-only inputs; scripts
must not modify `.vital` files.

## Installation

Use a supported Python environment and install the complete dependency set:

```sh
python -m pip install -r requirements.txt
```

The requirements file includes dependencies for the primary audits, waveform
processing, corrected Reflection Index extraction, NIBP analyses, legacy
feature/model analyses, and optional local figure generation.

## Staged workflow

The scripts are staged because later analyses consume intermediate files
created by earlier stages. Generated files are local working products and are
ignored by `.gitignore`.

### Stage 1 — primary task-window audit

```sh
python src/fixed_window_validity_audit.py --workers 4
```

This performs the propofol-anchored fixed-window numeric/raw audit and writes
local case-level and summary outputs. The fixed windows are operationally
anchored to the first positive propofol-rate observation.

### Stage 2 — coverage and source sensitivities

```sh
python src/native_coverage_audit.py --workers 4
python src/broader_arterial_cohort_audit.py --workers 4
python src/fem_waveform_audit.py --workers 4
python src/fem_mbp_source_sensitivity.py
python src/cadence_window_sensitivity_audit.py
```

These scripts perform native-record validation, pointwise-only comparison,
PPG-free arterial-source sensitivity, ART/FEM waveform sensitivity, FEM_MBP
numeric-source completeness sensitivity, and cadence/window sensitivity.

### Stage 3 — corrected Reflection Index analysis

```sh
python src/ri_reextract.py
python src/validate_ri_release.py
python src/render_ri_waveform_qc.py
```

The corrected RI workflow is gap-aware, same-pulse, and foot-referenced.
`validate_ri_release.py` is an analysis integrity script in `src/`; it is not
a release-package validator.

### Stage 4 — downstream models and publication metrics

The legacy feature and outcome tables must first be regenerated locally with
the numbered `step*` scripts. They are patient-level intermediate products
and are intentionally not included here. After those prerequisites exist,
run the relevant downstream analyses:

```sh
python src/regenerate_ri_models.py
python src/generate_canonical_metrics.py
python src/generate_nibp_metrics.py
python src/nibp_sampling_audit.py
```

Additional pipeline, sensitivity, stability, consequence, and optional figure
scripts are documented by their module docstrings and source code:

- `step1_cohort_selection.py` through `step8b_tables_shap.py`: cohort,
  induction, outcome, vascular-feature, modelling, sensitivity, and SHAP
  stages;
- `step11_nibp_corrected.py` and `step12_repeated_split_stability.py`:
  NIBP and stability analyses;
- `arc_consequence_analysis.py`, `internal_validation_analysis.py`, and
  `step13_fig3_arc_nested_roc.py`: consequence and validation analyses;
- `generate_subgroup_figure.py` and
  `regenerate_nibp_reference_figure.py`: optional local figure generation.

## Reproducibility boundary

The primary task-window audit can be regenerated from public VitalDB source
data using the included scripts. Downstream consequence and model analyses
also require locally generated feature and outcome tables from the staged
pipeline. These intermediate tables contain patient-level values and are not
distributed in this repository.

The repository intentionally contains no frozen result files. Any output
reported from a local rerun should be treated as locally generated and checked
against the user's VitalDB version, software environment, and analysis
configuration.

## Source layout

```text
.
├── src/                 # Python analysis source and shared utilities
├── README.md            # This document
├── LICENSE              # MIT license
├── requirements.txt     # Complete portable dependency set
└── .gitignore           # Excludes raw data and generated products
```

## License

Released under the MIT License; see `LICENSE`.
