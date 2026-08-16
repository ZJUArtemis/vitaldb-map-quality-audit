# Task-Window Validity of Pre-Induction Arterial Pressure in VitalDB

Reproducibility code for the IEEE Access manuscript:

> **Task-Window Validity of Pre-Induction Arterial Pressure for Hypotension
> Modeling in VitalDB**
> Ge Gao, Department of Anesthesiology, The First Affiliated Hospital,
> Zhejiang University School of Medicine, Hangzhou, China.

## Scope and principal result

This is a task-specific measurement-validity audit, not a claim that routine
range filtering is novel. Pointwise value plausibility is separated from
window-level measurement support.

The executable cohort criteria identified 926 track-eligible cases. Induction
onset was reconstructed directly from the first positive
`Orchestra/PPF20_RATE` observation in each local `.vital` file. The primary
audit then used fixed `[-300,0)`-s and `[0,+600)`-s windows and did not read the
ART-derived termination rule in the legacy segmentation.

Under the primary operational definition, 8/926 cases met the joint numeric
coverage, in-range, and continuity criteria (4 events and 4 non-events). The
50%–90% coverage/in-range grid retained 7–14 cases. In the independent raw ART
audit, only 11/926 cases had plausible pulsatile support for at least 80% of the
pre-induction window; median support was 0%.

The repository does **not** infer arterial-catheter placement, connection,
leveling, or zeroing times. VitalDB does not provide validated timestamps for
those clinical events. It also does not claim that MIMIC or another database
contains the same empirical pattern.

## Reproducibility release

The submission-frozen document-consistency release is tagged
`ieee-access-resubmission-v6`. The underlying fixed-window analysis remains
identified as v5 because v6 changes reporting consistency and terminology, not
the locked analysis or aggregate results.

```text
src/revision_v5_fixed_window_validity_audit.py
results/revision_v5/revision_v5_fixed_window_summary.json
results/revision_v5/revision_v5_threshold_grid.csv
results/revision_v5/revision_v5_source_independence.json
results/revision_v5/figures/
```

Patient-level derived tables and the case-level propofol-onset table are not
redistributed. They are regenerated from the public VitalDB source files.

## Repository layout

```text
src/
  step1_cohort_selection.py              Cohort construction
  step2_induction_segmentation.py        Legacy ART-derived segmentation
  step3_outcome_labeling.py              Legacy artifact-prone ART labels
  step5_vascular_features.py             PPG-derived vascular features
  step9_nested_models.py                 ARC consequence demonstration
  step11_nibp_corrected.py               Independent NIBP reference analysis
  revision_v5_fixed_window_validity_audit.py
                                          Primary 926-case fixed-window audit
results/
  metrics/                                Aggregate legacy model outputs
  figures/                                Earlier publication figures
  revision_v5/                            Frozen fixed-window aggregate outputs
```

## Data and ethics metadata

VitalDB is publicly available at <https://vitaldb.net> and is described by Lee
*et al.*, *Scientific Data* 2022, doi:10.1038/s41597-022-01411-5. The dataset
was collected under Seoul National University Hospital IRB H-1408-101-605 with
waiver of informed consent and was registered as NCT02914444. No additional
ethics approval was required for this secondary analysis of anonymized data.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline was tested with Python 3.10. No GPU is required.

## Running the fixed-window audit

Download the VitalDB `.vital` files and point `VITALDB_VITAL_DIR` to the folder
containing files such as `0001.vital`. Reconstruct only the cohort list before
running the primary audit; the legacy segmentation is not required.

```bash
export VITALDB_VITAL_DIR=/absolute/path/to/vital_files
python src/step1_cohort_selection.py
python src/revision_v5_fixed_window_validity_audit.py --workers 6
```

The fixed-window script:

- independently reconstructs propofol-defined induction onset;
- measures numeric coverage against complete fixed windows;
- estimates cadence from native `ART_MBP` record timestamps;
- evaluates 192 numeric validity definitions;
- audits raw `SNUADC/ART` in all 926 primary source files; and
- repeats the pulsatility assessment at 3-, 5-, and 10-mmHg amplitude
  thresholds.

## Interpretation boundaries

- Operational thresholds are not universal clinical standards.
- The 10-event/10-non-event criterion is a pragmatic model-fitting stop rule,
  not a sample-size sufficiency rule.
- The 528-case ARC models quantify consequences of insufficient measurement
  validation; their high AUROC is not clinically valid performance.
- The 910-case NIBP arm is an independently sourced, intermittent reference,
  not ground truth for a continuous invasive-arterial nadir.

## License

MIT — see [LICENSE](LICENSE).
