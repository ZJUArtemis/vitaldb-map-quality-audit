# Cadence-Aware Validation of Pre-Induction Arterial-Pressure Data in VitalDB

Reproducibility code for the IEEE Access manuscript:

> **Cadence-Aware Validation of Pre-Induction Arterial-Pressure Data for
> Hypotension Modeling in VitalDB**
> Ge Gao, Department of Anesthesiology, The First Affiliated Hospital,
> Zhejiang University School of Medicine, Hangzhou, China.

## Scope and principal result

This is a task-specific measurement-validity audit, not a claim that routine
range filtering is novel. The audit asks whether the `Solar8000/ART_MBP` stream
is sufficiently observed, physiologically plausible, and temporally continuous
across a specified 300-s pre-induction predictor window and up-to-600-s outcome
window.

The source set contains 910 induction segments. The original ART feature-table
pipeline contains 909 because one segment did not enter that merge. The
revision-wide validity and raw-waveform audits use all 910 source segments.
Under the primary operational definition, 9/910 segments met the joint numeric
coverage, in-range, and continuity criteria. The complete operational
sensitivity grid did not yield a viable modeling subset. A separate audit of
all 910 raw files corroborated that raw-track presence does not establish a
plausible pulsatile pre-induction waveform.

The repository does **not** infer arterial-catheter placement, connection,
leveling, or zeroing times. VitalDB does not provide a validated timestamp for
those clinical events. Monitor-track availability and validity are treated only
as operational data-state measures. The manuscript also does not claim that
MIMIC or another database contains the same empirical pattern.

## Reproducibility release

The submission-frozen release is tagged `ieee-access-resubmission-v4`.
The v4 files are:

```text
src/revision_v4_full_validity_audit.py
results/revision_v4/revision_v4_full_validity_summary.json
results/revision_v4/revision_v4_threshold_grid.csv
results/revision_v4/figures/
```

Patient-level derived tables are intentionally not redistributed. They are
regenerated from the public VitalDB source files. Aggregate counts, threshold
grids, and publication figures are provided for verification.

## Repository layout

```text
src/
  step1_cohort_selection.py           Original cohort construction
  step2_induction_segmentation.py     Original induction segmentation
  step3_outcome_labeling.py           Original artifact-prone ART labels
  step5_vascular_features.py          PPG-derived vascular features
  step9_nested_models.py              ARC consequence demonstration
  step11_nibp_corrected.py            Independent NIBP reference analysis
  revision_v4_full_validity_audit.py  Full 910-case numeric/raw ART audit
results/
  metrics/                             Earlier aggregate model outputs
  figures/                             Earlier publication figures
  revision_v4/                         Submission-frozen v4 aggregate outputs
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

## Running the v4 audit

Download the VitalDB `.vital` files and point `VITALDB_VITAL_DIR` to the folder
that contains files such as `0001.vital`. The cohort and induction-segment table
must first be reconstructed by the original pipeline.

```bash
export VITALDB_VITAL_DIR=/absolute/path/to/vital_files
python src/step1_cohort_selection.py
python src/step2_induction_segmentation.py
python src/revision_v4_full_validity_audit.py --workers 6
```

The v4 script:

- estimates numeric cadence from native `ART_MBP` record timestamps, not from
  finite values after 1-s materialization;
- measures coverage against the complete intended windows;
- evaluates 192 numeric validity definitions;
- audits raw `SNUADC/ART` in all source files; and
- repeats the pulsatility assessment at 3-, 5-, and 10-mmHg amplitude
  thresholds.

## Interpretation boundaries

- The 20–200-mmHg range, coverage cutoffs, continuity durations, and waveform
  amplitude cutoffs are operational audit definitions, not universal clinical
  standards.
- The criterion of at least 10 events and 10 non-events is only a pragmatic
  stop rule preventing model fitting in an obviously non-viable subset; it is
  not a sample-size sufficiency rule.
- The artefact-reduced cohort models quantify consequences of insufficient
  measurement validation; their high AUROC is not clinically valid performance.
- The NIBP arm is an independently sourced, intermittent reference outcome, not
  ground truth for a continuous invasive-arterial nadir.

## License

MIT — see [LICENSE](LICENSE).
