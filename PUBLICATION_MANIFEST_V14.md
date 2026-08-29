# Publication provenance manifest — v14

This is a code-only release: no result files are distributed. The table below
maps each publication item to its authoritative script, the inputs it needs,
and the output files it generates **locally** (under git-ignored `outputs/`
and `results/` directories). Generated output filenames retain historical
generation suffixes (`revision_v5_...`, `..._v9`, `..._v14`) so locally
regenerated outputs remain byte-comparable with the archived submission
record.

| Publication item | Authoritative script | Principal input | Locally generated output |
|---|---|---|---|
| Main task-validity counts and complete 192-definition grid | `fixed_window_validity_audit.py` | eligible IDs + raw `.vital` files | `revision_v5_fixed_window_summary.json`, `revision_v5_threshold_grid.csv`, `revision_v5_fixed_window_case_results.csv` |
| Native coverage and outcome-continuity verification | `native_coverage_audit.py` | fixed-window case results + native ART_MBP records | `native_numeric_coverage_summary_v9.json`, `outcome_continuity_sensitivity_v9.csv` |
| PPG-free arterial-source sensitivity | `broader_arterial_cohort_audit.py` | track availability + fixed-window case results + raw `.vital` files | `broader_arterial_cohort_summary_v9.json` |
| Raw ART/FEM sensitivity | `fixed_window_validity_audit.py`, `fem_waveform_audit.py` | fixed-window case results + raw ART/FEM waveforms | `fem_waveform_v7_summary.json`, `fem_waveform_v7_case_results.csv` |
| Window-length sensitivity | `cadence_window_sensitivity_audit.py` | fixed-window case results + raw `.vital` files | `window_length_sensitivity_v8.csv`, `cadence_window_v8_summary.json` |
| Corrected RI extraction and integrity audit | `ri_reextract.py`, `validate_ri_release.py` | raw PPG waveforms | `ri_reextraction_summary_v14.json`, `ri_case_features_v14.parquet`, `ri_beat_fiducials_v14.parquet` |
| RI waveform visual QC | `render_ri_waveform_qc.py` | locally generated RI tables | `ri_waveform_qc_summary_v14.json` + local PDF/PNG pages |
| Corrected-RI ARC, complete-case, noisy-label, NIBP, repeated-split, and ROC results | `regenerate_ri_models.py` | locally regenerated corrected-RI feature tables + legacy feature/outcome tables | `ri_model_results_v14.json` |
| Noisy-label reference and ARC threshold sensitivity | `generate_canonical_metrics.py` | locally regenerated vascular feature table + fixed-window case results | `canonical_publication_metrics_v9.json`, `canonical_noisy_label_models_v9.csv`, `canonical_arc_threshold_m1_v9.csv` |
| NIBP held-out estimates and paired delta-AUROC CIs | `regenerate_ri_models.py` | NIBP labels + corrected-RI feature table | `ri_model_results_v14.json` |
| NIBP source derivation | `step11_nibp_corrected.py` | raw `.vital` files + legacy segments | NIBP source counts consolidated by `generate_nibp_metrics.py` into `nibp_model_metrics_v10.json` |
| NIBP value-state sampling density | `nibp_sampling_audit.py` | raw NIBP records + legacy segments | `nibp_value_state_summary_v14.json` |
| Exploratory subgroup forest | `generate_subgroup_figure.py` | locally regenerated vascular feature table | local figure with canonical seed-0 CI convention |

Legacy feature/outcome tables are regenerated with the `step*` pipeline
(`step1_cohort_selection.py` → `step8b_tables_shap.py`); they contain
patient-level values and are never distributed. `CHECKSUMS_SHA256.txt` covers
the distributed files only, uses paths relative to the repository root, and
excludes the checksum file itself.
