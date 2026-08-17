# PROMIS-validation

Validation workflow for comparing canonical PROMIS pipeline outputs against Aabila's PSU and Geisinger outputs.

Canonical pipeline commit used for this validation:

`7cd577c69d4df839b426478e31c7a4caf5105cd8`

## Scope

Cross-pipeline equivalence intentionally excludes imputed datasets because the advanced imputation procedure is stochastic. The comparison focuses on:

- `stroke_cohort_final.parquet`
- `stroke_cohort_unimputed.parquet`
- `outcomes_flags.parquet`
- `cohort_with_outcomes.parquet`

For each artifact, the validation checks patient-set overlap, schema and dtypes, missingness, cell-level agreement on shared variables, and discordant patient IDs.

## Run

From this repository on the PROMIS workstation:

```bash
python scripts/compare_aabila.py
```

Default local roots are:

- Canonical PSU: `/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd`
- Canonical Geisinger: `/home/asadr/works/promis2/PROMIS-validation/reproduced/geisinger_997d6bd`
- Aabila PSU: `/home/asadr/works/promis2/aabila/PROMIS-ML-PennState/outputs`
- Aabila Geisinger: `/home/asadr/works/promis2/aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs`

All roots can be overridden with command-line arguments. Run `python scripts/compare_aabila.py --help` for options.

## Outputs

Results are written by default to `results/aabila_comparison/`, including:

- `summary.md` and `summary.csv`
- per-site/per-artifact `summary.json`
- `schema.csv`
- `column_comparison.csv`
- `patients_only_canonical.csv`
- `patients_only_aabila.csv`
- `discordant_patients.csv`
- `provenance.json`

Patient identifiers are retained only in local discrepancy files. Generated results and reproduced datasets should not be committed.
