# Validation experiment log

This page summarizes the major experiments in the order they were performed. It is meant to help a future analyst understand **why** each script exists and which results are still considered valid.

## Phase 1 — baseline cross-pipeline comparison

### Question

How different are the canonical PSU/Geisinger outputs from Aabila's saved outputs?

### Main scripts

- `compare_aabila.py`
- `compare_geisinger.py`
- `compare_psu_registry_only.py`
- `compare_psu_registry_outcomes.py`

### Findings

- PSU legacy output was much smaller than the canonical broad cohort and was effectively registry centered.
- Among shared PSU registry patients, clinical outcomes matched after proper deduplication.
- Geisinger legacy output was also smaller than the broad canonical cohort.
- Diagnosis-code disagreement was much larger than expected at both sites.

### Decision

Do not immediately change the canonical cohort. Investigate whether differences come from phenotype logic, source data, or row-order/tie behavior.

---

## Phase 2 — raw discrepancy adjudication

### Question

When canonical and legacy `DX` differ, which code is actually present in the source data?

### Main scripts

- `source_truth_audit.py`
- `raw_discrepancy_adjudication.py`
- `audit_psu_exceptions.py`
- `final_targeted_adjudication.py`

### PSU finding

For almost all reviewed PSU diagnosis discordances, both the canonical and Aabila diagnosis codes were present in the raw diagnosis data on the relevant encounter. One exceptional case involved a synthetic registry diagnosis versus a retinal-artery diagnosis.

### Geisinger finding

Many diagnosis discordances also contained multiple qualifying primary stroke codes on the same encounter.

### Decision

Most DX differences are **selection/tie-break provenance differences**, not proof that the canonical selected code is clinically wrong.

Do not overwrite the broad `DX` just to reproduce legacy row ordering. Add deterministic and ambiguity companion fields instead.

---

## Phase 3 — understand the Geisinger legacy phenotype

### Question

Why does the Aabila Geisinger cohort have 11,910 patients instead of the broader canonical count?

### Main scripts

- `audit_aabila_geisinger_code.py`
- `audit_geisinger_cohort.py`
- `inspect_aabila_strict_semantics.py`
- `trace_aabila_candidate_selection.py`
- `compare_geisinger_aabila_aligned.py`
- `compare_geisinger_aligned_full.py`

### Early hypotheses that were rejected

- A ±7-day evidence window was initially suspected. This was incorrect and must not be reported as the validated legacy rule.
- The ATN registry file was suspected to gate the Geisinger cohort. Direct identifier-overlap testing showed that the ATN file uses a different identifier namespace and does not overlap the transformed patient IDs used in the compared data.

### Key discovery

The saved legacy `all_stroke_encounters.parquet` was produced before all evidence filters, so using it alone to infer final selection logic was misleading.

The actual legacy selection applies evidence filters before final first-patient collapse.

---

## Phase 4 — exact Geisinger replay

### Question

Can the old Geisinger selection be reproduced exactly from raw data and old code semantics?

### Scripts

- `replay_aabila_exact_selection_v3.py`
- `replay_aabila_exact_selection_v4.py`
- `replay_aabila_exact_selection_v5.py`

### Why there are multiple versions

Each version tested a more exact understanding of the legacy logic. The final validated version is **v5**.

### Final validated logic

1. identify qualifying stroke encounters;
2. apply minimum encounter duration;
3. require qualifying CPT4 neuroimaging from 2 days before admission through discharge;
4. require qualifying lipid LOINC from admission through discharge;
5. use the validated `lipid_corrected_by_harold.csv` whitelist with **214 codes**;
6. merge qualifying stroke diagnosis rows;
7. derive the stroke diagnosis date;
8. sort by that date;
9. use the legacy `groupby(PATID).first()` behavior.

### Final replay result

- Replay patients: **11,910**
- Saved Aabila patients: **11,910**
- Encounter-ID mismatches: **0**
- Replay-only patients: **0**
- Saved-only patients: **0**

This established that the legacy Geisinger cohort is an evidence-qualified **strict phenotype**, not the same definition as the broad canonical cohort.

---

## Phase 5 — prototype sensitivity fields

### Question

How should the strict legacy behavior be represented without replacing the broad canonical cohort?

### Scripts

- `prototype_sensitivity_flags.py`
- `prototype_sensitivity_flags_v2.py`
- `prototype_sensitivity_flags_v3.py`
- `prototype_sensitivity_flags_v4.py`
- `tests/test_sensitivity_flag_semantics.py`

### Evolution

Early prototypes tested different ways of defining strict eligibility and alternative encounters. Later versions clarified the correct **filter-before-index** semantics.

### Final concepts moved to production

- `phenotype_strict_eligible`
- `has_alt_qualifying_stroke_encounter`
- `index_selection_sensitive`
- `strict_index_encounter_id`
- `strict_index_stroke_date`
- `strict_index_dx`
- diagnosis ambiguity/tie-break companion fields

### Important distinction

`has_alt_qualifying_stroke_encounter` and `index_selection_sensitive` are not the same thing.

A patient can have another strict-qualified encounter even when the selected strict first encounter is still the same as the broad index.

---

## Phase 6 — production harmonization regressions found during validation

### PSU registry gating

A harmonized run unexpectedly returned only **4,120** PSU patients.

Cause: the historical source YAML still had registry restriction enabled.

Fix: harmonized runtime configuration now disables registry gating while preserving historical behavior for explicit non-harmonized reproduction.

### PSU imaging semantic type

A later PSU run returned zero strict-eligible patients.

Cause: the normalized `PX_TYPE` field contained PCORnet categories, not the actual CPT/HCPCS semantic type.

Fix: harmonized PSU logic uses `RAW_PX_TYPE` for imaging/rehabilitation evidence.

### Step 6 metadata/lab issues

Several implementation fixes were made so that:

- immutable phenotype/sensitivity metadata are protected during data preparation;
- patient-level sensitivity metadata are propagated only to patient-level products;
- lab detection does not accidentally treat administrative/provenance fields as analytic labs.

---

## Phase 7 — final validated sensitivity results

### PSU

- Broad: **9,835**
- Broad-index strict evidence: **5,848**
- Final filter-before-index strict: **6,009**
- Strict only through another encounter: **161**
- `has_alt_qualifying_stroke_encounter`: **491**
- `index_selection_sensitive`: **162**
- `multiple_primary_stroke_dx`: **3,417**
- deterministic DX differs from current: **31**

### Geisinger

- Broad: **13,872**
- Strict: **11,910**
- `has_alt_qualifying_stroke_encounter`: **1,144**
- `index_selection_sensitive`: **168**
- `multiple_primary_stroke_dx`: **1,707**
- deterministic DX differs from current: **1,035**

---

## Phase 8 — Geisinger ICU outcome investigation

This investigation happened after the main phenotype validation but affects the current production outcome release.

### Initial problem

The current Geisinger broad outcome run produced `icu_admission = 0` for all 13,872 patients.

### Diagnostic findings

- ICU departments clearly exist in the raw encounter data.
- 1,515 broad-cohort patients had at least one ICU-like department record somewhere in their record.
- Relevant ICU rows had `ENC_ADM_DTTM` and `ENC_DIS_DTTM` missing.
- `ENC_DT` was populated for all 3,063 clean inpatient ICU rows inspected.
- Naive substring matching caused false positives because `CCU` can match text inside `OCCUP`.

### Corrected rule

For Geisinger, ICU exposure during the index hospitalization requires:

1. a clean ICU/critical-care department name using word-boundary matching;
2. `DEP_GROUPING == INPATIENT`;
3. `ENC_DT` within the broad index admission/discharge interval.

### Result

- `icu_admission`: **431 / 13,872 (3.11%)**

This is the validated current broad-cohort benchmark.

---

## Final interpretation

The validation work supports a two-level analysis strategy:

- **Broad canonical cohort for primary analyses**
- **Strict evidence-qualified phenotype for sensitivity/replication analyses**

The validation did not produce evidence that the broad canonical cohort should be replaced wholesale by either legacy pipeline.
