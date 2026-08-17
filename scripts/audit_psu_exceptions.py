#!/usr/bin/env python3
"""Audit the small PSU membership exceptions after registry-restriction is established.

The large canonical-vs-Aabila difference is handled separately by
``trace_psu_cohort.py``.  This script focuses on the edge cases:

* canonical registry-positive patients missing from Aabila
* Aabila patients absent from the canonical cohort

It also checks whether the canonical-only registry patients are associated with
registry augmentation / synthetic diagnoses, which is the part of the current
pipeline affected by the configurable +/- encounter-date flexibility.

Patient-level outputs are written only under the ignored local ``results/``
directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_CANONICAL = Path(
    "/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd/final/stroke_cohort_final.parquet"
)
DEFAULT_AABILA = Path(
    "/home/asadr/works/promis2/aabila/PROMIS-ML-PennState/outputs/final/stroke_cohort_final.parquet"
)
DEFAULT_FIRST = Path(
    "/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd/final/first_stroke_encounters.parquet"
)
DEFAULT_SYNTHETIC = Path(
    "/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd/intermediate/synthetic_diagnoses.parquet"
)
DEFAULT_MISSING = Path(
    "/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd/intermediate/missing_encounters.parquet"
)

KEY_CANDIDATES = ("PATID", "STUDY_ID", "PT_ID", "patient_id")
PREFERRED_COLUMNS = [
    "PATID", "STUDY_ID", "PT_ID", "patient_id",
    "ENCOUNTERID", "ENC_ID", "ADMIT_DATE", "DISCHARGE_DATE", "DX_DATE_stroke",
    "DX_DATE", "ENC_TYPE", "AGE_AT_STROKE", "age",
    "in_stroke_registry", "synthetic_registry_dx", "ehr_stroke_dx_present",
    "has_neuroimaging", "has_lipid_panel", "has_rehabilitation",
    "in_pcori", "local_area",
]


def choose_key(df: pd.DataFrame) -> str:
    for col in KEY_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"No patient identifier found among {KEY_CANDIDATES}")


def norm(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def add_id(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    key = choose_key(df)
    df["__id"] = norm(df[key])
    return df, key


def selected_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in PREFERRED_COLUMNS if c in df.columns]
    if "__id" not in cols:
        cols.insert(0, "__id")
    # Include any registry/synthetic/date fields not anticipated above.
    for c in df.columns:
        low = c.lower()
        if c not in cols and (
            "registry" in low
            or "synthetic" in low
            or "stroke_date" in low
            or "encounter_date" in low
        ):
            cols.append(c)
    return cols


def subset_for_ids(df: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    if "__id" not in df.columns:
        df, _ = add_id(df)
    out = df[df["__id"].isin(ids)].copy()
    cols = selected_columns(out)
    return out[cols].sort_values("__id")


def read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df, _ = add_id(df)
    return df


def count_flag(df: pd.DataFrame, ids: set[str], col: str, value=1) -> int | None:
    if col not in df.columns:
        return None
    s = df[df["__id"].isin(ids)][col]
    return int((s == value).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    p.add_argument("--aabila", type=Path, default=DEFAULT_AABILA)
    p.add_argument("--first-encounters", type=Path, default=DEFAULT_FIRST)
    p.add_argument("--synthetic-diagnoses", type=Path, default=DEFAULT_SYNTHETIC)
    p.add_argument("--missing-encounters", type=Path, default=DEFAULT_MISSING)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aabila_comparison/psu/exception_audit"),
    )
    args = p.parse_args()

    canonical, _ = add_id(pd.read_parquet(args.canonical))
    aabila, _ = add_id(pd.read_parquet(args.aabila))

    canonical_ids = set(canonical["__id"].dropna())
    aabila_ids = set(aabila["__id"].dropna())
    registry_ids = set(
        canonical.loc[canonical.get("in_stroke_registry", 0) == 1, "__id"].dropna()
    ) if "in_stroke_registry" in canonical.columns else set()

    canonical_registry_missing_aabila = registry_ids - aabila_ids
    aabila_outside_canonical = aabila_ids - canonical_ids
    aabila_outside_registry = aabila_ids - registry_ids

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    subset_for_ids(canonical, canonical_registry_missing_aabila).to_csv(
        out / "canonical_registry_missing_from_aabila.csv", index=False
    )
    subset_for_ids(aabila, aabila_outside_canonical).to_csv(
        out / "aabila_not_in_canonical.csv", index=False
    )

    first = read_optional(args.first_encounters)
    synthetic = read_optional(args.synthetic_diagnoses)
    missing = read_optional(args.missing_encounters)

    if first is not None:
        subset_for_ids(first, canonical_registry_missing_aabila).to_csv(
            out / "canonical_registry_missing_first_encounters.csv", index=False
        )
    if synthetic is not None:
        subset_for_ids(synthetic, canonical_registry_missing_aabila).to_csv(
            out / "canonical_registry_missing_synthetic_records.csv", index=False
        )
    if missing is not None:
        subset_for_ids(missing, canonical_registry_missing_aabila).to_csv(
            out / "canonical_registry_missing_missing_encounters.csv", index=False
        )

    lines = []
    lines.append("PSU EXCEPTION AUDIT")
    lines.append("=" * 72)
    lines.append(f"Canonical patients: {len(canonical_ids):,}")
    lines.append(f"Aabila patients: {len(aabila_ids):,}")
    lines.append(f"Canonical registry-positive: {len(registry_ids):,}")
    lines.append("")
    lines.append("EDGE GROUPS")
    lines.append(f"Canonical registry-positive missing from Aabila: {len(canonical_registry_missing_aabila):,}")
    lines.append(f"Aabila patients absent from canonical: {len(aabila_outside_canonical):,}")
    lines.append(f"Aabila patients outside canonical registry: {len(aabila_outside_registry):,}")
    lines.append("")

    lines.append("CURRENT-PIPELINE AUGMENTATION SIGNALS AMONG CANONICAL REGISTRY PATIENTS MISSING FROM AABILA")
    for col in ["synthetic_registry_dx", "ehr_stroke_dx_present", "in_pcori"]:
        n = count_flag(canonical, canonical_registry_missing_aabila, col, 1)
        if n is not None:
            lines.append(f"{col}=1: {n:,} / {len(canonical_registry_missing_aabila):,}")
    if "ehr_stroke_dx_present" in canonical.columns:
        n0 = count_flag(canonical, canonical_registry_missing_aabila, "ehr_stroke_dx_present", 0)
        lines.append(f"ehr_stroke_dx_present=0: {n0:,} / {len(canonical_registry_missing_aabila):,}")

    if synthetic is not None:
        synth_ids = set(synthetic["__id"].dropna())
        overlap = canonical_registry_missing_aabila & synth_ids
        lines.append(f"Present in synthetic_diagnoses.parquet: {len(overlap):,} / {len(canonical_registry_missing_aabila):,}")
    if missing is not None:
        miss_ids = set(missing["__id"].dropna())
        overlap = canonical_registry_missing_aabila & miss_ids
        lines.append(f"Present in missing_encounters.parquet: {len(overlap):,} / {len(canonical_registry_missing_aabila):,}")

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append(
        "The registry restriction explains the large cohort-size difference if nearly all Aabila patients "
        "are registry-positive. The +/-1-day encounter-date flexibility can only explain patients touched "
        "by registry augmentation/synthetic matching, so it should be evaluated within the small edge groups, "
        "not as an explanation for the thousands of canonical non-registry patients absent from Aabila."
    )

    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, end="")
    print(f"Wrote patient-level audit files to: {out.resolve()}")


if __name__ == "__main__":
    main()
