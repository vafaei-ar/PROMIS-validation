#!/usr/bin/env python3
"""Trace why Aabila's PSU cohort differs from the canonical PSU cohort.

This focuses on cohort membership, especially registry/provenance flags, before
investigating downstream variable-level differences. Patient IDs are written
only to the ignored local results directory.
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
FLAGS = [
    "in_stroke_registry",
    "synthetic_registry_dx",
    "ehr_stroke_dx_present",
    "has_neuroimaging",
    "has_lipid_panel",
    "has_rehabilitation",
    "in_pcori",
    "local_area",
]


def norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def key(df: pd.DataFrame) -> str:
    for c in ("PATID", "STUDY_ID", "PT_ID", "patient_id"):
        if c in df.columns:
            return c
    raise KeyError("No patient identifier found")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    p.add_argument("--aabila", type=Path, default=DEFAULT_AABILA)
    p.add_argument("--output-dir", type=Path, default=Path("results/aabila_comparison/psu/cohort_trace"))
    args = p.parse_args()

    canonical = pd.read_parquet(args.canonical)
    aabila = pd.read_parquet(args.aabila)
    ck, ak = key(canonical), key(aabila)
    canonical = canonical.copy()
    aabila = aabila.copy()
    canonical["__id"] = norm(canonical[ck])
    aabila["__id"] = norm(aabila[ak])

    a_ids = set(aabila["__id"].dropna())
    c_ids = set(canonical["__id"].dropna())
    shared = a_ids & c_ids
    c_only = c_ids - a_ids
    a_only = a_ids - c_ids

    canonical["membership"] = canonical["__id"].map(
        lambda x: "shared_with_aabila" if x in shared else "canonical_only"
    )

    rows = []
    for flag in FLAGS:
        if flag not in canonical.columns:
            continue
        tab = (
            canonical.groupby(["membership", flag], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        tab.insert(0, "flag", flag)
        tab = tab.rename(columns={flag: "value"})
        rows.append(tab)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out / "flag_by_membership.csv", index=False)
    pd.DataFrame({"patient_id": sorted(c_only)}).to_csv(out / "canonical_only_patients.csv", index=False)
    pd.DataFrame({"patient_id": sorted(a_only)}).to_csv(out / "aabila_only_patients.csv", index=False)

    print(f"Canonical patients: {len(c_ids):,}")
    print(f"Aabila patients:    {len(a_ids):,}")
    print(f"Shared:             {len(shared):,}")
    print(f"Canonical only:     {len(c_only):,}")
    print(f"Aabila only:        {len(a_only):,}")
    print()

    if "in_stroke_registry" in canonical.columns:
        registry = canonical[canonical["in_stroke_registry"] == 1]
        registry_ids = set(registry["__id"].dropna())
        print(f"Canonical registry-positive: {len(registry_ids):,}")
        print(f"Aabila in canonical registry: {len(a_ids & registry_ids):,} / {len(a_ids):,}")
        print(f"Canonical registry missing from Aabila: {len(registry_ids - a_ids):,}")
        print(f"Aabila outside canonical registry: {len(a_ids - registry_ids):,}")
        print()

    print("Flag distributions by membership:")
    for flag in FLAGS:
        if flag not in canonical.columns:
            continue
        print(f"\n[{flag}]")
        print(pd.crosstab(canonical["membership"], canonical[flag], dropna=False).to_string())

    print(f"\nWrote trace files to: {out.resolve()}")


if __name__ == "__main__":
    main()
