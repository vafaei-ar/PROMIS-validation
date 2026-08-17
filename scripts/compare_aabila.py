#!/usr/bin/env python3
"""Compare canonical PROMIS outputs against Aabila's outputs.

Validation intentionally excludes imputed datasets.  It compares four layers:
  * stroke_cohort_final
  * stroke_cohort_unimputed
  * outcomes_flags
  * cohort_with_outcomes

The script is designed for the current PROMIS workspace layout, while allowing
all roots and patient-key columns to be overridden from the command line.
Patient identifiers are written only to local result files; do not commit the
results directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_PIPELINE_COMMIT = "7cd577c69d4df839b426478e31c7a4caf5105cd8"

DEFAULT_MINE_ROOTS = {
    "psu": Path("/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd"),
    "geisinger": Path("/home/asadr/works/promis2/PROMIS-validation/reproduced/geisinger_997d6bd"),
}

DEFAULT_AABILA_ROOTS = {
    "psu": Path("/home/asadr/works/promis2/aabila/PROMIS-ML-PennState/outputs"),
    "geisinger": Path(
        "/home/asadr/works/promis2/aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs"
    ),
}

ARTIFACTS = {
    "final": "stroke_cohort_final.parquet",
    "unimputed": "stroke_cohort_unimputed.parquet",
    "outcomes": "outcomes_flags.parquet",
    "cohort_with_outcomes": "cohort_with_outcomes.parquet",
}

KEY_CANDIDATES = ("PATID", "STUDY_ID", "PT_ID", "patient_id", "PatientID")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", choices=["psu", "geisinger", "all"], default="all")
    p.add_argument("--mine-psu", type=Path, default=DEFAULT_MINE_ROOTS["psu"])
    p.add_argument("--mine-geisinger", type=Path, default=DEFAULT_MINE_ROOTS["geisinger"])
    p.add_argument("--aabila-psu", type=Path, default=DEFAULT_AABILA_ROOTS["psu"])
    p.add_argument("--aabila-geisinger", type=Path, default=DEFAULT_AABILA_ROOTS["geisinger"])
    p.add_argument("--mine-key", default=None, help="Override identifier column in canonical files")
    p.add_argument("--aabila-key", default=None, help="Override identifier column in Aabila files")
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aabila_comparison"),
        help="Local output directory. Patient-level discrepancy files are written here.",
    )
    return p.parse_args()


def first_existing(root: Path, filename: str) -> Path:
    candidates = [
        root / filename,
        root / "final" / filename,
        root / "intermediate" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {filename} under {root}. Searched:\n  - {searched}")


def choose_key(df: pd.DataFrame, override: str | None, label: str) -> str:
    if override:
        if override not in df.columns:
            raise KeyError(f"{label}: requested key {override!r} is not present")
        return override
    for col in KEY_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"{label}: no patient identifier found among {KEY_CANDIDATES}")


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def is_datetime_like(left: pd.Series, right: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(left) and pd.api.types.is_datetime64_any_dtype(right)


def differing_mask(left: pd.Series, right: pd.Series, atol: float, rtol: float) -> pd.Series:
    left_missing = left.isna()
    right_missing = right.isna()
    one_missing = left_missing ^ right_missing
    both_present = ~(left_missing | right_missing)
    diff = one_missing.copy()

    if not both_present.any():
        return diff

    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        lv = pd.to_numeric(left.loc[both_present], errors="coerce").astype(float)
        rv = pd.to_numeric(right.loc[both_present], errors="coerce").astype(float)
        neq = ~np.isclose(lv.to_numpy(), rv.to_numpy(), atol=atol, rtol=rtol, equal_nan=True)
        diff.loc[both_present] = neq
    elif is_datetime_like(left, right):
        diff.loc[both_present] = (
            left.loc[both_present].astype("datetime64[ns]")
            != right.loc[both_present].astype("datetime64[ns]")
        ).to_numpy()
    else:
        diff.loc[both_present] = (
            normalized_text(left.loc[both_present]).to_numpy()
            != normalized_text(right.loc[both_present]).to_numpy()
        )
    return diff.fillna(False)


def safe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compare_artifact(
    site: str,
    artifact: str,
    mine_path: Path,
    aabila_path: Path,
    output_dir: Path,
    mine_key_override: str | None,
    aabila_key_override: str | None,
    atol: float,
    rtol: float,
) -> dict:
    mine = pd.read_parquet(mine_path)
    other = pd.read_parquet(aabila_path)

    mine_key = choose_key(mine, mine_key_override, "canonical")
    other_key = choose_key(other, aabila_key_override, "Aabila")

    mine = mine.copy()
    other = other.copy()
    mine["__patient_id"] = normalize_ids(mine[mine_key])
    other["__patient_id"] = normalize_ids(other[other_key])

    duplicate_mine = int(mine["__patient_id"].duplicated().sum())
    duplicate_other = int(other["__patient_id"].duplicated().sum())
    if duplicate_mine or duplicate_other:
        raise ValueError(
            f"{site}/{artifact}: duplicate patient IDs prevent one-to-one comparison "
            f"(canonical={duplicate_mine}, Aabila={duplicate_other})"
        )

    mine_ids = set(mine["__patient_id"].dropna())
    other_ids = set(other["__patient_id"].dropna())
    shared_ids = mine_ids & other_ids
    only_mine = sorted(mine_ids - other_ids)
    only_other = sorted(other_ids - mine_ids)

    artifact_dir = output_dir / site / artifact
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_csv(pd.DataFrame({"patient_id": only_mine}), artifact_dir / "patients_only_canonical.csv")
    safe_csv(pd.DataFrame({"patient_id": only_other}), artifact_dir / "patients_only_aabila.csv")

    ignored_keys = {mine_key, other_key, "__patient_id"}
    mine_cols = set(mine.columns) - ignored_keys
    other_cols = set(other.columns) - ignored_keys
    shared_cols = sorted(mine_cols & other_cols)
    only_mine_cols = sorted(mine_cols - other_cols)
    only_other_cols = sorted(other_cols - mine_cols)

    schema_rows = []
    all_cols = sorted(mine_cols | other_cols)
    for col in all_cols:
        schema_rows.append(
            {
                "column": col,
                "in_canonical": col in mine_cols,
                "in_aabila": col in other_cols,
                "canonical_dtype": str(mine[col].dtype) if col in mine.columns else "",
                "aabila_dtype": str(other[col].dtype) if col in other.columns else "",
            }
        )
    safe_csv(pd.DataFrame(schema_rows), artifact_dir / "schema.csv")

    mine_shared = mine[mine["__patient_id"].isin(shared_ids)].set_index("__patient_id").sort_index()
    other_shared = other[other["__patient_id"].isin(shared_ids)].set_index("__patient_id").sort_index()

    column_rows = []
    discordant_sets: list[set[str]] = []
    for col in shared_cols:
        left = mine_shared[col]
        right = other_shared[col]
        mask = differing_mask(left, right, atol=atol, rtol=rtol)
        ids = set(mask.index[mask])
        if ids:
            discordant_sets.append(ids)

        left_missing = int(left.isna().sum())
        right_missing = int(right.isna().sum())
        column_rows.append(
            {
                "column": col,
                "canonical_dtype": str(left.dtype),
                "aabila_dtype": str(right.dtype),
                "canonical_missing": left_missing,
                "aabila_missing": right_missing,
                "missing_difference_aabila_minus_canonical": right_missing - left_missing,
                "different_cells": int(mask.sum()),
                "pct_shared_patients_different": (100.0 * float(mask.sum()) / len(shared_ids)) if shared_ids else np.nan,
            }
        )

    column_df = pd.DataFrame(column_rows)
    if not column_df.empty:
        column_df = column_df.sort_values(
            ["different_cells", "column"], ascending=[False, True]
        )
    safe_csv(column_df, artifact_dir / "column_comparison.csv")

    discordant_ids = sorted(set().union(*discordant_sets)) if discordant_sets else []
    safe_csv(pd.DataFrame({"patient_id": discordant_ids}), artifact_dir / "discordant_patients.csv")

    n_diff_cols = int((column_df["different_cells"] > 0).sum()) if not column_df.empty else 0
    total_diff_cells = int(column_df["different_cells"].sum()) if not column_df.empty else 0

    summary = {
        "site": site,
        "artifact": artifact,
        "canonical_path": str(mine_path),
        "aabila_path": str(aabila_path),
        "canonical_shape": list(mine.drop(columns=["__patient_id"]).shape),
        "aabila_shape": list(other.drop(columns=["__patient_id"]).shape),
        "canonical_key": mine_key,
        "aabila_key": other_key,
        "canonical_unique_patients": len(mine_ids),
        "aabila_unique_patients": len(other_ids),
        "shared_patients": len(shared_ids),
        "patients_only_canonical": len(only_mine),
        "patients_only_aabila": len(only_other),
        "shared_columns": len(shared_cols),
        "columns_only_canonical": only_mine_cols,
        "columns_only_aabila": only_other_cols,
        "columns_with_value_differences": n_diff_cols,
        "total_different_cells": total_diff_cells,
        "discordant_shared_patients": len(discordant_ids),
        "same_patient_set": mine_ids == other_ids,
        "same_nonkey_schema": not only_mine_cols and not only_other_cols,
        "exact_on_shared_values": total_diff_cells == 0,
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def markdown_summary(rows: Iterable[dict]) -> str:
    rows = list(rows)
    lines = [
        "# PROMIS canonical vs Aabila validation",
        "",
        f"Canonical pipeline commit: `{CANONICAL_PIPELINE_COMMIT}`",
        "",
        "Imputed datasets are intentionally excluded because advanced imputation is stochastic.",
        "",
        "| Site | Layer | Canonical patients | Aabila patients | Shared | Only canonical | Only Aabila | Schema-only cols (C/A) | Diff cols | Diff cells | Discordant patients |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in rows:
        lines.append(
            "| {site} | {artifact} | {canonical_unique_patients:,} | {aabila_unique_patients:,} | "
            "{shared_patients:,} | {patients_only_canonical:,} | {patients_only_aabila:,} | "
            "{cm}/{ca} | {columns_with_value_differences:,} | {total_different_cells:,} | "
            "{discordant_shared_patients:,} |".format(
                cm=len(s["columns_only_canonical"]),
                ca=len(s["columns_only_aabila"]),
                **s,
            )
        )

    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- `same_patient_set=false` is a cohort-selection discrepancy and should be investigated first.",
        "- Schema-only columns indicate extraction/harmonization differences; inspect `schema.csv`.",
        "- Value differences are calculated only for shared patients and shared variables.",
        "- Missingness differences and per-variable disagreement counts are in `column_comparison.csv`.",
        "- Patient identifiers appear only in local discrepancy CSVs and should not be committed.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sites = ["psu", "geisinger"] if args.site == "all" else [args.site]
    mine_roots = {"psu": args.mine_psu, "geisinger": args.mine_geisinger}
    aabila_roots = {"psu": args.aabila_psu, "geisinger": args.aabila_geisinger}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for site in sites:
        print(f"\n{'=' * 80}\n{site.upper()}\n{'=' * 80}")
        for artifact, filename in ARTIFACTS.items():
            mine_path = first_existing(mine_roots[site], filename)
            aabila_path = first_existing(aabila_roots[site], filename)
            print(f"Comparing {artifact}:\n  canonical: {mine_path}\n  Aabila:    {aabila_path}")
            summary = compare_artifact(
                site=site,
                artifact=artifact,
                mine_path=mine_path,
                aabila_path=aabila_path,
                output_dir=args.output_dir,
                mine_key_override=args.mine_key,
                aabila_key_override=args.aabila_key,
                atol=args.atol,
                rtol=args.rtol,
            )
            summaries.append(summary)
            print(
                "  patients shared={shared_patients:,}, only canonical={patients_only_canonical:,}, "
                "only Aabila={patients_only_aabila:,}; diff columns={columns_with_value_differences:,}; "
                "diff cells={total_different_cells:,}".format(**summary)
            )

    summary_df = pd.DataFrame(summaries)
    safe_csv(summary_df, args.output_dir / "summary.csv")
    (args.output_dir / "summary.md").write_text(markdown_summary(summaries))
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "canonical_pipeline_commit": CANONICAL_PIPELINE_COMMIT,
                "imputed_included": False,
                "sites": sites,
                "canonical_roots": {k: str(v) for k, v in mine_roots.items() if k in sites},
                "aabila_roots": {k: str(v) for k, v in aabila_roots.items() if k in sites},
                "atol": args.atol,
                "rtol": args.rtol,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote comparison results to: {args.output_dir.resolve()}")
    print(f"Open: {(args.output_dir / 'summary.md').resolve()}")


if __name__ == "__main__":
    main()
