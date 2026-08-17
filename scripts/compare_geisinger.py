#!/usr/bin/env python3
"""Compare canonical Geisinger outputs against Aabila's Geisinger outputs.

Validation scope intentionally excludes imputed data. The script compares:
  * stroke_cohort_final.parquet
  * stroke_cohort_unimputed.parquet
  * outcomes_flags.parquet
  * cohort_with_outcomes.parquet

Exact duplicate rows are collapsed automatically. If duplicate patient IDs
remain with conflicting records, those rows are written locally and that layer
is marked unsafe for one-to-one value comparison rather than being aggregated
silently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COMMIT = "7cd577c69d4df839b426478e31c7a4caf5105cd8"
DEFAULT_CANONICAL_ROOT = Path(
    "/home/asadr/works/promis2/PROMIS-validation/reproduced/geisinger_997d6bd"
)
DEFAULT_AABILA_ROOT = Path(
    "/home/asadr/works/promis2/aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs"
)
DEFAULT_OUTPUT = Path("results/aabila_comparison/geisinger")

ARTIFACTS = {
    "final": "stroke_cohort_final.parquet",
    "unimputed": "stroke_cohort_unimputed.parquet",
    "outcomes": "outcomes_flags.parquet",
    "cohort_with_outcomes": "cohort_with_outcomes.parquet",
}
KEYS = ("PATID", "STUDY_ID", "PT_ID", "patient_id", "PatientID")


def first_existing(root: Path, filename: str) -> Path | None:
    for p in (root / filename, root / "final" / filename, root / "intermediate" / filename):
        if p.exists():
            return p
    return None


def choose_key(df: pd.DataFrame) -> str:
    for c in KEYS:
        if c in df.columns:
            return c
    raise KeyError(f"No patient identifier found among {KEYS}")


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def prepare_unique(
    df: pd.DataFrame, label: str, layer_dir: Path
) -> tuple[pd.DataFrame | None, dict]:
    k = choose_key(df)
    x = df.copy()
    x["__id"] = norm_id(x[k])

    dup_mask = x["__id"].duplicated(keep=False)
    dup_rows = x.loc[dup_mask].copy()
    info = {
        f"{label}_rows_before": int(len(x)),
        f"{label}_unique_ids_before": int(x["__id"].nunique()),
        f"{label}_duplicate_ids": int(dup_rows["__id"].nunique()) if not dup_rows.empty else 0,
        f"{label}_duplicate_rows": int(len(dup_rows)),
        f"{label}_exact_duplicate_rows_removed": 0,
        f"{label}_conflicting_duplicate_ids": 0,
    }

    if not dup_rows.empty:
        before = len(x)
        x = x.drop_duplicates()
        info[f"{label}_exact_duplicate_rows_removed"] = before - len(x)
        still = x[x["__id"].duplicated(keep=False)].copy()
        if not still.empty:
            info[f"{label}_conflicting_duplicate_ids"] = int(still["__id"].nunique())
            still.sort_values("__id").to_csv(
                layer_dir / f"{label}_conflicting_duplicates.csv", index=False
            )
            still.groupby("__id").size().rename("rows").reset_index().to_csv(
                layer_dir / f"{label}_conflicting_duplicate_counts.csv", index=False
            )
            return None, info

    return x, info


def diff_mask(a: pd.Series, b: pd.Series, atol: float, rtol: float) -> pd.Series:
    a_na, b_na = a.isna(), b.isna()
    out = a_na ^ b_na
    both = ~(a_na | b_na)
    if not both.any():
        return out.fillna(False)

    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av = pd.to_numeric(a.loc[both], errors="coerce").astype(float).to_numpy()
        bv = pd.to_numeric(b.loc[both], errors="coerce").astype(float).to_numpy()
        out.loc[both] = ~np.isclose(av, bv, atol=atol, rtol=rtol, equal_nan=True)
    elif pd.api.types.is_datetime64_any_dtype(a) and pd.api.types.is_datetime64_any_dtype(b):
        out.loc[both] = (
            a.loc[both].astype("datetime64[ns]").to_numpy()
            != b.loc[both].astype("datetime64[ns]").to_numpy()
        )
    else:
        out.loc[both] = (
            a.loc[both].astype("string").str.strip().to_numpy()
            != b.loc[both].astype("string").str.strip().to_numpy()
        )
    return out.fillna(False)


def compare_layer(
    layer: str,
    canonical_path: Path,
    aabila_path: Path,
    output_dir: Path,
    atol: float,
    rtol: float,
) -> dict:
    layer_dir = output_dir / layer
    layer_dir.mkdir(parents=True, exist_ok=True)

    c_raw = pd.read_parquet(canonical_path)
    a_raw = pd.read_parquet(aabila_path)
    ck, ak = choose_key(c_raw), choose_key(a_raw)

    c, ci = prepare_unique(c_raw, "canonical", layer_dir)
    a, ai = prepare_unique(a_raw, "aabila", layer_dir)
    summary = {
        "layer": layer,
        "canonical_path": str(canonical_path),
        "aabila_path": str(aabila_path),
        **ci,
        **ai,
    }

    if c is None or a is None:
        summary["status"] = "conflicting_duplicates"
        (layer_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    c_ids = set(c["__id"].dropna())
    a_ids = set(a["__id"].dropna())
    shared = c_ids & a_ids
    c_only = sorted(c_ids - a_ids)
    a_only = sorted(a_ids - c_ids)

    pd.DataFrame({"patient_id": c_only}).to_csv(
        layer_dir / "patients_only_canonical.csv", index=False
    )
    pd.DataFrame({"patient_id": a_only}).to_csv(
        layer_dir / "patients_only_aabila.csv", index=False
    )

    ignored = {ck, ak, "__id"}
    c_cols = set(c.columns) - ignored
    a_cols = set(a.columns) - ignored
    shared_cols = sorted(c_cols & a_cols)
    c_only_cols = sorted(c_cols - a_cols)
    a_only_cols = sorted(a_cols - c_cols)

    schema_rows = []
    for col in sorted(c_cols | a_cols):
        schema_rows.append(
            {
                "column": col,
                "in_canonical": col in c_cols,
                "in_aabila": col in a_cols,
                "canonical_dtype": str(c[col].dtype) if col in c.columns else "",
                "aabila_dtype": str(a[col].dtype) if col in a.columns else "",
            }
        )
    pd.DataFrame(schema_rows).to_csv(layer_dir / "schema.csv", index=False)

    cs = c[c["__id"].isin(shared)].set_index("__id").sort_index()
    aa = a[a["__id"].isin(shared)].set_index("__id").sort_index()

    rows = []
    discordant = set()
    for col in shared_cols:
        m = diff_mask(cs[col], aa[col], atol=atol, rtol=rtol)
        discordant.update(m.index[m])
        cm = int(cs[col].isna().sum())
        am = int(aa[col].isna().sum())
        rows.append(
            {
                "column": col,
                "canonical_dtype": str(cs[col].dtype),
                "aabila_dtype": str(aa[col].dtype),
                "canonical_missing": cm,
                "aabila_missing": am,
                "missing_difference_aabila_minus_canonical": am - cm,
                "different_cells": int(m.sum()),
                "pct_shared_patients_different": (
                    100.0 * float(m.sum()) / len(shared) if shared else np.nan
                ),
            }
        )

    col_df = pd.DataFrame(rows)
    if not col_df.empty:
        col_df = col_df.sort_values(["different_cells", "column"], ascending=[False, True])
    col_df.to_csv(layer_dir / "column_comparison.csv", index=False)
    pd.DataFrame({"patient_id": sorted(discordant)}).to_csv(
        layer_dir / "discordant_patients.csv", index=False
    )

    summary.update(
        {
            "status": "compared",
            "canonical_patients": len(c_ids),
            "aabila_patients": len(a_ids),
            "shared_patients": len(shared),
            "canonical_only": len(c_only),
            "aabila_only": len(a_only),
            "shared_columns": len(shared_cols),
            "columns_only_canonical": c_only_cols,
            "columns_only_aabila": a_only_cols,
            "columns_with_differences": int((col_df["different_cells"] > 0).sum())
            if not col_df.empty
            else 0,
            "total_different_cells": int(col_df["different_cells"].sum())
            if not col_df.empty
            else 0,
            "discordant_shared_patients": len(discordant),
            "same_patient_set": c_ids == a_ids,
            "same_nonkey_schema": not c_only_cols and not a_only_cols,
            "exact_on_shared_values": bool(col_df.empty or (col_df["different_cells"] == 0).all()),
        }
    )
    (layer_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    p.add_argument("--aabila-root", type=Path, default=DEFAULT_AABILA_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--rtol", type=float, default=1e-7)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for layer, filename in ARTIFACTS.items():
        cp = first_existing(args.canonical_root, filename)
        ap = first_existing(args.aabila_root, filename)
        if cp is None or ap is None:
            s = {
                "layer": layer,
                "status": "missing_artifact",
                "filename": filename,
                "canonical_found": cp is not None,
                "aabila_found": ap is not None,
            }
            summaries.append(s)
            print(json.dumps(s, indent=2))
            continue

        print(f"\nComparing {layer}:\n  canonical: {cp}\n  Aabila:    {ap}")
        s = compare_layer(layer, cp, ap, args.output_dir, args.atol, args.rtol)
        summaries.append(s)
        print(json.dumps(s, indent=2))

    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    lines = [
        "# Geisinger canonical vs Aabila validation",
        "",
        f"Canonical pipeline commit: `{CANONICAL_COMMIT}`",
        "",
        "Imputed datasets are intentionally excluded.",
        "",
        "| Layer | Status | Canonical | Aabila | Shared | C-only | A-only | Diff cols | Diff cells | Discordant patients |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        if s.get("status") == "compared":
            lines.append(
                "| {layer} | compared | {canonical_patients:,} | {aabila_patients:,} | {shared_patients:,} | "
                "{canonical_only:,} | {aabila_only:,} | {columns_with_differences:,} | {total_different_cells:,} | "
                "{discordant_shared_patients:,} |".format(**s)
            )
        else:
            lines.append(f"| {s['layer']} | {s['status']} |  |  |  |  |  |  |  |  |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nResults written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
