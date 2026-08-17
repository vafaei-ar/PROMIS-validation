#!/usr/bin/env python3
"""Compare canonical PSU registry-positive patients against Aabila PSU outputs.

This is the primary apples-to-apples PSU equivalence analysis because Aabila's
cohort is effectively registry-restricted, whereas the canonical harmonized PSU
cohort also includes non-registry EHR stroke patients.

Imputed data are intentionally excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COMMIT = "7cd577c69d4df839b426478e31c7a4caf5105cd8"

DEFAULT_CANONICAL_ROOT = Path(
    "/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd"
)
DEFAULT_AABILA_ROOT = Path(
    "/home/asadr/works/promis2/aabila/PROMIS-ML-PennState/outputs"
)
DEFAULT_OUTPUT = Path("results/aabila_comparison/psu_registry_only")

ARTIFACTS = {
    "final": "stroke_cohort_final.parquet",
    "unimputed": "stroke_cohort_unimputed.parquet",
    "outcomes": "outcomes_flags.parquet",
    "cohort_with_outcomes": "cohort_with_outcomes.parquet",
}
KEY_CANDIDATES = ("PATID", "STUDY_ID", "PT_ID", "patient_id", "PatientID")


def first_existing(root: Path, filename: str) -> Path | None:
    for p in (root / filename, root / "final" / filename, root / "intermediate" / filename):
        if p.exists():
            return p
    return None


def choose_key(df: pd.DataFrame) -> str:
    for c in KEY_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"No patient identifier found among {KEY_CANDIDATES}")


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def diff_mask(a: pd.Series, b: pd.Series, atol: float, rtol: float) -> pd.Series:
    a_na = a.isna()
    b_na = b.isna()
    out = a_na ^ b_na
    both = ~(a_na | b_na)
    if not both.any():
        return out

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
    out_dir: Path,
    atol: float,
    rtol: float,
) -> dict:
    c = pd.read_parquet(canonical_path).copy()
    a = pd.read_parquet(aabila_path).copy()
    ck = choose_key(c)
    ak = choose_key(a)

    if "in_stroke_registry" not in c.columns:
        raise KeyError(f"{canonical_path} has no in_stroke_registry column")

    c = c[c["in_stroke_registry"] == 1].copy()
    c["__id"] = norm_id(c[ck])
    a["__id"] = norm_id(a[ak])

    if c["__id"].duplicated().any() or a["__id"].duplicated().any():
        raise ValueError(f"Duplicate patient IDs found in {layer}")

    c_ids = set(c["__id"].dropna())
    a_ids = set(a["__id"].dropna())
    shared = c_ids & a_ids
    c_only = sorted(c_ids - a_ids)
    a_only = sorted(a_ids - c_ids)

    layer_dir = out_dir / layer
    layer_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": c_only}).to_csv(layer_dir / "patients_only_canonical_registry.csv", index=False)
    pd.DataFrame({"patient_id": a_only}).to_csv(layer_dir / "patients_only_aabila.csv", index=False)

    ignored = {ck, ak, "__id"}
    c_cols = set(c.columns) - ignored
    a_cols = set(a.columns) - ignored
    shared_cols = sorted(c_cols & a_cols)
    c_only_cols = sorted(c_cols - a_cols)
    a_only_cols = sorted(a_cols - c_cols)

    schema = []
    for col in sorted(c_cols | a_cols):
        schema.append({
            "column": col,
            "in_canonical_registry": col in c_cols,
            "in_aabila": col in a_cols,
            "canonical_dtype": str(c[col].dtype) if col in c.columns else "",
            "aabila_dtype": str(a[col].dtype) if col in a.columns else "",
        })
    pd.DataFrame(schema).to_csv(layer_dir / "schema.csv", index=False)

    cs = c[c["__id"].isin(shared)].set_index("__id").sort_index()
    aa = a[a["__id"].isin(shared)].set_index("__id").sort_index()

    rows = []
    discordant_sets: list[set[str]] = []
    for col in shared_cols:
        mask = diff_mask(cs[col], aa[col], atol=atol, rtol=rtol)
        ids = set(mask.index[mask])
        if ids:
            discordant_sets.append(ids)
        rows.append({
            "column": col,
            "canonical_missing": int(cs[col].isna().sum()),
            "aabila_missing": int(aa[col].isna().sum()),
            "missing_difference_aabila_minus_canonical": int(aa[col].isna().sum() - cs[col].isna().sum()),
            "different_cells": int(mask.sum()),
            "pct_shared_patients_different": (100.0 * float(mask.sum()) / len(shared)) if shared else np.nan,
        })

    col_df = pd.DataFrame(rows)
    if not col_df.empty:
        col_df = col_df.sort_values(["different_cells", "column"], ascending=[False, True])
    col_df.to_csv(layer_dir / "column_comparison.csv", index=False)

    discordant = sorted(set().union(*discordant_sets)) if discordant_sets else []
    pd.DataFrame({"patient_id": discordant}).to_csv(layer_dir / "discordant_patients.csv", index=False)

    summary = {
        "layer": layer,
        "canonical_registry_patients": len(c_ids),
        "aabila_patients": len(a_ids),
        "shared_patients": len(shared),
        "canonical_registry_only": len(c_only),
        "aabila_only": len(a_only),
        "shared_columns": len(shared_cols),
        "columns_only_canonical_registry": c_only_cols,
        "columns_only_aabila": a_only_cols,
        "columns_with_value_differences": int((col_df["different_cells"] > 0).sum()) if not col_df.empty else 0,
        "total_different_cells": int(col_df["different_cells"].sum()) if not col_df.empty else 0,
        "discordant_shared_patients": len(discordant),
        "same_patient_set": c_ids == a_ids,
        "same_nonkey_schema": not c_only_cols and not a_only_cols,
        "canonical_path": str(canonical_path),
        "aabila_path": str(aabila_path),
    }
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
    missing = []

    for layer, filename in ARTIFACTS.items():
        cpath = first_existing(args.canonical_root, filename)
        apath = first_existing(args.aabila_root, filename)
        if cpath is None or apath is None:
            missing.append({
                "layer": layer,
                "canonical_found": cpath is not None,
                "aabila_found": apath is not None,
                "filename": filename,
            })
            print(f"Skipping {layer}: canonical={cpath is not None}, Aabila={apath is not None}")
            continue

        print(f"Comparing {layer}\n  canonical: {cpath}\n  Aabila:    {apath}")
        s = compare_layer(layer, cpath, apath, args.output_dir, args.atol, args.rtol)
        summaries.append(s)
        print(
            "  registry={canonical_registry_patients:,}, Aabila={aabila_patients:,}, "
            "shared={shared_patients:,}, C-only={canonical_registry_only:,}, A-only={aabila_only:,}, "
            "diff cols={columns_with_value_differences:,}, diff cells={total_different_cells:,}".format(**s)
        )

    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(missing).to_csv(args.output_dir / "missing_artifacts.csv", index=False)

    lines = [
        "# PSU registry-only canonical vs Aabila",
        "",
        f"Canonical pipeline commit: `{CANONICAL_COMMIT}`",
        "",
        "Canonical rows are filtered to `in_stroke_registry == 1` before comparison.",
        "Imputed data are intentionally excluded.",
        "",
        "| Layer | Canonical registry | Aabila | Shared | C-only | A-only | Diff cols | Diff cells | Discordant patients |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {layer} | {canonical_registry_patients:,} | {aabila_patients:,} | {shared_patients:,} | "
            "{canonical_registry_only:,} | {aabila_only:,} | {columns_with_value_differences:,} | "
            "{total_different_cells:,} | {discordant_shared_patients:,} |".format(**s)
        )
    if missing:
        lines += ["", "## Missing artifacts", ""]
        for m in missing:
            lines.append(
                f"- `{m['layer']}` ({m['filename']}): canonical_found={m['canonical_found']}, "
                f"Aabila_found={m['aabila_found']}"
            )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nResults written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
