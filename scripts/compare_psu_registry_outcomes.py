#!/usr/bin/env python3
"""Compare PSU registry-only outcomes against Aabila with duplicate diagnostics.

The registry ID set is derived from canonical stroke_cohort_final. For Aabila
outcome files, exact duplicate rows are collapsed automatically. If duplicate
patient IDs contain conflicting rows, the script writes those rows locally and
skips value-level comparison for that layer rather than guessing how to reduce
them.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

CANONICAL_ROOT = Path("/home/asadr/works/promis2/PROMIS-validation/reproduced/psu_997d6bd")
AABILA_ROOT = Path("/home/asadr/works/promis2/aabila/PROMIS-ML-PennState/outputs")
OUT = Path("results/aabila_comparison/psu_registry_outcomes")
KEYS = ("PATID", "STUDY_ID", "PT_ID", "patient_id", "PatientID")
LAYERS = {
    "outcomes": "outcomes_flags.parquet",
    "cohort_with_outcomes": "cohort_with_outcomes.parquet",
}


def first_existing(root: Path, name: str) -> Path | None:
    for p in (root / name, root / "final" / name, root / "intermediate" / name):
        if p.exists():
            return p
    return None


def key(df: pd.DataFrame) -> str:
    for c in KEYS:
        if c in df.columns:
            return c
    raise KeyError(f"No patient identifier found among {KEYS}")


def norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def registry_ids() -> set[str]:
    p = first_existing(CANONICAL_ROOT, "stroke_cohort_final.parquet")
    if p is None:
        raise FileNotFoundError("Canonical stroke_cohort_final.parquet not found")
    df = pd.read_parquet(p)
    k = key(df)
    return set(norm(df.loc[df["in_stroke_registry"] == 1, k]).dropna())


def prepare_unique(df: pd.DataFrame, label: str, layer_dir: Path) -> tuple[pd.DataFrame | None, dict]:
    k = key(df)
    x = df.copy()
    x["__id"] = norm(x[k])

    dup_mask = x["__id"].duplicated(keep=False)
    duplicated_rows = x.loc[dup_mask].copy()
    duplicated_ids = int(duplicated_rows["__id"].nunique())
    duplicate_row_count = int(len(duplicated_rows))

    info = {
        f"{label}_rows_before": int(len(x)),
        f"{label}_unique_ids_before": int(x["__id"].nunique()),
        f"{label}_duplicate_ids": duplicated_ids,
        f"{label}_duplicate_rows": duplicate_row_count,
        f"{label}_exact_duplicate_rows_removed": 0,
        f"{label}_conflicting_duplicate_ids": 0,
    }

    if not duplicated_rows.empty:
        # Remove exact full-row duplicates first (ignoring dataframe index).
        before = len(x)
        x = x.drop_duplicates()
        info[f"{label}_exact_duplicate_rows_removed"] = before - len(x)

        still = x[x["__id"].duplicated(keep=False)].copy()
        if not still.empty:
            conflict_ids = int(still["__id"].nunique())
            info[f"{label}_conflicting_duplicate_ids"] = conflict_ids
            # Patient-level identifiers stay only in ignored local result files.
            still.sort_values("__id").to_csv(layer_dir / f"{label}_conflicting_duplicates.csv", index=False)
            counts = still.groupby("__id").size().rename("rows").reset_index()
            counts.to_csv(layer_dir / f"{label}_conflicting_duplicate_counts.csv", index=False)
            return None, info

    return x, info


def diff_mask(a: pd.Series, b: pd.Series, atol: float = 1e-8, rtol: float = 1e-7) -> pd.Series:
    a_na, b_na = a.isna(), b.isna()
    out = a_na ^ b_na
    both = ~(a_na | b_na)
    if not both.any():
        return out
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av = pd.to_numeric(a.loc[both], errors="coerce").astype(float).to_numpy()
        bv = pd.to_numeric(b.loc[both], errors="coerce").astype(float).to_numpy()
        out.loc[both] = ~np.isclose(av, bv, atol=atol, rtol=rtol, equal_nan=True)
    elif pd.api.types.is_datetime64_any_dtype(a) and pd.api.types.is_datetime64_any_dtype(b):
        out.loc[both] = a.loc[both].astype("datetime64[ns]").to_numpy() != b.loc[both].astype("datetime64[ns]").to_numpy()
    else:
        out.loc[both] = a.loc[both].astype("string").str.strip().to_numpy() != b.loc[both].astype("string").str.strip().to_numpy()
    return out.fillna(False)


def compare(layer: str, filename: str, reg_ids: set[str]) -> dict:
    cp = first_existing(CANONICAL_ROOT, filename)
    ap = first_existing(AABILA_ROOT, filename)
    layer_dir = OUT / layer
    layer_dir.mkdir(parents=True, exist_ok=True)

    if cp is None or ap is None:
        return {"layer": layer, "status": "missing_artifact", "canonical_found": cp is not None, "aabila_found": ap is not None}

    c = pd.read_parquet(cp).copy()
    a = pd.read_parquet(ap).copy()
    ck, ak = key(c), key(a)
    c["__id"] = norm(c[ck])
    a["__id"] = norm(a[ak])
    c = c[c["__id"].isin(reg_ids)].copy()

    c, ci = prepare_unique(c, "canonical", layer_dir)
    a, ai = prepare_unique(a, "aabila", layer_dir)
    base = {"layer": layer, **ci, **ai}

    if c is None or a is None:
        base["status"] = "conflicting_duplicates"
        (layer_dir / "summary.json").write_text(json.dumps(base, indent=2) + "\n")
        return base

    c_ids, a_ids = set(c["__id"].dropna()), set(a["__id"].dropna())
    shared = c_ids & a_ids
    pd.DataFrame({"patient_id": sorted(c_ids - a_ids)}).to_csv(layer_dir / "patients_only_canonical_registry.csv", index=False)
    pd.DataFrame({"patient_id": sorted(a_ids - c_ids)}).to_csv(layer_dir / "patients_only_aabila.csv", index=False)

    ignored = {ck, ak, "__id"}
    c_cols, a_cols = set(c.columns) - ignored, set(a.columns) - ignored
    shared_cols = sorted(c_cols & a_cols)
    cs = c[c["__id"].isin(shared)].set_index("__id").sort_index()
    aa = a[a["__id"].isin(shared)].set_index("__id").sort_index()

    rows, discordant = [], set()
    for col in shared_cols:
        m = diff_mask(cs[col], aa[col])
        discordant.update(m.index[m])
        rows.append({
            "column": col,
            "canonical_missing": int(cs[col].isna().sum()),
            "aabila_missing": int(aa[col].isna().sum()),
            "different_cells": int(m.sum()),
            "pct_shared_patients_different": 100.0 * float(m.sum()) / len(shared) if shared else np.nan,
        })
    col_df = pd.DataFrame(rows).sort_values(["different_cells", "column"], ascending=[False, True]) if rows else pd.DataFrame()
    col_df.to_csv(layer_dir / "column_comparison.csv", index=False)
    pd.DataFrame({"patient_id": sorted(discordant)}).to_csv(layer_dir / "discordant_patients.csv", index=False)

    base.update({
        "status": "compared",
        "canonical_registry_patients": len(c_ids),
        "aabila_patients": len(a_ids),
        "shared_patients": len(shared),
        "canonical_only": len(c_ids - a_ids),
        "aabila_only": len(a_ids - c_ids),
        "shared_columns": len(shared_cols),
        "columns_only_canonical": sorted(c_cols - a_cols),
        "columns_only_aabila": sorted(a_cols - c_cols),
        "columns_with_differences": int((col_df["different_cells"] > 0).sum()) if not col_df.empty else 0,
        "total_different_cells": int(col_df["different_cells"].sum()) if not col_df.empty else 0,
        "discordant_shared_patients": len(discordant),
    })
    (layer_dir / "summary.json").write_text(json.dumps(base, indent=2) + "\n")
    return base


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg_ids = registry_ids()
    print(f"Registry-positive reference IDs: {len(reg_ids):,}")
    summaries = []
    for layer, filename in LAYERS.items():
        print(f"\nComparing {layer}...")
        s = compare(layer, filename, reg_ids)
        summaries.append(s)
        print(json.dumps(s, indent=2))
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nResults written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
