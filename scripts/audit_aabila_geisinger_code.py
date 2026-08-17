#!/usr/bin/env python3
"""Audit Aabila's Geisinger code/config for rules that may explain cohort differences.

This script treats Aabila's actual local code and configuration as primary evidence.
It searches the local Geisinger pipeline for cohort filters, lipid/imaging rules,
index-stroke/date handling, encounter selection, and outcome definitions, and writes
compact excerpts for review. It also tests simple phenotype/evidence filters against
the canonical final cohort to see which rules reproduce Aabila's observed cohort size.

No source files are modified.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_AABILA_ROOT = Path(
    "/home/asadr/works/promis2/aabila/PROMIS-ML-pipeline-master-Geisinger_07162026"
)
DEFAULT_CANONICAL_FINAL = Path(
    "/home/asadr/works/promis2/PROMIS-validation/reproduced/geisinger_997d6bd/final/stroke_cohort_final.parquet"
)
DEFAULT_AABILA_FINAL = DEFAULT_AABILA_ROOT / "outputs/final/stroke_cohort_final.parquet"
DEFAULT_OUT = Path("results/aabila_comparison/geisinger_code_audit")

TEXT_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".txt", ".md", ".sh"}
PATTERNS = {
    "lipid": re.compile(r"lipid|ldl|hdl|triglycer", re.I),
    "imaging": re.compile(r"imag|ct\b|mri\b|neuro", re.I),
    "cohort_filter": re.compile(r"filter|eligib|inclusion|exclude|exclusion|require|dropna|query\(|\[.*==.*\]", re.I),
    "stroke_date": re.compile(r"DX_DATE_stroke|stroke.*date|index.*date|dx_date|diagnos.*date", re.I),
    "encounter": re.compile(r"encounter|admit|discharge|length.?of.?stay|los\b", re.I),
    "outcome": re.compile(r"mort|death|readmission|revisit|icu|outcome", re.I),
    "one_day": re.compile(r"1\s*day|one\s*day|timedelta\s*\(\s*days\s*=\s*1|days\s*<=?\s*1|abs\(.*day", re.I),
}


def text_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS:
            # Ignore bulky/generated environments if present.
            if any(part in {".git", ".venv", "venv", "__pycache__", "outputs", "dataset"} for part in p.parts):
                continue
            yield p


def collect_hits(root: Path) -> list[dict]:
    hits = []
    for path in text_files(root):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, start=1):
            matched = [name for name, pat in PATTERNS.items() if pat.search(line)]
            if not matched:
                continue
            # Context window gives enough evidence without dumping entire files.
            lo = max(1, i - 2)
            hi = min(len(lines), i + 2)
            context = "\n".join(f"{j}: {lines[j-1]}" for j in range(lo, hi + 1))
            hits.append({
                "file": str(path.relative_to(root)),
                "line": i,
                "categories": ",".join(matched),
                "text": line.strip(),
                "context": context,
            })
    return hits


def choose_key(df: pd.DataFrame) -> str:
    for c in ("PATID", "STUDY_ID", "PT_ID", "patient_id", "PatientID"):
        if c in df.columns:
            return c
    raise KeyError("No patient identifier found")


def bool_mask(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)
    return s.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


def test_candidate_filters(canonical_path: Path, aabila_path: Path | None) -> pd.DataFrame:
    c = pd.read_parquet(canonical_path).copy()
    target = None
    if aabila_path and aabila_path.exists():
        a = pd.read_parquet(aabila_path)
        ak = choose_key(a)
        target = int(a[ak].astype("string").str.strip().nunique())

    candidates: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=c.index))]
    if "has_lipid_panel" in c.columns:
        lipid = bool_mask(c["has_lipid_panel"])
        candidates.append(("has_lipid_panel == 1", lipid))
    else:
        lipid = None
    if "has_neuroimaging" in c.columns:
        imaging = bool_mask(c["has_neuroimaging"])
        candidates.append(("has_neuroimaging == 1", imaging))
    else:
        imaging = None
    if lipid is not None and imaging is not None:
        candidates.append(("lipid AND imaging", lipid & imaging))
        candidates.append(("lipid OR imaging", lipid | imaging))
    if "has_rehabilitation" in c.columns:
        rehab = bool_mask(c["has_rehabilitation"])
        candidates.append(("has_rehabilitation == 1", rehab))
        if lipid is not None:
            candidates.append(("lipid AND rehabilitation", lipid & rehab))
        if imaging is not None:
            candidates.append(("imaging AND rehabilitation", imaging & rehab))
        if lipid is not None and imaging is not None:
            candidates.append(("lipid AND imaging AND rehabilitation", lipid & imaging & rehab))

    rows = []
    for name, mask in candidates:
        n = int(mask.sum())
        rows.append({
            "candidate_rule": name,
            "canonical_patients": n,
            "aabila_unique_patients_target": target,
            "difference_from_aabila": (n - target) if target is not None else None,
            "absolute_difference": abs(n - target) if target is not None else None,
        })
    df = pd.DataFrame(rows)
    if target is not None:
        df = df.sort_values(["absolute_difference", "candidate_rule"])
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aabila-root", type=Path, default=DEFAULT_AABILA_ROOT)
    p.add_argument("--canonical-final", type=Path, default=DEFAULT_CANONICAL_FINAL)
    p.add_argument("--aabila-final", type=Path, default=DEFAULT_AABILA_FINAL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if not args.aabila_root.exists():
        raise FileNotFoundError(f"Aabila Geisinger root not found: {args.aabila_root}")
    if not args.canonical_final.exists():
        raise FileNotFoundError(f"Canonical final not found: {args.canonical_final}")

    hits = collect_hits(args.aabila_root)
    hit_df = pd.DataFrame(hits)
    hit_df.to_csv(out / "code_config_hits.csv", index=False)

    # Produce category-specific text excerpts that are easy to upload/read.
    for category in PATTERNS:
        category_hits = [h for h in hits if category in h["categories"].split(",")]
        lines = []
        for h in category_hits:
            lines.append(f"### {h['file']}:{h['line']}")
            lines.append(h["context"])
            lines.append("")
        (out / f"{category}_hits.txt").write_text("\n".join(lines))

    candidate_df = test_candidate_filters(
        args.canonical_final,
        args.aabila_final if args.aabila_final.exists() else None,
    )
    candidate_df.to_csv(out / "candidate_filter_counts.csv", index=False)

    summary = {
        "aabila_root": str(args.aabila_root),
        "canonical_final": str(args.canonical_final),
        "aabila_final": str(args.aabila_final),
        "text_files_scanned": sum(1 for _ in text_files(args.aabila_root)),
        "matching_lines": len(hits),
        "hits_by_category": {
            cat: sum(cat in h["categories"].split(",") for h in hits) for cat in PATTERNS
        },
        "closest_candidate_filters": candidate_df.head(10).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Aabila Geisinger code/config audit")
    print(f"Root: {args.aabila_root}")
    print(f"Text files scanned: {summary['text_files_scanned']:,}")
    print(f"Matching source/config lines: {len(hits):,}")
    print("\nHits by category:")
    for k, v in summary["hits_by_category"].items():
        print(f"  {k}: {v:,}")
    print("\nCandidate canonical filters closest to Aabila cohort size:")
    print(candidate_df.head(10).to_string(index=False))
    print(f"\nResults written to: {out.resolve()}")


if __name__ == "__main__":
    main()
