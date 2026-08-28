#!/usr/bin/env python3
"""Fail if tracked validation-repository content looks like patient-level data.

This repository is public. Generated comparison outputs, reproduced datasets,
and any patient-level identifiers must remain local and outside Git.
This guard is conservative and is not a legal determination of HIPAA compliance.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED_DIR_PREFIXES = (
    "results/",
    "reproduced/",
    "logs/",
    "outputs/",
    "configs/reproduction/",
)

BANNED_EXTENSIONS = {
    ".parquet", ".feather", ".arrow", ".rds", ".sas7bdat", ".dta", ".sav",
    ".xlsx", ".xls", ".zip", ".tar", ".gz", ".ipynb", ".csv",
}

PATIENT_ID_PATTERNS = {
    "PSU-style patient ID": re.compile(r"\bPSU\d{8,}\b"),
    "hashed patient ID": re.compile(r"\bpsu-[0-9a-f]{10,}\b", re.I),
    "long encounter ID": re.compile(r"\bENC\d{6,}\b", re.I),
}

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini",
    ".cfg", ".sh", ".r", ".sql", ".html", ".xml",
}


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [p.decode("utf-8") for p in out.split(b"\0") if p]


def main() -> int:
    problems: list[str] = []

    for rel in tracked_files():
        low = rel.lower()
        path = ROOT / rel

        if low.startswith(BANNED_DIR_PREFIXES):
            problems.append(f"tracked generated/sensitive path: {rel}")

        suffix = path.suffix.lower()
        if suffix in BANNED_EXTENSIONS:
            problems.append(f"tracked data/notebook artifact: {rel}")

        if suffix not in TEXT_EXTENSIONS:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for label, pattern in PATIENT_ID_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label} found in tracked text: {rel}")

    if problems:
        print("Potential patient-level information detected in Git-tracked content:")
        for item in sorted(set(problems)):
            print(f"  - {item}")
        print("\nRemove/sanitize the content before pushing.")
        return 1

    print("No obvious patient-level identifiers or prohibited clinical-data artifacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
