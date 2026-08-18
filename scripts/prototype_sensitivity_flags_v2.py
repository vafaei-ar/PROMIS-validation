#!/usr/bin/env python3
"""Robust wrapper for prototype_sensitivity_flags.py.

Changes from v1:
- Auto-discovers a Geisinger raw diagnosis parquet when the historical hard-coded
  path does not exist, requiring PATID/ENCOUNTERID/DX columns.
- Reuses v1 semantics unchanged for strict-index and DX ambiguity flags.
- Writes a fresh summary under results/sensitivity_flag_prototype/<site>/summary_v2.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

import prototype_sensitivity_flags as base


def discover_geisinger_raw_dx() -> Path:
    root = base.ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
    candidates = []
    for p in root.rglob('*.parquet'):
        low = p.name.lower()
        if not any(k in low for k in ('diagn', 'dx', 'condition')):
            continue
        pathlow = str(p).lower().replace('\\', '/')
        if '/outputs/' in pathlow or '/results/' in pathlow:
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        required = {'PATID', 'ENCOUNTERID', 'DX'}
        if required.issubset(df.columns):
            candidates.append((0 if '/dataset/' in pathlow else 1, len(df), p))
    if not candidates:
        raise FileNotFoundError(
            'Could not auto-discover a Geisinger raw diagnosis parquet with '
            'PATID, ENCOUNTERID, and DX columns.'
        )
    candidates.sort(key=lambda x: (x[0], -x[1], str(x[2])))
    return candidates[0][2]


def run(site: str, canonical_root: Path | None = None, raw_dx: Path | None = None):
    if canonical_root is None:
        canonical_root = base.DEFAULTS[site]['canonical_root']
    if raw_dx is None:
        default = base.DEFAULTS[site]['raw_dx']
        raw_dx = default if default.exists() else (
            discover_geisinger_raw_dx() if site == 'geisinger' else default
        )

    outdir = base.RESULTS / site
    outdir.mkdir(parents=True, exist_ok=True)

    first, all_enc, dx, first_path, all_path = base.load_inputs(site, canonical_root, raw_dx)
    strict = base.strict_index_flags(first, all_enc)
    dxf, dxmeta = base.dx_ambiguity_flags(first, dx)
    flags = strict.merge(dxf.drop(columns=['ENCOUNTERID'], errors='ignore'), on='PATID', how='left')

    errors = base.validate_flags(flags, first)
    prototype = first.merge(
        flags.drop(columns=['broad_index_encounter_id', 'broad_index_stroke_date', 'broad_index_dx'], errors='ignore'),
        on='PATID', how='left', validate='1:1'
    )

    flags.to_parquet(outdir / 'patient_sensitivity_flags.parquet', index=False)
    flags.to_csv(outdir / 'patient_sensitivity_flags.csv', index=False)
    prototype.to_parquet(outdir / 'prototype_first_stroke_with_flags.parquet', index=False)

    summary = {
        'site': site,
        'canonical_first_file': str(first_path),
        'canonical_all_encounters_file': str(all_path),
        'raw_dx_file': str(raw_dx),
        'patients': int(first['PATID'].nunique()),
        'phenotype_strict_eligible': int(flags['phenotype_strict_eligible'].sum()),
        'has_alt_qualifying_stroke_encounter': int(flags['has_alt_qualifying_stroke_encounter'].sum()),
        'index_selection_sensitive': int(flags['index_selection_sensitive'].sum()),
        'dx': dxmeta,
        'validation_errors': errors,
    }
    if site == 'geisinger':
        summary['geisinger_edge_check'] = base.optional_geisinger_edge_check(flags)

    (outdir / 'summary_v2.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))
    if errors:
        raise SystemExit('Validation errors: ' + '; '.join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', choices=['psu', 'geisinger'], required=True)
    ap.add_argument('--canonical-root', type=Path)
    ap.add_argument('--raw-dx', type=Path)
    args = ap.parse_args()
    run(args.site, args.canonical_root, args.raw_dx)


if __name__ == '__main__':
    main()
