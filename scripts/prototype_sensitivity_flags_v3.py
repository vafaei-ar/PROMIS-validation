#!/usr/bin/env python3
"""Alias-aware Geisinger sensitivity-flag prototype.

This wrapper fixes raw diagnosis discovery for site-specific schemas by accepting
common patient/encounter/diagnosis aliases and normalizing them to PATID,
ENCOUNTERID, and DX before running the validated prototype logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

import prototype_sensitivity_flags as base

ROOT = Path('/home/asadr/works/promis2')
GEI_ROOT = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'

ID_ALIASES = ('PATID','STUDY_ID','PT_ID','PAT_ID','patient_id','PatientID')
ENC_ALIASES = ('ENCOUNTERID','ENCOUNTER_ID','ENC_ID','ENCOUNTERID_RAW')
DX_ALIASES = ('DX','DX_CODE','DIAGNOSIS_CODE','DIAGNOSIS','ICD_CODE')
PRIMARY_ALIASES = ('PDX','DX_PRIMARY_YN','PRIMARY_DX','IS_PRIMARY')
DATE_ALIASES = ('DX_DATE','DIAGNOSIS_DATE','DATE_OF_DIAGNOSIS')


def first_present(cols, aliases):
    return next((c for c in aliases if c in cols), None)


def discover_and_normalize_raw_dx() -> tuple[pd.DataFrame, Path, dict]:
    roots = [GEI_ROOT / 'dataset', GEI_ROOT]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*.parquet'):
            low = str(p).lower().replace('\\','/')
            if '/outputs/' in low:
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            cols = set(df.columns)
            ic = first_present(cols, ID_ALIASES)
            ec = first_present(cols, ENC_ALIASES)
            dc = first_present(cols, DX_ALIASES)
            if ic and ec and dc:
                score = 0
                name = p.name.lower()
                if 'diagn' in name or 'dx' in name:
                    score += 10
                score += min(len(df), 1_000_000) / 1_000_000
                candidates.append((score, p, df, ic, ec, dc))
    if not candidates:
        raise FileNotFoundError('No alias-compatible Geisinger raw diagnosis parquet was found.')
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, path, df, ic, ec, dc = candidates[0]
    rename = {}
    if ic != 'PATID': rename[ic] = 'PATID'
    if ec != 'ENCOUNTERID': rename[ec] = 'ENCOUNTERID'
    if dc != 'DX': rename[dc] = 'DX'
    out = df.rename(columns=rename).copy()

    # Normalize one primary indicator if the source uses an alias not understood by base.
    if not any(c in out.columns for c in ('PDX','DX_PRIMARY_YN','PRIMARY_DX')):
        pc = first_present(set(out.columns), PRIMARY_ALIASES)
        if pc and pc not in ('PDX','DX_PRIMARY_YN','PRIMARY_DX'):
            out = out.rename(columns={pc:'PRIMARY_DX'})
    if 'DX_DATE' not in out.columns:
        dc2 = first_present(set(out.columns), DATE_ALIASES)
        if dc2 and dc2 != 'DX_DATE':
            out = out.rename(columns={dc2:'DX_DATE'})

    meta = {
        'source_file': str(path),
        'source_rows': int(len(out)),
        'source_columns': int(len(out.columns)),
        'original_id_col': ic,
        'original_encounter_col': ec,
        'original_dx_col': dc,
    }
    return out, path, meta


def run_geisinger(canonical_root: Path | None = None):
    canonical_root = canonical_root or base.DEFAULTS['geisinger']['canonical_root']
    first_path = base.find_file(canonical_root, 'first_stroke_encounters.parquet')
    all_path = base.find_file(canonical_root, 'all_stroke_encounters.parquet')
    first = pd.read_parquet(first_path).copy()
    all_enc = pd.read_parquet(all_path).copy()
    for df in (first, all_enc):
        df['PATID'] = base.norm(df['PATID'])
        if 'ENCOUNTERID' in df.columns:
            df['ENCOUNTERID'] = base.norm(df['ENCOUNTERID'])

    dx, raw_path, raw_meta = discover_and_normalize_raw_dx()
    dx['PATID'] = base.norm(dx['PATID'])
    dx['ENCOUNTERID'] = base.norm(dx['ENCOUNTERID'])

    strict = base.strict_index_flags(first, all_enc)
    dxf, dxmeta = base.dx_ambiguity_flags(first, dx)
    flags = strict.merge(dxf.drop(columns=['ENCOUNTERID'], errors='ignore'), on='PATID', how='left')
    errors = base.validate_flags(flags, first)

    outdir = base.RESULTS / 'geisinger'
    outdir.mkdir(parents=True, exist_ok=True)
    prototype = first.merge(
        flags.drop(columns=['broad_index_encounter_id','broad_index_stroke_date','broad_index_dx'], errors='ignore'),
        on='PATID', how='left', validate='1:1'
    )
    flags.to_parquet(outdir/'patient_sensitivity_flags.parquet', index=False)
    flags.to_csv(outdir/'patient_sensitivity_flags.csv', index=False)
    prototype.to_parquet(outdir/'prototype_first_stroke_with_flags.parquet', index=False)

    summary = {
        'site':'geisinger',
        'canonical_first_file':str(first_path),
        'canonical_all_encounters_file':str(all_path),
        'raw_dx':raw_meta,
        'patients':int(first['PATID'].nunique()),
        'phenotype_strict_eligible':int(flags['phenotype_strict_eligible'].sum()),
        'has_alt_qualifying_stroke_encounter':int(flags['has_alt_qualifying_stroke_encounter'].sum()),
        'index_selection_sensitive':int(flags['index_selection_sensitive'].sum()),
        'dx':dxmeta,
        'validation_errors':errors,
        'geisinger_edge_check':base.optional_geisinger_edge_check(flags),
    }
    (outdir/'summary_v3.json').write_text(json.dumps(summary, indent=2, default=str)+'\n')
    print(json.dumps(summary, indent=2, default=str))
    if errors:
        raise RuntimeError('; '.join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical-root', type=Path)
    args = ap.parse_args()
    run_geisinger(args.canonical_root)


if __name__ == '__main__':
    main()
