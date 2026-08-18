#!/usr/bin/env python3
"""Robust Geisinger sensitivity prototype.

Key improvement over v3: strict-index and edge validation are completed even if
raw diagnosis schema normalization fails. Raw DX ambiguity analysis is a separate,
non-blocking section.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

import prototype_sensitivity_flags as base

ROOT = Path('/home/asadr/works/promis2')
GEI_ROOT = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
KNOWN_RAW_DX = GEI_ROOT / 'dataset/parquet/diagnosis.parquet'

ID_ALIASES = ('PATID','STUDY_ID','PT_ID','PAT_ID','patient_id','PatientID')
ENC_ALIASES = ('ENCOUNTERID','ENCOUNTER_ID','ENC_ID','ENCOUNTERID_RAW')
DX_ALIASES = (
    'DX','DX_CODE','DX_CD','DIAGNOSIS_CODE','DIAGNOSIS_CD','DIAG_CD',
    'DIAGNOSIS','ICD_CODE','ICD_CD','ICD10_CODE','ICD9_CODE'
)
PRIMARY_ALIASES = ('PDX','DX_PRIMARY_YN','PRIMARY_DX','IS_PRIMARY','PRIMARY_YN','PRINCIPAL_DX')
DATE_ALIASES = ('DX_DATE','DIAGNOSIS_DATE','DATE_OF_DIAGNOSIS','DX_DT','DIAG_DATE')


def first_present(cols, aliases):
    return next((c for c in aliases if c in cols), None)


def infer_dx_col(columns):
    cols = list(columns)
    exact = first_present(set(cols), DX_ALIASES)
    if exact:
        return exact
    excluded = ('DATE','DT','TIME','TYPE','PRIMARY','PRINCIPAL','POA','SOURCE','ORIGIN','ID')
    scored = []
    for c in cols:
        u = c.upper()
        if not any(k in u for k in ('DX','DIAG','ICD')):
            continue
        if any(k in u for k in excluded):
            continue
        score = 0
        if 'CODE' in u or 'CD' in u: score += 5
        if u.startswith('DX') or u.startswith('DIAG'): score += 3
        scored.append((score, c))
    return sorted(scored, reverse=True)[0][1] if scored else None


def load_raw_dx():
    paths = [KNOWN_RAW_DX]
    if not KNOWN_RAW_DX.exists():
        paths = [p for p in (GEI_ROOT/'dataset').rglob('diagnosis.parquet')]
    if not paths:
        raise FileNotFoundError('Geisinger diagnosis.parquet not found under dataset tree')
    path = paths[0]
    df = pd.read_parquet(path).copy()
    cols = set(df.columns)
    ic = first_present(cols, ID_ALIASES)
    ec = first_present(cols, ENC_ALIASES)
    dc = infer_dx_col(df.columns)
    if not ic or not ec or not dc:
        raise KeyError(f'Could not map raw DX schema. columns={list(df.columns)} id={ic} enc={ec} dx={dc}')
    rename = {ic:'PATID', ec:'ENCOUNTERID', dc:'DX'}
    out = df.rename(columns={k:v for k,v in rename.items() if k != v}).copy()
    if not any(c in out.columns for c in ('PDX','DX_PRIMARY_YN','PRIMARY_DX')):
        pc = first_present(set(out.columns), PRIMARY_ALIASES)
        if pc and pc not in ('PDX','DX_PRIMARY_YN','PRIMARY_DX'):
            out = out.rename(columns={pc:'PRIMARY_DX'})
    if 'DX_DATE' not in out.columns:
        d = first_present(set(out.columns), DATE_ALIASES)
        if d and d != 'DX_DATE':
            out = out.rename(columns={d:'DX_DATE'})
    meta = {
        'source_file': str(path),
        'source_rows': int(len(out)),
        'source_columns': int(len(out.columns)),
        'original_columns': list(df.columns),
        'mapped_id_col': ic,
        'mapped_encounter_col': ec,
        'mapped_dx_col': dc,
    }
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical-root', type=Path)
    args = ap.parse_args()
    canonical_root = args.canonical_root or base.DEFAULTS['geisinger']['canonical_root']

    first_path = base.find_file(canonical_root, 'first_stroke_encounters.parquet')
    all_path = base.find_file(canonical_root, 'all_stroke_encounters.parquet')
    first = pd.read_parquet(first_path).copy()
    all_enc = pd.read_parquet(all_path).copy()
    for df in (first, all_enc):
        df['PATID'] = base.norm(df['PATID'])
        if 'ENCOUNTERID' in df.columns:
            df['ENCOUNTERID'] = base.norm(df['ENCOUNTERID'])

    # This is the primary scientific question and must never depend on raw-DX mapping.
    flags = base.strict_index_flags(first, all_enc)
    errors = base.validate_flags(flags, first)
    edge = base.optional_geisinger_edge_check(flags)

    dx_summary = {'available': False}
    try:
        dx, raw_meta = load_raw_dx()
        dx['PATID'] = base.norm(dx['PATID'])
        dx['ENCOUNTERID'] = base.norm(dx['ENCOUNTERID'])
        dxf, dxmeta = base.dx_ambiguity_flags(first, dx)
        flags = flags.merge(dxf.drop(columns=['ENCOUNTERID'], errors='ignore'), on='PATID', how='left')
        dx_summary = {'available': True, 'raw': raw_meta, **dxmeta}
    except Exception as e:
        dx_summary = {'available': False, 'error': repr(e)}

    outdir = base.RESULTS / 'geisinger'
    outdir.mkdir(parents=True, exist_ok=True)
    flags.to_csv(outdir/'patient_sensitivity_flags_v4.csv', index=False)
    flags.to_parquet(outdir/'patient_sensitivity_flags_v4.parquet', index=False)

    summary = {
        'site':'geisinger',
        'canonical_first_file':str(first_path),
        'canonical_all_encounters_file':str(all_path),
        'patients':int(first['PATID'].nunique()),
        'phenotype_strict_eligible':int(flags['phenotype_strict_eligible'].sum()),
        'has_alt_qualifying_stroke_encounter':int(flags['has_alt_qualifying_stroke_encounter'].sum()),
        'index_selection_sensitive':int(flags['index_selection_sensitive'].sum()),
        'geisinger_edge_check':edge,
        'dx':dx_summary,
        'validation_errors':errors,
    }
    (outdir/'summary_v4.json').write_text(json.dumps(summary, indent=2, default=str)+'\n')
    print(json.dumps(summary, indent=2, default=str))
    if errors:
        raise RuntimeError('; '.join(errors))


if __name__ == '__main__':
    main()
