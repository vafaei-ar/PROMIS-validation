#!/usr/bin/env python3
"""Prototype scientifically explicit phenotype-sensitivity flags.

This script DOES NOT modify the production pipeline or canonical outputs.
It reads the canonical first/all stroke encounter outputs plus raw diagnosis data,
constructs a strict evidence-qualified alternative index encounter, derives diagnosis
ambiguity flags, and writes patient-level prototype outputs under results/.

Design principles
-----------------
* Preserve the current broad/canonical index encounter.
* Expose whether a stricter imaging+lipid phenotype would select a different index.
* Do not silently replace a valid DX when multiple primary stroke diagnoses exist.
* Provide a deterministic representative DX as an additional field.
* Keep patient-level outputs under results/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
RESULTS = VAL / 'results' / 'sensitivity_flag_prototype'

DEFAULTS = {
    'psu': {
        'canonical_root': VAL / 'reproduced/psu_997d6bd',
        'raw_dx': ROOT / 'datasets/psu/diagnosis.parquet',
    },
    'geisinger': {
        'canonical_root': VAL / 'reproduced/geisinger_997d6bd',
        'raw_dx': ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/dataset/diagnosis.parquet',
    },
}


def norm(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip()


def find_file(root: Path, name: str) -> Path:
    candidates = [root / name, root / 'final' / name, root / 'intermediate' / name]
    for p in candidates:
        if p.exists():
            return p
    hits = list(root.rglob(name)) if root.exists() else []
    if not hits:
        raise FileNotFoundError(f'{name} not found under {root}')
    return hits[0]


def stroke_like(dx: pd.Series) -> pd.Series:
    """Broad ischemic-stroke code rule matching the harmonized phenotype families."""
    x = norm(dx).str.upper().str.replace('.', '', regex=False)
    icd10 = x.str.startswith(('I63', 'H341'), na=False)
    icd9 = (
        x.str.startswith(('433', '434'), na=False)
        & x.str.len().gt(4)
        & x.str[4].eq('1')
    )
    return icd10 | icd9


def primary_mask(df: pd.DataFrame) -> pd.Series:
    if 'PDX' in df.columns:
        x = norm(df['PDX']).str.upper()
        return x.eq('P') | x.eq('1') | x.eq('Y') | x.eq('YES') | x.eq('TRUE')
    if 'DX_PRIMARY_YN' in df.columns:
        x = norm(df['DX_PRIMARY_YN']).str.upper()
        return x.isin({'1', 'Y', 'YES', 'TRUE', 'P'})
    if 'PRIMARY_DX' in df.columns:
        x = norm(df['PRIMARY_DX']).str.upper()
        return x.isin({'1', 'Y', 'YES', 'TRUE', 'P'})
    # If source lacks a primary indicator, do not fabricate one. Return all and
    # make this limitation visible in the summary.
    return pd.Series(True, index=df.index)


def choose_date(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    for c in ['DX_DATE_stroke', 'DX_DATE', 'ADMIT_DATE', 'ENC_DT', 'DISCHARGE_DATE']:
        if c in df.columns:
            result = result.combine_first(pd.to_datetime(df[c], errors='coerce'))
    return result


def load_inputs(site: str, canonical_root: Path, raw_dx: Path):
    first_path = find_file(canonical_root, 'first_stroke_encounters.parquet')
    all_path = find_file(canonical_root, 'all_stroke_encounters.parquet')
    first = pd.read_parquet(first_path).copy()
    all_enc = pd.read_parquet(all_path).copy()
    dx = pd.read_parquet(raw_dx).copy()
    for df in [first, all_enc, dx]:
        if 'PATID' not in df.columns:
            raise KeyError('PATID is required')
        df['PATID'] = norm(df['PATID'])
        if 'ENCOUNTERID' in df.columns:
            df['ENCOUNTERID'] = norm(df['ENCOUNTERID'])
    return first, all_enc, dx, first_path, all_path


def strict_index_flags(first: pd.DataFrame, all_enc: pd.DataFrame) -> pd.DataFrame:
    needed = {'PATID', 'ENCOUNTERID', 'has_neuroimaging', 'has_lipid_panel'}
    missing = needed - set(all_enc.columns)
    if missing:
        raise KeyError(f'all_stroke_encounters missing required fields: {sorted(missing)}')

    cand = all_enc.copy()
    cand['_index_date'] = choose_date(cand)
    cand['_strict_ok'] = (
        pd.to_numeric(cand['has_neuroimaging'], errors='coerce').eq(1)
        & pd.to_numeric(cand['has_lipid_panel'], errors='coerce').eq(1)
    )

    # Stable ordering: clinical date, then encounter ID. This is deterministic and
    # does not change the broad index; it only defines the proposed strict index.
    strict = (
        cand[cand['_strict_ok']]
        .sort_values(['PATID', '_index_date', 'ENCOUNTERID'], na_position='last')
        .drop_duplicates('PATID', keep='first')
    )

    first2 = first.copy()
    first2['_broad_index_date'] = choose_date(first2)
    broad_cols = ['PATID', 'ENCOUNTERID', '_broad_index_date']
    if 'DX' in first2.columns:
        broad_cols.append('DX')
    broad = first2[broad_cols].drop_duplicates('PATID').rename(columns={
        'ENCOUNTERID': 'broad_index_encounter_id',
        '_broad_index_date': 'broad_index_stroke_date',
        'DX': 'broad_index_dx',
    })

    strict_cols = ['PATID', 'ENCOUNTERID', '_index_date']
    if 'DX' in strict.columns:
        strict_cols.append('DX')
    strict_small = strict[strict_cols].rename(columns={
        'ENCOUNTERID': 'strict_index_encounter_id',
        '_index_date': 'strict_index_stroke_date',
        'DX': 'strict_index_dx',
    })

    flags = broad.merge(strict_small, on='PATID', how='left')
    flags['phenotype_strict_eligible'] = flags['strict_index_encounter_id'].notna().astype('int8')
    flags['index_selection_sensitive'] = (
        flags['strict_index_encounter_id'].notna()
        & flags['strict_index_encounter_id'].ne(flags['broad_index_encounter_id'])
    ).astype('int8')

    strict_pairs = cand.loc[cand['_strict_ok'], ['PATID', 'ENCOUNTERID']].drop_duplicates()
    broad_pairs = broad[['PATID', 'broad_index_encounter_id']].rename(columns={'broad_index_encounter_id': 'ENCOUNTERID'})
    alt = strict_pairs.merge(broad_pairs.assign(_broad=1), on=['PATID', 'ENCOUNTERID'], how='left')
    alt_patients = set(alt.loc[alt['_broad'].isna(), 'PATID'])
    flags['has_alt_qualifying_stroke_encounter'] = flags['PATID'].isin(alt_patients).astype('int8')
    return flags


def dx_ambiguity_flags(first: pd.DataFrame, raw_dx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if 'DX' not in raw_dx.columns or 'ENCOUNTERID' not in raw_dx.columns:
        raise KeyError('raw diagnosis requires DX and ENCOUNTERID')

    selected = first[['PATID', 'ENCOUNTERID']].drop_duplicates()
    x = raw_dx.merge(selected, on=['PATID', 'ENCOUNTERID'], how='inner')
    x = x[stroke_like(x['DX']) & primary_mask(x)].copy()
    x['_dx_norm'] = norm(x['DX']).str.upper()
    x['_dx_date'] = choose_date(x)

    # Collapse exact duplicate diagnosis records, then count distinct qualifying codes.
    unique = x[['PATID', 'ENCOUNTERID', '_dx_norm', '_dx_date']].drop_duplicates()
    counts = (
        unique.groupby(['PATID', 'ENCOUNTERID'], as_index=False)
        .agg(n_qualifying_primary_stroke_dx=('_dx_norm', 'nunique'))
    )

    # Deterministic representative: earliest available diagnosis date, then
    # lexical normalized code as a stable final tie-break. This is a companion
    # field and does not overwrite the canonical DX.
    chosen = (
        unique.sort_values(['PATID', 'ENCOUNTERID', '_dx_date', '_dx_norm'], na_position='last')
        .drop_duplicates(['PATID', 'ENCOUNTERID'], keep='first')
        [['PATID', 'ENCOUNTERID', '_dx_norm']]
        .rename(columns={'_dx_norm': 'dx_deterministic'})
    )

    out = selected.merge(counts, on=['PATID', 'ENCOUNTERID'], how='left').merge(
        chosen, on=['PATID', 'ENCOUNTERID'], how='left'
    )
    out['n_qualifying_primary_stroke_dx'] = out['n_qualifying_primary_stroke_dx'].fillna(0).astype('int16')
    out['multiple_primary_stroke_dx'] = out['n_qualifying_primary_stroke_dx'].gt(1).astype('int8')
    out['dx_tiebreak_applied'] = out['multiple_primary_stroke_dx'].copy()

    current = first[['PATID', 'ENCOUNTERID'] + (['DX'] if 'DX' in first.columns else [])].drop_duplicates('PATID')
    if 'DX' in current.columns:
        current['DX'] = norm(current['DX']).str.upper()
        out = out.merge(current, on=['PATID', 'ENCOUNTERID'], how='left')
        out['dx_deterministic_differs_from_current'] = (
            out['dx_deterministic'].notna() & out['DX'].notna() & out['dx_deterministic'].ne(out['DX'])
        ).astype('int8')
    else:
        out['dx_deterministic_differs_from_current'] = 0

    meta = {
        'raw_primary_indicator_present': bool(any(c in raw_dx.columns for c in ['PDX', 'DX_PRIMARY_YN', 'PRIMARY_DX'])),
        'selected_encounters_with_qualifying_raw_dx': int(out['n_qualifying_primary_stroke_dx'].gt(0).sum()),
        'multiple_primary_stroke_dx_patients': int(out['multiple_primary_stroke_dx'].sum()),
        'deterministic_dx_differs_from_current': int(out['dx_deterministic_differs_from_current'].sum()),
    }
    return out, meta


def validate_flags(flags: pd.DataFrame, first: pd.DataFrame) -> list[str]:
    errors = []
    if flags['PATID'].duplicated().any():
        errors.append('patient flags are not one row per PATID')
    if len(flags) != first['PATID'].nunique():
        errors.append('patient flag row count does not equal canonical patient count')
    bad = flags['phenotype_strict_eligible'].eq(0) & flags['strict_index_encounter_id'].notna()
    if bad.any():
        errors.append('strict_index_encounter_id populated for non-eligible patient')
    expected_sensitive = (
        flags['strict_index_encounter_id'].notna()
        & flags['strict_index_encounter_id'].ne(flags['broad_index_encounter_id'])
    )
    if not flags['index_selection_sensitive'].eq(expected_sensitive.astype('int8')).all():
        errors.append('index_selection_sensitive inconsistent with encounter IDs')
    return errors


def optional_geisinger_edge_check(flags: pd.DataFrame) -> dict:
    edge_path = VAL / 'results/source_truth_audit/geisinger_edges/edge_patients.csv'
    aabila_root = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs'
    if not edge_path.exists() or not aabila_root.exists():
        return {'available': False}
    edges = pd.read_csv(edge_path, low_memory=False)
    idc = '__id' if '__id' in edges.columns else ('patient_id' if 'patient_id' in edges.columns else None)
    if not idc:
        return {'available': False}
    edge_ids = set(norm(edges[idc]))
    try:
        aabila = pd.read_parquet(find_file(aabila_root, 'stroke_cohort_final.parquet'))
    except Exception:
        try:
            aabila = pd.read_parquet(find_file(aabila_root, 'first_stroke_encounters.parquet'))
        except Exception:
            return {'available': False}
    aabila['PATID'] = norm(aabila['PATID'])
    if 'ENCOUNTERID' not in aabila.columns:
        return {'available': False}
    aabila['ENCOUNTERID'] = norm(aabila['ENCOUNTERID'])
    cmp = flags[flags['PATID'].isin(edge_ids)].merge(
        aabila[['PATID', 'ENCOUNTERID']].drop_duplicates('PATID').rename(columns={'ENCOUNTERID': 'aabila_index_encounter_id'}),
        on='PATID', how='left'
    )
    return {
        'available': True,
        'edge_patients': int(cmp['PATID'].nunique()),
        'strict_index_matches_aabila': int(cmp['strict_index_encounter_id'].eq(cmp['aabila_index_encounter_id']).sum()),
        'broad_index_matches_aabila': int(cmp['broad_index_encounter_id'].eq(cmp['aabila_index_encounter_id']).sum()),
        'index_selection_sensitive': int(cmp['index_selection_sensitive'].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', choices=['psu', 'geisinger'], required=True)
    ap.add_argument('--canonical-root', type=Path)
    ap.add_argument('--raw-dx', type=Path)
    args = ap.parse_args()

    site = args.site
    canonical_root = args.canonical_root or DEFAULTS[site]['canonical_root']
    raw_dx = args.raw_dx or DEFAULTS[site]['raw_dx']
    outdir = RESULTS / site
    outdir.mkdir(parents=True, exist_ok=True)

    first, all_enc, dx, first_path, all_path = load_inputs(site, canonical_root, raw_dx)
    strict = strict_index_flags(first, all_enc)
    dxf, dxmeta = dx_ambiguity_flags(first, dx)
    flags = strict.merge(dxf.drop(columns=['ENCOUNTERID'], errors='ignore'), on='PATID', how='left')

    errors = validate_flags(flags, first)
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
        summary['geisinger_edge_check'] = optional_geisinger_edge_check(flags)

    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))
    if errors:
        raise SystemExit('Validation failed: ' + '; '.join(errors))


if __name__ == '__main__':
    main()
