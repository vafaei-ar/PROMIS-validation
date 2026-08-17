#!/usr/bin/env python3
"""Build source-of-truth discrepancy audit tables for PROMIS validation.

Initial scope:
  * Geisinger DX_DATE_stroke and DX discordance on aligned patients
  * Geisinger 167 Aabila-only imaging/lipid edge patients
  * PSU DX discordance on registry-aligned patients

This script does not adjudicate automatically. It assembles canonical values,
Aabila values, key encounter fields, phenotype evidence flags, and any nearby
intermediate/raw-support rows it can locate. Patient-level outputs remain under
results/ and must not be committed.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
OUT = VAL / 'results/source_truth_audit'

CANON = {
    'psu': VAL / 'reproduced/psu_997d6bd',
    'geisinger': VAL / 'reproduced/geisinger_997d6bd',
}
AABILA = {
    'psu': ROOT / 'aabila/PROMIS-ML-PennState/outputs',
    'geisinger': ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs',
}
AABILA_GEI_ROOT = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'

KEYS = ('PATID','STUDY_ID','PT_ID','patient_id','PatientID')


def first_existing(root: Path, name: str) -> Path | None:
    for p in (root/name, root/'final'/name, root/'intermediate'/name):
        if p.exists():
            return p
    return None


def key(df: pd.DataFrame) -> str:
    for c in KEYS:
        if c in df.columns:
            return c
    raise KeyError(f'No patient identifier found among {KEYS}')


def norm(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip()


def load_final(site: str, who: str) -> pd.DataFrame:
    root = CANON[site] if who == 'canonical' else AABILA[site]
    p = first_existing(root, 'stroke_cohort_final.parquet')
    if p is None:
        raise FileNotFoundError(p)
    df = pd.read_parquet(p).copy()
    k = key(df)
    df['__id'] = norm(df[k])
    return df


def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def geisinger_dx_audit() -> dict:
    c = load_final('geisinger', 'canonical')
    a = load_final('geisinger', 'aabila')

    # keep canonical patients satisfying the rule-aligned imaging+lipid criteria
    aligned = c[(c.get('has_neuroimaging', 0) == 1) & (c.get('has_lipid_panel', 0) == 1)].copy()
    a_dup = set(a.loc[a['__id'].duplicated(keep=False), '__id'])
    if a_dup:
        a = a[~a['__id'].isin(a_dup)].copy()
        aligned = aligned[~aligned['__id'].isin(a_dup)].copy()

    ids = sorted(set(aligned['__id']) & set(a['__id']))
    cs = aligned[aligned['__id'].isin(ids)].set_index('__id').sort_index()
    aa = a[a['__id'].isin(ids)].set_index('__id').sort_index()

    rows = []
    candidate_cols = [
        'DX','DX_DATE_stroke','ENCOUNTERID','ADMIT_DATE','DISCHARGE_DATE','los_days',
        'has_neuroimaging','has_lipid_panel','HAS_IMAGING','HAS_LIPID','ENC_TYPE','AGE'
    ]
    for pid in ids:
        cdx = cs.at[pid, 'DX'] if 'DX' in cs.columns else pd.NA
        adx = aa.at[pid, 'DX'] if 'DX' in aa.columns else pd.NA
        cdt = cs.at[pid, 'DX_DATE_stroke'] if 'DX_DATE_stroke' in cs.columns else pd.NA
        adt = aa.at[pid, 'DX_DATE_stroke'] if 'DX_DATE_stroke' in aa.columns else pd.NA
        dx_diff = (pd.isna(cdx) != pd.isna(adx)) or (not pd.isna(cdx) and not pd.isna(adx) and str(cdx).strip() != str(adx).strip())
        date_diff = (pd.isna(cdt) != pd.isna(adt)) or (not pd.isna(cdt) and not pd.isna(adt) and str(cdt).strip() != str(adt).strip())
        if not (dx_diff or date_diff):
            continue
        rec = {
            'patient_id': pid,
            'site': 'geisinger',
            'dx_diff': dx_diff,
            'dx_date_diff': date_diff,
            'canonical_DX': cdx,
            'aabila_DX': adx,
            'canonical_DX_DATE_stroke': cdt,
            'aabila_DX_DATE_stroke': adt,
            'adjudication': 'needs_review',
            'reason': '',
        }
        for col in candidate_cols:
            if col in cs.columns: rec[f'canonical_{col}'] = cs.at[pid, col]
            if col in aa.columns: rec[f'aabila_{col}'] = aa.at[pid, col]
        rows.append(rec)

    out = OUT/'geisinger_dx'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'discrepancy_audit.csv', index=False)
    return {'rows': len(rows), 'duplicate_ids_excluded': len(a_dup)}


def geisinger_edge_audit() -> dict:
    c = load_final('geisinger', 'canonical')
    a = load_final('geisinger', 'aabila')
    c_ids = set(c['__id'])
    aligned_ids = set(c.loc[(c.get('has_neuroimaging',0)==1)&(c.get('has_lipid_panel',0)==1), '__id'])
    edge_ids = sorted(set(a['__id']) - aligned_ids)

    acols = safe_cols(a, ['__id','DX','DX_DATE_stroke','ENCOUNTERID','ADMIT_DATE','DISCHARGE_DATE','HAS_IMAGING','HAS_LIPID'])
    ccols = safe_cols(c, ['__id','DX','DX_DATE_stroke','ENCOUNTERID','ADMIT_DATE','DISCHARGE_DATE','has_neuroimaging','has_lipid_panel'])
    ae = a[a['__id'].isin(edge_ids)][acols].copy()
    ce = c[c['__id'].isin(edge_ids)][ccols].copy()
    merged = ae.merge(ce, on='__id', how='left', suffixes=('_aabila','_canonical'))
    merged['in_canonical_full_cohort'] = merged['__id'].isin(c_ids)
    merged['adjudication'] = 'needs_review'
    merged['reason'] = ''

    out = OUT/'geisinger_edges'
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out/'edge_patients.csv', index=False)

    # Search likely intermediate outputs from both pipelines for those IDs.
    search_roots = [CANON['geisinger'], AABILA['geisinger'], AABILA_GEI_ROOT/'outputs']
    found = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob('*.parquet'):
            if p.name in {'stroke_cohort_final.parquet','stroke_cohort_imputed.parquet'}:
                continue
            try:
                df = pd.read_parquet(p)
                try: k = key(df)
                except Exception: continue
                ids = norm(df[k])
                hit = df[ids.isin(edge_ids)].copy()
                if hit.empty: continue
                hit.insert(0, '__source_file', str(p))
                hit.insert(1, '__id', norm(hit[k]))
                # cap width but preserve fields most likely useful
                keep = ['__source_file','__id'] + safe_cols(hit, ['DX','DX_DATE','DX_DATE_stroke','ENCOUNTERID','ADMIT_DATE','DISCHARGE_DATE','ENC_TYPE','HAS_IMAGING','HAS_LIPID','has_neuroimaging','has_lipid_panel'])
                found.append(hit[keep].drop_duplicates())
            except Exception:
                continue
    if found:
        pd.concat(found, ignore_index=True).to_csv(out/'intermediate_support_rows.csv', index=False)
    else:
        pd.DataFrame(columns=['__source_file','__id']).to_csv(out/'intermediate_support_rows.csv', index=False)
    return {'edge_ids': len(edge_ids), 'rows': len(merged)}


def psu_dx_audit() -> dict:
    c = load_final('psu', 'canonical')
    a = load_final('psu', 'aabila')
    c = c[c.get('in_stroke_registry', 0) == 1].copy()
    ids = sorted(set(c['__id']) & set(a['__id']))
    cs = c[c['__id'].isin(ids)].set_index('__id').sort_index()
    aa = a[a['__id'].isin(ids)].set_index('__id').sort_index()
    rows = []
    for pid in ids:
        if 'DX' not in cs.columns or 'DX' not in aa.columns:
            break
        cv, av = cs.at[pid,'DX'], aa.at[pid,'DX']
        diff = (pd.isna(cv) != pd.isna(av)) or (not pd.isna(cv) and not pd.isna(av) and str(cv).strip()!=str(av).strip())
        if not diff: continue
        rec = {'patient_id':pid,'site':'psu','canonical_DX':cv,'aabila_DX':av,'adjudication':'needs_review','reason':''}
        for col in ['DX_DATE_stroke','ENCOUNTERID','ADMIT_DATE','DISCHARGE_DATE','in_stroke_registry','synthetic_registry_dx','ehr_stroke_dx_present']:
            if col in cs.columns: rec[f'canonical_{col}']=cs.at[pid,col]
            if col in aa.columns: rec[f'aabila_{col}']=aa.at[pid,col]
        rows.append(rec)
    out = OUT/'psu_dx'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'discrepancy_audit.csv', index=False)
    return {'rows': len(rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'geisinger_dx': geisinger_dx_audit(),
        'geisinger_edges': geisinger_edge_audit(),
        'psu_dx': psu_dx_audit(),
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {OUT.resolve()}')

if __name__ == '__main__':
    main()
