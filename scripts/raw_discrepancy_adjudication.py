#!/usr/bin/env python3
"""Independent raw-record adjudication for PROMIS validation discrepancies.

Scope
-----
1) Geisinger DX disagreements among the imaging+lipid aligned cohort.
2) PSU DX disagreements among registry-aligned patients.
3) Geisinger Aabila-only edge patients caused by imaging/lipid/index-encounter logic.

The script intentionally does NOT call production phenotype-selection functions.
It searches raw/intermediate source files, extracts records for discordant patients,
and produces local evidence tables for manual/independent adjudication.

Patient-level outputs remain under results/ and must not be committed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
OUT = VAL / 'results/raw_discrepancy_adjudication'

CANON = {
    'psu': VAL / 'reproduced/psu_997d6bd',
    'geisinger': VAL / 'reproduced/geisinger_997d6bd',
}
AABILA = {
    'psu': ROOT / 'aabila/PROMIS-ML-PennState/outputs',
    'geisinger': ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs',
}
AABILA_GEI_ROOT = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
CANON_PIPE = ROOT / 'codes/PROMIS-ML-pipeline'

KEYS = ('PATID','STUDY_ID','PT_ID','patient_id','PatientID','PERSON_ID','MRN')
DX_HINTS = ('dx','diagnos','condition')
ENC_HINTS = ('enc','visit','admit','discharge')
IMG_HINTS = ('imaging','image','radiol','ct','mri','procedure','proc')
LAB_HINTS = ('lab','lipid','chol','ldl','hdl','trig')


def first_existing(root: Path, name: str) -> Path:
    for p in (root/name, root/'final'/name, root/'intermediate'/name):
        if p.exists():
            return p
    raise FileNotFoundError(f'{name} not found under {root}')


def key(df: pd.DataFrame) -> str | None:
    for c in KEYS:
        if c in df.columns:
            return c
    return None


def norm(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip()


def load_final(site: str, who: str) -> pd.DataFrame:
    root = CANON[site] if who == 'canonical' else AABILA[site]
    df = pd.read_parquet(first_existing(root, 'stroke_cohort_final.parquet')).copy()
    k = key(df)
    if k is None:
        raise KeyError(f'No patient ID in final file for {site}/{who}')
    df['__id'] = norm(df[k])
    return df


def values_differ(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return False
    if pd.isna(a) != pd.isna(b):
        return True
    return str(a).strip() != str(b).strip()


def aligned_geisinger() -> tuple[pd.DataFrame,pd.DataFrame,set[str]]:
    c = load_final('geisinger','canonical')
    a = load_final('geisinger','aabila')
    c = c[(c.get('has_neuroimaging',0)==1) & (c.get('has_lipid_panel',0)==1)].copy()
    dup = set(a.loc[a['__id'].duplicated(keep=False),'__id'])
    if dup:
        c = c[~c['__id'].isin(dup)].copy()
        a = a[~a['__id'].isin(dup)].copy()
    ids = set(c['__id']) & set(a['__id'])
    return c[c['__id'].isin(ids)].copy(), a[a['__id'].isin(ids)].copy(), ids


def discordant_dx_ids(site: str) -> set[str]:
    if site == 'geisinger':
        c,a,_ = aligned_geisinger()
    else:
        c = load_final('psu','canonical')
        a = load_final('psu','aabila')
        c = c[c.get('in_stroke_registry',0)==1].copy()
        ids = set(c['__id']) & set(a['__id'])
        c = c[c['__id'].isin(ids)]
        a = a[a['__id'].isin(ids)]
    cs = c.drop_duplicates('__id').set_index('__id')
    aa = a.drop_duplicates('__id').set_index('__id')
    ids = set(cs.index) & set(aa.index)
    return {pid for pid in ids if 'DX' in cs.columns and 'DX' in aa.columns and values_differ(cs.at[pid,'DX'],aa.at[pid,'DX'])}


def geisinger_edge_ids() -> set[str]:
    c = load_final('geisinger','canonical')
    a = load_final('geisinger','aabila')
    aligned = set(c.loc[(c.get('has_neuroimaging',0)==1)&(c.get('has_lipid_panel',0)==1),'__id'])
    return set(a['__id']) - aligned


def candidate_files(roots: Iterable[Path], hints: tuple[str,...]) -> list[Path]:
    seen = set()
    out = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ('*.parquet','*.csv'):
            for p in root.rglob(ext):
                low = str(p).lower()
                if not any(h in low for h in hints):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                out.append(p)
    return sorted(out)


def read_source(p: Path) -> pd.DataFrame | None:
    try:
        if p.suffix.lower() == '.parquet':
            return pd.read_parquet(p)
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return None


def extract_hits(files: list[Path], ids: set[str], category: str, max_files: int = 250) -> tuple[pd.DataFrame,pd.DataFrame]:
    hits = []
    manifest = []
    for p in files[:max_files]:
        df = read_source(p)
        if df is None:
            manifest.append({'category':category,'file':str(p),'status':'read_failed','rows':None,'hits':None})
            continue
        k = key(df)
        if k is None:
            manifest.append({'category':category,'file':str(p),'status':'no_patient_key','rows':len(df),'hits':0})
            continue
        sid = norm(df[k])
        mask = sid.isin(ids)
        n = int(mask.sum())
        manifest.append({'category':category,'file':str(p),'status':'ok','rows':len(df),'hits':n,'key':k})
        if n == 0:
            continue
        h = df.loc[mask].copy()
        h.insert(0,'__source_file',str(p))
        h.insert(1,'__category',category)
        h.insert(2,'__id',norm(h[k]))
        hits.append(h)
    if not hits:
        return pd.DataFrame(), pd.DataFrame(manifest)
    return pd.concat(hits,ignore_index=True,sort=False), pd.DataFrame(manifest)


def compact_columns(df: pd.DataFrame, category: str) -> pd.DataFrame:
    if df.empty:
        return df
    always = ['__source_file','__category','__id']
    category_terms = {
        'dx': ('dx','diag','code','type','position','primary','date','enc','admit','discharge','source','poa'),
        'encounter': ('enc','visit','admit','discharge','type','los','date'),
        'imaging': ('image','imaging','procedure','proc','code','date','enc','ct','mri','cpt','hcpcs'),
        'lab': ('lab','lipid','chol','ldl','hdl','trig','result','value','unit','date','enc','name'),
    }
    terms = category_terms[category]
    keep = list(always)
    for c in df.columns:
        lc = str(c).lower()
        if c not in keep and any(t in lc for t in terms):
            keep.append(c)
    # preserve a bounded amount of contextual data if naming is unusual
    if len(keep) < 8:
        for c in df.columns:
            if c not in keep:
                keep.append(c)
            if len(keep) >= 20:
                break
    return df[keep]


def build_final_reference(site: str, ids: set[str]) -> pd.DataFrame:
    c = load_final(site,'canonical')
    a = load_final(site,'aabila')
    cols = ['__id','DX','DX_DATE_stroke','ENCOUNTERID','ADMIT_DATE','DISCHARGE_DATE','ENC_TYPE',
            'has_neuroimaging','has_lipid_panel','HAS_IMAGING','HAS_LIPID']
    cc = c[c['__id'].isin(ids)][[x for x in cols if x in c.columns]].drop_duplicates('__id')
    aa = a[a['__id'].isin(ids)][[x for x in cols if x in a.columns]].drop_duplicates('__id')
    return cc.merge(aa,on='__id',how='outer',suffixes=('_canonical','_aabila'))


def audit_dx(site: str) -> dict:
    ids = discordant_dx_ids(site)
    od = OUT/f'{site}_dx'
    od.mkdir(parents=True,exist_ok=True)
    build_final_reference(site,ids).to_csv(od/'final_reference.csv',index=False)

    if site == 'geisinger':
        roots = [AABILA_GEI_ROOT/'dataset', AABILA_GEI_ROOT/'outputs', CANON['geisinger'], CANON_PIPE]
    else:
        roots = [ROOT/'aabila/PROMIS-ML-PennState', CANON['psu'], CANON_PIPE]

    files = candidate_files(roots, DX_HINTS + ENC_HINTS)
    raw, manifest = extract_hits(files,ids,'dx')
    manifest.to_csv(od/'source_manifest.csv',index=False)
    compact_columns(raw,'dx').to_csv(od/'raw_dx_support.csv',index=False)

    # Patient-level summary: where each final DX appears in extracted source records.
    ref = build_final_reference(site,ids)
    rows = []
    if not raw.empty:
        text_cols = [c for c in raw.columns if raw[c].dtype == 'object' or pd.api.types.is_string_dtype(raw[c])]
        grouped = {pid:g for pid,g in raw.groupby('__id')}
    else:
        text_cols, grouped = [], {}
    for _,r in ref.iterrows():
        pid = r['__id']
        g = grouped.get(pid)
        rec = {'patient_id':pid,'site':site,'canonical_DX':r.get('DX_canonical'), 'aabila_DX':r.get('DX_aabila')}
        for label,val in [('canonical',r.get('DX_canonical')),('aabila',r.get('DX_aabila'))]:
            found = False
            files_found = set()
            if g is not None and pd.notna(val):
                target = str(val).strip().lower()
                for c in text_cols:
                    s = g[c].astype('string').str.strip().str.lower()
                    m = s.eq(target)
                    if m.any():
                        found = True
                        files_found.update(g.loc[m,'__source_file'].astype(str).tolist())
            rec[f'{label}_dx_found_in_extracted_sources'] = found
            rec[f'{label}_support_file_count'] = len(files_found)
        rec['adjudication'] = 'needs_review'
        rec['reason'] = ''
        rows.append(rec)
    pd.DataFrame(rows).to_csv(od/'dx_support_summary.csv',index=False)
    return {'discordant_patients':len(ids),'candidate_files_scanned':len(manifest),'source_rows_extracted':len(raw)}


def audit_geisinger_edges() -> dict:
    ids = geisinger_edge_ids()
    od = OUT/'geisinger_edges'
    od.mkdir(parents=True,exist_ok=True)
    build_final_reference('geisinger',ids).to_csv(od/'final_reference.csv',index=False)
    roots = [AABILA_GEI_ROOT/'dataset', AABILA_GEI_ROOT/'outputs', CANON['geisinger'], CANON_PIPE]

    results = {}
    for category,hints in [('encounter',ENC_HINTS),('imaging',IMG_HINTS),('lab',LAB_HINTS)]:
        files = candidate_files(roots,hints)
        raw, manifest = extract_hits(files,ids,category)
        manifest.to_csv(od/f'{category}_source_manifest.csv',index=False)
        compact_columns(raw,category).to_csv(od/f'raw_{category}_support.csv',index=False)
        results[category] = {'candidate_files_scanned':len(manifest),'source_rows_extracted':len(raw)}

    # Produce a compact encounter-alignment table suitable for manual adjudication.
    ref = build_final_reference('geisinger',ids)
    ref['same_final_encounter'] = (
        ref.get('ENCOUNTERID_canonical',pd.Series(index=ref.index,dtype='object')).astype('string') ==
        ref.get('ENCOUNTERID_aabila',pd.Series(index=ref.index,dtype='object')).astype('string')
    )
    ref['adjudication'] = 'needs_review'
    ref['reason'] = ''
    ref.to_csv(od/'edge_adjudication.csv',index=False)
    return {'edge_patients':len(ids), **results}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    summary = {
        'geisinger_dx': audit_dx('geisinger'),
        'psu_dx': audit_dx('psu'),
        'geisinger_edges': audit_geisinger_edges(),
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(f'Outputs: {OUT.resolve()}')


if __name__ == '__main__':
    main()
