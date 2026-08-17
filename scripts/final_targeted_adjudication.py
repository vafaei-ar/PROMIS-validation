#!/usr/bin/env python3
"""Final targeted adjudication for remaining PROMIS validation discrepancies.

Scope
-----
1. Geisinger DX discordance: reconstruct all diagnosis rows on the selected
   encounter for discordant patients and characterize deterministic tie-breaking.
2. Geisinger 167 edge patients: reconstruct candidate encounters and attach raw
   imaging/lipid evidence so order-of-operations can be adjudicated.
3. PSU DX discordance: search raw PSU diagnosis-like source files directly and
   extract support for both canonical and Aabila codes on the selected encounter.

This script deliberately avoids importing production phenotype-selection code.
Patient-level outputs remain under results/ and must not be committed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
OUT = VAL / 'results/final_targeted_adjudication'

CANON_GEI = VAL / 'reproduced/geisinger_997d6bd'
CANON_PSU = VAL / 'reproduced/psu_997d6bd'
AABILA_GEI = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
AABILA_PSU = ROOT / 'aabila/PROMIS-ML-PennState'
PIPELINE = ROOT / 'codes/PROMIS-ML-pipeline'

KEY_CANDIDATES = ('PATID','STUDY_ID','PT_ID','patient_id','PatientID')
ENCOUNTER_CANDIDATES = ('ENCOUNTERID','ENCOUNTER_ID','ENC_ID','ENCID')
DX_CANDIDATES = ('DX','DX_CODE','DIAGNOSIS_CODE','ICD_CODE','CODE')
DATE_CANDIDATES = ('DX_DATE','DX_DATE_stroke','ADMIT_DATE','ENC_DT','ENC_ADM_DTTM','DATE')
PRIMARY_CANDIDATES = ('DX_TYPE','DIAGNOSIS_TYPE','PRIMARY_DX','IS_PRIMARY','DX_POSITION','PDX')


def first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        for p in (root/name, root/'outputs'/name, root/'final'/name, root/'intermediate'/name):
            if p.exists():
                return p
    return None


def load_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == '.parquet':
        return pd.read_parquet(path)
    if suf in {'.csv', '.txt'}:
        return pd.read_csv(path, low_memory=False)
    if suf in {'.tsv'}:
        return pd.read_csv(path, sep='\t', low_memory=False)
    raise ValueError(path)


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    upper = {str(c).upper(): c for c in df.columns}
    for c in candidates:
        if c.upper() in upper:
            return upper[c.upper()]
    return None


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip()


def norm_code(v):
    if pd.isna(v):
        return pd.NA
    return re.sub(r'[^A-Z0-9]', '', str(v).upper().strip())


def norm_date(v):
    if pd.isna(v):
        return pd.NaT
    return pd.to_datetime(v, errors='coerce').normalize()


def load_final(root: Path) -> pd.DataFrame:
    p = first_existing(root, ['stroke_cohort_final.parquet'])
    if p is None:
        raise FileNotFoundError(f'Cannot find final cohort under {root}')
    df = pd.read_parquet(p).copy()
    k = find_col(df, KEY_CANDIDATES)
    if not k:
        raise KeyError(f'No patient key in {p}')
    df['__id'] = norm_id(df[k])
    return df


def diagnosis_like_files(root: Path, limit_mb: int = 500) -> list[Path]:
    out = []
    if not root.exists():
        return out
    pats = ('dx','diagn','stroke','encounter')
    for ext in ('*.parquet','*.csv','*.tsv'):
        for p in root.rglob(ext):
            low = p.name.lower()
            if not any(x in low for x in pats):
                continue
            try:
                if p.stat().st_size > limit_mb * 1024 * 1024:
                    continue
            except OSError:
                continue
            out.append(p)
    return sorted(set(out))


def extract_dx_support(files: list[Path], target_ids: set[str], target_encounters: set[str] | None = None) -> tuple[pd.DataFrame,list[dict]]:
    frames = []
    inventory = []
    for p in files:
        try:
            df = load_any(p)
        except Exception as e:
            inventory.append({'file':str(p),'status':'read_error','detail':str(e)[:200]})
            continue
        k = find_col(df, KEY_CANDIDATES)
        dxc = find_col(df, DX_CANDIDATES)
        if not k or not dxc:
            inventory.append({'file':str(p),'status':'skip_no_key_or_dx','detail':''})
            continue
        ids = norm_id(df[k])
        hit = df[ids.isin(target_ids)].copy()
        if hit.empty:
            inventory.append({'file':str(p),'status':'no_target_rows','detail':''})
            continue
        hit['__id'] = norm_id(hit[k])
        ec = find_col(hit, ENCOUNTER_CANDIDATES)
        if target_encounters is not None and ec:
            encs = norm_id(hit[ec])
            hit = hit[encs.isin(target_encounters)].copy()
            if hit.empty:
                inventory.append({'file':str(p),'status':'target_ids_but_no_target_encounter','detail':''})
                continue
        keep = ['__id', dxc]
        for cands in (ENCOUNTER_CANDIDATES, DATE_CANDIDATES, PRIMARY_CANDIDATES):
            c = find_col(hit, cands)
            if c and c not in keep:
                keep.append(c)
        # preserve original row order explicitly
        hit = hit.reset_index().rename(columns={'index':'__source_row'})
        hit['__source_file'] = str(p)
        hit['__dx_norm'] = hit[dxc].map(norm_code)
        cols = ['__source_file','__source_row','__id','__dx_norm'] + keep[1:]
        frames.append(hit[cols])
        inventory.append({'file':str(p),'status':'used','detail':f'{len(hit)} rows'})
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False), inventory
    return pd.DataFrame(), inventory


def geisinger_dx_tiebreak() -> dict:
    c = load_final(CANON_GEI)
    a = load_final(AABILA_GEI / 'outputs')
    c = c[(c.get('has_neuroimaging',0)==1) & (c.get('has_lipid_panel',0)==1)].copy()
    dup = set(a.loc[a['__id'].duplicated(keep=False),'__id'])
    if dup:
        c = c[~c['__id'].isin(dup)]
        a = a[~a['__id'].isin(dup)]
    ids = sorted(set(c['__id']) & set(a['__id']))
    cs = c[c['__id'].isin(ids)].set_index('__id')
    aa = a[a['__id'].isin(ids)].set_index('__id')
    discord = []
    encs = set()
    for pid in ids:
        cv = cs.at[pid,'DX'] if 'DX' in cs.columns else pd.NA
        av = aa.at[pid,'DX'] if 'DX' in aa.columns else pd.NA
        if norm_code(cv) == norm_code(av):
            continue
        ce = str(cs.at[pid,'ENCOUNTERID']).strip() if 'ENCOUNTERID' in cs.columns else ''
        ae = str(aa.at[pid,'ENCOUNTERID']).strip() if 'ENCOUNTERID' in aa.columns else ''
        if ce: encs.add(ce)
        if ae: encs.add(ae)
        discord.append({'patient_id':pid,'canonical_DX':cv,'aabila_DX':av,'canonical_ENCOUNTERID':ce,'aabila_ENCOUNTERID':ae})
    disc = pd.DataFrame(discord)
    out = OUT/'geisinger_dx_tiebreak'
    out.mkdir(parents=True, exist_ok=True)
    disc.to_csv(out/'discordant_patients.csv', index=False)

    files = diagnosis_like_files(AABILA_GEI/'dataset') + diagnosis_like_files(AABILA_GEI/'outputs')
    support, inventory = extract_dx_support(files, set(disc.patient_id.astype(str)), encs)
    pd.DataFrame(inventory).to_csv(out/'source_inventory.csv', index=False)
    support.to_csv(out/'raw_dx_rows.csv', index=False)

    adjud = []
    if not support.empty:
        ec = find_col(support, ENCOUNTER_CANDIDATES)
        for r in discord:
            pid = r['patient_id']
            rows = support[support['__id'].astype(str)==str(pid)].copy()
            if ec and r['canonical_ENCOUNTERID']:
                rows = rows[norm_id(rows[ec])==r['canonical_ENCOUNTERID']]
            cn = norm_code(r['canonical_DX']); an = norm_code(r['aabila_DX'])
            cr = rows[rows['__dx_norm']==cn]
            ar = rows[rows['__dx_norm']==an]
            cfirst = cr['__source_row'].min() if not cr.empty else pd.NA
            afirst = ar['__source_row'].min() if not ar.empty else pd.NA
            adjud.append({
                **r,
                'canonical_raw_supported': not cr.empty,
                'aabila_raw_supported': not ar.empty,
                'canonical_first_source_row': cfirst,
                'aabila_first_source_row': afirst,
                'canonical_appears_first': (pd.notna(cfirst) and pd.notna(afirst) and cfirst < afirst),
                'aabila_appears_first': (pd.notna(cfirst) and pd.notna(afirst) and afirst < cfirst),
                'same_selected_encounter': r['canonical_ENCOUNTERID']==r['aabila_ENCOUNTERID'],
            })
    adf = pd.DataFrame(adjud)
    adf.to_csv(out/'tiebreak_adjudication.csv', index=False)
    summary = {
        'discordant_patients': len(disc),
        'raw_support_rows': len(support),
        'both_codes_raw_supported': int(((adf.get('canonical_raw_supported',False)==True)&(adf.get('aabila_raw_supported',False)==True)).sum()) if not adf.empty else 0,
        'canonical_appears_first': int(adf.get('canonical_appears_first',pd.Series(dtype=bool)).sum()) if not adf.empty else 0,
        'aabila_appears_first': int(adf.get('aabila_appears_first',pd.Series(dtype=bool)).sum()) if not adf.empty else 0,
        'same_selected_encounter': int(adf.get('same_selected_encounter',pd.Series(dtype=bool)).sum()) if not adf.empty else 0,
    }
    return summary


def geisinger_edges() -> dict:
    c = load_final(CANON_GEI)
    a = load_final(AABILA_GEI/'outputs')
    aligned = set(c.loc[(c.get('has_neuroimaging',0)==1)&(c.get('has_lipid_panel',0)==1),'__id'])
    edge = sorted(set(a['__id']) - aligned)
    out = OUT/'geisinger_edges'
    out.mkdir(parents=True, exist_ok=True)

    ae = a[a['__id'].isin(edge)].copy()
    ce = c[c['__id'].isin(edge)].copy()
    ae.to_csv(out/'aabila_edge_final_rows.csv', index=False)
    ce.to_csv(out/'canonical_edge_final_rows.csv', index=False)

    # Pull raw candidate encounters and likely evidence tables without applying production logic.
    raw_root = AABILA_GEI/'dataset'
    wanted = set(edge)
    candidates = []
    inventory = []
    for p in sorted(raw_root.rglob('*')) if raw_root.exists() else []:
        if not p.is_file() or p.suffix.lower() not in {'.parquet','.csv','.tsv'}:
            continue
        low = p.name.lower()
        kind = None
        if 'encounter' in low: kind = 'encounter'
        elif 'procedure' in low or 'imaging' in low: kind = 'imaging_or_procedure'
        elif 'lab' in low or 'lipid' in low: kind = 'lab_or_lipid'
        elif 'diagn' in low or re.search(r'(^|_)dx(_|\.)', low): kind = 'diagnosis'
        if kind is None: continue
        try:
            if p.stat().st_size > 700*1024*1024: continue
            df = load_any(p)
        except Exception as e:
            inventory.append({'file':str(p),'kind':kind,'status':'read_error','detail':str(e)[:150]})
            continue
        k = find_col(df, KEY_CANDIDATES)
        if not k:
            inventory.append({'file':str(p),'kind':kind,'status':'skip_no_patient_key','detail':''})
            continue
        ids = norm_id(df[k])
        hit = df[ids.isin(wanted)].copy()
        if hit.empty:
            inventory.append({'file':str(p),'kind':kind,'status':'no_target_rows','detail':''})
            continue
        hit.insert(0,'__source_file',str(p))
        hit.insert(1,'__source_kind',kind)
        hit.insert(2,'__id',norm_id(hit[k]))
        # keep all columns for these 167 only; this is local ignored evidence.
        candidates.append(hit)
        inventory.append({'file':str(p),'kind':kind,'status':'used','detail':f'{len(hit)} rows'})
    pd.DataFrame(inventory).to_csv(out/'source_inventory.csv', index=False)
    if candidates:
        # Different source schemas cannot be safely concatenated into one wide table;
        # write one CSV per source plus compact manifest.
        manifest = []
        for i, hit in enumerate(candidates, 1):
            kind = str(hit['__source_kind'].iloc[0])
            name = f'{i:03d}_{kind}.csv'
            hit.to_csv(out/name, index=False)
            manifest.append({'file':name,'source':str(hit['__source_file'].iloc[0]),'kind':kind,'rows':len(hit),'columns':len(hit.columns)})
        pd.DataFrame(manifest).to_csv(out/'raw_support_manifest.csv', index=False)
    return {'edge_patients':len(edge),'canonical_edge_rows':len(ce),'aabila_edge_rows':len(ae),'raw_source_tables_used':len(candidates)}


def psu_raw_dx() -> dict:
    c = load_final(CANON_PSU)
    a = load_final(AABILA_PSU/'outputs')
    c = c[c.get('in_stroke_registry',0)==1].copy()
    ids = sorted(set(c['__id']) & set(a['__id']))
    cs = c[c['__id'].isin(ids)].set_index('__id')
    aa = a[a['__id'].isin(ids)].set_index('__id')
    discord = []
    encs = set()
    for pid in ids:
        if 'DX' not in cs.columns or 'DX' not in aa.columns: break
        cv, av = cs.at[pid,'DX'], aa.at[pid,'DX']
        if norm_code(cv)==norm_code(av): continue
        ce = str(cs.at[pid,'ENCOUNTERID']).strip() if 'ENCOUNTERID' in cs.columns else ''
        ae = str(aa.at[pid,'ENCOUNTERID']).strip() if 'ENCOUNTERID' in aa.columns else ''
        if ce: encs.add(ce)
        if ae: encs.add(ae)
        discord.append({'patient_id':pid,'canonical_DX':cv,'aabila_DX':av,'canonical_ENCOUNTERID':ce,'aabila_ENCOUNTERID':ae})
    disc = pd.DataFrame(discord)
    out = OUT/'psu_raw_dx'
    out.mkdir(parents=True, exist_ok=True)
    disc.to_csv(out/'discordant_patients.csv', index=False)

    # Search candidate raw roots first; do not rely on reproduced/intermediate outputs.
    roots = [
        PIPELINE/'dataset', PIPELINE/'data', PIPELINE/'stroke'/'dataset',
        AABILA_PSU/'dataset', AABILA_PSU/'data',
    ]
    files = []
    root_inventory = []
    for r in roots:
        fs = diagnosis_like_files(r)
        root_inventory.append({'root':str(r),'exists':r.exists(),'diagnosis_like_files':len(fs)})
        files.extend(fs)
    files = sorted(set(files))
    support, inventory = extract_dx_support(files, set(disc.patient_id.astype(str)), encs)
    pd.DataFrame(root_inventory).to_csv(out/'raw_root_inventory.csv', index=False)
    pd.DataFrame(inventory).to_csv(out/'source_inventory.csv', index=False)
    support.to_csv(out/'raw_dx_rows.csv', index=False)

    adjud = []
    if not support.empty:
        ec = find_col(support, ENCOUNTER_CANDIDATES)
        for r in discord:
            rows = support[support['__id'].astype(str)==str(r['patient_id'])].copy()
            if ec and r['canonical_ENCOUNTERID']:
                rows = rows[norm_id(rows[ec])==r['canonical_ENCOUNTERID']]
            cn, an = norm_code(r['canonical_DX']), norm_code(r['aabila_DX'])
            cr, ar = rows[rows['__dx_norm']==cn], rows[rows['__dx_norm']==an]
            adjud.append({**r,'canonical_raw_supported':not cr.empty,'aabila_raw_supported':not ar.empty,'raw_rows_on_selected_encounter':len(rows)})
    adf = pd.DataFrame(adjud)
    adf.to_csv(out/'raw_support_adjudication.csv', index=False)
    return {
        'discordant_patients':len(disc),
        'raw_diagnosis_files_found':len(files),
        'raw_support_rows':len(support),
        'canonical_raw_supported':int(adf.get('canonical_raw_supported',pd.Series(dtype=bool)).sum()) if not adf.empty else 0,
        'aabila_raw_supported':int(adf.get('aabila_raw_supported',pd.Series(dtype=bool)).sum()) if not adf.empty else 0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'geisinger_dx_tiebreak': geisinger_dx_tiebreak(),
        'geisinger_edges': geisinger_edges(),
        'psu_raw_dx': psu_raw_dx(),
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {OUT.resolve()}')

if __name__ == '__main__':
    main()
