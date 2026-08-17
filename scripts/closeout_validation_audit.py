#!/usr/bin/env python3
"""Final focused validation audit for PROMIS-validation.

Goals
-----
1. Locate and inspect direct PSU raw diagnosis source candidates for the 732 DX discordances.
2. Reconstruct Geisinger candidate encounters for the 167 edge patients and attach
   imaging/lipid evidence independently from raw diagnosis/encounter/procedure/lab tables.
3. Summarize deterministic diagnosis tie-break candidates for Geisinger DX discordances.

Patient-level outputs are written only under results/ and should remain uncommitted.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
OUT = VAL / 'results/closeout_validation_audit'
SRC = VAL / 'results/source_truth_audit'
RAW_ADJ = VAL / 'results/raw_discrepancy_adjudication'
FINAL_ADJ = VAL / 'results/final_targeted_adjudication'

PSU_SEARCH_ROOTS = [
    ROOT / 'codes/PROMIS-ML-pipeline',
    ROOT / 'aabila/PROMIS-ML-PennState',
]
GEI_ROOT = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'

ID_CANDIDATES = ('PATID','STUDY_ID','PT_ID','patient_id','PatientID')
ENC_CANDIDATES = ('ENCOUNTERID','ENCOUNTER_ID','ENC_ID')
DX_CANDIDATES = ('DX','DX_CODE','DIAGNOSIS_CODE','DIAGNOSIS','ICD_CODE')
DXTYPE_CANDIDATES = ('DX_TYPE','DXTYPE','DIAGNOSIS_TYPE')
DXSRC_CANDIDATES = ('DX_SOURCE','DIAGNOSIS_SOURCE','SOURCE')
PRIMARY_CANDIDATES = ('DX_ORIGIN','DX_TYPE','PRIMARY_DX','IS_PRIMARY','PDX')
DATE_CANDIDATES = ('DX_DATE','DX_DATE_stroke','ADMIT_DATE','ENC_DT','ENC_ADM_DTTM','RESULT_DATE','SPECIMEN_DATE','PROCEDURE_DATE')


def id_col(df):
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def enc_col(df):
    for c in ENC_CANDIDATES:
        if c in df.columns:
            return c
    return None


def norm(s):
    return s.astype('string').str.strip()


def safe_read(path: Path):
    try:
        if path.suffix.lower() == '.parquet':
            return pd.read_parquet(path)
        if path.suffix.lower() in {'.csv','.txt'}:
            return pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    return None


def candidate_files(roots, keywords):
    exts = {'.parquet','.csv','.txt'}
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            low = p.name.lower()
            if any(k in low for k in keywords):
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    yield p


def load_discordant_ids(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    for c in ('patient_id','__id','PATID','STUDY_ID'):
        if c in df.columns:
            return set(norm(df[c]).dropna())
    return set()


def psu_raw_dx_locator():
    ids = load_discordant_ids(SRC/'psu_dx/discrepancy_audit.csv')
    outdir = OUT/'psu_raw_dx'
    outdir.mkdir(parents=True, exist_ok=True)
    inventory = []
    hits = []
    keywords = ['diagn','dx','condition']
    for p in candidate_files(PSU_SEARCH_ROOTS, keywords):
        df = safe_read(p)
        if df is None:
            continue
        k = id_col(df)
        inventory.append({
            'file': str(p),
            'rows': len(df),
            'cols': len(df.columns),
            'id_col': k or '',
            'enc_col': enc_col(df) or '',
            'dx_cols': '|'.join([c for c in DX_CANDIDATES if c in df.columns]),
            'date_cols': '|'.join([c for c in DATE_CANDIDATES if c in df.columns]),
        })
        if not k or not ids:
            continue
        mask = norm(df[k]).isin(ids)
        if mask.any():
            h = df.loc[mask].copy()
            h.insert(0,'__source_file',str(p))
            h.insert(1,'__id',norm(h[k]))
            keep = ['__source_file','__id'] + [c for c in list(ENC_CANDIDATES)+list(DX_CANDIDATES)+list(DXTYPE_CANDIDATES)+list(DXSRC_CANDIDATES)+list(PRIMARY_CANDIDATES)+list(DATE_CANDIDATES) if c in h.columns]
            hits.append(h[keep].drop_duplicates())
    pd.DataFrame(inventory).to_csv(outdir/'candidate_raw_dx_files.csv', index=False)
    if hits:
        all_hits = pd.concat(hits, ignore_index=True)
        all_hits.to_csv(outdir/'discordant_patient_raw_dx_rows.csv', index=False)
        files_with_hits = all_hits['__source_file'].nunique()
        patients_hit = all_hits['__id'].nunique()
    else:
        pd.DataFrame(columns=['__source_file','__id']).to_csv(outdir/'discordant_patient_raw_dx_rows.csv', index=False)
        files_with_hits = 0
        patients_hit = 0
    return {'discordant_ids': len(ids), 'candidate_files': len(inventory), 'files_with_hits': files_with_hits, 'patients_hit': patients_hit}


def find_gei_raw_tables():
    tables = defaultdict(list)
    mapping = {
        'diagnosis':['diagnosis','dx'],
        'encounter':['encounter'],
        'procedure':['procedure','imaging'],
        'lab':['lab_result','labresult','lab_order','laboratory','lab'],
    }
    for kind, kws in mapping.items():
        for p in candidate_files([GEI_ROOT/'dataset', GEI_ROOT], kws):
            # prioritize dataset/raw-like files; avoid outputs where possible
            if '/outputs/' in str(p).replace('\\','/'):
                continue
            tables[kind].append(p)
    return tables


def choose_columns(df):
    cols=[]
    for group in (ID_CANDIDATES, ENC_CANDIDATES, DX_CANDIDATES, DXTYPE_CANDIDATES, DXSRC_CANDIDATES, PRIMARY_CANDIDATES, DATE_CANDIDATES):
        cols.extend([c for c in group if c in df.columns])
    for c in df.columns:
        lc=c.lower()
        if any(x in lc for x in ['cpt','hcpcs','procedure','proc_','lab','result','test','loinc','lipid','chol','trig','hdl','ldl','imaging','ct','mri']):
            cols.append(c)
    return list(dict.fromkeys(cols))


def geisinger_edge_reconstruction():
    edge_path = SRC/'geisinger_edges/edge_patients.csv'
    if not edge_path.exists():
        return {'edge_ids':0,'raw_tables':0,'rows_extracted':0}
    edges = pd.read_csv(edge_path)
    idc = '__id' if '__id' in edges.columns else 'patient_id'
    edge_ids = set(norm(edges[idc]))
    outdir = OUT/'geisinger_edges'
    outdir.mkdir(parents=True, exist_ok=True)
    tables = find_gei_raw_tables()
    rows=[]
    table_inventory=[]
    for kind, paths in tables.items():
        for p in paths:
            df=safe_read(p)
            if df is None:
                continue
            k=id_col(df)
            table_inventory.append({'kind':kind,'file':str(p),'rows':len(df),'cols':len(df.columns),'id_col':k or '', 'enc_col':enc_col(df) or ''})
            if not k:
                continue
            m=norm(df[k]).isin(edge_ids)
            if not m.any():
                continue
            h=df.loc[m].copy()
            h.insert(0,'__kind',kind)
            h.insert(1,'__source_file',str(p))
            h.insert(2,'__id',norm(h[k]))
            keep=['__kind','__source_file','__id'] + choose_columns(h)
            rows.append(h[keep].drop_duplicates())
    pd.DataFrame(table_inventory).to_csv(outdir/'raw_table_inventory.csv', index=False)
    if rows:
        raw=pd.concat(rows, ignore_index=True, sort=False)
        raw.to_csv(outdir/'edge_raw_evidence.csv', index=False)
    else:
        raw=pd.DataFrame(columns=['__kind','__source_file','__id'])
        raw.to_csv(outdir/'edge_raw_evidence.csv', index=False)

    # Encounter-level compact support table based on explicit encounter IDs wherever possible.
    compact=[]
    if not raw.empty:
        for pid,g in raw.groupby('__id', dropna=False):
            enc_fields=[c for c in ENC_CANDIDATES if c in g.columns]
            enc_values=set()
            for c in enc_fields:
                enc_values.update(norm(g[c]).dropna().tolist())
            if not enc_values:
                compact.append({'patient_id':pid,'encounter_id':'','n_dx_rows':int((g['__kind']=='diagnosis').sum()),'n_proc_rows':int((g['__kind']=='procedure').sum()),'n_lab_rows':int((g['__kind']=='lab').sum())})
                continue
            for enc in sorted(enc_values):
                gm=pd.Series(False,index=g.index)
                for c in enc_fields:
                    gm = gm | (norm(g[c])==enc)
                gg=g.loc[gm]
                compact.append({
                    'patient_id':pid,
                    'encounter_id':enc,
                    'n_dx_rows':int((gg['__kind']=='diagnosis').sum()),
                    'n_proc_rows':int((gg['__kind']=='procedure').sum()),
                    'n_lab_rows':int((gg['__kind']=='lab').sum()),
                    'has_any_imaging_like_text': int(gg.astype('string').apply(lambda s: s.str.contains('MRI|CT|IMAG',case=False,regex=True,na=False)).any(axis=1).any()),
                    'has_any_lipid_like_text': int(gg.astype('string').apply(lambda s: s.str.contains('LIPID|LDL|HDL|TRIG|CHOLESTEROL',case=False,regex=True,na=False)).any(axis=1).any()),
                })
    pd.DataFrame(compact).to_csv(outdir/'encounter_level_support.csv', index=False)
    return {'edge_ids':len(edge_ids),'raw_tables':len(table_inventory),'rows_extracted':len(raw),'encounter_support_rows':len(compact)}


def geisinger_dx_tiebreak_summary():
    # Prefer detailed rows from the final targeted adjudication if available.
    candidates=[]
    for root in [FINAL_ADJ, RAW_ADJ, SRC/'geisinger_dx']:
        if not root.exists():
            continue
        for p in root.rglob('*.csv'):
            low=p.name.lower()
            if 'dx' in low or 'diagn' in low:
                candidates.append(p)
    outdir=OUT/'geisinger_dx_tiebreak'
    outdir.mkdir(parents=True, exist_ok=True)
    inventory=[]
    combined=[]
    for p in candidates:
        try: df=pd.read_csv(p, low_memory=False)
        except Exception: continue
        cols=set(df.columns)
        if not ({'canonical_DX','aabila_DX'} & cols) and not any(c in cols for c in DX_CANDIDATES):
            continue
        inventory.append({'file':str(p),'rows':len(df),'cols':len(df.columns)})
        temp=df.copy(); temp.insert(0,'__source_file',str(p)); combined.append(temp)
    pd.DataFrame(inventory).to_csv(outdir/'input_inventory.csv', index=False)
    if combined:
        all_df=pd.concat(combined, ignore_index=True, sort=False)
        all_df.to_csv(outdir/'combined_dx_evidence.csv', index=False)
        summary={
            'input_files':len(inventory),
            'rows':len(all_df),
            'patients': int(all_df['patient_id'].astype('string').nunique()) if 'patient_id' in all_df.columns else None,
        }
    else:
        pd.DataFrame().to_csv(outdir/'combined_dx_evidence.csv', index=False)
        summary={'input_files':0,'rows':0,'patients':0}
    (outdir/'policy_note.txt').write_text(
        'Do not adjudicate DX from raw row order alone. A defensible production rule should explicitly prioritize qualifying ischemic-stroke diagnoses by documented primary-status semantics, encounter/date linkage, and a stable deterministic final tie-break (for example normalized code order) rather than incidental file order.\n'
    )
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary={
        'psu_raw_dx': psu_raw_dx_locator(),
        'geisinger_edges': geisinger_edge_reconstruction(),
        'geisinger_dx_tiebreak': geisinger_dx_tiebreak_summary(),
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {OUT.resolve()}')

if __name__=='__main__':
    main()
