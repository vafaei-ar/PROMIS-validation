#!/usr/bin/env python3
"""Trace Aabila Geisinger candidate construction and first-index selection exactly.

This is a read-only diagnostic. It extracts the relevant source-code region from
Aabila's 02_extract_stroke_patients.py, inventories intermediate outputs, and
reconstructs the candidate-to-first-index ordering from Aabila's own files.
Patient-level outputs stay under results/ and are gitignored.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
GEI = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
SCRIPT = GEI / 'stroke/stroke_extraction_scripts/02_extract_stroke_patients.py'
OUT = VAL / 'results/aabila_candidate_selection_trace'


def norm(s):
    return s.astype('string').str.strip()


def extract_main_flow(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines()
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            start = node.lineno
            end = getattr(node, 'end_lineno', start)
            chunks.append('\n'.join(lines[start-1:end]))
    if chunks:
        return '\n\n'.join(chunks)
    # fallback: grab wide context around filter calls / first collapse
    idx=[]
    for i,line in enumerate(lines):
        if any(k in line for k in ['apply_imaging_filter(', 'apply_lipid_filter(', 'first_stroke', 'groupby(', 'drop_duplicates(']):
            idx.append(i)
    if not idx:
        return text
    a=max(0,min(idx)-120); b=min(len(lines),max(idx)+160)
    return '\n'.join(lines[a:b])


def extract_key_lines(text: str):
    patterns = [
        r'apply_imaging_filter', r'apply_lipid_filter', r'first_stroke',
        r'all_stroke_encounters', r'DX_DATE', r'ADMIT_DATE', r'sort_values',
        r'groupby', r'drop_duplicates', r'HAS_IMAGING', r'HAS_LIPID',
        r'stroke_encounters', r'stroke_dx_filtered', r'merge\('
    ]
    rx = re.compile('|'.join(patterns))
    rows=[]
    for i,line in enumerate(text.splitlines(),1):
        if rx.search(line):
            rows.append({'line':i,'text':line.rstrip()})
    return pd.DataFrame(rows)


def find_output(name):
    roots=[GEI/'outputs', GEI/'outputs/final', GEI/'outputs/intermediate']
    for root in roots:
        p=root/name
        if p.exists(): return p
    hits=list((GEI/'outputs').rglob(name))
    return hits[0] if hits else None


def date_choice(df):
    out=pd.Series(pd.NaT,index=df.index,dtype='datetime64[ns]')
    for c in ['DX_DATE_stroke','DX_DATE','ADMIT_DATE','ENC_DT','DISCHARGE_DATE']:
        if c in df.columns:
            out=out.combine_first(pd.to_datetime(df[c],errors='coerce'))
    return out


def infer_from_outputs():
    result={}
    allp=find_output('all_stroke_encounters.parquet')
    firstp=find_output('first_stroke_encounters.parquet')
    if not allp or not firstp:
        return {'available':False,'all_path':str(allp) if allp else None,'first_path':str(firstp) if firstp else None}
    all_df=pd.read_parquet(allp).copy(); first=pd.read_parquet(firstp).copy()
    for df in (all_df,first):
        if 'PATID' in df: df['PATID']=norm(df['PATID'])
        if 'ENCOUNTERID' in df: df['ENCOUNTERID']=norm(df['ENCOUNTERID'])
    result.update({
        'available':True,
        'all_path':str(allp),'first_path':str(firstp),
        'all_rows':int(len(all_df)),'all_patients':int(all_df.PATID.nunique()),
        'all_encounters':int(all_df.ENCOUNTERID.nunique()) if 'ENCOUNTERID' in all_df else None,
        'first_rows':int(len(first)),'first_patients':int(first.PATID.nunique()),
        'all_columns':list(all_df.columns),'first_columns':list(first.columns),
    })

    # Test plausible first-event orderings against Aabila's saved first-stroke file.
    all_df['_date_choice']=date_choice(all_df)
    tests=[]
    order_specs=[]
    if 'DX_DATE' in all_df: order_specs.append(('DX_DATE',['PATID','DX_DATE','ENCOUNTERID']))
    if 'ADMIT_DATE' in all_df: order_specs.append(('ADMIT_DATE',['PATID','ADMIT_DATE','ENCOUNTERID']))
    order_specs.append(('date_choice',['PATID','_date_choice','ENCOUNTERID']))
    for label,cols in order_specs:
        x=all_df.copy()
        for c in cols:
            if c in ['DX_DATE','ADMIT_DATE','_date_choice']:
                x[c]=pd.to_datetime(x[c],errors='coerce')
        cand=x.sort_values(cols,na_position='last').drop_duplicates('PATID',keep='first')
        cmp=cand[['PATID','ENCOUNTERID']].merge(first[['PATID','ENCOUNTERID']],on='PATID',how='inner',suffixes=('_recon','_saved'))
        tests.append({'ordering':label,'patients_compared':int(len(cmp)),'encounter_matches':int(cmp.ENCOUNTERID_recon.eq(cmp.ENCOUNTERID_saved).sum())})
    pd.DataFrame(tests).to_csv(OUT/'ordering_reconstruction_tests.csv',index=False)
    result['ordering_tests']=tests

    # Duplicate diagnosis rows per encounter can change row-level sorting; characterize them.
    if {'PATID','ENCOUNTERID'}.issubset(all_df.columns):
        enc_counts=(all_df.groupby(['PATID','ENCOUNTERID']).size().rename('n_rows').reset_index())
        enc_counts.to_csv(OUT/'rows_per_candidate_encounter.csv',index=False)
        result['candidate_encounters_with_multiple_rows']=int((enc_counts.n_rows>1).sum())
        result['max_rows_per_candidate_encounter']=int(enc_counts.n_rows.max())
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    text=SCRIPT.read_text(errors='ignore')
    (OUT/'main_flow.txt').write_text(extract_main_flow(text))
    extract_key_lines(text).to_csv(OUT/'key_selection_lines.csv',index=False)
    inferred=infer_from_outputs()
    summary={
        'script':str(SCRIPT),
        'exact_imaging_window':'ADMIT_DATE - 2 days through DISCHARGE_DATE',
        'exact_lipid_window':'ADMIT_DATE through DISCHARGE_DATE',
        'output_inference':inferred,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps(summary,indent=2,default=str))
    print(f'Outputs: {OUT.resolve()}')

if __name__=='__main__':
    main()
