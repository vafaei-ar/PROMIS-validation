#!/usr/bin/env python3
"""Final scientific validation tasks for PROMIS-validation.

This script intentionally does not modify production PROMIS data. It produces local,
patient-level validation evidence under results/ only.

Tasks
-----
1. Geisinger 167 edge patients: reconstruct imaging/lipid evidence around every
   candidate stroke encounter using temporal linkage rather than encounter-ID-only joins.
   Config/code text from Aabila's Geisinger workspace is copied into the audit output so
   the temporal-window assumptions are visible and reviewable.
2. PSU DX discordances: broaden direct-source discovery beyond outputs/intermediates and
   classify candidate diagnosis files as upstream/raw-like versus derived/output-like.
3. DX tie-break evaluation: compare deterministic policies for encounters containing
   multiple qualifying diagnosis codes without treating incidental file order as truth.

No patient-level output should be committed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from collections import defaultdict
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
OUT = VAL / 'results/final_scientific_validation'
SRC = VAL / 'results/source_truth_audit'
RAW_ADJ = VAL / 'results/raw_discrepancy_adjudication'
FINAL_ADJ = VAL / 'results/final_targeted_adjudication'

CAN_GEI = VAL / 'reproduced/geisinger_997d6bd'
AAB_GEI = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
AAB_GEI_OUT = AAB_GEI / 'outputs'
CAN_PSU = VAL / 'reproduced/psu_997d6bd'
AAB_PSU = ROOT / 'aabila/PROMIS-ML-PennState'
PIPE = ROOT / 'codes/PROMIS-ML-pipeline'

ID_COLS = ('PATID','STUDY_ID','PT_ID','patient_id','PatientID')
ENC_COLS = ('ENCOUNTERID','ENCOUNTER_ID','ENC_ID')
DX_COLS = ('DX','DX_CODE','DIAGNOSIS_CODE','DIAGNOSIS','ICD_CODE')
DATE_HINTS = ('DATE','DTTM','DT','TIME')


def norm(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip()


def id_col(df: pd.DataFrame):
    return next((c for c in ID_COLS if c in df.columns), None)


def enc_col(df: pd.DataFrame):
    return next((c for c in ENC_COLS if c in df.columns), None)


def safe_read(path: Path):
    try:
        if path.suffix.lower() == '.parquet':
            return pd.read_parquet(path)
        if path.suffix.lower() in {'.csv','.txt'}:
            return pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    return None


def first_existing(root: Path, names):
    for name in names:
        for p in (root/name, root/'final'/name, root/'intermediate'/name, root/'dataset'/name):
            if p.exists():
                return p
    return None


def load_final(root: Path):
    p = first_existing(root, ['stroke_cohort_final.parquet'])
    if p is None:
        raise FileNotFoundError(f'No final cohort under {root}')
    df = pd.read_parquet(p).copy()
    k = id_col(df)
    if not k:
        raise KeyError(f'No ID in {p}')
    df['__id'] = norm(df[k])
    return df


def parse_config_and_code_text():
    """Collect Aabila config/code snippets mentioning temporal/evidence rules."""
    outdir = OUT/'geisinger_temporal_rules'
    outdir.mkdir(parents=True, exist_ok=True)
    pats = re.compile(r'(imaging|lipid|window|day|tolerance|match|procedure|lab)', re.I)
    rows=[]
    roots=[AAB_GEI/'stroke/stroke_extraction_scripts', AAB_GEI]
    seen=set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in {'.py','.yaml','.yml','.json','.toml','.ini'}:
                continue
            if '/outputs/' in str(p).replace('\\','/'):
                continue
            rp=str(p.resolve())
            if rp in seen: continue
            seen.add(rp)
            try: text=p.read_text(errors='ignore')
            except Exception: continue
            lines=text.splitlines()
            for i,line in enumerate(lines, start=1):
                if pats.search(line):
                    rows.append({'file':rp,'line':i,'text':line[:1000]})
    pd.DataFrame(rows).to_csv(outdir/'temporal_rule_snippets.csv', index=False)
    return rows


def infer_candidate_date_columns(df):
    cols=[]
    for c in df.columns:
        uc=c.upper()
        if any(h in uc for h in DATE_HINTS):
            try:
                parsed=pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().any(): cols.append(c)
            except Exception:
                pass
    return cols


def find_raw_tables(root: Path, keywords):
    exts={'.parquet','.csv','.txt'}
    for p in root.rglob('*') if root.exists() else []:
        if p.is_file() and p.suffix.lower() in exts:
            low=p.name.lower()
            if any(k in low for k in keywords) and '/outputs/' not in str(p).replace('\\','/'):
                yield p


def geisinger_temporal_reconstruction():
    """Reconstruct evidence around both canonical and Aabila index encounters.

    We deliberately output several symmetric windows rather than silently hard-coding a
    single window if Aabila's config is ambiguous. The snippets file records the actual
    code/config text so the final report can distinguish configured from sensitivity windows.
    """
    outdir=OUT/'geisinger_edge_temporal'
    outdir.mkdir(parents=True, exist_ok=True)
    parse_config_and_code_text()

    edge_path=SRC/'geisinger_edges/edge_patients.csv'
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)
    edges=pd.read_csv(edge_path, low_memory=False)
    eid='__id' if '__id' in edges.columns else 'patient_id'
    edges['__id']=norm(edges[eid])
    edge_ids=set(edges['__id'])

    # Candidate encounters from canonical/Aabila final rows plus raw encounter table.
    cf=load_final(CAN_GEI)
    af=load_final(AAB_GEI_OUT)
    cf=cf[cf['__id'].isin(edge_ids)].copy()
    af=af[af['__id'].isin(edge_ids)].copy()

    raw_enc=[]
    for p in find_raw_tables(AAB_GEI/'dataset', ['encounter']):
        df=safe_read(p)
        if df is None: continue
        k=id_col(df)
        if not k: continue
        m=norm(df[k]).isin(edge_ids)
        if not m.any(): continue
        h=df.loc[m].copy(); h['__id']=norm(h[k]); h['__source_file']=str(p)
        raw_enc.append(h)
    enc=pd.concat(raw_enc, ignore_index=True, sort=False) if raw_enc else pd.DataFrame()

    # Raw procedures and labs. Procedure tables may lack encounter IDs; date linkage is primary.
    proc=[]; lab=[]
    for kind,kws,collector in [('procedure',['procedure','imaging'],proc),('lab',['lab_result','lab_order','laboratory'],lab)]:
        for p in find_raw_tables(AAB_GEI/'dataset', kws):
            df=safe_read(p)
            if df is None: continue
            k=id_col(df)
            if not k: continue
            m=norm(df[k]).isin(edge_ids)
            if not m.any(): continue
            h=df.loc[m].copy(); h['__id']=norm(h[k]); h['__source_file']=str(p); h['__kind']=kind
            collector.append(h)
    proc_df=pd.concat(proc, ignore_index=True, sort=False) if proc else pd.DataFrame()
    lab_df=pd.concat(lab, ignore_index=True, sort=False) if lab else pd.DataFrame()

    def row_text(df):
        if df.empty: return pd.Series(dtype='string')
        return df.astype('string').agg(' | '.join, axis=1)
    if not proc_df.empty: proc_df['__text']=row_text(proc_df)
    if not lab_df.empty: lab_df['__text']=row_text(lab_df)

    # Broad phenotype text cues are audit aids, not production definitions.
    proc_pat=re.compile(r'\b(MRI|CT|CTA|MRA|IMAG|BRAIN|HEAD)\b', re.I)
    lipid_pat=re.compile(r'\b(LIPID|LDL|HDL|TRIG|CHOLESTEROL)\b', re.I)

    windows=[0,1,3,7,30]
    records=[]
    for pid in sorted(edge_ids):
        c=cf[cf['__id']==pid]
        a=af[af['__id']==pid]
        anchors=[]
        for label,d in [('canonical',c),('aabila',a)]:
            if d.empty: continue
            r=d.iloc[0]
            dt=None
            for col in ['DX_DATE_stroke','ADMIT_DATE','ENC_DT','ENC_ADM_DTTM']:
                if col in d.columns:
                    x=pd.to_datetime(r.get(col), errors='coerce')
                    if pd.notna(x): dt=x; break
            anchors.append((label, str(r.get('ENCOUNTERID','')), dt))
        # also include raw encounters for full candidate spectrum
        if not enc.empty:
            ge=enc[enc['__id']==pid]
            ec=enc_col(ge)
            for _,r in ge.iterrows():
                dt=None
                for col in ['ADMIT_DATE','ENC_DT','ENC_ADM_DTTM','DISCHARGE_DATE']:
                    if col in ge.columns:
                        x=pd.to_datetime(r.get(col), errors='coerce')
                        if pd.notna(x): dt=x; break
                anchors.append(('raw_candidate', str(r.get(ec,'')) if ec else '', dt))
        seen=set()
        for label,encid,dt in anchors:
            key=(label,encid,str(dt))
            if key in seen or pd.isna(dt): continue
            seen.add(key)
            pp=proc_df[proc_df['__id']==pid] if not proc_df.empty else pd.DataFrame()
            ll=lab_df[lab_df['__id']==pid] if not lab_df.empty else pd.DataFrame()
            pdates=[]
            for col in infer_candidate_date_columns(pp) if not pp.empty else []:
                x=pd.to_datetime(pp[col], errors='coerce'); pdates.append(x)
            ldates=[]
            for col in infer_candidate_date_columns(ll) if not ll.empty else []:
                x=pd.to_datetime(ll[col], errors='coerce'); ldates.append(x)
            pdate=pd.concat(pdates,axis=1).min(axis=1) if pdates else pd.Series(pd.NaT,index=pp.index)
            ldate=pd.concat(ldates,axis=1).min(axis=1) if ldates else pd.Series(pd.NaT,index=ll.index)
            ptxt=pp.get('__text',pd.Series('',index=pp.index))
            ltxt=ll.get('__text',pd.Series('',index=ll.index))
            for w in windows:
                pm=(pdate.dt.normalize()-pd.Timestamp(dt).normalize()).abs().dt.days.le(w) if len(pp) else pd.Series(dtype=bool)
                lm=(ldate.dt.normalize()-pd.Timestamp(dt).normalize()).abs().dt.days.le(w) if len(ll) else pd.Series(dtype=bool)
                records.append({
                    'patient_id':pid,'anchor_type':label,'encounter_id':encid,'anchor_date':dt,
                    'window_days':w,
                    'imaging_like_rows':int((pm & ptxt.str.contains(proc_pat,na=False)).sum()) if len(pp) else 0,
                    'lipid_like_rows':int((lm & ltxt.str.contains(lipid_pat,na=False)).sum()) if len(ll) else 0,
                    'procedure_rows_in_window':int(pm.sum()) if len(pp) else 0,
                    'lab_rows_in_window':int(lm.sum()) if len(ll) else 0,
                })
    res=pd.DataFrame(records)
    res.to_csv(outdir/'encounter_temporal_evidence.csv', index=False)
    # compact paired summary for canonical vs Aabila anchors
    if not res.empty:
        paired=res[res['anchor_type'].isin(['canonical','aabila'])].copy()
        paired.to_csv(outdir/'canonical_vs_aabila_temporal_evidence.csv', index=False)
    return {'edge_patients':len(edge_ids),'rows':len(res),'procedure_source_rows':len(proc_df),'lab_source_rows':len(lab_df)}


def psu_raw_source_discovery():
    outdir=OUT/'psu_raw_source'
    outdir.mkdir(parents=True, exist_ok=True)
    disc=SRC/'psu_dx/discrepancy_audit.csv'
    d=pd.read_csv(disc); ids=set(norm(d['patient_id']))

    # Broaden search to workspace-level roots that plausibly contain imports/raw extracts.
    roots=[ROOT/'codes', ROOT/'aabila', ROOT/'data', ROOT/'dataset', ROOT/'datasets', ROOT/'raw', ROOT/'extracts']
    keywords=['diagn','dx','condition']
    inventory=[]; hits=[]
    seen=set()
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in {'.parquet','.csv','.txt'}: continue
            low=p.name.lower()
            if not any(k in low for k in keywords): continue
            rp=str(p.resolve())
            if rp in seen: continue
            seen.add(rp)
            df=safe_read(p)
            if df is None: continue
            k=id_col(df)
            pathlow=rp.lower().replace('\\','/')
            derived=('/outputs/' in pathlow or '/results/' in pathlow or '/intermediate/' in pathlow or 'harmonized' in pathlow)
            rawlike=any(x in pathlow for x in ['/dataset/','/datasets/','/raw/','/extract/','/source/']) and not derived
            hitn=0; phit=0
            if k:
                m=norm(df[k]).isin(ids); hitn=int(m.sum()); phit=int(norm(df.loc[m,k]).nunique()) if m.any() else 0
                if m.any():
                    h=df.loc[m].copy(); h['__id']=norm(h[k]); h['__source_file']=rp; h['__rawlike']=rawlike; h['__derived']=derived
                    hits.append(h)
            inventory.append({'file':rp,'rows':len(df),'cols':len(df.columns),'id_col':k or '',
                              'rawlike':rawlike,'derived':derived,'matching_rows':hitn,'matching_patients':phit})
    inv=pd.DataFrame(inventory).sort_values(['rawlike','matching_patients','matching_rows'],ascending=[False,False,False]) if inventory else pd.DataFrame()
    inv.to_csv(outdir/'diagnosis_source_inventory.csv', index=False)
    if hits:
        # schema-union output; no arbitrary narrowing of diagnosis semantics
        pd.concat(hits, ignore_index=True, sort=False).to_csv(outdir/'discordant_dx_source_rows.csv', index=False)
    return {'discordant_patients':len(ids),'candidate_files':len(inv),
            'rawlike_files_with_hits':int(((inv.get('rawlike',False)==True)&(inv.get('matching_patients',0)>0)).sum()) if not inv.empty else 0,
            'derived_files_with_hits':int(((inv.get('derived',False)==True)&(inv.get('matching_patients',0)>0)).sum()) if not inv.empty else 0}


def normalize_dx_code(x):
    if pd.isna(x): return ''
    return re.sub(r'[^A-Z0-9]','',str(x).upper())


def dx_tiebreak_evaluation():
    outdir=OUT/'dx_tiebreak'
    outdir.mkdir(parents=True, exist_ok=True)
    summaries=[]
    for site,path in [('geisinger',SRC/'geisinger_dx/discrepancy_audit.csv'),('psu',SRC/'psu_dx/discrepancy_audit.csv')]:
        if not path.exists(): continue
        df=pd.read_csv(path, low_memory=False)
        if 'canonical_DX' not in df.columns or 'aabila_DX' not in df.columns: continue
        df['canonical_norm']=df['canonical_DX'].map(normalize_dx_code)
        df['aabila_norm']=df['aabila_DX'].map(normalize_dx_code)
        # Evaluate transparent deterministic lexical policies as reproducibility baselines.
        df['lexical_min']=df[['canonical_norm','aabila_norm']].min(axis=1)
        df['lexical_max']=df[['canonical_norm','aabila_norm']].max(axis=1)
        df['lexical_min_matches_canonical']=(df['lexical_min']==df['canonical_norm'])
        df['lexical_min_matches_aabila']=(df['lexical_min']==df['aabila_norm'])
        df['same_after_normalization']=(df['canonical_norm']==df['aabila_norm'])
        df.to_csv(outdir/f'{site}_dx_policy_evaluation.csv', index=False)
        summaries.append({
            'site':site,'rows':len(df),
            'same_after_normalization':int(df['same_after_normalization'].sum()),
            'lexical_min_canonical':int(df['lexical_min_matches_canonical'].sum()),
            'lexical_min_aabila':int(df['lexical_min_matches_aabila'].sum()),
        })
    note=(
        'Recommended scientific rule: first restrict to diagnosis records that satisfy the documented stroke phenotype and primary-status semantics on the selected encounter/date. '
        'If more than one code remains clinically equivalent, apply a stable deterministic tie-break independent of raw file row order (for example normalized ICD code ascending). '
        'This audit evaluates lexical ordering only as a reproducibility baseline; it does not claim lexical order is clinically superior.\n'
    )
    (outdir/'recommended_policy.txt').write_text(note)
    pd.DataFrame(summaries).to_csv(outdir/'summary.csv', index=False)
    return summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary={
        'geisinger_temporal':geisinger_temporal_reconstruction(),
        'psu_raw_source':psu_raw_source_discovery(),
        'dx_tiebreak':dx_tiebreak_evaluation(),
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps(summary,indent=2,default=str))
    print(f'Outputs: {OUT.resolve()}')

if __name__=='__main__':
    main()
