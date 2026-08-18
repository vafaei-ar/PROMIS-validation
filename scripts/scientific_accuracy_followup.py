#!/usr/bin/env python3
"""Scientific-accuracy follow-up for PROMIS validation.

This script addresses two remaining validation questions without modifying the
production pipeline:

1. Geisinger: reconstruct the 167 Aabila-only edge patients using the exact
   imaging/lipid timing semantics discoverable from Aabila's code/config and raw
   procedure/lab data. The output is designed to distinguish encounter-selection
   policy from genuine evidence errors.
2. PSU: search broadly for an upstream/raw diagnosis extract for the 732 DX
   discordances, explicitly excluding derived outputs where possible.

It also writes a proposed set of final-data indicator columns that could be
added later if the validation demonstrates clinically meaningful ambiguity.
Patient-level outputs remain under results/ and should not be committed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
OUT = VAL / 'results/scientific_accuracy_followup'
SRC = VAL / 'results/source_truth_audit'
GEI_ROOT = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
PSU_CANON = VAL / 'reproduced/psu_997d6bd'
PSU_AAB = ROOT / 'aabila/PROMIS-ML-PennState/outputs'

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


def read_table(path: Path):
    try:
        if path.suffix.lower() == '.parquet':
            return pd.read_parquet(path)
        if path.suffix.lower() in {'.csv','.txt'}:
            return pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    return None


def find_file(root: Path, name: str):
    for p in (root/name, root/'final'/name, root/'intermediate'/name):
        if p.exists():
            return p
    hits = list(root.rglob(name)) if root.exists() else []
    return hits[0] if hits else None


def load_final(root: Path):
    p = find_file(root, 'stroke_cohort_final.parquet')
    if not p:
        raise FileNotFoundError(f'stroke_cohort_final.parquet not found under {root}')
    df = pd.read_parquet(p).copy()
    k = id_col(df)
    if not k:
        raise KeyError(f'No patient ID in {p}')
    df['__id'] = norm(df[k])
    return df


def discover_aabila_timing_logic():
    """Inventory timing/filter expressions from Aabila Geisinger code/config.

    This deliberately records the actual source text rather than silently
    assuming a window. Numeric values are extracted when they can be linked to
    imaging/lipid/match/window/tolerance concepts.
    """
    outdir = OUT/'geisinger_timing_logic'
    outdir.mkdir(parents=True, exist_ok=True)
    records=[]
    numeric=[]
    patterns = re.compile(r'(imaging|lipid|window|tolerance|match_tolerance|days_before|days_after|lookback|lookahead)', re.I)
    numpat = re.compile(r'(?P<key>[A-Za-z0-9_\-]*(?:window|tolerance|days|lookback|lookahead)[A-Za-z0-9_\-]*)\s*[:=]\s*["\']?(?P<value>-?\d+(?:\.\d+)?)', re.I)
    for p in GEI_ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.py','.yaml','.yml','.json','.toml','.ini','.cfg','.txt'}:
            continue
        try:
            text=p.read_text(errors='ignore')
        except Exception:
            continue
        for i,line in enumerate(text.splitlines(),1):
            if patterns.search(line):
                records.append({'file':str(p),'line':i,'text':line.strip()})
                m=numpat.search(line)
                if m:
                    numeric.append({'file':str(p),'line':i,'key':m.group('key'),'value':float(m.group('value')),'text':line.strip()})
    pd.DataFrame(records).to_csv(outdir/'timing_logic_lines.csv', index=False)
    pd.DataFrame(numeric).to_csv(outdir/'numeric_timing_parameters.csv', index=False)
    return records, numeric


def date_columns(df: pd.DataFrame):
    out=[]
    for c in df.columns:
        uc=c.upper()
        if any(h in uc for h in DATE_HINTS):
            try:
                parsed=pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().any():
                    out.append((c, parsed))
            except Exception:
                pass
    return out


def classify_imaging_text(df: pd.DataFrame):
    txt=df.astype('string').agg(' | '.join, axis=1)
    return txt.str.contains(r'\b(MRI|MRA|CT|CTA|HEAD CT|BRAIN|NEUROIMAG|IMAGING)\b', case=False, regex=True, na=False)


def classify_lipid_text(df: pd.DataFrame):
    txt=df.astype('string').agg(' | '.join, axis=1)
    return txt.str.contains(r'\b(LIPID|LDL|HDL|TRIG(?:LYCERIDE)?|CHOLESTEROL)\b', case=False, regex=True, na=False)


def geisinger_edge_exact_reconstruction():
    edge_path = SRC/'geisinger_edges/edge_patients.csv'
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)
    edges=pd.read_csv(edge_path, low_memory=False)
    eid='__id' if '__id' in edges.columns else 'patient_id'
    edges['__id']=norm(edges[eid])
    edge_ids=set(edges['__id'])

    logic_lines, numeric = discover_aabila_timing_logic()

    # Candidate raw tables from Aabila's Geisinger dataset.
    tables=[]
    dataset=GEI_ROOT/'dataset'
    for p in dataset.rglob('*') if dataset.exists() else []:
        if p.is_file() and p.suffix.lower() in {'.parquet','.csv','.txt'}:
            low=p.name.lower()
            if any(x in low for x in ['diagn','encounter','procedure','imaging','lab']):
                tables.append(p)

    outdir=OUT/'geisinger_edges'
    outdir.mkdir(parents=True, exist_ok=True)
    inventory=[]; extracted=[]
    for p in tables:
        df=read_table(p)
        if df is None: continue
        k=id_col(df)
        inventory.append({'file':str(p),'rows':len(df),'cols':len(df.columns),'id_col':k or '', 'enc_col':enc_col(df) or ''})
        if not k: continue
        m=norm(df[k]).isin(edge_ids)
        if not m.any(): continue
        h=df.loc[m].copy(); h.insert(0,'__source_file',str(p)); h.insert(1,'__id',norm(h[k]))
        extracted.append(h)
    pd.DataFrame(inventory).to_csv(outdir/'raw_table_inventory.csv',index=False)
    raw=pd.concat(extracted,ignore_index=True,sort=False) if extracted else pd.DataFrame()
    raw.to_parquet(outdir/'edge_raw_evidence.parquet',index=False) if not raw.empty else None

    # Use the actual final encounters from both pipelines as anchors. We calculate
    # evidence proximity around each anchor across all available raw date columns.
    cfinal=load_final(VAL/'reproduced/geisinger_997d6bd')
    afinal=load_final(GEI_ROOT/'outputs')
    cfinal=cfinal[cfinal['__id'].isin(edge_ids)].drop_duplicates('__id')
    afinal=afinal[afinal['__id'].isin(edge_ids)].drop_duplicates('__id')
    anchors=cfinal.merge(afinal,on='__id',how='outer',suffixes=('_canonical','_aabila'))

    # Find timing windows. We report all discovered numbers and evaluate a small
    # transparent grid as sensitivity if code/config does not expose one clean pair.
    discovered_days=sorted({abs(int(r['value'])) for r in numeric if abs(r['value']) <= 365})
    windows=sorted(set([0,1,3,7,14,30] + discovered_days))

    detail=[]
    if not raw.empty:
        bypid={pid:g for pid,g in raw.groupby('__id')}
        for _,a in anchors.iterrows():
            pid=str(a['__id'])
            g=bypid.get(pid)
            if g is None: continue
            imaging_mask=classify_imaging_text(g)
            lipid_mask=classify_lipid_text(g)
            dcols=date_columns(g)
            for which in ('canonical','aabila'):
                anchor=None
                for col in [f'ADMIT_DATE_{which}', f'DX_DATE_stroke_{which}', f'ENC_DT_{which}']:
                    if col in a.index and pd.notna(a[col]):
                        anchor=pd.to_datetime(a[col],errors='coerce');
                        if pd.notna(anchor): break
                if anchor is None or pd.isna(anchor): continue
                for dcol,parsed in dcols:
                    delta=(parsed-anchor).dt.total_seconds()/86400.0
                    for w in windows:
                        near=delta.abs() <= w
                        detail.append({
                            'patient_id':pid,'anchor':which,'anchor_date':anchor,
                            'source_date_col':dcol,'window_days':w,
                            'imaging_evidence':int((near & imaging_mask).any()),
                            'lipid_evidence':int((near & lipid_mask).any()),
                            'n_imaging_rows':int((near & imaging_mask).sum()),
                            'n_lipid_rows':int((near & lipid_mask).sum()),
                        })
    det=pd.DataFrame(detail)
    det.to_csv(outdir/'window_reconstruction_detail.csv',index=False)
    if not det.empty:
        summary=(det.groupby(['anchor','window_days'])[['imaging_evidence','lipid_evidence']]
                   .agg(['sum','mean']))
        summary.to_csv(outdir/'window_reconstruction_summary.csv')
        both=(det.assign(both=(det.imaging_evidence.eq(1)&det.lipid_evidence.eq(1)).astype(int))
                 .groupby(['anchor','window_days'])['both'].agg(['sum','mean']).reset_index())
        both.to_csv(outdir/'both_evidence_by_anchor_window.csv',index=False)
    else:
        both=pd.DataFrame()

    return {'edge_ids':len(edge_ids),'raw_tables':len(inventory),'raw_rows':len(raw),'windows_tested':windows,'timing_logic_lines':len(logic_lines),'numeric_timing_params':len(numeric)}


def psu_upstream_raw_dx_search():
    ids_path=SRC/'psu_dx/discrepancy_audit.csv'
    idsdf=pd.read_csv(ids_path)
    pidc='patient_id' if 'patient_id' in idsdf.columns else idsdf.columns[0]
    ids=set(norm(idsdf[pidc]))
    outdir=OUT/'psu_raw_dx'; outdir.mkdir(parents=True,exist_ok=True)

    # Search the whole project tree, but explicitly classify derived/output paths.
    inventory=[]; hits=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.parquet','.csv','.txt'}: continue
        low=p.name.lower()
        if not any(k in low for k in ['diagn','dx','condition']): continue
        pathlow=str(p).lower().replace('\\','/')
        derived=any(x in pathlow for x in ['/outputs/','/results/','/reproduced/','/intermediate/'])
        df=read_table(p)
        if df is None: continue
        k=id_col(df)
        inventory.append({'file':str(p),'derived_path':derived,'rows':len(df),'cols':len(df.columns),'id_col':k or '', 'enc_col':enc_col(df) or '', 'dx_cols':'|'.join(c for c in DX_COLS if c in df.columns)})
        if not k: continue
        m=norm(df[k]).isin(ids)
        if not m.any(): continue
        h=df.loc[m].copy(); h.insert(0,'__source_file',str(p)); h.insert(1,'__derived_path',derived); h.insert(2,'__id',norm(h[k]))
        # retain useful columns but avoid duplicate names
        keep=list(dict.fromkeys(['__source_file','__derived_path','__id'] + [c for c in df.columns if c in ENC_COLS+DX_COLS or 'DATE' in c.upper() or 'PRIMARY' in c.upper() or 'ORIGIN' in c.upper() or 'TYPE' in c.upper()]))
        hits.append(h.loc[:,keep].drop_duplicates())
    inv=pd.DataFrame(inventory); inv.to_csv(outdir/'diagnosis_file_inventory.csv',index=False)
    allhits=pd.concat(hits,ignore_index=True,sort=False) if hits else pd.DataFrame(columns=['__source_file','__derived_path','__id'])
    allhits.to_csv(outdir/'discordant_patient_diagnosis_rows.csv',index=False)
    direct=allhits[allhits['__derived_path'].eq(False)] if '__derived_path' in allhits.columns else pd.DataFrame()
    direct.to_csv(outdir/'direct_upstream_hits.csv',index=False)
    return {'discordant_ids':len(ids),'candidate_files':len(inv),'patients_any_hit':int(allhits['__id'].nunique()) if len(allhits) else 0,'direct_upstream_files_with_hits':int(direct['__source_file'].nunique()) if len(direct) else 0,'patients_direct_upstream_hit':int(direct['__id'].nunique()) if len(direct) else 0}


def write_indicator_proposal():
    outdir=OUT/'indicator_proposal'; outdir.mkdir(parents=True,exist_ok=True)
    rows=[
        {'column':'has_neuroimaging','meaning':'Imaging evidence around selected index stroke encounter','current_status':'already present'},
        {'column':'has_lipid_panel','meaning':'Lipid-panel evidence around selected index stroke encounter','current_status':'already present'},
        {'column':'has_alt_qualifying_stroke_encounter','meaning':'Another qualifying stroke encounter exists that could become index under stricter evidence rules','current_status':'proposed'},
        {'column':'alt_index_has_neuroimaging','meaning':'Alternative qualifying index encounter has imaging evidence under configured window','current_status':'proposed'},
        {'column':'alt_index_has_lipid_panel','meaning':'Alternative qualifying index encounter has lipid evidence under configured window','current_status':'proposed'},
        {'column':'index_selection_sensitive','meaning':'Patient index encounter changes under evidence-required phenotype definition','current_status':'proposed'},
        {'column':'multiple_primary_stroke_dx','meaning':'Selected encounter contains >1 qualifying primary ischemic-stroke diagnosis','current_status':'proposed'},
        {'column':'dx_tiebreak_applied','meaning':'Final DX required deterministic tie-breaking among qualifying diagnoses','current_status':'proposed'},
        {'column':'phenotype_strict_eligible','meaning':'Patient satisfies strict evidence-required phenotype definition','current_status':'proposed'},
    ]
    pd.DataFrame(rows).to_csv(outdir/'proposed_final_data_indicators.csv',index=False)
    (outdir/'design_note.txt').write_text(
        'Recommendation: preserve a broad harmonized cohort and expose evidence/ambiguity flags rather than silently dropping patients. '
        'Users can then define a strict primary cohort or sensitivity cohorts explicitly. Production changes should be made only after the validation results demonstrate the precise semantics of each flag.\n'
    )


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={
        'geisinger_edge_reconstruction':geisinger_edge_exact_reconstruction(),
        'psu_upstream_raw_dx':psu_upstream_raw_dx_search(),
    }
    write_indicator_proposal()
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps(summary,indent=2,default=str))
    print(f'Outputs: {OUT.resolve()}')

if __name__=='__main__':
    main()
