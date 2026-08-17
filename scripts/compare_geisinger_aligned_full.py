#!/usr/bin/env python3
"""Full Geisinger comparison on canonical imaging+lipid-positive patients, plus audit of 167 Aabila-only edge patients.

Imputed data are excluded. Patient IDs are written only to local ignored results.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

CROOT=Path('/home/asadr/works/promis2/PROMIS-validation/reproduced/geisinger_997d6bd')
AROOT=Path('/home/asadr/works/promis2/aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs')
OUT=Path('results/aabila_comparison/geisinger_aligned_full')
KEYS=('PATID','STUDY_ID','PT_ID','patient_id','PatientID')
LAYERS={'final':'stroke_cohort_final.parquet','unimputed':'stroke_cohort_unimputed.parquet','outcomes':'outcomes_flags.parquet','cohort_with_outcomes':'cohort_with_outcomes.parquet'}

def find(root,name):
    for p in (root/name,root/'final'/name,root/'intermediate'/name):
        if p.exists(): return p
    return None

def key(df):
    for k in KEYS:
        if k in df.columns:return k
    raise KeyError('No patient key')

def norm(s): return s.astype('string').str.strip()

def diffmask(a,b):
    am,bm=a.isna(),b.isna(); out=am^bm; both=~(am|bm)
    if not both.any(): return out
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av=pd.to_numeric(a[both],errors='coerce').astype(float).to_numpy(); bv=pd.to_numeric(b[both],errors='coerce').astype(float).to_numpy()
        out.loc[both]=~np.isclose(av,bv,atol=1e-8,rtol=1e-7,equal_nan=True)
    else:
        out.loc[both]=a[both].astype('string').str.strip().to_numpy()!=b[both].astype('string').str.strip().to_numpy()
    return out.fillna(False)

def unique_or_drop_exact(df,label,layerdir):
    before=len(df); x=df.drop_duplicates().copy(); removed=before-len(x)
    bad=x[x['__id'].duplicated(keep=False)].copy()
    if not bad.empty:
        bad.sort_values('__id').to_csv(layerdir/f'{label}_conflicting_duplicates.csv',index=False)
        bad_ids=set(bad['__id'])
        x=x[~x['__id'].isin(bad_ids)].copy()
    else: bad_ids=set()
    return x,removed,bad_ids

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cp=find(CROOT,'stroke_cohort_final.parquet'); ap=find(AROOT,'stroke_cohort_final.parquet')
    cfinal=pd.read_parquet(cp).copy(); afinal=pd.read_parquet(ap).copy(); ck,ak=key(cfinal),key(afinal)
    cfinal['__id']=norm(cfinal[ck]); afinal['__id']=norm(afinal[ak])
    aligned_ids=set(cfinal.loc[(cfinal['has_neuroimaging']==1)&(cfinal['has_lipid_panel']==1),'__id'].dropna())
    a_ids=set(afinal['__id'].dropna())
    edge_ids=sorted(a_ids-aligned_ids)
    pd.DataFrame({'patient_id':sorted(aligned_ids)}).to_csv(OUT/'aligned_ids.csv',index=False)
    pd.DataFrame({'patient_id':edge_ids}).to_csv(OUT/'aabila_only_edge_ids.csv',index=False)

    edge=afinal[afinal['__id'].isin(edge_ids)].copy()
    edge.to_csv(OUT/'aabila_only_edge_rows_final.csv',index=False)
    cedge=cfinal[cfinal['__id'].isin(edge_ids)].copy()
    keep=[x for x in [ck,'has_neuroimaging','has_lipid_panel','DX','DX_DATE_stroke','ADMIT_DATE','DISCHARGE_DATE','ENCOUNTERID'] if x in cedge.columns]
    cedge[keep].to_csv(OUT/'canonical_flags_for_aabila_edges.csv',index=False)
    if not cedge.empty:
        print('Edge canonical flag combinations:')
        print(cedge.groupby(['has_neuroimaging','has_lipid_panel'],dropna=False).size().to_string())

    summaries=[]
    for layer,name in LAYERS.items():
        cpath,apath=find(CROOT,name),find(AROOT,name)
        if cpath is None or apath is None:
            summaries.append({'layer':layer,'status':'missing','canonical_found':cpath is not None,'aabila_found':apath is not None}); continue
        c=pd.read_parquet(cpath).copy(); a=pd.read_parquet(apath).copy(); ck2,ak2=key(c),key(a)
        c['__id']=norm(c[ck2]); a['__id']=norm(a[ak2]); c=c[c['__id'].isin(aligned_ids)].copy(); a=a[a['__id'].isin(aligned_ids)].copy()
        ld=OUT/layer; ld.mkdir(parents=True,exist_ok=True)
        c,cr,cb=unique_or_drop_exact(c,'canonical',ld); a,ar,ab=unique_or_drop_exact(a,'aabila',ld)
        shared=set(c['__id'])&set(a['__id']); conly=set(c['__id'])-set(a['__id']); aonly=set(a['__id'])-set(c['__id'])
        pd.DataFrame({'patient_id':sorted(conly)}).to_csv(ld/'patients_only_canonical.csv',index=False); pd.DataFrame({'patient_id':sorted(aonly)}).to_csv(ld/'patients_only_aabila.csv',index=False)
        ignored={ck2,ak2,'__id'}; cc=set(c.columns)-ignored; ac=set(a.columns)-ignored; sc=sorted(cc&ac)
        cs=c[c['__id'].isin(shared)].set_index('__id').sort_index(); aa=a[a['__id'].isin(shared)].set_index('__id').sort_index()
        rows=[]; discord=set()
        for col in sc:
            m=diffmask(cs[col],aa[col]); discord.update(m.index[m]); rows.append({'column':col,'canonical_missing':int(cs[col].isna().sum()),'aabila_missing':int(aa[col].isna().sum()),'different_cells':int(m.sum()),'pct_shared_patients_different':100*float(m.sum())/len(shared) if shared else np.nan})
        comp=pd.DataFrame(rows).sort_values(['different_cells','column'],ascending=[False,True]) if rows else pd.DataFrame(); comp.to_csv(ld/'column_comparison.csv',index=False)
        pd.DataFrame({'patient_id':sorted(discord)}).to_csv(ld/'discordant_patients.csv',index=False)
        s={'layer':layer,'status':'compared','aligned_reference_ids':len(aligned_ids),'canonical_patients':len(set(c['__id'])),'aabila_patients':len(set(a['__id'])),'shared_patients':len(shared),'canonical_only':len(conly),'aabila_only':len(aonly),'canonical_exact_dups_removed':cr,'aabila_exact_dups_removed':ar,'canonical_conflicting_dup_ids':len(cb),'aabila_conflicting_dup_ids':len(ab),'shared_columns':len(sc),'columns_with_differences':int((comp['different_cells']>0).sum()) if not comp.empty else 0,'total_different_cells':int(comp['different_cells'].sum()) if not comp.empty else 0,'discordant_shared_patients':len(discord),'columns_only_canonical':sorted(cc-ac),'columns_only_aabila':sorted(ac-cc)}
        (ld/'summary.json').write_text(json.dumps(s,indent=2)+'\n'); summaries.append(s)
        print(json.dumps(s,indent=2))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False); (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(f'\nAligned IDs: {len(aligned_ids):,}; Aabila edge IDs: {len(edge_ids):,}')
    print(f'Results: {OUT.resolve()}')

if __name__=='__main__': main()
