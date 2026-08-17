#!/usr/bin/env python3
"""Audit Geisinger canonical vs Aabila cohort differences.

Focuses on two issues found by compare_geisinger.py:
1) Aabila contains 11,910 unique patients while canonical contains 13,872.
2) Aabila final/unimputed/cohort_with_outcomes contain one duplicated patient
   whose duplicated rows differ in SDoH values.

The script does not guess which duplicated SDoH row is correct. It excludes
that ambiguous patient from value-level comparison and writes the conflicting
rows locally. Patient IDs remain only under results/ and should not be committed.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

CANONICAL_ROOT = Path('/home/asadr/works/promis2/PROMIS-validation/reproduced/geisinger_997d6bd')
AABILA_ROOT = Path('/home/asadr/works/promis2/aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs')
OUT = Path('results/aabila_comparison/geisinger_audit')
KEYS = ('PATID','STUDY_ID','PT_ID','patient_id','PatientID')
LAYERS = {
    'final': 'stroke_cohort_final.parquet',
    'unimputed': 'stroke_cohort_unimputed.parquet',
    'outcomes': 'outcomes_flags.parquet',
    'cohort_with_outcomes': 'cohort_with_outcomes.parquet',
}


def first_existing(root: Path, name: str) -> Path | None:
    for p in (root/name, root/'final'/name, root/'intermediate'/name):
        if p.exists():
            return p
    return None


def key(df: pd.DataFrame) -> str:
    for c in KEYS:
        if c in df.columns:
            return c
    raise KeyError(f'No patient identifier among {KEYS}')


def norm(s: pd.Series) -> pd.Series:
    return s.astype('string').str.strip()


def diff_mask(a: pd.Series, b: pd.Series, atol: float=1e-8, rtol: float=1e-7) -> pd.Series:
    a_na, b_na = a.isna(), b.isna()
    out = a_na ^ b_na
    both = ~(a_na | b_na)
    if not both.any():
        return out.fillna(False)
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        av = pd.to_numeric(a.loc[both], errors='coerce').astype(float).to_numpy()
        bv = pd.to_numeric(b.loc[both], errors='coerce').astype(float).to_numpy()
        out.loc[both] = ~np.isclose(av, bv, atol=atol, rtol=rtol, equal_nan=True)
    elif pd.api.types.is_datetime64_any_dtype(a) and pd.api.types.is_datetime64_any_dtype(b):
        out.loc[both] = a.loc[both].astype('datetime64[ns]').to_numpy() != b.loc[both].astype('datetime64[ns]').to_numpy()
    else:
        out.loc[both] = a.loc[both].astype('string').str.strip().to_numpy() != b.loc[both].astype('string').str.strip().to_numpy()
    return out.fillna(False)


def prepare_aabila(df: pd.DataFrame, layer_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    k = key(df)
    x = df.copy()
    x['__id'] = norm(x[k])
    dup_ids = sorted(x.loc[x['__id'].duplicated(keep=False), '__id'].dropna().unique().tolist())
    if dup_ids:
        x[x['__id'].isin(dup_ids)].sort_values('__id').to_csv(layer_dir/'aabila_conflicting_duplicate_rows.csv', index=False)
        pd.DataFrame({'patient_id': dup_ids}).to_csv(layer_dir/'excluded_ambiguous_aabila_ids.csv', index=False)
        x = x[~x['__id'].isin(dup_ids)].copy()
    return x, dup_ids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []

    # Cohort membership is anchored to final.
    cp = first_existing(CANONICAL_ROOT, 'stroke_cohort_final.parquet')
    ap = first_existing(AABILA_ROOT, 'stroke_cohort_final.parquet')
    if cp is None or ap is None:
        raise FileNotFoundError('Final cohort file missing')
    cfinal = pd.read_parquet(cp).copy(); afinal = pd.read_parquet(ap).copy()
    ck, ak = key(cfinal), key(afinal)
    cfinal['__id'] = norm(cfinal[ck]); afinal['__id'] = norm(afinal[ak])
    c_ids = set(cfinal['__id'].dropna()); a_ids = set(afinal['__id'].dropna())
    shared_ids = c_ids & a_ids; c_only = c_ids - a_ids; a_only = a_ids - c_ids
    pd.DataFrame({'patient_id': sorted(c_only)}).to_csv(OUT/'canonical_only_patients.csv', index=False)
    pd.DataFrame({'patient_id': sorted(a_only)}).to_csv(OUT/'aabila_only_patients.csv', index=False)

    # Profile canonical-only vs shared using useful cohort/provenance fields when present.
    cfinal['membership'] = cfinal['__id'].map(lambda x: 'shared_with_aabila' if x in shared_ids else 'canonical_only')
    fields = [
        'ENC_DT','ADMIT_DATE','DISCHARGE_DATE','AGE','AGE_AT_ENCOUNTER','ENC_TYPE','DX','DX_DATE_stroke',
        'in_stroke_registry','synthetic_registry_dx','ehr_stroke_dx_present','has_neuroimaging','has_lipid_panel',
        'has_rehabilitation','in_pcori','local_area'
    ]
    rows=[]
    for col in fields:
        if col not in cfinal.columns:
            continue
        s=cfinal[col]
        if pd.api.types.is_numeric_dtype(s):
            for grp,g in cfinal.groupby('membership'):
                vals=pd.to_numeric(g[col], errors='coerce')
                rows.append({'field':col,'membership':grp,'type':'numeric','n':len(g),'missing':int(vals.isna().sum()),'mean':float(vals.mean()) if vals.notna().any() else np.nan,'median':float(vals.median()) if vals.notna().any() else np.nan})
        elif pd.api.types.is_datetime64_any_dtype(s):
            for grp,g in cfinal.groupby('membership'):
                vals=pd.to_datetime(g[col], errors='coerce')
                rows.append({'field':col,'membership':grp,'type':'datetime','n':len(g),'missing':int(vals.isna().sum()),'min':str(vals.min()),'max':str(vals.max())})
        else:
            tab=(cfinal.groupby(['membership',col], dropna=False).size().rename('n').reset_index())
            for _,r in tab.iterrows():
                rows.append({'field':col,'membership':r['membership'],'type':'category','value':str(r[col]),'n':int(r['n'])})
    pd.DataFrame(rows).to_csv(OUT/'canonical_membership_profile.csv', index=False)

    print(f'Canonical unique patients: {len(c_ids):,}')
    print(f'Aabila unique patients:    {len(a_ids):,}')
    print(f'Shared:                    {len(shared_ids):,}')
    print(f'Canonical only:            {len(c_only):,}')
    print(f'Aabila only:               {len(a_only):,}')

    for layer,name in LAYERS.items():
        cp=first_existing(CANONICAL_ROOT,name); ap=first_existing(AABILA_ROOT,name)
        layer_dir=OUT/layer; layer_dir.mkdir(parents=True, exist_ok=True)
        if cp is None or ap is None:
            summaries.append({'layer':layer,'status':'missing_artifact','canonical_found':cp is not None,'aabila_found':ap is not None})
            continue
        c=pd.read_parquet(cp).copy(); a=pd.read_parquet(ap).copy()
        ck,ak=key(c),key(a); c['__id']=norm(c[ck]); a['__id']=norm(a[ak])
        a,excluded=prepare_aabila(a,layer_dir)
        if c['__id'].duplicated().any():
            raise ValueError(f'Canonical duplicate IDs in {layer}')
        if a['__id'].duplicated().any():
            raise ValueError(f'Aabila duplicate IDs remain in {layer}')
        ci=set(c['__id'].dropna()); ai=set(a['__id'].dropna()); sh=ci & ai
        pd.DataFrame({'patient_id':sorted(ci-ai)}).to_csv(layer_dir/'patients_only_canonical.csv', index=False)
        pd.DataFrame({'patient_id':sorted(ai-ci)}).to_csv(layer_dir/'patients_only_aabila.csv', index=False)
        ignored={ck,ak,'__id'}; cc=set(c.columns)-ignored; ac=set(a.columns)-ignored; shared_cols=sorted(cc&ac)
        cs=c[c['__id'].isin(sh)].set_index('__id').sort_index(); aa=a[a['__id'].isin(sh)].set_index('__id').sort_index()
        comp=[]; discord=set()
        for col in shared_cols:
            m=diff_mask(cs[col],aa[col]); discord.update(m.index[m])
            comp.append({'column':col,'canonical_missing':int(cs[col].isna().sum()),'aabila_missing':int(aa[col].isna().sum()),'different_cells':int(m.sum()),'pct_shared_patients_different':100*float(m.sum())/len(sh) if sh else np.nan})
        compdf=pd.DataFrame(comp)
        if not compdf.empty: compdf=compdf.sort_values(['different_cells','column'],ascending=[False,True])
        compdf.to_csv(layer_dir/'column_comparison.csv', index=False)
        pd.DataFrame({'patient_id':sorted(discord)}).to_csv(layer_dir/'discordant_patients.csv', index=False)
        s={'layer':layer,'status':'compared_excluding_ambiguous_duplicates','excluded_aabila_duplicate_ids':len(excluded),'canonical_patients':len(ci),'aabila_patients_after_exclusion':len(ai),'shared_patients':len(sh),'canonical_only':len(ci-ai),'aabila_only':len(ai-ci),'shared_columns':len(shared_cols),'columns_only_canonical':sorted(cc-ac),'columns_only_aabila':sorted(ac-cc),'columns_with_differences':int((compdf['different_cells']>0).sum()) if not compdf.empty else 0,'total_different_cells':int(compdf['different_cells'].sum()) if not compdf.empty else 0,'discordant_shared_patients':len(discord)}
        summaries.append(s)
        (layer_dir/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
        print('\n'+layer+':')
        print(json.dumps(s,indent=2))

    pd.DataFrame(summaries).to_csv(OUT/'summary.csv', index=False)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(f'\nResults written to: {OUT.resolve()}')

if __name__ == '__main__':
    main()
