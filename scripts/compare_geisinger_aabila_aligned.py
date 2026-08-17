#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

CANONICAL_ROOT = Path('/home/asadr/works/promis2/PROMIS-validation/reproduced/geisinger_997d6bd')
AABILA_ROOT = Path('/home/asadr/works/promis2/aabila/PROMIS-ML-pipeline-master-Geisinger_07162026/outputs')
OUT = Path('results/aabila_comparison/geisinger_aabila_aligned')
KEYS = ('PATID','STUDY_ID','PT_ID','patient_id','PatientID')


def first_existing(root, name):
    for p in (root / name, root / 'final' / name, root / 'intermediate' / name):
        if p.exists():
            return p
    return None


def key(df):
    for c in KEYS:
        if c in df.columns:
            return c
    raise KeyError('No patient identifier found')


def norm(s):
    return s.astype('string').str.strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cp = first_existing(CANONICAL_ROOT, 'stroke_cohort_final.parquet')
    ap = first_existing(AABILA_ROOT, 'stroke_cohort_final.parquet')
    if cp is None or ap is None:
        raise FileNotFoundError('Final cohort missing')

    c = pd.read_parquet(cp).copy()
    a = pd.read_parquet(ap).copy()
    ck, ak = key(c), key(a)
    c['__id'] = norm(c[ck])
    a['__id'] = norm(a[ak])
    a_ids = set(a['__id'].dropna())

    rules = {
        'all': pd.Series(True, index=c.index),
        'imaging_only': c['has_neuroimaging'].eq(1),
        'lipid_only': c['has_lipid_panel'].eq(1),
        'imaging_and_lipid': c['has_neuroimaging'].eq(1) & c['has_lipid_panel'].eq(1),
    }

    rows = []
    selected_name = None
    selected_ids = None
    best = None
    for name, mask in rules.items():
        ids = set(c.loc[mask, '__id'].dropna())
        score = len(ids - a_ids) + len(a_ids - ids)
        rows.append({
            'rule': name,
            'canonical_patients': len(ids),
            'aabila_patients': len(a_ids),
            'shared': len(ids & a_ids),
            'canonical_only': len(ids - a_ids),
            'aabila_only': len(a_ids - ids),
            'symmetric_difference': score,
        })
        if best is None or score < best:
            best = score
            selected_name = name
            selected_ids = ids

    pd.DataFrame(rows).sort_values('symmetric_difference').to_csv(OUT / 'candidate_rule_counts.csv', index=False)
    pd.DataFrame({'patient_id': sorted(selected_ids)}).to_csv(OUT / 'canonical_aligned_patient_ids.csv', index=False)
    summary = {
        'selected_rule': selected_name,
        'canonical_aligned_patients': len(selected_ids),
        'aabila_patients': len(a_ids),
        'shared': len(selected_ids & a_ids),
        'canonical_only': len(selected_ids - a_ids),
        'aabila_only': len(a_ids - selected_ids),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Candidate rule counts:', (OUT / 'candidate_rule_counts.csv').resolve())
    print('Aligned patient IDs:', (OUT / 'canonical_aligned_patient_ids.csv').resolve())


if __name__ == '__main__':
    main()
