#!/usr/bin/env python3
"""Fixed closeout validation audit.

Preserves the original closeout audit while fixing duplicate-column failures in
both PSU raw-DX extraction and Geisinger DX evidence consolidation. Patient-level
outputs remain under results/ and should not be committed.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

import closeout_validation_audit as base


def psu_raw_dx_locator_fixed():
    ids = base.load_discordant_ids(base.SRC/'psu_dx/discrepancy_audit.csv')
    outdir = base.OUT/'psu_raw_dx'
    outdir.mkdir(parents=True, exist_ok=True)
    inventory, hits = [], []
    keywords = ['diagn','dx','condition']

    for p in base.candidate_files(base.PSU_SEARCH_ROOTS, keywords):
        df = base.safe_read(p)
        if df is None:
            continue
        k = base.id_col(df)
        inventory.append({
            'file': str(p), 'rows': len(df), 'cols': len(df.columns),
            'id_col': k or '', 'enc_col': base.enc_col(df) or '',
            'dx_cols': '|'.join([c for c in base.DX_CANDIDATES if c in df.columns]),
            'date_cols': '|'.join([c for c in base.DATE_CANDIDATES if c in df.columns]),
            'is_output_or_intermediate': int('/outputs/' in str(p).replace('\\','/')),
        })
        if not k or not ids:
            continue
        mask = base.norm(df[k]).isin(ids)
        if not mask.any():
            continue
        h = df.loc[mask].copy()
        h.insert(0, '__source_file', str(p))
        h.insert(1, '__id', base.norm(h[k]))
        requested = (['__source_file','__id'] + list(base.ENC_CANDIDATES)
                     + list(base.DX_CANDIDATES) + list(base.DXTYPE_CANDIDATES)
                     + list(base.DXSRC_CANDIDATES) + list(base.PRIMARY_CANDIDATES)
                     + list(base.DATE_CANDIDATES))
        keep = list(dict.fromkeys(c for c in requested if c in h.columns))
        hits.append(h.loc[:, keep].drop_duplicates())

    inv = pd.DataFrame(inventory)
    inv.to_csv(outdir/'candidate_raw_dx_files.csv', index=False)
    if hits:
        all_hits = pd.concat(hits, ignore_index=True, sort=False)
        all_hits.to_csv(outdir/'discordant_patient_raw_dx_rows.csv', index=False)
        files_with_hits = int(all_hits['__source_file'].nunique())
        patients_hit = int(all_hits['__id'].nunique())
        direct_raw_files = int(sum('/outputs/' not in p.replace('\\','/') for p in all_hits['__source_file'].dropna().unique()))
    else:
        pd.DataFrame(columns=['__source_file','__id']).to_csv(outdir/'discordant_patient_raw_dx_rows.csv', index=False)
        files_with_hits = patients_hit = direct_raw_files = 0

    return {
        'discordant_ids': len(ids), 'candidate_files': len(inventory),
        'files_with_hits': files_with_hits, 'patients_hit': patients_hit,
        'direct_raw_files_with_hits': direct_raw_files,
        'limitation': 'If direct_raw_files_with_hits is 0, PSU DX remains independently unvalidated upstream of pipeline outputs.'
    }


def geisinger_dx_tiebreak_summary_fixed():
    candidates = []
    for root in [base.FINAL_ADJ, base.RAW_ADJ, base.SRC/'geisinger_dx']:
        if not root.exists():
            continue
        for p in root.rglob('*.csv'):
            low = p.name.lower()
            if 'dx' in low or 'diagn' in low:
                candidates.append(p)

    outdir = base.OUT/'geisinger_dx_tiebreak'
    outdir.mkdir(parents=True, exist_ok=True)
    inventory, combined = [], []
    for p in candidates:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        cols = set(df.columns)
        if not ({'canonical_DX','aabila_DX'} & cols) and not any(c in cols for c in base.DX_CANDIDATES):
            continue
        inventory.append({'file':str(p),'rows':len(df),'cols':len(df.columns)})
        temp = df.copy()
        if '__source_file' in temp.columns:
            temp.insert(0, '__audit_input_file', str(p))
        else:
            temp.insert(0, '__source_file', str(p))
        combined.append(temp)

    pd.DataFrame(inventory).to_csv(outdir/'input_inventory.csv', index=False)
    if combined:
        all_df = pd.concat(combined, ignore_index=True, sort=False)
        all_df.to_csv(outdir/'combined_dx_evidence.csv', index=False)
        patient_col = 'patient_id' if 'patient_id' in all_df.columns else ('__id' if '__id' in all_df.columns else None)
        summary = {
            'input_files':len(inventory), 'rows':len(all_df),
            'patients': int(all_df[patient_col].astype('string').nunique()) if patient_col else None,
        }
    else:
        pd.DataFrame().to_csv(outdir/'combined_dx_evidence.csv', index=False)
        summary = {'input_files':0,'rows':0,'patients':0}

    (outdir/'policy_note.txt').write_text(
        'Do not adjudicate DX from incidental raw row order. Both canonical and Aabila DX values are source-supported in the discordant Geisinger encounters. A production rule should explicitly prioritize qualifying ischemic-stroke diagnoses by documented primary-status semantics, encounter/date linkage, then a stable deterministic tie-break such as normalized ICD code order.\n'
    )
    return summary


def main():
    base.OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'psu_raw_dx': psu_raw_dx_locator_fixed(),
        'geisinger_edges': base.geisinger_edge_reconstruction(),
        'geisinger_dx_tiebreak': geisinger_dx_tiebreak_summary_fixed(),
        'geisinger_edge_limitation': 'Procedure sources do not carry encounter IDs; imaging evidence cannot be assigned to an encounter by ID alone and must be reconstructed using the configured date/time window. Lab orders do carry encounter IDs; other lab/procedure sources may require date linkage.'
    }
    (base.OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {base.OUT.resolve()}')


if __name__ == '__main__':
    main()
