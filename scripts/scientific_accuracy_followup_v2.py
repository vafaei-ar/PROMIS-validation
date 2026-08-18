#!/usr/bin/env python3
"""Patient-level scientific-accuracy follow-up.

Fixes two issues in the prior audit:
1) Geisinger evidence summaries are aggregated once per patient/anchor/window
   across all source date columns, rather than counting source-date-column rows.
2) PSU upstream search tolerates pre-existing bookkeeping columns.

All patient-level outputs remain under results/ and should not be committed.
"""
from __future__ import annotations

import json
import pandas as pd

import scientific_accuracy_followup_fixed as fixed
import scientific_accuracy_followup as base


def geisinger_patient_level():
    info = fixed.geisinger_edge_exact_reconstruction_fixed()
    outdir = base.OUT / 'geisinger_edges'
    detail_path = outdir / 'window_reconstruction_detail.csv'
    detail = pd.read_csv(detail_path, low_memory=False)
    if detail.empty:
        pd.DataFrame().to_csv(outdir / 'patient_level_evidence.csv', index=False)
        pd.DataFrame().to_csv(outdir / 'patient_level_summary.csv', index=False)
        return info | {'patient_level_rows': 0}

    # Collapse all source-date-column rows to one row per patient/anchor/window.
    patient = (
        detail.groupby(['patient_id', 'anchor', 'window_days'], as_index=False)
        .agg(
            imaging_evidence=('imaging_evidence', 'max'),
            lipid_evidence=('lipid_evidence', 'max'),
            n_imaging_rows=('n_imaging_rows', 'sum'),
            n_lipid_rows=('n_lipid_rows', 'sum'),
        )
    )
    patient['both_evidence'] = (
        patient['imaging_evidence'].eq(1) & patient['lipid_evidence'].eq(1)
    ).astype(int)
    patient.to_csv(outdir / 'patient_level_evidence.csv', index=False)

    summary = (
        patient.groupby(['anchor', 'window_days'], as_index=False)
        .agg(
            patients=('patient_id', 'nunique'),
            imaging_positive=('imaging_evidence', 'sum'),
            lipid_positive=('lipid_evidence', 'sum'),
            both_positive=('both_evidence', 'sum'),
        )
    )
    for c in ['imaging_positive', 'lipid_positive', 'both_positive']:
        summary[c + '_rate'] = summary[c] / summary['patients']
    summary.to_csv(outdir / 'patient_level_summary.csv', index=False)

    # Direct canonical-vs-Aabila comparison at the configured 7-day window.
    seven = patient[patient['window_days'].eq(7)].pivot_table(
        index='patient_id', columns='anchor', values='both_evidence', aggfunc='max'
    ).fillna(0).astype(int)
    if {'canonical', 'aabila'}.issubset(seven.columns):
        seven['aabila_only_both'] = ((seven['aabila'] == 1) & (seven['canonical'] == 0)).astype(int)
        seven['canonical_only_both'] = ((seven['canonical'] == 1) & (seven['aabila'] == 0)).astype(int)
        seven['both_anchors'] = ((seven['canonical'] == 1) & (seven['aabila'] == 1)).astype(int)
        seven['neither_anchor'] = ((seven['canonical'] == 0) & (seven['aabila'] == 0)).astype(int)
        seven.reset_index().to_csv(outdir / 'configured_7d_anchor_comparison.csv', index=False)

    return info | {'patient_level_rows': int(len(patient))}


def psu_upstream_raw_dx_search_fixed():
    ids_path = base.SRC / 'psu_dx/discrepancy_audit.csv'
    idsdf = pd.read_csv(ids_path)
    pidc = 'patient_id' if 'patient_id' in idsdf.columns else idsdf.columns[0]
    ids = set(base.norm(idsdf[pidc]))
    outdir = base.OUT / 'psu_raw_dx'
    outdir.mkdir(parents=True, exist_ok=True)

    inventory, hits = [], []
    for p in base.ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.parquet', '.csv', '.txt'}:
            continue
        low = p.name.lower()
        if not any(k in low for k in ['diagn', 'dx', 'condition']):
            continue
        pathlow = str(p).lower().replace('\\', '/')
        derived = any(x in pathlow for x in ['/outputs/', '/results/', '/reproduced/', '/intermediate/'])
        df = base.read_table(p)
        if df is None:
            continue
        k = base.id_col(df)
        inventory.append({
            'file': str(p), 'derived_path': derived, 'rows': len(df), 'cols': len(df.columns),
            'id_col': k or '', 'enc_col': base.enc_col(df) or '',
            'dx_cols': '|'.join(c for c in base.DX_COLS if c in df.columns),
        })
        if not k:
            continue
        m = base.norm(df[k]).isin(ids)
        if not m.any():
            continue
        h = df.loc[m].copy()
        # Assign instead of insert so reruns / nested derived files cannot collide.
        h['__source_file'] = str(p)
        h['__derived_path'] = derived
        h['__id'] = base.norm(h[k])
        requested = ['__source_file', '__derived_path', '__id'] + [
            c for c in df.columns
            if c in base.ENC_COLS + base.DX_COLS
            or 'DATE' in c.upper() or 'PRIMARY' in c.upper()
            or 'ORIGIN' in c.upper() or 'TYPE' in c.upper()
        ]
        keep = list(dict.fromkeys(c for c in requested if c in h.columns))
        hits.append(h.loc[:, keep].drop_duplicates())

    inv = pd.DataFrame(inventory)
    inv.to_csv(outdir / 'diagnosis_file_inventory.csv', index=False)
    allhits = pd.concat(hits, ignore_index=True, sort=False) if hits else pd.DataFrame(
        columns=['__source_file', '__derived_path', '__id']
    )
    allhits.to_csv(outdir / 'discordant_patient_diagnosis_rows.csv', index=False)
    direct = allhits[allhits['__derived_path'].eq(False)].copy() if len(allhits) else pd.DataFrame()
    direct.to_csv(outdir / 'direct_upstream_hits.csv', index=False)

    return {
        'discordant_ids': len(ids),
        'candidate_files': int(len(inv)),
        'patients_any_hit': int(allhits['__id'].nunique()) if len(allhits) else 0,
        'direct_upstream_files_with_hits': int(direct['__source_file'].nunique()) if len(direct) else 0,
        'patients_direct_upstream_hit': int(direct['__id'].nunique()) if len(direct) else 0,
    }


def main():
    base.OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'geisinger_edge_reconstruction': geisinger_patient_level(),
        'psu_upstream_raw_dx': psu_upstream_raw_dx_search_fixed(),
    }
    base.write_indicator_proposal()
    (base.OUT / 'summary_v2.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))
    print(f'Outputs: {base.OUT.resolve()}')


if __name__ == '__main__':
    main()
