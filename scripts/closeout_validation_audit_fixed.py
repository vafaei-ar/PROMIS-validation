#!/usr/bin/env python3
"""Fixed closeout validation audit.

This wrapper preserves the original closeout audit logic but fixes the PSU raw-DX
concatenation failure caused by duplicate column names in the selected output set.
Patient-level outputs remain under results/ and should not be committed.
"""
from __future__ import annotations

import json
import pandas as pd

import closeout_validation_audit as base


def psu_raw_dx_locator_fixed():
    ids = base.load_discordant_ids(base.SRC/'psu_dx/discrepancy_audit.csv')
    outdir = base.OUT/'psu_raw_dx'
    outdir.mkdir(parents=True, exist_ok=True)
    inventory = []
    hits = []
    keywords = ['diagn','dx','condition']

    for p in base.candidate_files(base.PSU_SEARCH_ROOTS, keywords):
        df = base.safe_read(p)
        if df is None:
            continue
        k = base.id_col(df)
        inventory.append({
            'file': str(p),
            'rows': len(df),
            'cols': len(df.columns),
            'id_col': k or '',
            'enc_col': base.enc_col(df) or '',
            'dx_cols': '|'.join([c for c in base.DX_CANDIDATES if c in df.columns]),
            'date_cols': '|'.join([c for c in base.DATE_CANDIDATES if c in df.columns]),
        })
        if not k or not ids:
            continue
        mask = base.norm(df[k]).isin(ids)
        if not mask.any():
            continue

        h = df.loc[mask].copy()
        h.insert(0, '__source_file', str(p))
        h.insert(1, '__id', base.norm(h[k]))

        requested = (
            ['__source_file', '__id']
            + list(base.ENC_CANDIDATES)
            + list(base.DX_CANDIDATES)
            + list(base.DXTYPE_CANDIDATES)
            + list(base.DXSRC_CANDIDATES)
            + list(base.PRIMARY_CANDIDATES)
            + list(base.DATE_CANDIDATES)
        )
        # Some semantic groups contain the same physical column name (for example
        # DX_TYPE), so de-duplicate the requested column list before concatenation.
        keep = list(dict.fromkeys(c for c in requested if c in h.columns))
        hits.append(h.loc[:, keep].drop_duplicates())

    pd.DataFrame(inventory).to_csv(outdir/'candidate_raw_dx_files.csv', index=False)
    if hits:
        all_hits = pd.concat(hits, ignore_index=True, sort=False)
        all_hits.to_csv(outdir/'discordant_patient_raw_dx_rows.csv', index=False)
        files_with_hits = int(all_hits['__source_file'].nunique())
        patients_hit = int(all_hits['__id'].nunique())
    else:
        pd.DataFrame(columns=['__source_file','__id']).to_csv(
            outdir/'discordant_patient_raw_dx_rows.csv', index=False
        )
        files_with_hits = 0
        patients_hit = 0

    return {
        'discordant_ids': len(ids),
        'candidate_files': len(inventory),
        'files_with_hits': files_with_hits,
        'patients_hit': patients_hit,
    }


def main():
    base.OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'psu_raw_dx': psu_raw_dx_locator_fixed(),
        'geisinger_edges': base.geisinger_edge_reconstruction(),
        'geisinger_dx_tiebreak': base.geisinger_dx_tiebreak_summary(),
    }
    (base.OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {base.OUT.resolve()}')


if __name__ == '__main__':
    main()
