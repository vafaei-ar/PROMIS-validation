#!/usr/bin/env python3
"""Robust wrapper for scientific_accuracy_followup.py.

Fixes pandas NA handling in row-wise text classification without changing the
scientific logic of the original validation workflow.
"""
from __future__ import annotations

import json
import pandas as pd

import scientific_accuracy_followup as base


def _row_text(df: pd.DataFrame) -> pd.Series:
    # Convert missing values to empty strings before joining row text. Using
    # pandas StringDtype directly leaves pd.NA objects that Python str.join
    # cannot concatenate.
    safe = df.astype('string').fillna('')
    return safe.apply(lambda row: ' | '.join(row.astype(str).tolist()), axis=1)


def classify_imaging_text_fixed(df: pd.DataFrame):
    txt = _row_text(df)
    return txt.str.contains(
        r'\b(MRI|MRA|CT|CTA|HEAD CT|BRAIN|NEUROIMAG|IMAGING)\b',
        case=False,
        regex=True,
        na=False,
    )


def classify_lipid_text_fixed(df: pd.DataFrame):
    txt = _row_text(df)
    return txt.str.contains(
        r'\b(LIPID|LDL|HDL|TRIG(?:LYCERIDE)?|CHOLESTEROL)\b',
        case=False,
        regex=True,
        na=False,
    )


def main():
    base.classify_imaging_text = classify_imaging_text_fixed
    base.classify_lipid_text = classify_lipid_text_fixed

    base.OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'geisinger_edge_reconstruction': base.geisinger_edge_exact_reconstruction(),
        'psu_upstream_raw_dx': base.psu_upstream_raw_dx_search(),
    }
    base.write_indicator_proposal()
    (base.OUT/'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))
    print(f'Outputs: {base.OUT.resolve()}')


if __name__ == '__main__':
    main()
