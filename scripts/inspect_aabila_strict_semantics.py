#!/usr/bin/env python3
"""Inspect Aabila Geisinger strict phenotype semantics without modifying data.

Outputs the exact imaging/lipid filter function bodies, relevant config lines,
raw table schemas, and any intermediate/output tables that already contain
encounter-level imaging/lipid flags. This is used to implement the strict
alternative index exactly rather than approximating the legacy semantics.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
AAB = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
OUT = ROOT / 'PROMIS-validation/results/aabila_strict_semantics'


def find_script() -> Path:
    candidates = [
        AAB / 'stroke/stroke_extraction_scripts/02_extract_stroke_patients.py',
        AAB / 'stroke_extraction_scripts/02_extract_stroke_patients.py',
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(AAB.rglob('02_extract_stroke_patients.py'))
    if not hits:
        raise FileNotFoundError('Aabila 02_extract_stroke_patients.py not found')
    return hits[0]


def extract_function_source(path: Path, names: set[str]) -> dict[str, str]:
    text = path.read_text(errors='ignore')
    tree = ast.parse(text)
    lines = text.splitlines()
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            out[node.name] = '\n'.join(lines[start:end])
    return out


def relevant_config_lines() -> list[dict]:
    patt = re.compile(r'(imaging|lipid|time_window|window|tolerance|match_tolerance|cpt|loinc)', re.I)
    rows = []
    for p in AAB.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in {'.yaml','.yml','.json','.toml','.ini','.cfg','.py'}:
            continue
        try:
            lines = p.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if patt.search(line):
                rows.append({'file': str(p), 'line': i, 'text': line.strip()})
    return rows


def parquet_schema_inventory() -> list[dict]:
    rows = []
    roots = [AAB/'dataset', AAB/'outputs', AAB/'stroke']
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*.parquet'):
            if p in seen:
                continue
            seen.add(p)
            low = p.name.lower()
            if not any(k in low for k in ('diagn','encounter','procedure','lab','stroke','cohort','patient')):
                continue
            try:
                df = pd.read_parquet(p)
            except Exception as e:
                rows.append({'file': str(p), 'error': str(e)})
                continue
            cols = list(map(str, df.columns))
            flag_cols = [c for c in cols if any(k in c.upper() for k in ('IMAGING','LIPID','HAS_IMG','HAS_LIP'))]
            rows.append({
                'file': str(p),
                'rows': int(len(df)),
                'columns': cols,
                'flag_columns': flag_cols,
            })
    return rows


def candidate_flag_tables(inv: list[dict]) -> list[dict]:
    out = []
    for r in inv:
        if r.get('error'):
            continue
        flags = r.get('flag_columns') or []
        cols = set(r.get('columns') or [])
        has_patient = bool(cols & {'PATID','STUDY_ID','PT_ID','PAT_ID'})
        has_enc = bool(cols & {'ENCOUNTERID','ENCOUNTER_ID','ENC_ID'})
        if flags and has_patient and has_enc:
            out.append({
                'file': r['file'],
                'rows': r['rows'],
                'flag_columns': flags,
                'id_columns': sorted(cols & {'PATID','STUDY_ID','PT_ID','PAT_ID'}),
                'encounter_columns': sorted(cols & {'ENCOUNTERID','ENCOUNTER_ID','ENC_ID'}),
            })
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    script = find_script()
    funcs = extract_function_source(script, {
        'apply_imaging_filter',
        'apply_lipid_filter',
        'filter_imaging',
        'filter_lipid',
    })
    (OUT/'filter_functions.txt').write_text(
        '\n\n'.join(f'### {k}\n{v}' for k,v in funcs.items()) + '\n'
    )

    cfg_rows = relevant_config_lines()
    pd.DataFrame(cfg_rows).to_csv(OUT/'relevant_config_lines.csv', index=False)

    inv = parquet_schema_inventory()
    pd.DataFrame([
        {
            'file': r.get('file'),
            'rows': r.get('rows'),
            'flag_columns': '|'.join(r.get('flag_columns') or []),
            'columns': '|'.join(r.get('columns') or []),
            'error': r.get('error',''),
        }
        for r in inv
    ]).to_csv(OUT/'parquet_schema_inventory.csv', index=False)

    flag_tables = candidate_flag_tables(inv)
    pd.DataFrame(flag_tables).to_csv(OUT/'candidate_encounter_flag_tables.csv', index=False)

    summary = {
        'script': str(script),
        'filter_functions_found': sorted(funcs),
        'relevant_config_lines': len(cfg_rows),
        'parquet_tables_inventoried': len(inv),
        'candidate_encounter_flag_tables': flag_tables,
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {OUT}')


if __name__ == '__main__':
    main()
