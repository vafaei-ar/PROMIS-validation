#!/usr/bin/env python3
"""Exact Aabila Geisinger replay with the two previously omitted gates.

Restores:
1) ATN_reg_ptid patient-pool override when present.
2) the lipid LOINC whitelist using an equivalent local copy when the original
   /home/atharzeen path is unavailable.

Writes comparison and mismatch diagnostics under results/aabila_exact_replay_v4.
"""
from pathlib import Path
import importlib.util, os, sys, json
import pandas as pd

ROOT = Path('/home/asadr/works/promis2')
VAL = ROOT / 'PROMIS-validation'
GEI = ROOT / 'aabila/PROMIS-ML-pipeline-master-Geisinger_07162026'
SCRIPT = GEI / 'stroke/stroke_extraction_scripts/02_extract_stroke_patients.py'
OUT = VAL / 'results/aabila_exact_replay_v4'


def norm(s):
    return s.astype('string').str.strip()


def loadmod():
    sys.path.insert(0, str(SCRIPT.parent))
    oldcwd = os.getcwd()
    real_makedirs = os.makedirs
    def safe_makedirs(path, *args, **kwargs):
        if str(path).startswith('/home/atharzeen/Research'):
            return None
        return real_makedirs(path, *args, **kwargs)
    os.makedirs = safe_makedirs
    os.chdir(GEI)
    try:
        spec = importlib.util.spec_from_file_location('aabila_extract02', SCRIPT)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        os.makedirs = real_makedirs
        os.chdir(oldcwd)


def local_input_dir():
    for p in [GEI / 'dataset/parquet', GEI / 'dataset']:
        if (p / 'encounter.parquet').exists() and (p / 'diagnosis.parquet').exists():
            return p
    raise FileNotFoundError('Could not locate local Geisinger parquet input directory')


def local_lipid_csv():
    candidates = [
        GEI / 'stroke/external_data/lipid_corrected_by_harold.csv',
        GEI / 'external_data/lipid_corrected_by_harold.csv',
        ROOT / 'codes/PROMIS-ML-pipeline/external_data/lipid_corrected_by_harold.csv',
        ROOT / 'PROMIS-ML-pipeline/external_data/lipid_corrected_by_harold.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(ROOT.rglob('lipid_corrected_by_harold.csv'))
    return hits[0] if hits else None


def patch_config(cfg, inp, lipid_csv):
    original_get = cfg.get
    def patched_get(key, default=None):
        if key == 'data_paths.input_data_dir':
            return str(inp)
        if key == 'data_paths.lipid_loinc_csv' and lipid_csv is not None:
            return str(lipid_csv)
        return original_get(key, default)
    try:
        cfg.get = patched_get
    except Exception:
        pass
    for attr in ('config', '_config', 'data', '_data', 'cfg'):
        obj = getattr(cfg, attr, None)
        if isinstance(obj, dict):
            obj.setdefault('data_paths', {})['input_data_dir'] = str(inp)
            if lipid_csv is not None:
                obj['data_paths']['lipid_loinc_csv'] = str(lipid_csv)


m = loadmod()
cfg = m.load_config()
inp = local_input_dir()
lipid_csv = local_lipid_csv()
patch_config(cfg, inp, lipid_csv)
crit = cfg.get_patient_selection_criteria()

# Load and normalize raw tables exactly as the Aabila script does.
en = m.standardize_encounter_df(m.apply_mapping(m.load_parquet_safe(str(inp/'encounter.parquet')), 'encounter.parquet'))
dx = m.apply_mapping(m.load_parquet_safe(str(inp/'diagnosis.parquet')), 'diagnosis.parquet')
for d in (en, dx):
    d['PATID'] = norm(d['PATID'])
    d['ENCOUNTERID'] = norm(d['ENCOUNTERID'])
dx['DX'] = dx['DX'].astype(str).str.strip().str.upper()

# Diagnosis-side encounter date merge used by Aabila.
ed = en[['ENCOUNTERID', 'ENC_DT']].copy()
ed['ENC_DT'] = pd.to_datetime(ed['ENC_DT'], errors='coerce')
ed = ed.dropna(subset=['ENC_DT']).sort_values(['ENCOUNTERID', 'ENC_DT']).drop_duplicates('ENCOUNTERID')
target = 'ENC_DT' if 'ENC_DT' not in dx.columns else 'ENC_DT_FROM_ENC'
ed = ed.rename(columns={'ENC_DT': target})
dx = dx.merge(ed, on='ENCOUNTERID', how='left', validate='m:1')
dx[target] = pd.to_datetime(dx[target], errors='coerce').dt.strftime('%Y-%m-%d')
if 'ENC_DT' in dx.columns:
    dx.rename(columns={'ENC_DT': 'DX_DATE'}, inplace=True)
elif 'DX_DATE' not in dx.columns and 'ENC_DT_FROM_ENC' in dx.columns:
    dx['DX_DATE'] = dx['ENC_DT_FROM_ENC']

# Reproduce the effective patient-pool restriction. Aabila explicitly overrides
# with this file when it exists.
reg_path = GEI / 'stroke/external_data/ATN_reg_ptid.csv'
registry_ids = None
if reg_path.exists():
    reg = pd.read_csv(reg_path)
    if 'PATID' in reg.columns:
        registry_ids = set(norm(reg['PATID']).dropna())
        dx = dx[dx['PATID'].isin(registry_ids)].copy()

if 'ENC_TYPE' in dx.columns:
    dx = dx[dx['ENC_TYPE'].isin(crit['encounter_types'])].copy()

matcher = getattr(m, '_is_stroke_dx', None) or getattr(m, 'stroke_dx_mask', None)
if matcher is None:
    raise AttributeError('Could not find Aabila stroke diagnosis matcher')
try:
    smask = matcher(dx['DX'])
except TypeError:
    smask = matcher(dx['DX'], cfg)
sdx = dx[smask].copy()

if crit['require_primary_diagnosis']:
    if 'DX_PRIMARY_YN' in sdx.columns:
        sdx = sdx[sdx['DX_PRIMARY_YN'] == 1].copy()
    elif 'PDX' in sdx.columns:
        sdx = sdx[norm(sdx['PDX']).str.upper().isin({'P','1','Y','YES','TRUE'})].copy()

cand = en[en['ENCOUNTERID'].isin(sdx['ENCOUNTERID'].unique())].copy()
cand['ADMIT_DATE'] = pd.to_datetime(cand['ADMIT_DATE'], errors='coerce')
cand['DISCHARGE_DATE'] = pd.to_datetime(cand['DISCHARGE_DATE'], errors='coerce')
cand['enc_duration'] = (cand['DISCHARGE_DATE'] - cand['ADMIT_DATE']).dt.days
if crit['min_encounter_duration'] is not None:
    cand = cand[cand['enc_duration'] >= crit['min_encounter_duration']].copy()
n0 = (len(cand), cand['PATID'].nunique())

cand = m.apply_imaging_filter(cand, cfg)
n1 = (len(cand), cand['PATID'].nunique())
cand = m.apply_lipid_filter(cand, cfg)
n2 = (len(cand), cand['PATID'].nunique())

sdx['DX_DATE'] = pd.to_datetime(sdx['DX_DATE'], errors='coerce')
r = cand.merge(sdx[['ENCOUNTERID','DX','DX_DATE']], on='ENCOUNTERID', how='left').rename(columns={'DX_DATE':'DX_DATE_stroke'})
adm = pd.to_datetime(r['ENC_ADM_DTTM'], errors='coerce').dt.normalize() if 'ENC_ADM_DTTM' in r.columns else pd.to_datetime(r['ADMIT_DATE'], errors='coerce').dt.normalize()
encdt = pd.to_datetime(r['ENC_DT'], errors='coerce') if 'ENC_DT' in r.columns else pd.Series(pd.NaT, index=r.index)
r['DX_DATE_stroke'] = (pd.to_datetime(r['DX_DATE_stroke'], errors='coerce')
    .combine_first(adm).combine_first(encdt)
    .combine_first(pd.to_datetime(r['ADMIT_DATE'], errors='coerce')))
first = r.sort_values('DX_DATE_stroke').groupby('PATID').first().reset_index()

# Post-selection age handling.
dm = m.apply_mapping(m.load_parquet_safe(str(inp/'demographic.parquet')), 'demographic.parquet')
dm['PATID'] = norm(dm['PATID'])
first = m.calculate_age_at_stroke(dm, first)
if 'AGE_AT_STROKE' in first.columns:
    first.loc[first['AGE_AT_STROKE'] > 89, 'AGE_AT_STROKE'] = 89
first = m.filter_by_age(first, crit['min_age'], crit['max_age'])

saved_path = GEI / 'outputs/first_stroke_encounters.parquet'
if not saved_path.exists():
    hits = list((GEI/'outputs').rglob('first_stroke_encounters.parquet'))
    if not hits:
        raise FileNotFoundError('Saved Aabila first_stroke_encounters.parquet not found')
    saved_path = hits[0]
saved = pd.read_parquet(saved_path).copy()
for d in (first, saved):
    d['PATID'] = norm(d['PATID'])
    d['ENCOUNTERID'] = norm(d['ENCOUNTERID'])

cmp = saved[['PATID','ENCOUNTERID']].drop_duplicates('PATID').rename(columns={'ENCOUNTERID':'saved'}).merge(
    first[['PATID','ENCOUNTERID']].drop_duplicates('PATID').rename(columns={'ENCOUNTERID':'replay'}),
    on='PATID', how='outer')
cmp['match'] = cmp['saved'].eq(cmp['replay'])

OUT.mkdir(parents=True, exist_ok=True)
cmp.to_csv(OUT/'encounter_comparison.csv', index=False)

mismatch_ids = set(cmp.loc[cmp['saved'].notna() & cmp['replay'].notna() & ~cmp['match'], 'PATID'])
diag_cols = [c for c in ['PATID','ENCOUNTERID','DX_DATE_stroke','ADMIT_DATE','DISCHARGE_DATE','ENC_DT','DX','HAS_IMAGING','HAS_LIPID'] if c in r.columns]
detail = r[r['PATID'].isin(mismatch_ids)][diag_cols].copy()
if not detail.empty:
    detail = detail.sort_values([c for c in ['PATID','DX_DATE_stroke','ADMIT_DATE','ENCOUNTERID'] if c in detail.columns])
detail.to_csv(OUT/'mismatch_candidate_detail.csv', index=False)

edge = VAL / 'results/source_truth_audit/geisinger_edges/edge_patients.csv'
edge_n = edge_match = None
if edge.exists():
    e = pd.read_csv(edge)
    idc = '__id' if '__id' in e.columns else ('patient_id' if 'patient_id' in e.columns else e.columns[0])
    ids = set(norm(e[idc]))
    ec = cmp[cmp['PATID'].isin(ids)]
    edge_n = int(ec['PATID'].nunique())
    edge_match = int(ec['match'].sum())
    ec.to_csv(OUT/'edge_encounter_comparison.csv', index=False)

summary = {
    'local_input_dir': str(inp),
    'lipid_whitelist': str(lipid_csv) if lipid_csv else None,
    'registry_override_file': str(reg_path) if reg_path.exists() else None,
    'registry_override_patients': len(registry_ids) if registry_ids is not None else None,
    'before_imaging_rows': n0[0], 'before_imaging_patients': n0[1],
    'after_imaging_rows': n1[0], 'after_imaging_patients': n1[1],
    'after_lipid_rows': n2[0], 'after_lipid_patients': n2[1],
    'replay_first_patients': int(first['PATID'].nunique()),
    'saved_first_patients': int(saved['PATID'].nunique()),
    'encounter_matches': int(cmp['match'].fillna(False).sum()),
    'saved_and_replay_encounter_mismatches': int((cmp['saved'].notna() & cmp['replay'].notna() & cmp['saved'].ne(cmp['replay'])).sum()),
    'replay_only_patients': int((cmp['saved'].isna() & cmp['replay'].notna()).sum()),
    'saved_only_patients': int((cmp['saved'].notna() & cmp['replay'].isna()).sum()),
    'edge_patients': edge_n,
    'edge_replay_matches_saved': edge_match,
}
(OUT/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
