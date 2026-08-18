from pathlib import Path
import importlib.util
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prototype_sensitivity_flags.py'
spec = importlib.util.spec_from_file_location('prototype_sensitivity_flags', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_strict_index_preserves_broad_and_flags_sensitive_change():
    first = pd.DataFrame({
        'PATID': ['A', 'B', 'C'],
        'ENCOUNTERID': ['A1', 'B1', 'C1'],
        'ADMIT_DATE': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-01']),
        'DX': ['I63.9', 'I63.9', 'I63.9'],
    })
    all_enc = pd.DataFrame({
        'PATID': ['A', 'A', 'B', 'C', 'C'],
        'ENCOUNTERID': ['A1', 'A2', 'B1', 'C1', 'C2'],
        'ADMIT_DATE': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-01', '2020-01-01', '2020-03-01']),
        'DX': ['I63.9'] * 5,
        'has_neuroimaging': [0, 1, 1, 1, 1],
        'has_lipid_panel': [0, 1, 1, 1, 1],
    })

    out = mod.strict_index_flags(first, all_enc).set_index('PATID')

    # A: broad index fails strict evidence, later encounter qualifies.
    assert out.loc['A', 'broad_index_encounter_id'] == 'A1'
    assert out.loc['A', 'strict_index_encounter_id'] == 'A2'
    assert out.loc['A', 'phenotype_strict_eligible'] == 1
    assert out.loc['A', 'index_selection_sensitive'] == 1
    assert out.loc['A', 'has_alt_qualifying_stroke_encounter'] == 1

    # B: broad index itself qualifies; no alternative.
    assert out.loc['B', 'strict_index_encounter_id'] == 'B1'
    assert out.loc['B', 'index_selection_sensitive'] == 0
    assert out.loc['B', 'has_alt_qualifying_stroke_encounter'] == 0

    # C: broad qualifies and a later qualifying alternative exists; index remains stable.
    assert out.loc['C', 'strict_index_encounter_id'] == 'C1'
    assert out.loc['C', 'index_selection_sensitive'] == 0
    assert out.loc['C', 'has_alt_qualifying_stroke_encounter'] == 1


def test_dx_ambiguity_is_flagged_without_overwriting_current_dx():
    first = pd.DataFrame({
        'PATID': ['A', 'B'],
        'ENCOUNTERID': ['A1', 'B1'],
        'DX': ['I63.9', 'I63.1'],
    })
    raw = pd.DataFrame({
        'PATID': ['A', 'A', 'B'],
        'ENCOUNTERID': ['A1', 'A1', 'B1'],
        'DX': ['I63.9', 'I63.1', 'I63.1'],
        'PDX': ['P', 'P', 'P'],
        'DX_DATE': pd.to_datetime(['2020-01-02', '2020-01-01', '2020-01-01']),
    })

    out, meta = mod.dx_ambiguity_flags(first, raw)
    a = out.set_index('PATID').loc['A']
    b = out.set_index('PATID').loc['B']

    assert a['n_qualifying_primary_stroke_dx'] == 2
    assert a['multiple_primary_stroke_dx'] == 1
    assert a['dx_tiebreak_applied'] == 1
    assert a['dx_deterministic'] == 'I63.1'
    assert a['DX'] == 'I63.9'  # original remains visible, not silently replaced
    assert a['dx_deterministic_differs_from_current'] == 1

    assert b['n_qualifying_primary_stroke_dx'] == 1
    assert b['multiple_primary_stroke_dx'] == 0
    assert b['dx_tiebreak_applied'] == 0
    assert meta['multiple_primary_stroke_dx_patients'] == 1


def test_validation_detects_no_semantic_errors_for_valid_flags():
    first = pd.DataFrame({'PATID': ['A'], 'ENCOUNTERID': ['A1']})
    flags = pd.DataFrame({
        'PATID': ['A'],
        'broad_index_encounter_id': ['A1'],
        'strict_index_encounter_id': ['A2'],
        'phenotype_strict_eligible': [1],
        'index_selection_sensitive': [1],
    })
    assert mod.validate_flags(flags, first) == []
