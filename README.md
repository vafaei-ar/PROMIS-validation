# PROMIS-validation

Validation workspace for comparing the canonical PROMIS stroke pipeline with earlier PSU and Geisinger implementations and for adjudicating important differences against source data.

> **Start here:** see [`docs/README.md`](docs/README.md) for the complete, plain-language validation record.

## What this repository is for

This repository is an **audit and scientific validation workspace**, not the production pipeline.

The production implementation is maintained in `TheDecodeLab/PROMIS-ML-pipeline`.

The validation work answers four main questions:

1. Can older PSU/Geisinger outputs be reproduced when their exact rules are replayed?
2. Which differences represent real bugs versus different phenotype definitions?
3. Which diagnosis differences are caused by multiple valid source rows and tie/order behavior?
4. Which findings should become production fixes, provenance fields, or sensitivity analyses?

## Current scientific conclusion

Use the harmonized **broad cohort** for primary analyses and the evidence-qualified **strict phenotype** for sensitivity/replication analyses.

Current validated counts:

| Site | Broad | Strict |
|---|---:|---:|
| PSU | 9,835 | 6,009 |
| Geisinger | 13,872 | 11,910 |

The exact Geisinger legacy strict selection was reproduced at 11,910 patients with zero encounter-ID mismatches after the correct imaging/lipid rules and 214-code lipid whitelist were identified.

## Documentation

- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — experiments in order, including rejected hypotheses
- [`docs/FINDINGS_AND_DECISIONS.md`](docs/FINDINGS_AND_DECISIONS.md) — final scientific conclusions
- [`docs/SCRIPT_MAP.md`](docs/SCRIPT_MAP.md) — which scripts are final, historical, diagnostic, or prototypes

## Important validation rule

A legacy implementation is a comparison target, not automatically the scientific truth. A production definition is changed only when source-code review, raw-data adjudication, or reproducible evidence supports the change.

## Data handling

Generated `results/` folders and reproduced datasets can contain patient identifiers. They should remain local and are intentionally excluded from version control.

## Historical baseline comparison

The original general comparison entry point remains:

```bash
python scripts/compare_aabila.py
```

Use `python scripts/compare_aabila.py --help` for local path overrides. For current interpretation, read the documentation above before treating any single historical script output as the final answer.
