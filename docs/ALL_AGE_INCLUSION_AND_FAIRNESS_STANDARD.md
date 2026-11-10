# Inclusive analysis and age-coverage standard

This repository contains projects across health, finance, retail, transport, public services and language technology. Demographic analysis must be broad, evidence-led and explicit about who a dataset and model can validly represent.

## All-age principle

Where age is relevant and the source data support it, analyse the full observed age range rather than selecting one convenient demographic. Include children, young people, working-age adults and older people in population descriptions and service-planning scenarios when those groups are present in the source.

Use the age bands supplied by the source wherever possible. If a project needs a different grouping, document the mapping and reconcile group totals to the source population. Do not manufacture detailed age splits from broad totals.

## Minimum age reporting

For every project that uses age or age-sensitive decisions, document:

1. The source-defined minimum and maximum ages and any exclusions.
2. The age-band definitions and sample counts, including missing/unknown age.
3. Performance or outcome differences across age groups when sample sizes permit.
4. Confidence intervals or uncertainty where appropriate, and small-cell suppression where disclosure risk exists.
5. Whether the model is intended for all ages, a narrower validated population, or descriptive analysis only.

Do not imply that a model supports an age group merely because the age field exists in the dataset. Report subgroup sample sizes and avoid strong conclusions from sparse groups.

## Model validity is not the same as population inclusion

A project can describe an all-age population while correctly limiting a model to the population for which it was developed and evaluated.

In particular, adult-derived clinical biomarkers, biological-age equations and risk models must not be extrapolated to children. Pediatric use requires an appropriate pediatric reference population, age-specific feature interpretation, independent validation and clinical governance. If these are unavailable, label the model unsupported for children and do not generate individual predictions for them.

Likewise, do not assume one threshold, calibration curve or error profile is equally suitable across every age group. Assess subgroup calibration, false-positive/false-negative rates and decision consequences before proposing deployment.

## Broader fairness checks

Where the source contains suitable, lawful and sufficiently populated fields, consider intersections with sex, geography, disability, socioeconomic circumstances, language and other relevant access factors. Do not infer sensitive characteristics. Explain missingness and source limitations.

For operational projects, ask whether the system creates unequal access, service quality, pricing, prioritisation or escalation. Keep human review available for high-impact or uncertain decisions.

## Evidence and reproducibility

- Cite the exact source, release, table and access date.
- Keep observed facts separate from scenario assumptions and synthetic examples.
- Set random seeds for stochastic analysis and record them.
- Preserve the original notebook or dataset snapshot; add enhancements as companion modules, tests or documentation.
- Add code comments beside non-obvious transformations, leakage controls, metric choices, thresholds and policy assumptions. Comments should explain *why* a decision is made, not merely restate the syntax.
- Test cohort reconciliation, missing values, boundary ages, subgroup calculations and deterministic outputs.

## Hiring-facing project checklist

A reviewer should be able to find the decision question, source provenance, intended population, method, baseline, validation design, key results, limitations, reproducibility instructions and practical implications without reverse-engineering the notebook.

This standard is a quality-control guide. It does not certify fairness, clinical validity or regulatory compliance.
