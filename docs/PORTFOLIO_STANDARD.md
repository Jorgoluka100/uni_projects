# Portfolio Standard — final hiring structure

This repository has three deliberately different kinds of evidence. They should **not** be flattened into one template.

## Tier 1 — protected original university/course work

These notebooks are historical evidence of work completed during university/course/laboratory study. They remain visible at repository root and are linked from the main README.

**Rule:** preserve them. Do not delete, rename, silently replace or rewrite them just to make the portfolio look more modern.

`protected_originals.json` records the required originals. `scripts/validate_protected_originals.py` runs in Portfolio Integrity CI and fails if a protected notebook disappears; the core university/course notebooks are also pinned to their current Git blob SHA so an automated cleanup cannot silently overwrite them.

Professional follow-on projects may strengthen the same skill alongside the original. They never replace the original.

## Tier 2 — DataCamp-style foundations

`skills/` contains compact, interview-friendly learning projects. These are intentionally smaller than the professional applications.

A good foundation notebook should make one skill easy to inspect from start to finish:

1. clear objective / question
2. small or built-in dataset / reproducible input
3. data inspection and necessary cleaning/preprocessing
4. direct code rather than abstraction-heavy wrappers
5. at least one useful table or visualisation where the skill supports it
6. method/model implementation
7. appropriate metric/check/output
8. example inference or worked result where relevant
9. concise conclusion / what the exercise proves

The point is the same as a strong guided project: apply a skill to a complete analysis, not pretend a small exercise is production software.

Current foundation coverage includes data cleaning, NumPy, classification, PyTorch, LSTM, TF-IDF NLP, CNNs, Linear Regression/Ridge, clustering and SQL.

## Tier 3 — full professional applications

`projects/` contains the recruiter-facing end-to-end applications. Each project must stand alone and answer a hiring manager's question: **can this candidate take a real data/AI problem and solve it responsibly?**

Where technically relevant, a full application should show directly in `project_notebook.ipynb`:

1. problem, stakeholder and success criteria
2. dataset provenance and reproducible acquisition
3. schema and data-quality checks
4. cleaning and preprocessing
5. substantial EDA and visualisation
6. feature engineering / transformations
7. simple baseline
8. meaningful alternative model/approach comparison
9. leakage-aware validation / temporal or grouped split where appropriate
10. tuning / ablation where useful
11. relevant metrics
12. residual, confusion, slice or failure analysis
13. robustness / missingness / outlier / concentration checks
14. explainability or feature interpretation where useful
15. uncertainty / calibration / confidence policy where useful
16. inference, business rule or operational decision layer
17. retained results and evidence
18. tests and sanity checks
19. API / pipeline / SQL / dbt / Spark / BI / Docker / CI / monitoring when role-relevant
20. limitations, risks and next steps
21. a concise final decision based on measured evidence

The notebook is the primary recruiter artifact. Modular `.py`, `src/`, SQL, APIs and tests remain because they demonstrate engineering quality, but they must not hide the actual analysis.

### Depth rule

For a major application, **roughly 800–1,200+ meaningful visible code lines** is a useful working range when the problem supports it. Around 1,000 is a guide, not a quota. A shorter complete project is better than a padded notebook.

The automated notebook pipeline now runs:

`build → preserve/enrich → visual analysis → robustness/slices/decision analysis → validate → commit`

The full-project validator checks all 21 applications for a notebook, README, Python/engineering evidence, project story and retained evidence route.

## Why the two project styles are different

DataCamp describes its Projects as applying skills in notebooks to solve real-world problems and complete an analysis from start to finish. MIT's Hands-On Deep Learning project guidance asks students to define the problem, dataset and approach, then report problem description, approach, results and lessons learned. This portfolio uses the same core idea while separating compact skill demonstrations from larger employer-facing systems.

References:
- DataCamp Projects: https://www.datacamp.com/projects
- MIT OpenCourseWare — Hands-On Deep Learning assignments/project: https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/pages/assignments/
- Harvard CS50 final project: https://cs50.harvard.edu/x/project/

## Hiring rule

Do not add projects merely to increase the count. Add or modify work only when it closes a real role/skill gap, improves evidence, or is needed for a specific application.

For the current hiring pass, GitHub work is complete when:
- protected originals pass
- focused skills notebooks pass
- all 21 professional applications pass
- notebook sync is green
- Portfolio Integrity is green
- the homepage links everything from the single primary portfolio URL
