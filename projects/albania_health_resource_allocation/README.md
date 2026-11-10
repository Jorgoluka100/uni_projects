# Albania Healthcare Resource Allocation — All-Age Extension

**A transparent policy-scenario model for demographic equity across all 12 Albanian prefectures.**

[Open the original executed 09B notebook](../../09B_Albania_Healthcare_Resource_Allocation_Optimisation.ipynb)

## Decision question

How could a hypothetical additional healthcare budget be distributed transparently between national programmes and regional service delivery, using a balanced view of the **whole population** rather than prioritising one age group?

This is a decision-support scenario, not a claim that an optimiser can determine the clinically optimal budget.

## What is included

- The original 09B notebook and its six-programme €100m scenario.
- Census 2023 population counts for all 12 prefectures.
- An all-age demographic extension covering the three mutually exclusive Census cohorts: **0–14, 15–64 and 65+**.
- Young-, 65+- and total-dependency ratios, reported as descriptive population indicators.
- A side-by-side comparison of a population-only baseline and a balanced all-age regional allocation. Each published age cohort receives the same scenario weight; no age group is singled out for extra priority.
- A 1,000-scenario sensitivity analysis over alternative demographic weights and equity floors.
- Validation tests for Census totals, age-group reconciliation, budget conservation and reproducibility.
- A repeatable export command that writes the prefecture profile, allocation comparison, sensitivity ranges and scenario metadata to CSV/JSON.

## Official demographic evidence

The all-age extension uses INSTAT *Regional Statistical Yearbook 2024*, Table 1.6, “Population by sex, broad age group and prefectures, Census 2023”. The published national totals reconcile to:

| Age group | Census population |
|---|---:|
| 0–14 | 373,929 |
| 15–64 | 1,555,080 |
| 65+ | 473,104 |
| **All ages** | **2,402,113** |

Source: [INSTAT Regional Statistical Yearbook 2024 (PDF)](https://www.instat.gov.al/media/vurntwyn/regional-statistical-yearbook-2024-fin.pdf), Table 1.6, pp. 40–41. The source table attributes the population counts to the 2023 Population and Housing Census.

## Run the analysis

From the repository root:

```bash
python -m pip install numpy pandas
python extensions/albania_all_age_allocation.py
```

The module prints a reproducible profile, scenario comparison and sensitivity summary when run directly. It also exposes reusable functions for the age profile, all-age need shares, budget allocation, scenario comparison and sensitivity analysis.

## Export machine-readable results

Generate a consistent set of files for further analysis, Power BI, Tableau or review:

```bash
python extensions/export_albania_all_age_outputs.py
```

By default, this writes to `artifacts/albania_all_age/`:

- `prefecture_age_profile.csv` — 12 prefectures, broad age cohorts, population shares and dependency ratios.
- `allocation_scenario_comparison.csv` — population-only baseline and balanced all-age allocations.
- `allocation_weight_sensitivity.csv` — 10th, median and 90th percentile allocations from the seeded sensitivity run.
- `scenario_metadata.json` — source, budget, assumptions, simulation count and random seed.

The run is configurable:

```bash
python extensions/export_albania_all_age_outputs.py \
  --regional-budget-eur 40000000 \
  --simulations 1000 \
  --seed 42 \
  --output-dir artifacts/albania_all_age
```

To inspect the results interactively:

```python
from extensions.albania_all_age_allocation import (
    build_age_profile,
    compare_allocation_scenarios,
    run_weight_sensitivity,
)

profile = build_age_profile()
comparison = compare_allocation_scenarios(regional_budget_eur=40_000_000)
sensitivity = run_weight_sensitivity(regional_budget_eur=40_000_000)
```

Run the Albania checks:

```bash
python -m unittest discover -s tests -p "test_albania_*" -v
```

## Modelling choices and limitations

- The all-age model uses the three mutually exclusive Census cohorts: 0–14, 15–64 and 65+. The 65+ cohort remains represented as one part of the whole population; there is no separate 80+ or older-only allocation scenario in this extension.
- Population shares and age-cohort shares are demographic allocation proxies. They are **not** measured disease burden, healthcare utilisation, disability prevalence, individual medical need or age-specific expenditure.
- The all-age weights and equal-prefecture equity floor are explicit scenario assumptions. They must be reviewed with Albanian health economists, clinicians and public authorities before policy use.
- The model does not infer an individual's healthcare spending from household expenditure. INSTAT's Household Budget Survey is collected at household level; individual-age spending claims require suitable microdata and a defensible attribution method.
- A robust real-world allocation would add age-specific disease prevalence, mortality, disability, workforce, facility capacity, travel time, waiting lists, medicine utilisation, poverty and programme cost-effectiveness.

## Why this is stronger

The extension makes the trade-offs visible across all three source-defined age cohorts, compares a population-only baseline with a balanced all-age scenario, and checks allocations against official Census totals rather than relying on an undocumented demographic split.

This is an auditable planning model—not a clinical recommendation, fiscal forecast or causal estimate of health outcomes.
