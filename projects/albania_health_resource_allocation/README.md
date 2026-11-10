# Albania Healthcare Resource Allocation — All-Age Extension

**A transparent policy-scenario model for demographic equity across all 12 Albanian prefectures.**

[Open the original executed 09B notebook](../../09B_Albania_Healthcare_Resource_Allocation_Optimisation.ipynb)

## Decision question

How could a hypothetical additional healthcare budget be distributed transparently between national programmes and regional service delivery, while considering residents of **all ages** rather than focusing primarily on older residents?

This is a decision-support scenario, not a claim that an optimiser can determine the clinically optimal budget.

## What is included

- The original 09B notebook and its six-programme €100m scenario.
- Census 2023 population counts for all 12 prefectures.
- An all-age demographic extension covering **0–14, 15–64 and 65+**.
- Young-, old- and total-dependency ratios.
- A side-by-side comparison of population-only, all-age and older-age-focused regional allocations; the older-age comparator retains the notebook’s 65+ and 80+ emphasis.
- A 1,000-scenario sensitivity analysis over alternative demographic weights and equity floors.
- Validation tests for Census totals, age-group reconciliation, budget conservation and reproducibility.

## Official demographic evidence

The all-age extension uses INSTAT *Regional Statistical Yearbook 2024*, Table 1.6, “Population by sex, broad age group and prefectures, Census 2023”. The published national totals reconcile to:

| Age group | Census population |
|---|---:|
| 0–14 | 373,929 |
| 15–64 | 1,555,080 |
| 65+ | 473,104 |
| **All ages** | **2,402,113** |

Source: [INSTAT Regional Statistical Yearbook 2024 (PDF)](https://www.instat.gov.al/media/vurntwyn/regional-statistical-yearbook-2024-fin.pdf), Table 1.6, pp. 40–41. The source table attributes the population counts to the 2023 Population and Housing Census.

## Run

From the repository root:

```bash
python -m pip install numpy pandas
python extensions/albania_all_age_allocation.py
```

The module exposes reusable functions for the age profile, all-age need shares, budget allocation, scenario comparison and sensitivity analysis. To inspect the results interactively:

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

Run tests:

```bash
python -m unittest discover -s tests -p "test_albania_all_age_allocation.py" -v
```

## Modelling choices and limitations

- The all-age model uses broad Census cohorts: 0–14, 15–64 and 65+. The original notebook's 65+ subgroups remain available for older-age analysis.
- Population shares and age-cohort shares are demographic allocation proxies. They are **not** measured disease burden, healthcare utilisation, disability prevalence, individual medical need or age-specific expenditure.
- The all-age weights and equal-prefecture equity floor are explicit scenario assumptions. They must be reviewed with Albanian health economists, clinicians and public authorities before policy use.
- The model does not infer an individual's healthcare spending from household expenditure. INSTAT's Household Budget Survey is collected at household level; individual-age spending claims require suitable microdata and a defensible attribution method.
- A robust real-world allocation would add age-specific disease prevalence, mortality, disability, workforce, facility capacity, travel time, waiting lists, medicine utilisation, poverty and programme cost-effectiveness.

## Why this is stronger

The extension makes the trade-offs visible: children and working-age residents are represented alongside older residents; population-only and older-age-focused scenarios can be compared with an all-age scenario; and the allocation is checked against official Census totals rather than relying on an undocumented demographic split.

This is an auditable planning model—not a clinical recommendation, fiscal forecast or causal estimate of health outcomes.
