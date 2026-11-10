# LongevityAI — age coverage, subgroup validity and responsible scope

## Population coverage

The project should be interpreted against the age range actually present in each NHANES extract and the population for which each published equation was developed and validated. Report the observed age range, counts by age band, exclusions and missing-age handling before presenting age-stratified results.

The existence of an age variable does not establish that every life stage is represented or that one biological-age equation is valid across childhood, adolescence and adulthood.

## All-age inclusion without invalid extrapolation

The wider portfolio's demographic perspective includes children, young people, working-age adults and older adults. LongevityAI must not convert that inclusive perspective into unsupported pediatric predictions.

- **Children and adolescents:** do not apply an adult-derived PhenoAge equation or adult biomarker thresholds. A pediatric analysis would require an appropriate pediatric cohort, age-specific reference ranges, a justified outcome and independent validation.
- **Adults:** report the exact age range represented in the source and the validated scope of each equation. Do not silently extrapolate beyond either range.
- **Older adults:** report subgroup sample size and uncertainty; examine calibration and error patterns rather than assuming adult-wide performance is uniform.
- **Across age groups:** use source-supported bands, preserve unknown/missing age as a visible category, and avoid comparisons where the data do not support them.

## Required evaluation notes

For any age-stratified model evaluation, report sample count, MAE/RMSE or other task-appropriate error, uncertainty intervals where feasible, and calibration or residual patterns by age band. Compare against a transparent baseline. Flag small or unrepresentative groups rather than presenting unstable metrics as evidence.

Biological-age estimates are model outputs, not diagnoses, proof of rejuvenation, life-expectancy predictions or treatment recommendations. Scenario perturbations are sensitivity analyses; they do not establish that changing a biomarker causes an equivalent change in biological age or healthspan.

## Hiring-facing explanation

This project demonstrates an important responsible-AI distinction: **inclusive population analysis does not justify using an adult model on children**. A strong data scientist identifies who is represented, who is not, what the model can support, and what additional evidence is needed before extending its use.

See [the repository-wide inclusive analysis standard](../../docs/ALL_AGE_INCLUSION_AND_FAIRNESS_STANDARD.md).
