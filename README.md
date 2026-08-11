# AI & Data Science Portfolio

## Jorgo Luka — MSc Artificial Intelligence & Data Science (Distinction)

I build reproducible data products that start with a real decision, audit and clean the source data, establish an honest baseline, and finish with evaluated models, uncertainty, tests, and limitations.

This repository contains work across Python, SQL, R, PyTorch, TensorFlow, PySpark, Pandas, NumPy, Matplotlib, Seaborn, DuckDB, and scikit-learn. The projects below are the best place to start: they are fully executed, preserve their outputs, and contain no stored notebook errors.

## Start here

| Project | Problem solved | Verified evidence | Main tools | Open |
|---|---|---|---|---|
| **UK House Price Analysis and Prediction** | Produce a forward-looking residential price range while being explicit about uncertainty and missing property characteristics. | 995,059 modelling transactions; 216,564-sale untouched 2026 test; CatBoost MAE **£81,805**, R² **0.604**; 90% interval coverage **91.6%**. | Python, Pandas, NumPy, DuckDB SQL, CatBoost | [Notebook](01_UK_House_Price_Analysis_and_Prediction.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/01_UK_House_Price_Analysis_and_Prediction.ipynb) |
| **London Air Quality Analysis with R** | Turn messy, daily-updated government files into an auditable monitoring panel and test whether past-only features improve hourly NO₂ estimates. | Official 2025–2026 data; 143,102 observed station-hours; untouched 2026 MAE **10.48** and R² **0.056**; 90% interval coverage **88.2%**. | R, data.table, ggplot2, mgcv | [Notebook](07_London_Air_Quality_Analysis_with_R.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/07_London_Air_Quality_Analysis_with_R.ipynb) |
| **Image Classification with CNNs and Transfer Learning** | Classify pet breeds and route uncertain predictions to human review instead of automating every case. | 7,349 real images and 37 breeds; test accuracy **58.5%**, macro-F1 **56.3%**, top-3 accuracy **83.1%**; accepted-case accuracy **80.9%** at **49.1%** coverage. | PyTorch, torchvision, MobileNetV3, Grad-CAM | [Notebook](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) |
| **Clickstream Analysis with PySpark** | Measure engagement drop-off at Spark scale and identify higher-intent shopping sessions without using post-purchase leakage. | 165,474 real click events across 24,026 sessions; separate 12,330-session conversion dataset; test PR-AUC **0.351** versus **0.155** prevalence and ROC-AUC **0.763**. | PySpark, Spark SQL, Spark ML | [Notebook](06_Clickstream_Analysis_with_PySpark.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/06_Clickstream_Analysis_with_PySpark.ipynb) |

## Additional verified work

### Energy Demand Forecasting with TensorFlow

A chronological 14-day electricity-demand forecast with an LSTM, two honest baselines, prediction intervals, saved-model verification, and automated checks. On the untouched test period, the LSTM achieved MAE **43.51**, improving on the 7-day seasonal baseline MAE of **53.18** by **18.2%**.

[Open notebook](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) · [Run in Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/05_Energy_Demand_Forecasting_with_TensorFlow.ipynb)

> Data-freshness note: this project uses the official Open Power System Data series from 2006–2017. It is presented as a time-series methodology project, not a description of today's German electricity market.

### Data Analyst Bootcamp Portfolio

The separate [Data Analyst Bootcamp repository](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) contains applied SQL, Python, data cleaning, reporting, regression, and Power BI work. Its strongest executed case study is the [Retail Margin Pipeline](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp/blob/main/01_retail_margin_pipeline.ipynb), which identifies where discounting, geography, and product mix erode profit.

## How I build a project

1. Define the decision, user, target, and failure cost.
2. Document source provenance, licence, time coverage, and known limitations.
3. Audit missingness, duplicates, invalid values, ranges, types, and join integrity.
4. Fit preprocessing on training data only and prevent temporal, group, or target leakage.
5. Compare against a simple baseline before selecting a more complex model.
6. Reserve an untouched test set and report metrics that match the decision.
7. Add uncertainty, error analysis, slices, or a human-review policy where appropriate.
8. Reload the saved artifact and run automated acceptance checks.
9. State what the result does **not** prove.

## Skills evidenced

| Capability | Evidence |
|---|---|
| Data cleaning and preprocessing | Explicit schemas, missing-value audits, duplicate controls, range checks, categorical handling, training-only transforms |
| SQL and analytical querying | DuckDB SQL in the house-price project; Spark SQL in clickstream analysis; applied SQL work in the bootcamp repository |
| Classical machine learning | CatBoost, histogram gradient boosting, regularised regression, baselines, tuning, calibration, error analysis |
| Deep learning | CNNs, transfer learning, TensorFlow LSTMs, staged fine-tuning, Grad-CAM, model export |
| Big-data processing | PySpark DataFrames, windows, sessionisation, Spark SQL, Spark ML, one-million-row load test |
| Statistics and R | Time-aware modelling, GAMs, conformal intervals, data.table, ggplot2 |
| Communication and governance | Decision framing, model cards, limitations, data licences, human escalation, reproducibility checks |

## Evidence policy

- A result is called **verified** only when the notebook has been run end to end, retains the produced outputs, contains no stored error, and passes its documented checks.
- Current official data are preferred when they suit the problem. Older public datasets are labelled clearly and used only when the project is primarily demonstrating a method.
- A large or complicated model is not automatically better. Every model must earn its place against a relevant baseline.
- Files not featured above should be treated as coursework, experiments, or work awaiting re-validation—not as finished portfolio claims.

## Current focus

I am curating and expanding the existing portfolio one project at a time. Each promoted project must solve a clear problem, teach the full workflow from raw data to decision, and be defensible in a technical interview.

For opportunities in data science, machine learning, AI, or analytics, contact me through my [GitHub profile](https://github.com/Jorgoluka100).
