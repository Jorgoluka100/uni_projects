# AI & Data Science Portfolio

## Jorgo Luka — MSc Artificial Intelligence & Data Science (Distinction)

I build reproducible data products that start with a real decision, audit and clean the source data, establish an honest baseline, and finish with evaluated models, uncertainty, tests, and limitations.

**`main` is intentionally curated:** it contains only seven projects that have been executed end to end and promoted after validation. Older coursework and experiments are preserved on the archive branch rather than mixed into the recruiter-facing portfolio.

## Start here

| Project | Problem solved | Verified evidence | Main tools | Open |
|---|---|---|---|---|
| **UK House Price Analysis and Prediction** | Produce a forward-looking residential price range while being explicit about uncertainty and missing property characteristics. | 995,059 modelling transactions; 216,564-sale untouched 2026 test; CatBoost MAE **£81,805**, R² **0.604**; 90% interval coverage **91.6%**. | Python, Pandas, NumPy, DuckDB SQL, CatBoost | [Notebook](01_UK_House_Price_Analysis_and_Prediction.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/01_UK_House_Price_Analysis_and_Prediction.ipynb) |
| **SQL Sales and Customer Analysis** | Build trustworthy marketplace KPIs from nine relational tables without multiplying money across incompatible grains. | **98,199** commercial orders, **94,983** customers and **R$13.49M** merchandise value reconciled; source hashes, key checks, semantic-layer tests and artifact tests all passed. | SQL, DuckDB, Python, Pandas, Parquet | [Notebook](02_SQL_Sales_and_Customer_Analysis.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/02_SQL_Sales_and_Customer_Analysis.ipynb) |
| **Customer Churn Prediction** | Build a decision-ready churn screening system while controlling duplicate-profile leakage, calibration and review capacity. | Holdout PR-AUC **0.955** (95% bootstrap **0.927–0.980**) and ROC-AUC **0.990**; **94.9%** recall at **73.2%** precision while flagging **20.2%** of customers. | Python, scikit-learn, calibration, bootstrap diagnostics | [Notebook](03_Customer_Churn_Prediction.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/03_Customer_Churn_Prediction.ipynb) |
| **Image Classification with CNNs and Transfer Learning** | Classify pet breeds and route uncertain predictions to human review instead of automating every case. | 7,349 real images and 37 breeds; test accuracy **58.5%**, macro-F1 **56.3%**, top-3 accuracy **83.1%**; accepted-case accuracy **80.9%** at **49.1%** coverage. | PyTorch, torchvision, MobileNetV3, Grad-CAM | [Notebook](04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) |
| **Energy Demand Forecasting with TensorFlow** | Forecast a chronological 14-day electricity-demand window and test whether the neural model actually beats honest seasonal baselines. | LSTM test MAE **43.51** versus **53.18** for the 7-day seasonal baseline — an **18.2%** improvement — with prediction intervals and saved-model verification. | TensorFlow, Keras, LSTM, time-series validation | [Notebook](05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) |
| **Clickstream Analysis with PySpark** | Measure engagement drop-off at Spark scale and identify higher-intent shopping sessions without using post-purchase leakage. | 165,474 real click events across 24,026 sessions; separate 12,330-session conversion dataset; test PR-AUC **0.351** versus **0.155** prevalence and ROC-AUC **0.763**. | PySpark, Spark SQL, Spark ML | [Notebook](06_Clickstream_Analysis_with_PySpark.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/06_Clickstream_Analysis_with_PySpark.ipynb) |
| **London Air Quality Analysis with R** | Turn messy, daily-updated government files into an auditable monitoring panel and test whether past-only features improve hourly NO₂ estimates. | Official 2025–2026 data; 143,102 observed station-hours; untouched 2026 MAE **10.48** and R² **0.056**; 90% interval coverage **88.2%**. | R, data.table, ggplot2, mgcv | [Notebook](07_London_Air_Quality_Analysis_with_R.ipynb) · [Colab](https://colab.research.google.com/github/Jorgoluka100/uni_projects/blob/main/07_London_Air_Quality_Analysis_with_R.ipynb) |

## Data-freshness notes

- **Customer Churn Prediction** uses the historical UCI Iranian Churn dataset. It demonstrates rigorous classification and decision design, not current telecom-market behaviour.
- **Energy Demand Forecasting** uses the official Open Power System Data series from 2006–2017. It is a time-series methodology project, not a description of today's German electricity market.
- Current official data are preferred where the problem supports them; older datasets are labelled rather than presented as current evidence.

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
| SQL and analytical querying | DuckDB SQL, relational grain controls and reconciliation; Spark SQL and session analytics |
| Classical machine learning | CatBoost, histogram gradient boosting, regularised regression, baselines, tuning, calibration, error analysis |
| Deep learning | CNN transfer learning, TensorFlow LSTMs, staged fine-tuning, Grad-CAM, model export |
| Big-data processing | PySpark DataFrames, windows, sessionisation, Spark SQL and Spark ML |
| Statistics and R | Time-aware modelling, GAMs, conformal intervals, data.table, ggplot2 |
| Communication and governance | Decision framing, limitations, data licences, human escalation, reproducibility and artifact checks |

## Additional analytics work

My separate [Data Analyst Bootcamp Portfolio](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp) contains applied Python, SQL, data cleaning, regression, reporting and business-analysis work. Its strongest executed case studies are now organised under [`notebooks/`](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp/tree/main/notebooks) with their source data under [`data/`](https://github.com/Jorgoluka100/primed-talent-data-analyst-bootcamp/tree/main/data).

## Evidence policy

- A result is called **verified** only when the notebook has been run end to end, retains the produced outputs, contains no stored error, and passes its documented checks.
- A complicated model is not automatically better. Every model must earn its place against a relevant baseline.
- `main` contains only the seven promoted notebooks above. Earlier coursework and experiments are preserved on `archive/pre-portfolio-curation-2026-08-11` so they remain recoverable without weakening the public portfolio surface.
- A project is promoted only when its claims are defensible in a technical interview.

## Current focus

I am expanding this portfolio selectively rather than by notebook count. New work must add a genuinely different hiring signal, solve a clear problem, and survive the same validation standard as the projects already on `main`.

For opportunities in data science, machine learning, applied AI, or analytics, contact me through my [GitHub profile](https://github.com/Jorgoluka100).
