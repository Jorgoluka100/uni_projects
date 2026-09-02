# Notebooks + Datasets — recruiter quick access

Every strengthened project has a Jupyter notebook entry point and a documented data source. This page is the fastest way to inspect the portfolio without searching through folders.

> **Data policy:** large, third-party, licensed or frequently refreshed datasets are not duplicated into Git history. Each project keeps the source/provenance and reproducible retrieval or generation route in its README/data card/run code. Small fixtures and derived evidence stay in the project where appropriate. This is deliberate: recruiters can reproduce the work without the repository becoming a data dump.

## End-to-end projects

| Project | Jupyter notebook | Dataset / data source |
| --- | --- | --- |
| **Flight Delay Risk Platform** | [`project_notebook.ipynb`](../projects/flight_delay_risk/project_notebook.ipynb) | Official 2026 U.S. Bureau of Transportation Statistics flight data — [`DATA_CARD.md`](../projects/flight_delay_risk/DATA_CARD.md) / [`README.md`](../projects/flight_delay_risk/README.md) |
| **Reliable Event Pipeline** | [`project_notebook.ipynb`](../projects/reliable_event_pipeline/project_notebook.ipynb) | Reproducible event fixtures and pipeline inputs — [`README.md`](../projects/reliable_event_pipeline/README.md) |
| **E-commerce SQL + dbt** | [`project_notebook.ipynb`](../projects/ecommerce_sql_analytics/project_notebook.ipynb) | Public multi-table e-commerce order/customer/product/payment/review data — [`DATA_MODEL.md`](../projects/ecommerce_sql_analytics/DATA_MODEL.md) / [`README.md`](../projects/ecommerce_sql_analytics/README.md) |
| **PySpark Clickstream** | [`project_notebook.ipynb`](../projects/pyspark_clickstream_analytics/project_notebook.ipynb) | Real clickstream events plus a clearly labelled generated load-test dataset — [`README.md`](../projects/pyspark_clickstream_analytics/README.md) |
| **Executive Commerce Intelligence — Power BI + Tableau** | [`project_notebook.ipynb`](../projects/executive_commerce_bi/project_notebook.ipynb) | Governed exports produced from the e-commerce analytics warehouse — [`README.md`](../projects/executive_commerce_bi/README.md) |
| **Retail Cleaning & Segmentation** | [`project_notebook.ipynb`](../projects/retail_customer_segmentation/project_notebook.ipynb) | 541,909-row public retail transaction dataset — [`README.md`](../projects/retail_customer_segmentation/README.md) |
| **Customer Churn** | [`project_notebook.ipynb`](../projects/customer_churn_prediction/project_notebook.ipynb) | Public telecom churn data with customer-level grouping and documented feature policy — [`README.md`](../projects/customer_churn_prediction/README.md) |
| **UK House Price Prediction** | [`project_notebook.ipynb`](../projects/uk_house_price_prediction/project_notebook.ipynb) | HM Land Registry Price Paid Data — [`README.md`](../projects/uk_house_price_prediction/README.md) |
| **Energy Demand Forecasting** | [`project_notebook.ipynb`](../projects/energy_demand_forecasting/project_notebook.ipynb) | Public chronological energy-demand time series — [`README.md`](../projects/energy_demand_forecasting/README.md) |
| **Image Classification + Confidence** | [`project_notebook.ipynb`](../projects/image_classification_confidence/project_notebook.ipynb) | Public image-classification data with reproducible train/evaluation setup — [`README.md`](../projects/image_classification_confidence/README.md) |
| **Grounded RAG** | [`project_notebook.ipynb`](../projects/grounded_rag/project_notebook.ipynb) | Versioned knowledge-base documents / evaluation fixtures used for retrieval and citation tests — [`README.md`](../projects/grounded_rag/README.md) |
| **ModelWatch** | [`project_notebook.ipynb`](../projects/model_watch/project_notebook.ipynb) | Versioned baseline/current scoring data used for drift, discrimination and calibration checks — [`README.md`](../projects/model_watch/README.md) |
| **ExperimentLab** | [`project_notebook.ipynb`](../projects/experiment_lab/project_notebook.ipynb) | Clearly labelled reproducible synthetic experiment data — [`README.md`](../projects/experiment_lab/README.md) |
| **Parkinson's Progression** | [`project_notebook.ipynb`](../projects/parkinsons_progression/project_notebook.ipynb) | Public Parkinson's progression/telemonitoring data with subject-grouped validation — [`README.md`](../projects/parkinsons_progression/README.md) |

## Original / university notebooks

The original executed `.ipynb` files are intentionally retained at repository root, including house-price prediction, SQL/customer analysis, customer churn, image classification, energy forecasting, PySpark clickstream, LLM work, medical AI, recommender systems and other university/laboratory projects.

**[Open the complete notebook index →](NOTEBOOK_INDEX.md)**

## What a recruiter should open first

1. **Data Scientist:** [`Flight Delay Risk`](../projects/flight_delay_risk/project_notebook.ipynb) → [`Customer Churn`](../projects/customer_churn_prediction/project_notebook.ipynb) → [`UK House Prices`](../projects/uk_house_price_prediction/project_notebook.ipynb)
2. **Data Engineer / Analytics Engineer:** [`Reliable Event Pipeline`](../projects/reliable_event_pipeline/project_notebook.ipynb) → [`PySpark Clickstream`](../projects/pyspark_clickstream_analytics/project_notebook.ipynb) → [`E-commerce SQL + dbt`](../projects/ecommerce_sql_analytics/project_notebook.ipynb)
3. **ML / AI Engineer:** [`Grounded RAG`](../projects/grounded_rag/project_notebook.ipynb) → [`Image Classification`](../projects/image_classification_confidence/project_notebook.ipynb) → [`Flight Delay Risk`](../projects/flight_delay_risk/project_notebook.ipynb)
4. **Data Analyst / BI:** [`Executive Commerce Intelligence`](../projects/executive_commerce_bi/project_notebook.ipynb) → [`E-commerce SQL + dbt`](../projects/ecommerce_sql_analytics/project_notebook.ipynb) → [`ExperimentLab`](../projects/experiment_lab/project_notebook.ipynb)

The notebook is the readable walkthrough; the `.py`, SQL, tests, BI assets, API code and CI in each project folder are the implementation evidence.