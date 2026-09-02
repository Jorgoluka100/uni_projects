# MSc University Projects

These are the original executed university/coursework notebooks retained from my MSc Artificial Intelligence & Data Science work. They remain visible as academic evidence, while several topics also have newer production-style follow-on projects with stronger validation, testing and reproducibility.

| University project | Original notebook(s) | Stronger follow-on evidence | Main skills |
| --- | --- | --- | --- |
| **Network Intrusion / Cyber-Attack Detection — KDD Cup** | [`KDDCup.ipynb`](../KDDCup.ipynb) · [`Logistic_Regression_PySpark.ipynb`](../Logistic_Regression_PySpark.ipynb) · [`Naive_Bayes_PySpark.ipynb`](../Naive_Bayes_PySpark.ipynb) | [`extensions/kdd_intrusion_v2.py`](../extensions/kdd_intrusion_v2.py) · [`verified/kdd_intrusion/`](../verified/kdd_intrusion/) · [`verified/spark_kdd/`](../verified/spark_kdd/) | PySpark, KDD Cup 99 network-intrusion data, preprocessing, logistic regression, Naive Bayes, attack classification, large-scale ML evaluation |
| **Parkinson's Progression Modelling** | [`Parkinsons_Progression_ML.ipynb`](../Parkinsons_Progression_ML.ipynb) | [`projects/parkinsons_progression/`](../projects/parkinsons_progression/) | data cleaning, regression, scikit-learn, leakage-aware grouped validation |
| **UK House Price Analysis & Prediction** | [`01_UK_House_Price_Analysis_and_Prediction.ipynb`](../01_UK_House_Price_Analysis_and_Prediction.ipynb) | [`projects/uk_house_price_prediction/`](../projects/uk_house_price_prediction/) | Python, regression, feature engineering, evaluation |
| **SQL Sales & Customer Analysis** | [`02_SQL_Sales_and_Customer_Analysis.ipynb`](../02_SQL_Sales_and_Customer_Analysis.ipynb) | [`projects/ecommerce_sql_analytics/`](../projects/ecommerce_sql_analytics/) | SQL, relational analysis, customer analytics, reporting grain |
| **Customer Churn Prediction** | [`03_Customer_Churn_Prediction.ipynb`](../03_Customer_Churn_Prediction.ipynb) | [`projects/customer_churn_prediction/`](../projects/customer_churn_prediction/) | classification, preprocessing, model evaluation, decision thresholds |
| **Image Classification with CNNs & Transfer Learning** | [`04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb`](../04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) | [`projects/image_classification_confidence/`](../projects/image_classification_confidence/) | computer vision, CNNs, transfer learning |
| **Energy Demand Forecasting with TensorFlow** | [`05_Energy_Demand_Forecasting_with_TensorFlow.ipynb`](../05_Energy_Demand_Forecasting_with_TensorFlow.ipynb) | [`projects/energy_demand_forecasting/`](../projects/energy_demand_forecasting/) | time series, TensorFlow/Keras, forecasting, temporal validation |
| **Clickstream Analysis with PySpark** | [`06_Clickstream_Analysis_with_PySpark.ipynb`](../06_Clickstream_Analysis_with_PySpark.ipynb) | [`projects/pyspark_clickstream_analytics/`](../projects/pyspark_clickstream_analytics/) | PySpark, distributed transformations, behavioural analytics, Spark ML |
| **London Air Quality Analysis with R** | [`07_London_Air_Quality_Analysis_with_R.ipynb`](../07_London_Air_Quality_Analysis_with_R.ipynb) | Original notebook retained as R evidence | R, exploratory analysis, environmental data, statistical visualisation |

## Cyber-security / KDD university suite

The KDD work is intentionally grouped here because the notebooks are related rather than three unrelated projects. The KDD Cup 1999 dataset is network-intrusion data for distinguishing normal traffic from attacks. The PySpark Logistic Regression and Naive Bayes notebooks are UEL teaching/coursework notebooks using that intrusion-detection dataset, while the retained extension/evidence folders show later hardening and evaluation.

## Why the originals stay

The academic notebooks show the original learning and executed analysis. The follow-on projects show how the same areas were later approached with stronger engineering discipline: explicit data contracts, leakage controls, realistic holdouts, tests, machine-readable results, CI and documented limitations.

The originals are not replaced by polished summaries. Recruiters can inspect both the academic foundation and the later engineering standard.

## Related course and specialist learning

The LLM Mastery/Udemy work is kept separately from MSc work so the provenance is clear:

- [`LLM_Mastery_Hands_on_Code.ipynb`](../LLM_Mastery_Hands_on_Code.ipynb) — hands-on PyTorch transformer/LLM training work.
- [`LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb`](../LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb) — Llama-style architecture and alignment study, including the supplied 138M-parameter pretrained-model workflow.

Return to the **[single portfolio homepage](../README.md)** for university work, course learning, all 20 professional projects, foundations, datasets and the complete catalog.