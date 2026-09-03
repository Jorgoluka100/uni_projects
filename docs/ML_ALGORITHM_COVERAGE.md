# Machine Learning & Deep Learning Coverage Map

This page makes the portfolio's algorithm coverage explicit. Recruiter-facing evidence is limited to **Jorgo Luka's own work or clearly identified Jorgo Luka contributions**. Lecturer tutorials and another student's notebooks are archive-only and are not used to prove authorship.

## Classical supervised learning

| Algorithm / family | Foundation / university evidence | Full applied evidence |
| --- | --- | --- |
| **Linear Regression** | [`08_regression_fundamentals.ipynb`](../skills/08_regression_fundamentals.ipynb) | [Building Energy Efficiency Decision Model](../projects/linear_regression_energy_efficiency/) · [UK House Prices](../projects/uk_house_price_prediction/) |
| **Ridge / Lasso regularisation** | regression foundations | [Building Energy Efficiency](../projects/linear_regression_energy_efficiency/) |
| **Logistic Regression** | [`03_sklearn_end_to_end_classification.ipynb`](../skills/03_sklearn_end_to_end_classification.ipynb) · [Jorgo Luka CN7030 intrusion coursework](../CN7030_Group3_Intrusion_Detection_Coursework.ipynb) | [Customer Churn](../projects/customer_churn_prediction/) · logistic baseline in [Deep Learning Marketing](../projects/deep_learning_marketing_response/) |
| **K-Nearest Neighbours (KNN)** | classification foundations / retained original learning evidence | [KNN Product Quality](../projects/knn_product_quality/) |
| **Naive Bayes** | text-classification foundation route | [NLP Document Intelligence](../projects/nlp_document_intelligence/) |
| **Support Vector Machines — Linear / RBF** | **[`11_support_vector_machines.ipynb`](../skills/11_support_vector_machines.ipynb)** · [Python](../skills/11_support_vector_machines.py) | calibrated `LinearSVC` in [NLP Document Intelligence](../projects/nlp_document_intelligence/) |
| **Decision Trees** | **[`12_tree_and_ensemble_models.ipynb`](../skills/12_tree_and_ensemble_models.ipynb)** · [Python](../skills/12_tree_and_ensemble_models.py) | tree-based comparisons and explainability across professional ML projects |
| **Random Forests / bagging** | [`12_tree_and_ensemble_models.ipynb`](../skills/12_tree_and_ensemble_models.ipynb) | classical ensemble comparator / feature-importance evidence |
| **Gradient Boosting** | [`12_tree_and_ensemble_models.ipynb`](../skills/12_tree_and_ensemble_models.ipynb) | advanced boosting extended through XGBoost and CatBoost projects |
| **XGBoost** | regression / ensemble foundation route | [XGBoost Bike Demand](../projects/xgboost_bike_demand/) |
| **CatBoost** | applied project evidence | [Flight Delay Risk](../projects/flight_delay_risk/) · [UK House Prices](../projects/uk_house_price_prediction/) |

## Unsupervised learning & representation

| Algorithm / family | Evidence |
| --- | --- |
| **K-Means clustering** | [`09_clustering_fundamentals.ipynb`](../skills/09_clustering_fundamentals.ipynb) · [`Clustering_Models.ipynb`](../Clustering_Models.ipynb) · [Retail Customer Segmentation](../projects/retail_customer_segmentation/) |
| **RFM + clustering** | [Retail Customer Segmentation](../projects/retail_customer_segmentation/) |
| **PCA / dimensionality reduction for inspection** | [SVM foundation](../skills/11_support_vector_machines.ipynb) and project-specific visual/error analysis where appropriate |

## Deep learning

| Architecture / concept | Foundation / original evidence | Full applied evidence |
| --- | --- | --- |
| **MLP / feed-forward neural networks** | [`04_pytorch_neural_network_fundamentals.ipynb`](../skills/04_pytorch_neural_network_fundamentals.ipynb) | [Deep Learning Marketing Response](../projects/deep_learning_marketing_response/) |
| **2D CNN / Conv2d** | [`07_cnn_image_fundamentals.ipynb`](../skills/07_cnn_image_fundamentals.ipynb) · [CNN & Transfer Learning](../04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb) | **[CNN 2D Image Classification — Convolutions, Padding & Confidence](../projects/cnn_retail_image_classification/)** |
| **Padding / kernels / feature maps / pooling** | CNN foundation | CNN 2D application uses `Conv2d`, `padding`, `MaxPool2d`, adaptive pooling and data augmentation |
| **Batch Normalisation / Dropout** | PyTorch/CNN foundations | deeper CNN in [CNN 2D Image Classification](../projects/cnn_retail_image_classification/) |
| **Transfer learning / ResNet / EfficientNet** | retained CNN transfer-learning evidence | ResNet18 comparator in [CNN 2D Image Classification](../projects/cnn_retail_image_classification/) · EfficientNet in [Image Classification Confidence](../projects/image_classification_confidence/) |
| **1D convolution** | sequence/time-series route | [Energy Demand Forecasting](../projects/energy_demand_forecasting/) |
| **LSTM / recurrent networks** | [`05_lstm_sequence_modelling.ipynb`](../skills/05_lstm_sequence_modelling.ipynb) | [Energy Demand Forecasting](../projects/energy_demand_forecasting/) |

## NLP, LLMs & applied AI

| Technique | Evidence |
| --- | --- |
| **TF-IDF** | [`06_text_classification_tfidf.ipynb`](../skills/06_text_classification_tfidf.ipynb) · retained CN7030 TF-IDF work · [NLP Document Intelligence](../projects/nlp_document_intelligence/) |
| **Naive Bayes + SVM text classification** | [NLP Document Intelligence](../projects/nlp_document_intelligence/) |
| **Embeddings / sequence representation** | retained CN7030 sequence-embedding work |
| **Transformers / LLM architecture** | hands-on course-learning notebooks + [Grounded RAG](../projects/grounded_rag/) |
| **Retrieval-Augmented Generation (RAG)** | [Grounded RAG](../projects/grounded_rag/) |

## Evaluation skills that cut across algorithms

The portfolio demonstrates train/test and temporal/grouped splits, cross-validation, GridSearchCV, baseline comparison, leakage controls, regression/classification metrics, confusion matrices, residual/error analysis, calibration, threshold/coverage trade-offs, uncertainty, feature importance/explainability, robustness slices, persistence/inference and role-relevant testing/CI.

## How to read the portfolio

1. **University/course notebooks** are surfaced only where Jorgo Luka's authorship or contribution is clear.
2. **`skills/` foundations** are concise DataCamp-style demonstrations of individual concepts and algorithms.
3. **`projects/` applications** prove that those methods can be combined into complete data/AI solutions with analysis, evaluation and decisions.
4. Lecturer tutorials, raw group copies and another student's work are preserved only in the historical archive.
