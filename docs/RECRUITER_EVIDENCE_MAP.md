# Recruiter Evidence Map

This page is a plain-English guide to the strongest evidence in this portfolio. It does not replace the project READMEs or notebooks; it helps a reviewer quickly see what each project demonstrates and gives me a clear structure for explaining the work in an interview.

## 1. Data science and model evaluation

### Flight Delay Risk Platform
**Evidence:** Python, CatBoost, temporal validation, classification, calibration, threshold selection, error analysis.

**What the work demonstrates:** I can start from an operational question, test a modelling formulation, reject an approach that does not beat a baseline, reframe the problem, evaluate on later unseen data and communicate the limits of the result.

**Interview discussion:** why chronological validation matters; why the original regression formulation was changed; PR-AUC versus prevalence; what a 1.58x top-decile lift means operationally.

### Customer Churn Prediction
**Evidence:** classification, grouped validation, feature engineering, probability calibration and decision thresholds.

**What the work demonstrates:** model quality is not only a headline accuracy score; the decision threshold has to reflect how the prediction will be used.

## 2. SQL, analytics and data quality

### E-commerce SQL + Customer Analysis
**Evidence:** SQL, DuckDB, Pandas, joins, key checks, reporting grain and reconciliation.

**What the work demonstrates:** I check whether the data can be trusted before drawing business conclusions. The analysis reconciles 98,199 orders, 94,983 customers and R$13.49M in merchandise value.

**Interview discussion:** join cardinality; duplicate risk; reporting grain; reconciliation; how I would turn the analysis into recurring reporting.

### UK House Price Analysis & Prediction
**Evidence:** Python, SQL/DuckDB, CatBoost, large tabular datasets, untouched test data and uncertainty.

**What the work demonstrates:** I can work with a large real-world dataset, structure validation carefully and report both prediction quality and uncertainty.

## 3. Machine learning breadth

The repository contains working evidence across regression, classification, clustering, KNN, SVMs, decision trees, random forests, gradient boosting, XGBoost, CatBoost and time-series methods.

The purpose is not to claim mastery of every algorithm. It shows that I understand the major families, can choose an approach for a problem and can compare it against a baseline.

## 4. NLP, retrieval and applied AI

### Grounded RAG
**Evidence:** BM25/TF-IDF, LSA retrieval, citations, abstention, prompt-injection checks, FastAPI, Docker and automated tests.

**What the work demonstrates:** I treat retrieval quality, grounding and failure behaviour as engineering problems rather than assuming an LLM answer is correct.

**Interview discussion:** why retrieval evaluation matters; when the system should abstain; how tool access is constrained; how I would extend the system to a production vector store or managed LLM.

### NLP Document Intelligence
**Evidence:** TF-IDF, Naive Bayes, LinearSVC, confidence routing and error analysis.

**What the work demonstrates:** I can build and compare classical NLP baselines as well as modern AI workflows.

## 5. Computer vision and deep learning

### Image Classification / VisionForge
**Evidence:** PyTorch, EfficientNet, CNNs, transfer learning, Grad-CAM, confidence-based review, TorchScript and ONNX checks.

**What the work demonstrates:** I can train and evaluate an image model, investigate confidence and errors, add explainability, and check that exported models behave consistently.

## 6. Data engineering and large-scale processing

### Reliable Event Pipeline / Spark projects
**Evidence:** Python, SQL, PySpark/Spark SQL, data pipelines, validation and reproducible processing.

**What the work demonstrates:** I can move beyond a single notebook and think about ingestion, transformations, data quality, repeatability and downstream analytics.

### E-commerce SQL + dbt
**Evidence:** relational modelling, SQL transformations, tests and analytics-engineering practices.

## 7. Experimentation, uncertainty and model monitoring

### ExperimentLab
**Evidence:** A/B testing, statistical inference, CUPED and decision-focused reporting.

### ModelWatch
**Evidence:** drift, calibration and monitoring gates.

**What these demonstrate:** I understand that a model is not finished at training time; experiments, uncertainty, monitoring and changing data all matter.

## 8. How I work

Across the recruiter-facing projects I try to make the same habits visible:

- define the decision or question before modelling;
- check source data, keys, leakage and assumptions;
- use a simple baseline before a more complex model;
- keep validation and test data separate;
- report limitations as well as positive results;
- make work reproducible with Git, notebooks, tests and documentation;
- explain technical results in language a non-specialist can use.

For full code, notebooks, project-specific results and limitations, use the main [portfolio index](../README.md).
