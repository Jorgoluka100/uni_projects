# NLP Document Intelligence — Classification & Insight Engine

A standalone natural-language-processing application for routing incoming text, surfacing category-level language patterns and escalating low-confidence documents for human review.

## Decision problem

A knowledge-operations team receives large volumes of unstructured text. It needs a reproducible NLP system that can classify documents into queues, explain which terms are associated with each category, and avoid blindly auto-routing uncertain examples.

## Dataset

Scikit-learn's 20 Newsgroups corpus, downloaded through `fetch_20newsgroups`. Headers, footers and quoted replies are removed to reduce source-specific leakage. The corpus is a public benchmark rather than proprietary support-ticket data, and that limitation is stated explicitly.

## What this project demonstrates

- text acquisition and cleaning
- corpus/data-quality profiling
- train/test separation supplied by the benchmark
- TF-IDF word and n-gram representation
- Multinomial Naive Bayes baseline
- linear SVM with probability calibration
- macro/weighted F1, accuracy and log loss
- confusion analysis
- low-confidence human-review policy
- category keyword extraction
- document-length error slices
- prediction/inference API function
- model persistence and parity checks
- reproducible tests and results

## Run

```bash
python run.py
```

## Portfolio files

- `project_notebook.ipynb` — recruiter-facing NLP workflow
- `run.py` — executable training/inference application
- `tests/test_nlp.py` — cleaning and decision-policy tests
- `results/` — metrics, confusion and category insights
- `artifacts/` — saved fitted NLP pipeline

## Limitations

20 Newsgroups is a historical benchmark and is not a live enterprise ticket stream. Real deployment would require domain-specific labelled text, privacy controls, continuous taxonomy management, calibration checks, language/drift monitoring and explicit human escalation for novel or high-impact documents.
