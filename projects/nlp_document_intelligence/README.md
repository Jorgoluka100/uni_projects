# NLP Document Intelligence — Classification & Insight Engine

A standalone natural-language-processing application for routing incoming text, surfacing category-level language patterns and escalating low-confidence documents for human review.

## Decision problem

A knowledge-operations team receives large volumes of unstructured text. It needs a reproducible NLP system that can classify documents into queues, explain which terms are associated with each category, and avoid blindly auto-routing uncertain examples.

## Dataset

Scikit-learn's 20 Newsgroups corpus, downloaded through `fetch_20newsgroups`. Headers, footers and quoted replies are removed to reduce source-specific leakage. The corpus is a public benchmark rather than proprietary support-ticket data, and that limitation is stated explicitly.

## What this project demonstrates

The executable pipeline includes short inline code notes explaining the purpose of preprocessing, leakage-safe vectorisation, baseline comparison, probability calibration, human-review thresholds, error slices and saved-model parity checks. Test notes explain the privacy and decision-policy guardrails.

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

Create an isolated environment and install the project's direct dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib pytest
```

Run the full benchmark pipeline (the first run downloads the public 20 Newsgroups corpus):

```bash
python run.py
pytest -q tests
```

The run writes machine-readable metrics, confusion pairs, confidence-threshold diagnostics, document-length error slices and a reloadable model artifact under `results/` and `artifacts/`. These are generated outputs, not hand-entered headline results.

## Portfolio files

- `project_notebook.ipynb` — recruiter-facing NLP workflow
- `run.py` — executable training/inference application
- `tests/test_nlp.py` — cleaning and decision-policy tests
- `results/` — metrics, confusion and category insights
- `artifacts/` — saved fitted NLP pipeline

## Limitations

20 Newsgroups is a historical benchmark and is not a live enterprise ticket stream. Real deployment would require domain-specific labelled text, privacy controls, continuous taxonomy management, calibration checks, language/drift monitoring and explicit human escalation for novel or high-impact documents.


## Evaluation and decision policy

- The Naive Bayes model is retained as a simple baseline; the calibrated linear SVM is compared against it on the benchmark's predefined held-out test split.
- The default 0.55 confidence threshold is an explicit demonstration setting, not a claim of a universally optimal operating point.
- The threshold sweep is a **held-out diagnostic** showing the coverage/review trade-off. It is not used to select the threshold. A real deployment should select policy thresholds on a separate validation set with operational stakeholders.
- Macro-F1 gives each topic equal weight; weighted-F1 reflects the observed class frequencies. Both are reported so class imbalance is visible.
- Confusion pairs and document-length slices make failure patterns inspectable instead of relying on one aggregate score.

## Privacy, fairness and deployment boundaries

The normaliser masks common email-address and URL patterns only. It is not a complete personally identifiable information (PII) detector or anonymisation system. Production text would need a tested privacy pipeline, access controls and retention policy before model processing.

The benchmark is an English-language, historical newsgroup corpus with a fixed 20-topic taxonomy. Its results do not establish performance on modern customer support, legal, healthcare or other high-impact documents. Before deployment, collect representative domain-labelled data, audit subgroup and language performance where lawful and meaningful, assess calibration and drift, and keep human review for uncertain or novel cases.
