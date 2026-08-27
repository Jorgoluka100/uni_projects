# Data & AI Skills Lab

The main portfolio answers **“can I build and evaluate useful projects?”**. This folder answers a different interview question: **“do I understand the building blocks?”**

These are deliberately small, focused notebooks. They are not presented as separate products or fake production systems. Each notebook isolates one capability so I can revise it, explain it at a whiteboard and point an interviewer to concrete code.

## Core notebooks

| # | Notebook | What it proves |
| --- | --- | --- |
| 01 | [`01_data_cleaning_preprocessing.ipynb`](01_data_cleaning_preprocessing.ipynb) | missing values, duplicates, types, outliers, categorical encoding and leakage-safe preprocessing |
| 02 | [`02_numpy_for_machine_learning.ipynb`](02_numpy_for_machine_learning.ipynb) | arrays, shapes, broadcasting, vectorisation, matrix multiplication and numerical thinking |
| 03 | [`03_sklearn_end_to_end_classification.ipynb`](03_sklearn_end_to_end_classification.ipynb) | train/test split, `ColumnTransformer`, pipelines, baseline, logistic regression and classification metrics |
| 04 | [`04_pytorch_neural_network_fundamentals.ipynb`](04_pytorch_neural_network_fundamentals.ipynb) | tensors, layers, forward pass, loss, backpropagation, optimiser and evaluation |
| 05 | [`05_lstm_sequence_modelling.ipynb`](05_lstm_sequence_modelling.ipynb) | what an LSTM is, sequence shapes, hidden state, training and sequence classification |
| 06 | [`06_text_classification_tfidf.ipynb`](06_text_classification_tfidf.ipynb) | text cleaning, TF-IDF, sparse features, logistic regression and evaluation |
| 07 | [`07_cnn_image_fundamentals.ipynb`](07_cnn_image_fundamentals.ipynb) | convolution, channels, pooling, flattening and a compact CNN training loop |

## How this fits the portfolio

The skill notebooks are **supporting evidence**, not the first thing a recruiter needs to read. The stronger end-to-end projects remain in [`../projects/`](../projects/), while the original MSc notebooks remain at repository root and in [`../docs/UNIVERSITY_PROJECTS.md`](../docs/UNIVERSITY_PROJECTS.md).

The intended progression is:

```text
focused skill notebook
        ↓
original university application
        ↓
strengthened / production-style project
```

That gives me both breadth and depth without pretending that every 30-line learning exercise is a flagship project.
