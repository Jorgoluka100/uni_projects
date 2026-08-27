# Data & AI Skills Lab

The main portfolio answers **“can I build and evaluate useful projects?”**. This folder answers a different interview question: **“do I understand the building blocks?”**

Every focused skill now has **both formats**:

- `.ipynb` for explanation, revision and Colab/Jupyter use
- `.py` for normal Python execution and code review

These are deliberately small learning artifacts, not fake production systems. The larger end-to-end implementations remain under [`../projects/`](../projects/).

## Core notebook + Python pairs

| # | Notebook | Python | What it proves |
| --- | --- | --- | --- |
| 01 | [`01_data_cleaning_preprocessing.ipynb`](01_data_cleaning_preprocessing.ipynb) | [`01_data_cleaning_preprocessing.py`](01_data_cleaning_preprocessing.py) | missing values, duplicates, types, encoding and leakage-safe preprocessing |
| 02 | [`02_numpy_for_machine_learning.ipynb`](02_numpy_for_machine_learning.ipynb) | [`02_numpy_for_machine_learning.py`](02_numpy_for_machine_learning.py) | arrays, shapes, broadcasting, vectorisation and matrix multiplication |
| 03 | [`03_sklearn_end_to_end_classification.ipynb`](03_sklearn_end_to_end_classification.ipynb) | [`03_sklearn_end_to_end_classification.py`](03_sklearn_end_to_end_classification.py) | split, preprocessing pipeline, dummy baseline, logistic regression and metrics |
| 04 | [`04_pytorch_neural_network_fundamentals.ipynb`](04_pytorch_neural_network_fundamentals.ipynb) | [`04_pytorch_neural_network_fundamentals.py`](04_pytorch_neural_network_fundamentals.py) | tensors, layers, forward pass, loss, backpropagation, optimiser and evaluation |
| 05 | [`05_lstm_sequence_modelling.ipynb`](05_lstm_sequence_modelling.ipynb) | [`05_lstm_sequence_modelling.py`](05_lstm_sequence_modelling.py) | Long Short-Term Memory, sequence shapes, hidden state and sequence classification |
| 06 | [`06_text_classification_tfidf.ipynb`](06_text_classification_tfidf.ipynb) | [`06_text_classification_tfidf.py`](06_text_classification_tfidf.py) | TF-IDF, sparse features, logistic regression and text evaluation |
| 07 | [`07_cnn_image_fundamentals.ipynb`](07_cnn_image_fundamentals.ipynb) | [`07_cnn_image_fundamentals.py`](07_cnn_image_fundamentals.py) | convolution, channels, pooling, flattening and a compact CNN training loop |

## How this fits the portfolio

The intended progression is:

```text
focused notebook + Python script
        ↓
original university / uploaded notebook
        ↓
strengthened project with notebook + Python package
```

The original MSc and previously uploaded notebooks remain at repository root. They are indexed in [`../docs/NOTEBOOK_INDEX.md`](../docs/NOTEBOOK_INDEX.md) and the complete project catalogue.
