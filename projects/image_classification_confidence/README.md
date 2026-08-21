# Image Classification with Confidence Checks

A PyTorch computer-vision project that goes beyond test accuracy by checking **calibration, uncertainty, human-review thresholds, Grad-CAM explanations and exported-model parity**.

The underlying task classifies bean-leaf images as `angular_leaf_spot`, `bean_rust` or `healthy` using the public Makerere Beans dataset.

## Retained result

| Metric | Independent test result |
|---|---:|
| Test images | 128 |
| Accuracy | **85.9%** |
| Balanced accuracy | **86.0%** |
| Macro-F1 | **85.8%** |
| Negative log likelihood | **0.311** |
| ECE after calibration | **0.056** |
| Accuracy 95% bootstrap interval | **79.7%–91.4%** |
| Macro-F1 95% bootstrap interval | **79.2%–91.3%** |

The independent verifier reproduced the retained headline metrics exactly from the saved checkpoint. See [`results/verified_metrics.json`](results/verified_metrics.json).

## Why confidence matters here

A classifier can be accurate on average and still be unreliable on individual examples. Instead of treating every prediction equally, this project saves a calibrated confidence threshold and uses it to decide when the model should **abstain and request review**.

On the retained test split:

- confidence threshold: **0.676**
- automatic coverage: **89.1%**
- routed to review: **10.9%**
- accuracy on accepted predictions: **90.4%**
- model errors escalated for review: **38.9%**

That is a more useful operational story than “the model got 86% accuracy.” It also makes the trade-off explicit: higher accepted accuracy comes at the cost of reviewing some cases.

## Engineering pieces exposed in this project

```text
image_classification_confidence/
├── src/
│   ├── model.py       # exact EfficientNet-B0 classifier head
│   ├── evaluation.py  # metrics, ECE, bootstrap intervals, selective prediction
│   ├── gradcam.py     # reusable Grad-CAM implementation
│   └── parity.py      # numerical export-parity checks
├── tests/
│   └── test_evaluation.py
├── results/
│   └── verified_metrics.json
├── MODEL_CARD.md
└── run.py
```

The original executed training/evaluation notebook is retained at [`12_VisionForge_PyTorch_Visual_Inspection.ipynb`](../../12_VisionForge_PyTorch_Visual_Inspection.ipynb), but the recruiter-facing project no longer depends on the old product-style name.

## Model architecture

The verified checkpoint uses EfficientNet-B0 with a custom head:

```text
Dropout(0.30)
→ Linear(backbone_features, 256)
→ SiLU
→ Dropout(0.20)
→ Linear(256, 3)
```

The architecture is exposed in [`src/model.py`](src/model.py) so a reviewer does not need to search notebook cells to understand the saved checkpoint.

## Calibration and selective prediction

[`src/evaluation.py`](src/evaluation.py) includes:

- numerically stable temperature-scaled softmax;
- accuracy, balanced accuracy, macro-F1 and negative log likelihood;
- expected calibration error (ECE);
- confidence-based coverage/review/error-escalation metrics; and
- non-parametric bootstrap confidence intervals.

The lightweight CI uses synthetic probabilities to test these behaviours without downloading a large deep-learning runtime.

## Explainability

[`src/gradcam.py`](src/gradcam.py) provides a small reusable Grad-CAM implementation for convolutional feature maps. I use explanations as a **debugging and inspection tool**, not as proof that the network has learned a biologically correct disease mechanism.

## Export evidence

The retained model was independently checked against exported inference formats:

- **TorchScript:** parity passed; max absolute error **0.0**
- **ONNX:** parity passed; max absolute error **4.77e-06**

[`src/parity.py`](src/parity.py) exposes the numerical comparison logic. The independent verifier also checks that the retained checkpoint hash matches the expected artifact.

## Run the fast checks

```bash
cd projects/image_classification_confidence
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python run.py --self-test
python -m unittest discover -s tests -v
python run.py
```

`python run.py` validates the compact retained evidence bundle. The heavier original checkpoint/test-set verification requires the computer-vision dependencies in `requirements-vision.txt` and is retained in the repository's hardened verifier.

## What I would improve next

The biggest question is **domain shift**, not another decimal point of test accuracy. Before field use I would collect images from different phones, lighting conditions, cultivars, locations and disease severities; then measure class performance, calibration and the review policy again on that external data.

I would also track calibration drift and review workload after deployment rather than fixing one confidence threshold permanently.

## Limitations

The public test split contains only **128 images**, so I report bootstrap uncertainty rather than presenting 85.9% as a precise population estimate. This is an educational plant-disease classifier and should not be the sole basis for agronomic treatment decisions. See [`MODEL_CARD.md`](MODEL_CARD.md) for the full intended-use and oversight notes.
