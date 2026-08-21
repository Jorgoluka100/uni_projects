# Image Classification with Confidence Checks

This is a PyTorch bean-leaf classifier, but I did not want the project to stop at a single accuracy number. I added calibration, confidence-based review, Grad-CAM and export checks so the project also shows what happens when the model is uncertain.

The task uses the public Makerere Beans dataset with three classes: `angular_leaf_spot`, `bean_rust` and `healthy`.

## Test result

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

The saved checkpoint was checked independently against the test set and reproduced the headline metrics. See [`results/verified_metrics.json`](results/verified_metrics.json).

## Confidence-based review

A model can be correct most of the time and still be unreliable on particular images. I therefore save a confidence threshold and route low-confidence predictions for review instead of treating every prediction the same way.

On the retained test split:

- confidence threshold: **0.676**
- automatic coverage: **89.1%**
- sent to review: **10.9%**
- accuracy on accepted predictions: **90.4%**
- model errors escalated for review: **38.9%**

That trade-off is more useful to me than simply saying the classifier reached about 86% accuracy.

## Project structure

```text
image_classification_confidence/
├── src/
│   ├── model.py
│   ├── evaluation.py
│   ├── gradcam.py
│   └── parity.py
├── tests/
│   └── test_evaluation.py
├── results/
│   └── verified_metrics.json
├── MODEL_CARD.md
└── run.py
```

The original executed training notebook is still available at [`12_VisionForge_PyTorch_Visual_Inspection.ipynb`](../../12_VisionForge_PyTorch_Visual_Inspection.ipynb). The current project folder uses a descriptive name instead of the old product-style one.

## Model

The verified checkpoint uses EfficientNet-B0 with this classifier head:

```text
Dropout(0.30)
→ Linear(backbone_features, 256)
→ SiLU
→ Dropout(0.20)
→ Linear(256, 3)
```

The architecture is kept in [`src/model.py`](src/model.py) so it is easy to inspect without digging through notebook cells.

## Calibration and evaluation

[`src/evaluation.py`](src/evaluation.py) contains:

- temperature-scaled softmax
- accuracy, balanced accuracy, macro-F1 and negative log likelihood
- expected calibration error (ECE)
- coverage/review/error-escalation metrics
- bootstrap confidence intervals

The lightweight CI tests these behaviours with small synthetic probability arrays rather than downloading the full deep-learning stack.

## Grad-CAM

[`src/gradcam.py`](src/gradcam.py) contains the Grad-CAM implementation used for inspection. I treat the heatmaps as a debugging aid, not as proof that the network has learned the correct biological reason for a disease class.

## Export checks

The saved model was also checked against exported inference formats:

- **TorchScript:** max absolute error **0.0**
- **ONNX:** max absolute error **4.77e-06**

[`src/parity.py`](src/parity.py) contains the comparison logic.

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

`python run.py` checks the compact saved evidence. Re-running the heavier checkpoint/test-set verification requires the computer-vision dependencies in `requirements-vision.txt`.

## What I would test next

The bigger question is domain shift. Before using a model like this outside the dataset, I would test images from different phones, lighting conditions, locations, cultivars and disease severities, then measure accuracy and calibration again.

## Limitations

The public test split contains only **128 images**, which is why I report bootstrap intervals instead of treating 85.9% as a perfectly precise estimate. This is an educational plant-disease classifier and should not be the only basis for agronomic treatment decisions. See [`MODEL_CARD.md`](MODEL_CARD.md).