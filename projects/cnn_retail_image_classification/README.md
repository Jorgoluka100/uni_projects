# CNN Retail Image Classification & Confidence Routing

A full computer-vision / deep-learning portfolio project built to make **convolutional neural networks** visible as a first-class skill rather than hiding them inside a generic image-classification project.

## Problem

A retail or visual-inspection team needs to classify incoming product images automatically, but it should not force a prediction when the model is uncertain. The project therefore treats image classification as a **model + confidence-routing decision system**.

## Dataset

**CIFAR-10** via `torchvision.datasets.CIFAR10`: 60,000 labelled colour images across 10 classes. The loader downloads the dataset at runtime, so raw data is not committed to GitHub.

## Deep-learning coverage

This project deliberately demonstrates the progression from fundamentals to stronger engineering:

1. image tensor inspection and class balance
2. train / validation / test separation
3. normalisation and data augmentation
4. **custom convolutional neural network**
5. convolution → ReLU → pooling → feature maps
6. deeper CNN with **BatchNorm**
7. **Dropout** regularisation
8. AdamW optimisation
9. learning-rate scheduling
10. label smoothing
11. training / validation learning curves
12. confusion matrix and per-class errors
13. **ResNet18 transfer learning** comparison
14. probability calibration / temperature scaling
15. expected calibration error
16. selective prediction / confidence thresholds
17. human-review routing for uncertain images
18. model persistence and reproducible inference
19. explainability route for Grad-CAM / activation inspection
20. limitations and production next steps

## Models

- `CompactCNN` — small from-scratch CNN baseline.
- `RegularizedCNN` — deeper convolution blocks with BatchNorm and Dropout.
- `ResNet18` — pretrained transfer-learning comparator adapted for 32×32 images.

The point is **not to assume the largest network wins**. The final choice should come from validation accuracy, calibration, error slices, compute cost and the confidence/coverage trade-off.

## Decision layer

Instead of forcing every image through automation, the project evaluates confidence thresholds. High-confidence predictions can be accepted automatically while ambiguous images are routed to manual review.

This makes the project relevant to visual quality inspection, e-commerce image routing, document/image triage and junior ML / AI engineering roles.

## Run

```bash
pip install -r requirements.txt
python run.py --model regularized --epochs 8
```

Alternative models:

```bash
python run.py --model compact --epochs 8
python run.py --model resnet18 --epochs 5
```

## Evidence produced

The run writes reproducible evidence to `artifacts/`:

- training history
- classification report
- per-class error table
- test predictions and confidence
- calibration metrics
- selective-prediction trade-off table
- saved model weights

## Recruiter notebook

Open [`project_notebook.ipynb`](project_notebook.ipynb). It is the main readable application and is expanded by the repository notebook workflow with direct EDA, visual analysis, robustness checks and decision logic.

## Limitations

CIFAR-10 is a benchmark rather than a real retailer's production catalogue. It is used here because it is reproducible and allows the CNN workflow to be evaluated end to end. Production deployment would require domain-specific images, class taxonomy governance, distribution-shift monitoring, privacy/licensing review and a documented human escalation policy.
