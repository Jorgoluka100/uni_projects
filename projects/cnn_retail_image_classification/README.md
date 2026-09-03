# CNN 2D Image Classification — Convolutions, Padding & Confidence Routing

A full computer-vision / deep-learning portfolio project built to make **2D convolutional neural networks** visible as a first-class skill rather than hiding them inside a generic image-classification project.

## Problem

An image-classification or visual-inspection team needs to classify incoming images automatically, but it should not force a prediction when the model is uncertain. The project therefore treats image classification as a **model + confidence-routing decision system**.

## Dataset

**CIFAR-10** via `torchvision.datasets.CIFAR10`: 60,000 labelled colour images across 10 classes. The loader downloads the dataset at runtime, so raw data is not committed to GitHub.

## Deep-learning coverage

This project deliberately demonstrates the progression from fundamentals to stronger engineering:

1. image tensors: batch × channel × height × width
2. train / validation / test separation
3. normalisation and data augmentation
4. **`nn.Conv2d` 2D convolution**
5. kernel size, input/output channels and learned feature maps
6. **padding (`padding=1`)** and its effect on spatial dimensions
7. stride and receptive-field reasoning
8. convolution → ReLU → pooling blocks
9. **`MaxPool2d`** spatial downsampling
10. deeper CNN with **`BatchNorm2d`**
11. **Dropout** regularisation
12. adaptive average pooling
13. flattening / linear classification heads
14. AdamW optimisation
15. learning-rate scheduling
16. label smoothing
17. training / validation learning curves
18. confusion matrix and per-class errors
19. **ResNet18 transfer learning** comparison
20. probability calibration / temperature scaling
21. expected calibration error
22. selective prediction / confidence thresholds
23. human-review routing for uncertain images
24. model persistence and reproducible inference
25. explainability route for Grad-CAM / activation inspection
26. limitations and production next steps

## Models

- `CompactCNN` — from-scratch `Conv2d` baseline with 3×3 kernels, padding, ReLU, max pooling and adaptive average pooling.
- `RegularizedCNN` — deeper convolution blocks with repeated `Conv2d`, BatchNorm, ReLU, MaxPool and Dropout.
- `ResNet18` — pretrained transfer-learning comparator adapted for 32×32 images, including a modified 3×3 padded first convolution.

The point is **not to assume the largest network wins**. The final choice should come from validation accuracy, calibration, error slices, compute cost and the confidence/coverage trade-off.

## Why padding matters here

For a 3×3 convolution, `padding=1` keeps height and width unchanged when stride is 1. That lets the network learn local features without shrinking the feature map after every convolution; pooling is then used deliberately when spatial downsampling is wanted. The implementation makes this visible in the custom networks rather than hiding it inside a pretrained model.

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

CIFAR-10 is a benchmark rather than a real production image catalogue. It is used here because it is reproducible and allows the CNN workflow to be evaluated end to end. Production deployment would require domain-specific images, class taxonomy governance, distribution-shift monitoring, privacy/licensing review and a documented human escalation policy.
