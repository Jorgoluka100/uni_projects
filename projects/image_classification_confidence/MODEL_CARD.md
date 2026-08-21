# Model card — bean leaf image classification

## Intended use

Educational computer-vision prototype for classifying bean-leaf images into three classes: `angular_leaf_spot`, `bean_rust` and `healthy`. The project is designed to demonstrate transfer learning, calibration, explainability, uncertainty-aware escalation and deployable model export.

It is **not** intended to replace an agronomist or to make crop-treatment decisions without human review.

## Model

EfficientNet-B0 with a custom classification head:

```text
EfficientNet-B0 backbone
→ Dropout(0.30)
→ Linear(features, 256)
→ SiLU
→ Dropout(0.20)
→ Linear(256, 3)
```

The retained checkpoint is independently reloaded and evaluated against the public `AI-Lab-Makerere/beans` test split.

## Retained test evidence

| Metric | Result |
|---|---:|
| Test images | 128 |
| Accuracy | 85.9% |
| Balanced accuracy | 86.0% |
| Macro-F1 | 85.8% |
| Negative log likelihood | 0.311 |
| ECE after calibration | 0.056 |
| Accuracy 95% bootstrap interval | 79.7%–91.4% |
| Macro-F1 95% bootstrap interval | 79.2%–91.3% |

Point estimates are not presented without the intervals because the retained test set is small.

## Selective prediction / human review

The saved confidence threshold is **0.676**. On the retained test split it:

- accepts **89.1%** of predictions automatically;
- routes **10.9%** to review;
- reaches **90.4%** accuracy on accepted predictions; and
- escalates **38.9%** of the model's errors.

This is a decision policy, not a claim that confidence perfectly measures correctness. The threshold should be revalidated when the deployment population changes.

## Explainability

Grad-CAM is used to inspect which image regions contribute to a prediction. Heatmaps are a debugging and communication aid; they are not proof that a model has learned a biologically valid disease mechanism.

## Export verification

The retained verification process checks fresh-process numerical parity against both exported formats:

- TorchScript: passed, max absolute error **0.0**
- ONNX: passed, max absolute error **4.77e-06**

An export existing on disk is therefore not treated as sufficient evidence by itself.

## Known limitations

- The retained test split contains only 128 images.
- Field lighting, camera type, cultivar, geography and disease severity can create domain shift.
- Calibration and confidence thresholds can drift over time.
- Grad-CAM can be visually plausible even when a prediction is wrong.
- This model should not be used as the sole basis for agricultural treatment decisions.

## Human oversight

Low-confidence cases should be reviewed by a knowledgeable person, and production monitoring should track performance, calibration, class balance and input shift before automated decisions are expanded.
