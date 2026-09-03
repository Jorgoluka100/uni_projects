"""Support Vector Machines — compact foundation project.

DataCamp-style evidence for scaling, linear and RBF kernels, margins,
hyperparameter tuning, support vectors, evaluation and inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

RANDOM_STATE = 42

# 1. Load and inspect a real classification dataset.
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()
print("Dataset shape:", X.shape)
print("Target balance:")
print(y.value_counts(normalize=True).sort_index().rename("share"))

# 2. Hold out an untouched test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# 3. Establish a trivial baseline.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# 4. Linear SVM. Scaling stays inside the pipeline to avoid leakage.
linear_svm = Pipeline(
    [
        ("scale", StandardScaler()),
        (
            "model",
            LinearSVC(
                C=1.0,
                dual="auto",
                random_state=RANDOM_STATE,
                max_iter=20000,
            ),
        ),
    ]
)
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)
linear_scores = linear_svm.decision_function(X_test)

# 5. Non-linear RBF SVM with cross-validated C and gamma.
rbf_pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", SVC(kernel="rbf")),
    ]
)
search = GridSearchCV(
    rbf_pipeline,
    {
        "model__C": [0.1, 1.0, 10.0, 100.0],
        "model__gamma": ["scale", 0.001, 0.01, 0.1],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
search.fit(X_train, y_train)
rbf_pred = search.predict(X_test)

# 6. Compare the baseline, linear margin classifier and tuned kernel SVM.
comparison = pd.DataFrame(
    [
        {
            "model": "Dummy most-frequent",
            "accuracy": accuracy_score(y_test, baseline_pred),
            "f1": f1_score(y_test, baseline_pred),
        },
        {
            "model": "Linear SVM",
            "accuracy": accuracy_score(y_test, linear_pred),
            "f1": f1_score(y_test, linear_pred),
        },
        {
            "model": "Tuned RBF SVM",
            "accuracy": accuracy_score(y_test, rbf_pred),
            "f1": f1_score(y_test, rbf_pred),
        },
    ]
).sort_values("f1", ascending=False)

print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))
print("\nBest RBF parameters:", search.best_params_)
print("\nRBF confusion matrix:")
print(confusion_matrix(y_test, rbf_pred))

# 7. Inspect support vectors and linear decision margins.
best_svc = search.best_estimator_.named_steps["model"]
print("\nSupport vectors per class:", best_svc.n_support_)
margin_summary = pd.Series(np.abs(linear_scores), name="absolute_margin").describe()
print("\nLinear-SVM absolute-margin summary:")
print(margin_summary.round(4))

# 8. PCA provides a 2D inspection view only; the classifiers use full features.
X_scaled = StandardScaler().fit_transform(X)
projection = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
projection_df = pd.DataFrame(
    {
        "pc1": projection[:, 0],
        "pc2": projection[:, 1],
        "target": y.map({0: data.target_names[0], 1: data.target_names[1]}),
    }
)
print("\n2D PCA sample:")
print(projection_df.head())

# 9. Inference example.
example = X_test.iloc[[0]]
predicted_label = int(search.predict(example)[0])
print("\nExample prediction:", data.target_names[predicted_label])
