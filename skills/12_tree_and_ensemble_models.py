"""Decision Trees, Random Forests & Gradient Boosting — foundation project.

A compact comparison of a single tree, bagging-style random forest and
sequential gradient boosting, including tuning and feature importance.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# 1. Load and inspect the dataset.
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()
print("Dataset shape:", X.shape)
print("Target distribution:")
print(y.value_counts().sort_index())

# 2. Create one shared stratified test set for a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# 3. Compare a dummy baseline with the three main tree-model families.
models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

rows = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    rows.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
    )

comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))

# 4. Inspect the complexity of the single interpretable tree.
tree = models["Decision Tree"]
print(f"\nDecision-tree depth={tree.get_depth()}, leaves={tree.get_n_leaves()}")

# 5. Tune the random forest with cross-validation.
rf_search = GridSearchCV(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    {
        "n_estimators": [200, 400],
        "max_depth": [None, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.7],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
rf_pred = rf_search.predict(X_test)

print("\nBest random-forest parameters:", rf_search.best_params_)
print("Tuned random-forest accuracy:", round(accuracy_score(y_test, rf_pred), 4))
print("Tuned random-forest F1:", round(f1_score(y_test, rf_pred), 4))
print("\nTuned random-forest confusion matrix:")
print(confusion_matrix(y_test, rf_pred))

# 6. Inspect model-driven feature importance.
best_rf = rf_search.best_estimator_
importance = (
    pd.Series(best_rf.feature_importances_, index=X.columns, name="importance")
    .sort_values(ascending=False)
    .head(12)
)
print("\nTop random-forest feature importances:")
print(importance.round(4))

gb = models["Gradient Boosting"]
gb_importance = (
    pd.Series(gb.feature_importances_, index=X.columns, name="importance")
    .sort_values(ascending=False)
    .head(10)
)
print("\nTop gradient-boosting feature importances:")
print(gb_importance.round(4))

# 7. Inference example.
example = X_test.iloc[[0]]
predicted_label = int(best_rf.predict(example)[0])
print("\nExample tuned-forest prediction:", data.target_names[predicted_label])
