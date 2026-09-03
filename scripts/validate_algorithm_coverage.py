"""Validate the ML algorithm coverage that recruiters should be able to find quickly.

This is deliberately a source-evidence check, not a claim generator. The three core
algorithms the portfolio must make unambiguous are Linear Regression, Logistic
Regression and Naive Bayes. The script also protects the wider classical-ML breadth
already present in the foundations/projects.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_APPLIED = {
    "Linear Regression": {
        "project": ROOT / "projects" / "linear_regression_energy_efficiency",
        "tokens": ("LinearRegression", "RidgeCV", "LassoCV"),
    },
    "Logistic Regression": {
        "project": ROOT / "projects" / "deep_learning_marketing_response",
        "tokens": ("LogisticRegression", "fit_logistic_baseline"),
    },
    "Naive Bayes": {
        "project": ROOT / "projects" / "nlp_document_intelligence",
        "tokens": ("MultinomialNB", "build_nb_baseline"),
    },
}

WIDER_COVERAGE = {
    "K-Nearest Neighbours": ("KNeighborsClassifier",),
    "Support Vector Machines": ("LinearSVC", "SVC"),
    "Decision Trees": ("DecisionTreeClassifier",),
    "Random Forests": ("RandomForestClassifier",),
    "Gradient Boosting": ("GradientBoostingClassifier",),
    "K-Means clustering": ("KMeans",),
    "XGBoost": ("XGBRegressor",),
    "CatBoost": ("CatBoost",),
}


def _all_python_under(path: Path) -> str:
    chunks: list[str] = []
    for file in sorted(path.rglob("*.py")):
        if "__pycache__" in file.parts:
            continue
        chunks.append(file.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _notebook_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        chunks.append("".join(source) if isinstance(source, list) else str(source or ""))
    return "\n".join(chunks)


def main() -> int:
    failures: list[str] = []

    coverage_doc = ROOT / "docs" / "ML_ALGORITHM_COVERAGE.md"
    doc_text = coverage_doc.read_text(encoding="utf-8") if coverage_doc.is_file() else ""

    print("Core applied algorithm evidence:")
    for name, spec in CORE_APPLIED.items():
        project: Path = spec["project"]
        notebook = project / "project_notebook.ipynb"
        if not project.is_dir():
            failures.append(f"{name}: missing project directory {project.relative_to(ROOT)}")
            continue
        if not notebook.is_file():
            failures.append(f"{name}: missing project notebook")
            continue

        python_text = _all_python_under(project)
        notebook_text = _notebook_text(notebook)
        missing_source = [token for token in spec["tokens"] if token not in python_text]
        missing_notebook = [token for token in spec["tokens"] if token not in notebook_text]

        if missing_source:
            failures.append(f"{name}: source evidence missing {missing_source}")
        if missing_notebook:
            failures.append(f"{name}: notebook evidence missing {missing_notebook}")
        if name not in doc_text:
            failures.append(f"{name}: docs/ML_ALGORITHM_COVERAGE.md does not name the algorithm")

        if not missing_source and not missing_notebook:
            print(f"  PASS {name}: source + notebook evidence")

    source_pool = "\n".join(
        [
            _all_python_under(ROOT / "projects"),
            _all_python_under(ROOT / "skills"),
        ]
    )

    print("Wider classical-ML coverage:")
    for name, tokens in WIDER_COVERAGE.items():
        if not any(token in source_pool for token in tokens):
            failures.append(f"{name}: no source evidence found for any of {list(tokens)}")
        else:
            print(f"  PASS {name}")

    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        return 1

    print("ML algorithm coverage validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
