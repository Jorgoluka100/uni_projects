"""Validate the focused Data & AI skills notebooks using only the stdlib."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
EXPECTED = [
    "01_data_cleaning_preprocessing.ipynb",
    "02_numpy_for_machine_learning.ipynb",
    "03_sklearn_end_to_end_classification.ipynb",
    "04_pytorch_neural_network_fundamentals.ipynb",
    "05_lstm_sequence_modelling.ipynb",
    "06_text_classification_tfidf.ipynb",
    "07_cnn_image_fundamentals.ipynb",
    "08_regression_fundamentals.ipynb",
    "09_clustering_fundamentals.ipynb",
    "10_sql_analytics_fundamentals.ipynb",
]


def main() -> int:
    failures: list[str] = []

    for name in EXPECTED:
        path = SKILLS / name
        if not path.is_file():
            failures.append(f"missing skills notebook: {name}")
            continue

        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"invalid JSON in {name}: {exc}")
            continue

        if notebook.get("nbformat") != 4:
            failures.append(f"{name}: expected nbformat 4")

        cells = notebook.get("cells")
        if not isinstance(cells, list) or len(cells) < 2:
            failures.append(f"{name}: expected at least two notebook cells")
            continue

        code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
        markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
        if not code_cells:
            failures.append(f"{name}: no code cells")
        if not markdown_cells:
            failures.append(f"{name}: no explanatory markdown cells")

        for index, cell in enumerate(cells):
            source = cell.get("source")
            if not isinstance(source, list) or not all(isinstance(line, str) for line in source):
                failures.append(f"{name}: cell {index} has invalid source")

    readme = SKILLS / "README.md"
    if not readme.is_file():
        failures.append("skills/README.md is missing")
    else:
        text = readme.read_text(encoding="utf-8")
        for name in EXPECTED:
            if name not in text:
                failures.append(f"skills README does not reference {name}")

    print(f"Skill notebooks checked: {len(EXPECTED)}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        return 1

    print("Skill notebook validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
