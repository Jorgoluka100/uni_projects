"""Lightweight integrity checks for the recruiter-facing portfolio.

This deliberately does not retrain models in CI. It verifies that the promoted
notebooks are present, valid notebook JSON, contain code, retain executed output,
and do not contain stored Python/Jupyter error outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

EXPECTED_NOTEBOOKS = [
    "01_UK_House_Price_Analysis_and_Prediction.ipynb",
    "02_SQL_Sales_and_Customer_Analysis.ipynb",
    "03_Customer_Churn_Prediction.ipynb",
    "04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb",
    "05_Energy_Demand_Forecasting_with_TensorFlow.ipynb",
    "06_Clickstream_Analysis_with_PySpark.ipynb",
    "07_London_Air_Quality_Analysis_with_R.ipynb",
]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def check_notebook(path: Path, failures: list[str]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - CI diagnostic
        fail(f"{path.name}: invalid notebook JSON ({exc})", failures)
        return

    if int(notebook.get("nbformat", 0)) < 4:
        fail(f"{path.name}: expected nbformat >= 4", failures)

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        fail(f"{path.name}: no notebook cells found", failures)
        return

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]

    if not code_cells:
        fail(f"{path.name}: no code cells found", failures)
    if not markdown_cells:
        fail(f"{path.name}: no markdown/documentation cells found", failures)

    if code_cells and not any(cell.get("execution_count") is not None for cell in code_cells):
        fail(f"{path.name}: no executed code cells retained", failures)

    stored_errors: list[str] = []
    for index, cell in enumerate(code_cells, start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                stored_errors.append(
                    f"code cell {index}: {output.get('ename', 'Error')} - "
                    f"{output.get('evalue', '')}"
                )

    if stored_errors:
        fail(
            f"{path.name}: stored execution errors found: " + "; ".join(stored_errors),
            failures,
        )
    else:
        print(f"PASS: {path.name}")


def main() -> int:
    failures: list[str] = []

    if not README.exists():
        fail("README.md is missing", failures)
        readme_text = ""
    else:
        readme_text = README.read_text(encoding="utf-8")

    for filename in EXPECTED_NOTEBOOKS:
        path = ROOT / filename
        if not path.exists():
            fail(f"missing promoted notebook: {filename}", failures)
            continue
        if filename not in readme_text:
            fail(f"README.md does not link to {filename}", failures)
        check_notebook(path, failures)

    if failures:
        print(f"\nPortfolio validation failed with {len(failures)} issue(s).")
        return 1

    print(f"\nPortfolio validation passed for {len(EXPECTED_NOTEBOOKS)} promoted notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
