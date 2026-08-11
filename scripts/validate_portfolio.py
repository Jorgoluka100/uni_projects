"""Portfolio integrity and completion-gate checks.

This script deliberately does not retrain models. It separates the public portfolio
into two evidence tiers:

1. VERIFIED_NOTEBOOKS must contain retained executed output and no stored errors.
2. ADVANCED_NOTEBOOKS must be valid, documented, code-bearing notebooks with no
   stored errors. They may be unexecuted while they are being upgraded/reverified.

The heuristic completion score is diagnostic only; it is not a substitute for a
clean end-to-end rerun and manual evidence review.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

VERIFIED_NOTEBOOKS = [
    "01_UK_House_Price_Analysis_and_Prediction.ipynb",
    "02_SQL_Sales_and_Customer_Analysis.ipynb",
    "03_Customer_Churn_Prediction.ipynb",
    "04_Image_Classification_with_CNNs_and_Transfer_Learning.ipynb",
    "05_Energy_Demand_Forecasting_with_TensorFlow.ipynb",
    "06_Clickstream_Analysis_with_PySpark.ipynb",
    "07_London_Air_Quality_Analysis_with_R.ipynb",
]

ADVANCED_NOTEBOOKS = [
    "01_ConsultAI_AI_Opportunity_Engine.ipynb",
    "12_VisionForge_PyTorch_Visual_Inspection.ipynb",
    "Advanced_Multi_Modal_Health_Analytics_Diagnostic_Suite.ipynb",
    "AeroFlow_AI_Engine.ipynb",
    "Aviation_Strategy_PostgreSQL_Optimization.ipynb",
    "CineIntelligence_NoSQL_DataEngineering.ipynb",
    "Clustering_Models.ipynb",
    "KDDCup.ipynb",
    "LLM_Mastery_Hands_on_Code,_Align_and_Master_LLMs_Alignment.ipynb",
    "LLM_Mastery_Hands_on_Code.ipynb",
    "Logistic_Regression_PySpark.ipynb",
    "Movie_Recommendation_System_A_Hybrid_DL_Pipeline.ipynb",
    "NYC_Airbnb_Market_Analysis (1).ipynb",
    "Naive_Bayes_PySpark.ipynb",
    "Parkinsons_Progression_ML.ipynb",
    "Pathfinding.ipynb",
    "PyTorch_medical_AI_xray_diagnosis.ipynb",
    "Strategic_Telecom_Churn_Analytics_Predictive_SQL.ipynb",
    "financial_fraud_aml_detection_system.ipynb",
]

COMPLETION_SIGNALS: dict[str, tuple[str, ...]] = {
    "decision framing": ("decision", "problem", "objective", "business question"),
    "data provenance": ("source", "provenance", "dataset", "data card"),
    "data quality": ("missing", "duplicate", "schema", "quality"),
    "leakage control": ("leakage", "train_test_split", "group", "time split", "temporal"),
    "baseline": ("baseline", "benchmark"),
    "evaluation": ("test set", "holdout", "roc", "auc", "mae", "rmse", "f1", "accuracy"),
    "uncertainty/error analysis": ("uncertainty", "calibration", "confidence", "error analysis", "interval"),
    "reproducibility": ("seed", "random_state", "requirements", "restart", "reproduc"),
    "engineering evidence": ("unittest", "pytest", "smoke", "artifact", "save", "api", "gradio", "fastapi"),
    "limitations": ("limitation", "failure mode", "does not prove", "next production"),
}


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"WARN: {message}")


def text_from_cells(cells: Iterable[dict]) -> str:
    pieces: list[str] = []
    for cell in cells:
        source = cell.get("source", [])
        if isinstance(source, list):
            pieces.extend(str(item) for item in source)
        elif isinstance(source, str):
            pieces.append(source)
    return "\n".join(pieces)


def stored_errors(code_cells: list[dict]) -> list[str]:
    errors: list[str] = []
    for index, cell in enumerate(code_cells, start=1):
        for output in cell.get("outputs", []) or []:
            if output.get("output_type") == "error":
                errors.append(
                    f"code cell {index}: {output.get('ename', 'Error')} - "
                    f"{output.get('evalue', '')}"
                )
    return errors


def completion_score(cells: list[dict]) -> tuple[int, list[str]]:
    corpus = text_from_cells(cells).lower()
    hits: list[str] = []
    for signal, patterns in COMPLETION_SIGNALS.items():
        if any(pattern.lower() in corpus for pattern in patterns):
            hits.append(signal)
    return len(hits), hits


def load_notebook(path: Path, failures: list[str]) -> dict | None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        fail(f"{path.name}: invalid notebook JSON ({exc})", failures)
        return None

    if int(notebook.get("nbformat", 0)) < 4:
        fail(f"{path.name}: expected nbformat >= 4", failures)

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        fail(f"{path.name}: no notebook cells found", failures)
        return None

    return notebook


def check_common(
    path: Path,
    notebook: dict,
    failures: list[str],
    warnings: list[str],
) -> tuple[list[dict], list[dict], bool]:
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]

    if not code_cells:
        fail(f"{path.name}: no code cells found", failures)
    if not markdown_cells:
        fail(f"{path.name}: no markdown/documentation cells found", failures)

    errors = stored_errors(code_cells)
    if errors:
        fail(f"{path.name}: stored execution errors: {'; '.join(errors)}", failures)

    executed = any(cell.get("execution_count") is not None for cell in code_cells)

    score, hits = completion_score(cells)
    missing = [name for name in COMPLETION_SIGNALS if name not in hits]
    print(
        f"INFO: {path.name}: completion signals {score}/{len(COMPLETION_SIGNALS)}"
        + (f"; missing {', '.join(missing)}" if missing else "; all heuristic signals found")
    )

    # Extremely low documentation depth is a useful corruption/incompleteness signal.
    markdown_text = text_from_cells(markdown_cells).strip()
    if len(markdown_text) < 150:
        warn(f"{path.name}: very little narrative documentation", warnings)

    # Flag unresolved authoring markers without making them fatal.
    corpus = text_from_cells(cells)
    unresolved = re.findall(r"\b(?:TODO|FIXME|TBD)\b", corpus, flags=re.IGNORECASE)
    if unresolved:
        warn(f"{path.name}: contains {len(unresolved)} TODO/FIXME/TBD marker(s)", warnings)

    return code_cells, markdown_cells, executed


def check_verified(
    path: Path,
    failures: list[str],
    warnings: list[str],
) -> None:
    notebook = load_notebook(path, failures)
    if notebook is None:
        return
    _, _, executed = check_common(path, notebook, failures, warnings)
    if not executed:
        fail(f"{path.name}: verified notebook has no retained executed code output", failures)
    else:
        print(f"PASS VERIFIED: {path.name}")


def check_advanced(
    path: Path,
    failures: list[str],
    warnings: list[str],
) -> None:
    notebook = load_notebook(path, failures)
    if notebook is None:
        return
    _, _, executed = check_common(path, notebook, failures, warnings)
    if executed:
        print(f"PASS ADVANCED + EXECUTED: {path.name}")
    else:
        warn(
            f"{path.name}: valid advanced notebook but execution evidence is not retained; "
            "rerun before promoting metrics",
            warnings,
        )
        print(f"PASS ADVANCED STRUCTURE: {path.name}")


def require_listed(filename: str, readme_text: str, failures: list[str]) -> None:
    if filename not in readme_text and filename.replace(" ", "%20") not in readme_text:
        fail(f"README.md does not reference {filename}", failures)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    if not README.exists():
        fail("README.md is missing", failures)
        readme_text = ""
    else:
        readme_text = README.read_text(encoding="utf-8")

    print("\n=== VERIFIED FLAGSHIPS ===")
    for filename in VERIFIED_NOTEBOOKS:
        path = ROOT / filename
        if not path.exists():
            fail(f"missing verified notebook: {filename}", failures)
            continue
        require_listed(filename, readme_text, failures)
        check_verified(path, failures, warnings)

    print("\n=== ADVANCED PROJECT LABORATORY ===")
    for filename in ADVANCED_NOTEBOOKS:
        path = ROOT / filename
        if not path.exists():
            fail(f"missing advanced notebook: {filename}", failures)
            continue
        require_listed(filename, readme_text, failures)
        check_advanced(path, failures, warnings)

    print("\n=== SUMMARY ===")
    print(f"Verified notebooks checked: {len(VERIFIED_NOTEBOOKS)}")
    print(f"Advanced notebooks checked: {len(ADVANCED_NOTEBOOKS)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("Portfolio validation FAILED. Fix hard failures before treating main as healthy.")
        return 1

    print("Portfolio integrity PASSED. Warnings identify projects still awaiting promotion evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
