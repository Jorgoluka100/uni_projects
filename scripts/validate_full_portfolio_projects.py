"""Validate that every professional project is a complete recruiter-facing application.

The goal is not raw line count. A project passes only when it has enough visible depth
and a complete story: problem, data provenance/quality, direct EDA/visualisation,
implementation, evaluation, robustness/failure analysis, decision logic,
reproducibility and limitations.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "projects"
MIN_MEANINGFUL_LINES = 800

REQUIRED_NOTEBOOK_TERMS = {
    "problem_or_objective": ("problem", "objective", "goal", "business", "stakeholder"),
    "data_and_provenance": ("dataset", "data source", "provenance", "source data", "data"),
    "quality_or_preprocessing": ("cleaning", "preprocessing", "data quality", "schema", "missing"),
    "eda_or_visualisation": ("eda", "exploratory", "visualisation", "visualization", "plot", "distribution"),
    "baseline_or_comparison": ("baseline", "benchmark", "comparison", "compare", "alternative"),
    "evaluation": ("evaluation", "metric", "validation", "result", "performance"),
    "error_or_robustness": ("error", "residual", "confusion", "robustness", "slice", "drift", "sensitivity"),
    "decision_or_solution": ("decision", "recommendation", "action", "solution", "inference", "policy"),
    "limitations": ("limitation", "risk", "caveat", "next step", "future work"),
    "reproducibility": ("reproduce", "reproducibility", "run the", "requirements", "environment"),
}


def source_text(notebook: dict) -> str:
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        parts.append(str(src))
    return "\n".join(parts).lower()


def meaningful_code_lines(notebook: dict) -> int:
    total = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for line in str(src).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def main() -> None:
    failures: list[str] = []
    report: list[dict[str, object]] = []

    for project in sorted(p for p in PROJECTS_ROOT.iterdir() if p.is_dir()):
        notebook_path = project / "project_notebook.ipynb"
        readme_path = project / "README.md"
        py_files = sorted(
            p for p in project.rglob("*.py")
            if "__pycache__" not in p.parts and ".venv" not in p.parts and "venv" not in p.parts
        )

        missing: list[str] = []
        if not notebook_path.exists():
            missing.append("project_notebook.ipynb")
        if not readme_path.exists():
            missing.append("README.md")
        if not py_files:
            missing.append("Python implementation")

        code_lines = 0
        missing_story: list[str] = []
        if notebook_path.exists():
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            text = source_text(notebook)
            code_lines = meaningful_code_lines(notebook)
            for label, alternatives in REQUIRED_NOTEBOOK_TERMS.items():
                if not any(term in text for term in alternatives):
                    missing_story.append(label)
            if code_lines < MIN_MEANINGFUL_LINES:
                missing.append(
                    f"full recruiter notebook depth (>={MIN_MEANINGFUL_LINES} meaningful visible code lines; no padding)"
                )

        evidence_exists = any((project / name).exists() for name in ("tests", "results", "artifacts", "outputs"))
        verified = ROOT / "verified" / project.name
        if verified.exists():
            evidence_exists = True
        if not evidence_exists:
            missing.append("tests/results/evidence route")

        if missing_story:
            missing.append("notebook story: " + ", ".join(missing_story))

        if missing:
            failures.append(f"{project.name}: " + "; ".join(missing))

        report.append({
            "project": project.name,
            "meaningful_code_lines": code_lines,
            "minimum_visible_depth": MIN_MEANINGFUL_LINES,
            "python_files": len(py_files),
            "has_readme": readme_path.exists(),
            "has_evidence_route": evidence_exists,
            "missing_story_components": missing_story,
            "status": "PASS" if not missing else "FAIL",
        })

    print(json.dumps({"projects": report, "failures": failures}, indent=2))
    if failures:
        raise SystemExit("\n".join(["Full portfolio project validation FAILED:", *failures]))
    print(f"Full portfolio project validation PASSED for {len(report)} projects.")


if __name__ == "__main__":
    main()
