"""Validate notebook/Python coverage for recruiter-facing and historical work."""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "portfolio_manifest.json"
CATALOG = ROOT / "docs" / "PROJECT_CATALOG.md"
NOTEBOOK_INDEX = ROOT / "docs" / "NOTEBOOK_INDEX.md"
SKILLS = ROOT / "skills"


def validate_notebook(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing notebook: {path.relative_to(ROOT)}")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid notebook JSON {path.relative_to(ROOT)}: {exc}")
        return
    if payload.get("nbformat") != 4:
        failures.append(f"unexpected nbformat in {path.relative_to(ROOT)}")
    if not isinstance(payload.get("cells"), list):
        failures.append(f"missing cells list in {path.relative_to(ROOT)}")


def main() -> int:
    failures: list[str] = []
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    production = payload.get("production_projects", [])
    for project in production:
        project_path = ROOT / project["path"]
        notebook = project_path / "project_notebook.ipynb"
        validate_notebook(notebook, failures)
        python_files = list(project_path.rglob("*.py")) if project_path.is_dir() else []
        if not python_files:
            failures.append(f"no Python source found under {project['path']}")

    skill_notebooks = sorted(SKILLS.glob("[0-9][0-9]_*.ipynb"))
    if not skill_notebooks:
        failures.append("no focused skills notebooks found")
    for notebook in skill_notebooks:
        validate_notebook(notebook, failures)
        script = notebook.with_suffix(".py")
        if not script.is_file():
            failures.append(f"missing Python twin for {notebook.relative_to(ROOT)}")

    if not CATALOG.is_file():
        failures.append("docs/PROJECT_CATALOG.md is missing")
    else:
        catalog_text = CATALOG.read_text(encoding="utf-8")
        historical_names = sorted(set(re.findall(r"`([^`]+\.ipynb)`", catalog_text)))
        for name in historical_names:
            path = ROOT / name
            if not path.is_file():
                failures.append(f"catalogued historical notebook is missing: {name}")

    if not NOTEBOOK_INDEX.is_file():
        failures.append("docs/NOTEBOOK_INDEX.md is missing")

    print(f"Production projects checked: {len(production)}")
    print(f"Focused notebook/Python pairs checked: {len(skill_notebooks)}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        return 1

    print("Notebook/Python coverage validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
