"""Validate notebook/Python coverage and recruiter-notebook quality."""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "portfolio_manifest.json"
CATALOG = ROOT / "docs" / "PROJECT_CATALOG.md"
NOTEBOOK_INDEX = ROOT / "docs" / "NOTEBOOK_INDEX.md"
PROJECTS = ROOT / "projects"
SKILLS = ROOT / "skills"


def read_notebook(path: Path, failures: list[str]) -> dict | None:
    if not path.is_file():
        failures.append(f"missing notebook: {path.relative_to(ROOT)}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid notebook JSON {path.relative_to(ROOT)}: {exc}")
        return None
    if payload.get("nbformat") != 4:
        failures.append(f"unexpected nbformat in {path.relative_to(ROOT)}")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        failures.append(f"missing/empty cells list in {path.relative_to(ROOT)}")
    return payload


def _cell_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source or "")


def validate_recruiter_notebook(path: Path, failures: list[str]) -> None:
    """Reject placeholders while allowing either full-code or recruiter-walkthrough notebooks."""
    payload = read_notebook(path, failures)
    if payload is None:
        return
    cells = payload.get("cells") or []
    if len(cells) < 8:
        failures.append(
            f"recruiter notebook is too thin ({len(cells)} cells): {path.relative_to(ROOT)}"
        )

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    if len(code_cells) < 3:
        failures.append(f"recruiter notebook needs >=3 code cells: {path.relative_to(ROOT)}")

    code_lines = sum(len(_cell_text(cell).splitlines()) for cell in code_cells)
    full_code_notebook = code_lines >= 80

    # A substantial full project notebook may be intentionally code-first. Smaller
    # walkthrough notebooks must carry more recruiter-facing explanation.
    minimum_markdown = 1 if full_code_notebook else 4
    if len(markdown_cells) < minimum_markdown:
        failures.append(
            f"recruiter notebook needs >={minimum_markdown} markdown cells: {path.relative_to(ROOT)}"
        )

    source_text = "\n".join(_cell_text(cell) for cell in cells).lower()
    if "data" not in source_text:
        failures.append(
            f"recruiter notebook missing 'data' context: {path.relative_to(ROOT)}"
        )

    # Thin walkthroughs must explicitly coach interview discussion. A substantial
    # full-code notebook already demonstrates the implementation directly and is
    # therefore validated on code depth instead.
    if not full_code_notebook and "interview" not in source_text:
        failures.append(
            f"recruiter notebook missing 'interview' context: {path.relative_to(ROOT)}"
        )


def validate_notebook(path: Path, failures: list[str]) -> None:
    read_notebook(path, failures)


def project_directories() -> list[Path]:
    """Return every immediate project folder that contains Python source."""
    if not PROJECTS.is_dir():
        return []
    return sorted(
        directory
        for directory in PROJECTS.iterdir()
        if directory.is_dir() and any(directory.rglob("*.py"))
    )


def main() -> int:
    failures: list[str] = []
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    index_text = NOTEBOOK_INDEX.read_text(encoding="utf-8") if NOTEBOOK_INDEX.is_file() else ""

    # Every Python-backed end-to-end project must have a substantive recruiter notebook.
    projects = project_directories()
    if not projects:
        failures.append("no Python-backed project directories found")
    for project_path in projects:
        notebook = project_path / "project_notebook.ipynb"
        validate_recruiter_notebook(notebook, failures)
        python_files = [path for path in project_path.rglob("*.py") if "__pycache__" not in path.parts]
        if not python_files:
            failures.append(f"no Python source found under {project_path.relative_to(ROOT)}")
        relative_notebook = str(notebook.relative_to(ROOT))
        if index_text and relative_notebook not in index_text:
            failures.append(f"notebook index does not reference {relative_notebook}")

    # Manifest projects must still resolve to real checked project folders.
    checked_paths = {str(path.relative_to(ROOT)) for path in projects}
    for project in payload.get("production_projects", []):
        manifest_path = project.get("path")
        if manifest_path not in checked_paths:
            failures.append(f"manifest project not covered by notebook/Python check: {manifest_path}")

    # Focused skills keep explicit notebook/Python twins, without imposing the larger
    # recruiter-notebook cell-count standard used for end-to-end project folders.
    skill_notebooks = sorted(SKILLS.glob("[0-9][0-9]_*.ipynb"))
    if not skill_notebooks:
        failures.append("no focused skills notebooks found")
    for notebook in skill_notebooks:
        validate_notebook(notebook, failures)
        script = notebook.with_suffix(".py")
        if not script.is_file():
            failures.append(f"missing Python twin for {notebook.relative_to(ROOT)}")

    # Keep catalogue-listed historical notebooks in place.
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

    print(f"Python-backed projects checked: {len(projects)}")
    print(f"Focused notebook/Python pairs checked: {len(skill_notebooks)}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        return 1

    print("Notebook/Python coverage and quality validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
