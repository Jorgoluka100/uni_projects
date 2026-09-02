"""Enrich every recruiter notebook without deleting the original project work.

The portfolio rule is deliberately simple:

    one project -> one substantial notebook -> one complete recruiter story

Existing notebook cells are preserved. Missing canonical Python modules from the
same project directory are appended as an application-engineering layer so a
recruiter can inspect the original analysis AND the production-style code in one
place.

A roughly 500-line notebook is a useful depth guide for many projects, not a quota.
Some focused applications are naturally smaller and some production systems are
much larger. This script reports code depth but never manufactures filler.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "projects"
TARGET_LOW = 250
TARGET_IDEAL = 500
TARGET_HIGH = 1200
ENRICHMENT_TAG = "portfolio-enrichment"

EXCLUDED_PARTS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}

PROJECT_LIMITATION_NOTES = {
    "parkinsons_progression": (
        "This project is educational and non-clinical. The UCI telemonitoring data is historical, "
        "the cohort is limited, and model performance does not establish safety or clinical utility. "
        "A real clinical workflow would require external validation across sites and populations, "
        "prospective evaluation, governance, clinician oversight and regulatory review before any use in care."
    ),
}


def _tags(cell: dict) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [ENRICHMENT_TAG]},
        "source": text.splitlines(keepends=True),
    }


def _code(text: str, rel_path: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "tags": [ENRICHMENT_TAG, "source-mirror", "skip-execution"],
            "source_file": rel_path,
        },
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _meaningful_code_lines(cells: list[dict]) -> int:
    total = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for line in _source(cell).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def _discover_python(project_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_dir.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda p: (len(p.relative_to(project_dir).parts), str(p)))


def _already_mirrored(existing_code: str, source: str) -> bool:
    """Use a strong textual check so repeated workflow runs remain idempotent."""
    normalized = source.strip()
    if not normalized:
        return True
    return normalized in existing_code


def _has_limitations_story(cells: list[dict]) -> bool:
    text = "\n".join(_source(cell).lower() for cell in cells)
    terms = ("limitation", "caveat", "risk", "future work", "next step")
    return any(term in text for term in terms)


def _limitations_cell(project_dir: Path) -> dict:
    note = PROJECT_LIMITATION_NOTES.get(
        project_dir.name,
        (
            "This portfolio project is bounded by the documented dataset, validation design and retained "
            "evidence. Important limitations include dataset representativeness, possible distribution shift, "
            "measurement quality and the gap between offline evaluation and production use. Next steps should "
            "include stronger external or time-separated validation, monitoring, operational cost calibration "
            "and human review where the application can affect consequential decisions."
        ),
    )
    return _markdown(f"## Limitations and next steps\n\n{note}\n")


def enrich_notebook(project_dir: Path, notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    # Remove only content previously created by this enrichment pass. Original
    # coursework/project cells and the code-first builder cells are preserved.
    preserved = [cell for cell in cells if ENRICHMENT_TAG not in _tags(cell)]
    existing_code = "\n".join(
        _source(cell) for cell in preserved if cell.get("cell_type") == "code"
    )

    additions: list[dict] = []
    added_files: list[str] = []

    for path in _discover_python(project_dir):
        rel = str(path.relative_to(project_dir))
        source = path.read_text(encoding="utf-8")
        if _already_mirrored(existing_code, source):
            continue
        if not additions:
            additions.append(_markdown(
                "## Application engineering layer\n\n"
                "The original project above is intentionally preserved. The cells below expose additional "
                "canonical Python from this same project—pipelines, APIs, feature code, evaluation, tests, "
                "monitoring and other application logic—so the notebook works as a single recruiter-facing "
                "project while the modular files remain the production source of truth.\n"
            ))
        additions.append(_markdown(f"### Canonical source: `{rel}`\n"))
        additions.append(_code(source, rel))
        added_files.append(rel)
        existing_code += "\n" + source

    final_cells = preserved + additions

    # A complete portfolio story must surface limitations in the notebook itself,
    # not only in a README. Add a concise project-aware section when the original
    # notebook and mirrored source do not already contain one.
    if not _has_limitations_story(final_cells):
        final_cells.append(_limitations_cell(project_dir))

    code_lines = _meaningful_code_lines(final_cells)

    if TARGET_LOW <= code_lines <= TARGET_HIGH:
        depth = "within the substantial-project guide"
    elif code_lines > TARGET_HIGH:
        depth = "larger than the usual guide; retain the extra code when it is genuinely project-specific"
    else:
        depth = "compact and should only grow through substantive project-specific functionality, not filler"

    final_cells.append(_markdown(
        "## Portfolio depth check\n\n"
        f"**Meaningful code lines currently visible in this notebook:** {code_lines:,}. "
        f"A useful portfolio guide is roughly **{TARGET_IDEAL:,} meaningful lines** for a major notebook, "
        f"with project-specific depth often ranging from about {TARGET_LOW:,} to {TARGET_HIGH:,} lines. "
        f"This notebook is {depth}.\n\n"
        "Line count is not a quality metric by itself. Additional code should only be added when it strengthens "
        "the real application: data validation, cleaning, EDA, feature engineering, modelling, tuning, error "
        "analysis, explainability, inference, testing, monitoring, APIs, reproducibility or business logic.\n"
    ))

    notebook["cells"] = final_cells
    notebook.setdefault("metadata", {})["portfolio_depth"] = {
        "meaningful_code_lines": code_lines,
        "target_ideal": TARGET_IDEAL,
        "target_band": [TARGET_LOW, TARGET_HIGH],
        "original_cells_preserved": len(preserved),
        "canonical_python_files_added": added_files,
    }
    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "project": project_dir.name,
        "code_lines": code_lines,
        "added_files": added_files,
        "original_cells_preserved": len(preserved),
    }


def main() -> None:
    reports: list[dict[str, object]] = []
    for project_dir in sorted(p for p in PROJECTS_ROOT.iterdir() if p.is_dir()):
        notebook_path = project_dir / "project_notebook.ipynb"
        if not notebook_path.exists():
            continue
        reports.append(enrich_notebook(project_dir, notebook_path))

    print(json.dumps({
        "rule": "preserve original project + expose full application code",
        "ideal_meaningful_code_lines": TARGET_IDEAL,
        "projects": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
