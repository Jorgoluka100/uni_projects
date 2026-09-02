"""Enrich every recruiter notebook without deleting the original project work.

The portfolio rule is deliberately simple:

    one project -> one substantial notebook -> one complete recruiter story

Existing notebook cells are preserved.  Missing canonical Python modules from the
same project directory are appended as an application-engineering layer so a
recruiter can inspect the original analysis AND the production-style code in one
place.

The ~1,000-line target is a depth target, never a padding target.  This script
reports code depth but does not manufacture filler just to hit a number.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "projects"
TARGET_LOW = 800
TARGET_IDEAL = 1000
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


def enrich_notebook(project_dir: Path, notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    # Remove only content previously created by this enrichment pass.  Original
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
    code_lines = _meaningful_code_lines(final_cells)

    if TARGET_LOW <= code_lines <= TARGET_HIGH:
        depth = "in the preferred substantial-project band"
    elif code_lines > TARGET_HIGH:
        depth = "above the preferred band; retain it when the extra code is genuinely project-specific"
    else:
        depth = "below the preferred band and should gain substantive project-specific depth rather than filler"

    final_cells.append(_markdown(
        "## Portfolio depth check\n\n"
        f"**Meaningful code lines currently visible in this notebook:** {code_lines:,}.  "
        f"The portfolio aims for roughly **{TARGET_IDEAL:,} meaningful lines** per major project "
        f"(normally about {TARGET_LOW:,}–{TARGET_HIGH:,}), and this notebook is {depth}.\n\n"
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
