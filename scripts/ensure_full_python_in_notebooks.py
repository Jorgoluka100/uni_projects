"""Ensure every recruiter-facing project notebook contains the complete Python source.

The portfolio uses modular .py files for engineering quality, but recruiters should not
have to leave project_notebook.ipynb to inspect the implementation. This script makes
that guarantee deterministic:

- every current project .py file is represented by one dedicated notebook code cell;
- the cell content is verbatim current source, not a shortened excerpt;
- source path and SHA-256 are stored in cell metadata for validation;
- previously generated full-source cells are replaced idempotently on every sync.

Original/university notebook cells are not deleted. Existing exact source cells are
reused and annotated where possible; otherwise a generated engineering appendix is
added.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "projects"
MIRROR_TAG = "full-python-mirror"

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


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source or "")


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _tags(cell: dict) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [MIRROR_TAG]},
        "source": text.splitlines(keepends=True),
    }


def _code(text: str, rel_path: str) -> dict:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "tags": [MIRROR_TAG, "source-mirror", "skip-execution"],
            "source_file": rel_path,
            "source_sha256": digest,
        },
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _python_files(project_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_dir.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda p: (len(p.relative_to(project_dir).parts), str(p.relative_to(project_dir))))


def _exact_source_cell(cells: list[dict], source_text: str) -> int | None:
    wanted = _normalise(source_text)
    if not wanted:
        return None
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if _normalise(_source(cell)) == wanted:
            return index
    return None


def _annotate_existing_cell(cell: dict, rel_path: str, source_text: str) -> None:
    metadata = cell.setdefault("metadata", {})
    tags = list(metadata.get("tags", []))
    for tag in ("source-mirror", "skip-execution"):
        if tag not in tags:
            tags.append(tag)
    metadata["tags"] = tags
    metadata["source_file"] = rel_path
    metadata["source_sha256"] = hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def ensure_project(project_dir: Path) -> dict[str, object] | None:
    notebook_path = project_dir / "project_notebook.ipynb"
    if not notebook_path.is_file():
        return None

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    original_cells = notebook.get("cells", [])

    # Remove only cells created by this script. Everything else is preserved.
    cells = [cell for cell in original_cells if MIRROR_TAG not in _tags(cell)]
    python_files = _python_files(project_dir)

    reused: list[str] = []
    added: list[str] = []
    appendix: list[dict] = []

    for path in python_files:
        rel_path = str(path.relative_to(project_dir)).replace("\\", "/")
        source_text = path.read_text(encoding="utf-8")
        existing_index = _exact_source_cell(cells, source_text)
        if existing_index is not None:
            _annotate_existing_cell(cells[existing_index], rel_path, source_text)
            reused.append(rel_path)
            continue

        if not appendix:
            appendix.append(
                _markdown(
                    "# Complete Python source appendix\n\n"
                    "Every current `.py` file in this project is reproduced verbatim below. "
                    "These cells are source mirrors for recruiter inspection; the modular files "
                    "remain the canonical executable implementation.\n"
                )
            )
        appendix.append(_markdown(f"## Full source — `{rel_path}`\n"))
        appendix.append(_code(source_text, rel_path))
        added.append(rel_path)

    cells.extend(appendix)
    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["full_python_source_audit"] = {
        "python_files": len(python_files),
        "reused_exact_source_cells": reused,
        "generated_exact_source_cells": added,
        "guarantee": "each current project .py file appears verbatim in one dedicated notebook code cell",
    }

    rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    before = notebook_path.read_text(encoding="utf-8")
    changed = rendered != before
    if changed:
        notebook_path.write_text(rendered, encoding="utf-8")

    return {
        "project": project_dir.name,
        "python_files": len(python_files),
        "reused": reused,
        "added": added,
        "changed": changed,
    }


def main() -> None:
    reports: list[dict[str, object]] = []
    for project_dir in sorted(path for path in PROJECTS_ROOT.iterdir() if path.is_dir()):
        report = ensure_project(project_dir)
        if report is not None:
            reports.append(report)

    print(json.dumps({
        "projects_checked": len(reports),
        "projects": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
