"""Fail CI if a project notebook omits or shortens any current Python source file."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "projects"

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

OBVIOUS_PLACEHOLDERS = (
    "raise NotImplementedError",
    "TODO: implement",
    "TODO implement",
    "FIXME: implement",
)


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source or "")


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _python_files(project_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_dir.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def _looks_like_placeholder(source_text: str) -> bool:
    if any(marker in source_text for marker in OBVIOUS_PLACEHOLDERS):
        return True
    return bool(re.search(r"(?m)^\s*\.\.\.\s*(?:#.*)?$", source_text))


def main() -> int:
    failures: list[str] = []
    project_reports: list[tuple[str, int]] = []

    project_dirs = sorted(
        path for path in PROJECTS_ROOT.iterdir()
        if path.is_dir() and _python_files(path)
    )

    for project_dir in project_dirs:
        notebook_path = project_dir / "project_notebook.ipynb"
        if not notebook_path.is_file():
            failures.append(f"{project_dir.name}: missing project_notebook.ipynb")
            continue

        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{project_dir.name}: invalid notebook JSON: {exc}")
            continue

        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        matched = 0

        for python_path in _python_files(project_dir):
            rel_path = str(python_path.relative_to(project_dir)).replace("\\", "/")
            source_text = python_path.read_text(encoding="utf-8")
            wanted = _normalise(source_text)
            digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

            if _looks_like_placeholder(source_text) and "test" not in python_path.parts:
                failures.append(f"{project_dir.name}/{rel_path}: obvious placeholder remains in canonical Python")

            candidates = []
            for cell in code_cells:
                if _normalise(_source(cell)) != wanted:
                    continue
                metadata = cell.get("metadata", {})
                if metadata.get("source_file") == rel_path:
                    candidates.append(cell)

            if not candidates:
                failures.append(
                    f"{project_dir.name}: {rel_path} is not reproduced verbatim in a dedicated notebook code cell"
                )
                continue

            if not any(cell.get("metadata", {}).get("source_sha256") == digest for cell in candidates):
                failures.append(f"{project_dir.name}: {rel_path} notebook source hash metadata is stale/missing")
                continue

            matched += 1

        project_reports.append((project_dir.name, matched))

    for project, matched in project_reports:
        print(f"{project}: {matched} Python files mirrored verbatim")
    print(f"Projects checked: {len(project_dirs)}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")

    if failures:
        return 1

    print("Full Python notebook mirror validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
