"""Validate the recruiter-facing portfolio manifest.

The manifest is deliberately simple: it is a single source of truth for the
university notebooks and production-style projects that should remain discoverable.
This check prevents broken local paths, duplicate project IDs and accidental removal
of university work from the public index.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "portfolio_manifest.json"
README = ROOT / "README.md"
UNIVERSITY_INDEX = ROOT / "docs" / "UNIVERSITY_PROJECTS.md"


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def require_path(relative_path: str, failures: list[str]) -> None:
    path = ROOT / relative_path
    if not path.exists():
        fail(f"missing manifest path: {relative_path}", failures)


def main() -> int:
    failures: list[str] = []

    if not MANIFEST.is_file():
        fail("portfolio_manifest.json is missing", failures)
        return 1

    try:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"portfolio_manifest.json is invalid JSON: {exc}", failures)
        return 1

    university = payload.get("university_projects")
    production = payload.get("production_projects")
    if not isinstance(university, list) or not university:
        fail("university_projects must be a non-empty list", failures)
        university = []
    if not isinstance(production, list) or not production:
        fail("production_projects must be a non-empty list", failures)
        production = []

    all_ids: list[str] = []
    for project in university + production:
        project_id = project.get("id") if isinstance(project, dict) else None
        if not isinstance(project_id, str) or not project_id.strip():
            fail(f"project entry has no valid id: {project!r}", failures)
        else:
            all_ids.append(project_id)

    duplicates = sorted({item for item in all_ids if all_ids.count(item) > 1})
    if duplicates:
        fail(f"duplicate project ids: {', '.join(duplicates)}", failures)

    readme_text = README.read_text(encoding="utf-8") if README.is_file() else ""
    university_text = (
        UNIVERSITY_INDEX.read_text(encoding="utf-8")
        if UNIVERSITY_INDEX.is_file()
        else ""
    )

    if not UNIVERSITY_INDEX.is_file():
        fail("docs/UNIVERSITY_PROJECTS.md is missing", failures)

    for project in university:
        if not isinstance(project, dict):
            fail(f"invalid university project entry: {project!r}", failures)
            continue
        notebook = project.get("notebook")
        if not isinstance(notebook, str) or not notebook:
            fail(f"university project {project.get('id')} has no notebook", failures)
            continue
        require_path(notebook, failures)
        if notebook not in university_text:
            fail(f"university index does not reference {notebook}", failures)

        follow_on = project.get("follow_on")
        if follow_on is not None:
            if not isinstance(follow_on, str) or not follow_on:
                fail(f"invalid follow_on path for {project.get('id')}", failures)
            else:
                require_path(follow_on, failures)

    for project in production:
        if not isinstance(project, dict):
            fail(f"invalid production project entry: {project!r}", failures)
            continue
        path = project.get("path")
        if not isinstance(path, str) or not path:
            fail(f"production project {project.get('id')} has no path", failures)
            continue
        require_path(path, failures)
        if f"{path}/" not in readme_text and f"{path})" not in readme_text:
            print(f"INFO: production project {project.get('id')} is catalogued but not a root README flagship")

    if "docs/UNIVERSITY_PROJECTS.md" not in readme_text:
        fail("root README does not link to docs/UNIVERSITY_PROJECTS.md", failures)

    print(f"University projects checked: {len(university)}")
    print(f"Production projects checked: {len(production)}")
    print(f"Failures: {len(failures)}")

    if failures:
        return 1

    print("Portfolio manifest validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
