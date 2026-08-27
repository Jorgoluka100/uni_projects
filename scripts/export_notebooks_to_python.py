"""Export Jupyter notebook code cells to plain Python without changing notebooks.

The exporter preserves the repository's original .ipynb files. Each output script
contains notebook code cells in order, separated by VS Code/Jupyter-style `# %%`
markers. Markdown is kept as short comments so the resulting file remains readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return ""


def markdown_as_comments(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        lines.append(f"# {stripped}" if stripped else "#")
    return "\n".join(lines)


def notebook_to_python(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"Unsupported notebook format in {notebook_path}")

    chunks = [
        f'"""Python export of {notebook_path.name}.\nGenerated from notebook cells; original .ipynb remains the source artifact.\n"""',
        "",
    ]

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        text = source_text(cell).rstrip()
        if not text:
            continue
        if cell_type == "code":
            chunks.extend([f"# %% [code cell {index}]", text, ""])
        elif cell_type == "markdown":
            chunks.extend([f"# %% [markdown cell {index}]", markdown_as_comments(text), ""])

    return "\n".join(chunks).rstrip() + "\n"


def discover(root: Path) -> list[Path]:
    notebooks: list[Path] = []
    for path in root.rglob("*.ipynb"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        notebooks.append(path)
    return sorted(notebooks)


def export_all(root: Path, output: Path) -> int:
    root = root.resolve()
    output = output.resolve()
    notebooks = discover(root)
    count = 0

    for notebook in notebooks:
        relative = notebook.relative_to(root)
        destination = output / relative.with_suffix(".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(notebook_to_python(notebook), encoding="utf-8")
        count += 1
        print(f"exported: {relative} -> {destination.relative_to(output)}")

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Export all repository notebooks to Python scripts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("notebook_python_exports"))
    args = parser.parse_args()

    count = export_all(args.root, args.output)
    print(f"Notebook Python exports created: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
