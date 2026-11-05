"""Build each Markdown notebook's .ipynb from its .md source (jupytext).

The .md file is the source: edit it, never the .ipynb. The .ipynb is what gets
run and published; `jupytext --update` rewrites its inputs from the .md and
keeps the outputs and cell ids already in it. (Not `--sync`: it takes whichever
side is newer, so running the .ipynb would make it overwrite the .md.)

Usage:
    uv run tools/build_notebooks.py            # every notebook under maps/ and chart/
    uv run tools/build_notebooks.py maps/2026/day04/clusters.md
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ("maps", "chart")


def is_notebook_source(path: Path) -> bool:
    """A Markdown file is a notebook source if its header has a jupytext block."""
    head = path.read_text(encoding="utf-8")[:2000]
    return head.startswith("---") and "jupytext:" in head


def main() -> int:
    if len(sys.argv) > 1:
        sources = [Path(arg).resolve() for arg in sys.argv[1:]]
    else:
        sources = sorted(p for folder in FOLDERS for p in (ROOT / folder).rglob("*.md"))
    sources = [p for p in sources if is_notebook_source(p)]
    if not sources:
        print("No jupytext Markdown notebooks found.")
        return 0
    for source in sources:
        print(f"Building {source.relative_to(ROOT)} -> .ipynb")
        subprocess.run(["jupytext", "--update", "--to", "ipynb", str(source)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
