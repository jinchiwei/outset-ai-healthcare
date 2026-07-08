"""Helpers for programmatic notebook construction.

The course notebooks are generated from Python source (one builder per day) so the
lab notebook (with `# TODO` blanks) and the solution notebook stay in sync from a
single source of truth. Never hand-edit two notebooks.
"""
from __future__ import annotations
from pathlib import Path

import nbformat as nbf


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def new_nb() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    })


def save(nb: nbf.NotebookNode, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    return path


def code_with_todos(solution_src: str, blanks: list[tuple[str, str]], mark_solution: bool = True):
    """Build a (solution_cell, lab_cell) pair from one source.

    `solution_src` is the complete, runnable code. `blanks` is a list of
    (line_substring, hint) tuples. In the LAB cell, the first line containing
    each `line_substring` is replaced with `# TODO: <hint>` (indentation kept).

    In the SOLUTION cell (when `mark_solution`), that same line keeps its code but
    gets a trailing marker so it's obvious which lines were the fill-in TODOs:
    `<code>   # <== TODO (you fill this in): <hint>`.

    Returns (solution_cell, lab_cell), both code cells.
    """
    orig = solution_src.strip("\n").splitlines()
    lab, sol = list(orig), list(orig)
    used = set()
    for sub, hint in blanks:
        for i, line in enumerate(orig):
            if i in used:
                continue
            if sub in line:
                indent = line[: len(line) - len(line.lstrip())]
                lab[i] = f"{indent}# TODO: {hint}"
                if mark_solution:
                    sol[i] = f"{line}   # <== TODO (you fill this in): {hint}"
                used.add(i)
                break
        else:
            raise ValueError(f"code_with_todos: no line contains {sub!r}")
    return code("\n".join(sol)), code("\n".join(lab))
