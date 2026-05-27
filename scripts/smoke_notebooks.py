"""Execute course notebooks headless and fail on cell errors.

Runs:
  - every `*_solution.ipynb` (the filled-in solutions carry smoke coverage)
  - every capstone `starter.ipynb` (working baselines)

Skips the TODO-blank lab notebooks (`day1.ipynb`, `day2.ipynb`): a notebook is
skipped if a sibling `<name>_solution.ipynb` exists, since the solution covers it.
Also skips the _shared warmup notebook (it pulls large weights; not part of CI).
"""
from __future__ import annotations
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SKIP_NAMES = {"pretrain_warmup.ipynb"}


def should_run(p: Path) -> bool:
    if ".ipynb_checkpoints" in p.parts:
        return False
    if p.name in SKIP_NAMES:
        return False
    # Skip a blank lab notebook when its solution exists alongside.
    if not p.stem.endswith("_solution"):
        if (p.parent / f"{p.stem}_solution.ipynb").exists():
            return False
    return True


def run_one(path: Path) -> tuple[bool, str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    try:
        client.execute()
    except Exception as e:  # noqa: BLE001 - report any cell failure
        return False, f"{path.relative_to(ROOT)}: {type(e).__name__}: {e}"
    return True, f"{path.relative_to(ROOT)}: ok"


def main() -> int:
    nbs = sorted(p for p in ROOT.glob("notebooks/**/*.ipynb") if should_run(p))
    if not nbs:
        print("no notebooks to smoke test")
        return 0
    fails = 0
    for nb_path in nbs:
        ok, msg = run_one(nb_path)
        print(("OK   " if ok else "FAIL "), msg)
        fails += 0 if ok else 1
    print(f"\n{len(nbs) - fails}/{len(nbs)} passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
