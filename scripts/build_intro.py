"""Build the Day 0 intro notebook (lab + solution).

Dead-simple, accessible primer. Branded plots come from nbfig helpers so the student's
code stays a one-liner. Each idea is a worked EXAMPLE, then YOUR TURN on new points.

Currently: TASK 1 only (draw a line to separate two clusters). More sections to follow.

Run:  python scripts/build_intro.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, md, new_nb, save  # noqa: E402

LAB, SOL = [], []
def both(cell): LAB.append(cell); SOL.append(cell)
def pair(lab_cell, sol_cell): LAB.append(lab_cell); SOL.append(sol_cell)


# --------------------------------------------------------------------------- #
both(md("""
# Machine learning starts with a line

Here are **blue** points and **pink** points. A huge amount of machine learning is just this:
**find a line that separates the groups.** You write the line as `y = m*x + b` -- pick the slope
`m` and the height `b`. A helper draws it and counts how many points land on the right side.
"""))

both(code("""
# grab the course helpers (branded plots). On Colab this clones the repo; locally it's a no-op.
import os, sys
if not os.path.exists("../_shared/nbfig.py"):
    os.system("git clone -q https://github.com/jinchiwei/outset-ai-healthcare.git")
    os.chdir("outset-ai-healthcare/notebooks/day0_intro")
sys.path.insert(0, "../_shared")
import nbfig
nbfig.use()
"""))

both(md("""
## Example

Two clusters are made for you. The line below is `y = 1*x + 0`. Run it and see how it does.
"""))

both(code("""
blue, pink = nbfig.two_clusters()      # a blue cluster + a pink cluster

m, b = 1, 0                            # your line:  y = m*x + b
nbfig.show_line(blue, pink, m, b)      # draws the line and scores it
"""))

both(md("""
## Your turn

These groups are stacked **top and bottom**, so `y = x` won't work. Change `m` and `b` (just two
numbers) until the line sits between the colors and separates all the points.
"""))

pair(
    code("""
blue, pink = nbfig.two_clusters(blue_center=(3, 1), pink_center=(3, 6))

m, b = 1, 0        # your turn: change these two numbers so the line separates blue from pink
nbfig.show_line(blue, pink, m, b)
"""),
    code("""
blue, pink = nbfig.two_clusters(blue_center=(3, 1), pink_center=(3, 6))

m, b = 0, 3.5      # a flat line at height 3.5 splits bottom (blue) from top (pink)
nbfig.show_line(blue, pink, m, b)
"""),
)


def build():
    out = ROOT / "notebooks/day0_intro"
    out.mkdir(parents=True, exist_ok=True)
    lab = new_nb(); lab.cells = LAB; save(lab, out / "intro.ipynb")
    sol = new_nb(); sol.cells = SOL; save(sol, out / "intro_solution.ipynb")
    print(f"wrote intro.ipynb ({len(LAB)} cells) + intro_solution.ipynb")


if __name__ == "__main__":
    build()
