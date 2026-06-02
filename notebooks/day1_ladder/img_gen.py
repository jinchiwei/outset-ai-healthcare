"""Author-time concept diagrams for the Day 1 notebook, via the build-figure skill.

These are the static, MIT-style concept images embedded in the notebook (the live
plots are generated in-notebook with nbfig). Run on a machine where the build-figure
skill is installed; commit the PNGs so Colab gets them with the repo clone.

    python notebooks/day1_ladder/img_gen.py
"""
import sys
from pathlib import Path

# the build-figure skill (brandfig) lives in the installed skills dir
sys.path.insert(0, str(Path.home() / ".claude/skills/build-figure"))
import brandfig as bf
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

bf.use("bone")
OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)
INK = bf.ink()


def transfer_learning():
    fig, ax = bf.fig(figsize=(9, 3.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    # frozen backbone (big), reused
    ax.add_patch(FancyBboxPatch((0.5, 1.0), 6.5, 2.4, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=bf.BLUEVIOLET, edgecolor="none"))
    ax.text(3.75, 2.85, "pretrained backbone", ha="center", fontsize=14, fontweight="bold",
            color="white", family="Geist Mono")
    ax.text(3.75, 2.25, "trained on a million everyday photos\n(ImageNet): already knows edges,\ntextures, shapes",
            ha="center", va="center", fontsize=10.5, color="white")
    ax.text(3.75, 1.25, "FROZEN  -- we do not retrain it", ha="center", fontsize=10,
            color="white", family="Geist Mono", style="italic")
    ax.annotate("", xy=(8.3, 2.2), xytext=(7.1, 2.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    # new head (small), trained
    ax.add_patch(FancyBboxPatch((8.4, 1.3), 3.1, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=bf.TURQUOISE, edgecolor="none"))
    ax.text(9.95, 2.6, "new head", ha="center", fontsize=13, fontweight="bold",
            color=bf.txt_on(bf.TURQUOISE), family="Geist Mono")
    ax.text(9.95, 1.95, "just this part learns\n'referable?'", ha="center", va="center",
            fontsize=10.5, color=bf.txt_on(bf.TURQUOISE))
    bf.figtitle(fig, "Transfer learning: reuse the vision, train a tiny head")
    bf.save(fig, OUT / "transfer_learning.png")


def vit_patches():
    fig, ax = bf.fig(figsize=(9.5, 3.2))
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4)
    # a 3x3 grid of patches
    import numpy as np
    rng = np.random.RandomState(2)
    for r in range(3):
        for c in range(3):
            shade = 0.45 + 0.5 * rng.rand()
            ax.add_patch(FancyBboxPatch((0.5 + c * 0.7, 0.9 + r * 0.7), 0.62, 0.62,
                                        boxstyle="round,pad=0.01,rounding_size=0.02",
                                        facecolor=(shade, shade, shade), edgecolor="white", lw=1.5))
    ax.text(1.55, 0.45, "image -> patches", ha="center", fontsize=10.5, color=bf.muted())
    ax.annotate("", xy=(4.4, 2.0), xytext=(3.0, 2.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    ax.text(3.7, 2.35, "embed each", ha="center", fontsize=9.5, color=bf.muted(), family="Geist Mono")
    # attention box
    ax.add_patch(FancyBboxPatch((4.6, 1.1), 3.6, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=bf.DEEPPINK, edgecolor="none"))
    ax.text(6.4, 2.45, "attention", ha="center", fontsize=14, fontweight="bold",
            color="white", family="Geist Mono")
    ax.text(6.4, 1.75, "which patches matter\nfor this eye?", ha="center", va="center",
            fontsize=10.5, color="white")
    ax.annotate("", xy=(9.6, 2.0), xytext=(8.3, 2.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    ax.add_patch(FancyBboxPatch((9.7, 1.3), 2.9, 1.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=bf.AMBER, edgecolor="none"))
    ax.text(11.15, 2.0, "referable?", ha="center", va="center", fontsize=13, fontweight="bold",
            color=bf.txt_on(bf.AMBER), family="Geist Mono")
    bf.figtitle(fig, "A Vision Transformer: patches that pay attention")
    bf.save(fig, OUT / "vit_patches.png")


if __name__ == "__main__":
    transfer_learning()
    vit_patches()
    print("day1 concept diagrams ->", OUT)
