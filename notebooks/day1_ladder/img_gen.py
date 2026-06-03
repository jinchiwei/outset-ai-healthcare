"""Author-time concept diagrams for the Day 1 notebook.

The static, MIT-style concept images embedded in the notebook (the live plots are
generated in-notebook). Rendered with the in-repo nbfig brand helper so there's no
external dependency; commit the PNGs so Colab gets them with the repo clone.

    python notebooks/day1_ladder/img_gen.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import nbfig  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

nbfig.use()
OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)
INK, MUTED = nbfig.INK, nbfig.MUTED


def _save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=nbfig.CANVAS)
    print("wrote", OUT / name)


def transfer_learning():
    fig, ax = nbfig.fig(figsize=(9, 3.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.grid(False)
    ax.add_patch(FancyBboxPatch((0.5, 1.0), 6.5, 2.4, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=nbfig.BLUEVIOLET, edgecolor="none"))
    ax.text(3.75, 2.85, "pretrained backbone", ha="center", fontsize=14, fontweight="bold",
            color="white", family="Geist Mono")
    ax.text(3.75, 2.25, "trained on a million everyday photos\n(ImageNet): already knows edges,\ntextures, shapes",
            ha="center", va="center", fontsize=10.5, color="white")
    ax.text(3.75, 1.25, "FROZEN  -- we do not retrain it", ha="center", fontsize=10,
            color="white", family="Geist Mono", style="italic")
    ax.annotate("", xy=(8.3, 2.2), xytext=(7.1, 2.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    ax.add_patch(FancyBboxPatch((8.4, 1.3), 3.1, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=nbfig.TURQUOISE, edgecolor="none"))
    ax.text(9.95, 2.6, "new head", ha="center", fontsize=13, fontweight="bold",
            color=nbfig.txt_on(nbfig.TURQUOISE), family="Geist Mono")
    ax.text(9.95, 1.95, "just this part learns\n'referable?'", ha="center", va="center",
            fontsize=10.5, color=nbfig.txt_on(nbfig.TURQUOISE))
    nbfig.title(fig, "Transfer learning: reuse the vision, train a tiny head")
    _save(fig, "transfer_learning.png")


def vit_patches():
    import numpy as np
    fig, ax = nbfig.fig(figsize=(9.5, 3.2))
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.grid(False)
    rng = np.random.RandomState(2)
    for r in range(3):
        for c in range(3):
            shade = 0.45 + 0.5 * rng.rand()
            ax.add_patch(FancyBboxPatch((0.5 + c * 0.7, 0.9 + r * 0.7), 0.62, 0.62,
                                        boxstyle="round,pad=0.01,rounding_size=0.02",
                                        facecolor=(shade, shade, shade), edgecolor="white", lw=1.5))
    ax.text(1.55, 0.45, "image -> patches", ha="center", fontsize=10.5, color=MUTED)
    ax.annotate("", xy=(4.4, 2.0), xytext=(3.0, 2.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    ax.text(3.7, 2.35, "embed each", ha="center", fontsize=9.5, color=MUTED, family="Geist Mono")
    ax.add_patch(FancyBboxPatch((4.6, 1.1), 3.6, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=nbfig.DEEPPINK, edgecolor="none"))
    ax.text(6.4, 2.45, "attention", ha="center", fontsize=14, fontweight="bold",
            color="white", family="Geist Mono")
    ax.text(6.4, 1.75, "which patches matter\nfor this eye?", ha="center", va="center",
            fontsize=10.5, color="white")
    ax.annotate("", xy=(9.6, 2.0), xytext=(8.3, 2.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    ax.add_patch(FancyBboxPatch((9.7, 1.3), 2.9, 1.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=nbfig.AMBER, edgecolor="none"))
    ax.text(11.15, 2.0, "referable?", ha="center", va="center", fontsize=13, fontweight="bold",
            color=nbfig.txt_on(nbfig.AMBER), family="Geist Mono")
    nbfig.title(fig, "A Vision Transformer: patches that pay attention")
    _save(fig, "vit_patches.png")


if __name__ == "__main__":
    transfer_learning()
    vit_patches()
    print("day1 concept diagrams ->", OUT)
