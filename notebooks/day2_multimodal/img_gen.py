"""Author-time concept diagrams for the Day 2 notebook.

Rendered with the in-repo nbfig brand helper (no external dependency); commit the PNGs
so Colab gets them with the repo clone. Live, data-driven plots are done in-notebook.

    python notebooks/day2_multimodal/img_gen.py
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


def multimodal_stack():
    fig, ax = nbfig.fig(figsize=(9.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4.4); ax.grid(False)
    rows = [("IMAGE", "image model -> img_pred", nbfig.TURQUOISE),
            ("TEXT", "LLM -> yes/no findings", nbfig.DEEPPINK),
            ("DEMOGRAPHICS", "age, sex, smoker", nbfig.AMBER)]
    for i, (name, desc, c) in enumerate(rows):
        y = 3.3 - i * 1.15
        ax.add_patch(FancyBboxPatch((0.4, y - 0.42), 5.2, 0.9,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=c, edgecolor="none"))
        tc = nbfig.txt_on(c)
        ax.text(0.7, y + 0.12, name, fontsize=12, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(0.7, y - 0.22, desc, fontsize=10, color=tc)
        ax.annotate("", xy=(7.4, 2.05), xytext=(5.7, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
    ax.add_patch(FancyBboxPatch((7.5, 1.4), 2.2, 1.3, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#FFFFFF", edgecolor="#CCC", lw=1.5))
    ax.text(8.6, 2.05, "one\ntable", ha="center", va="center", fontsize=13, fontweight="bold",
            color=INK, family="Geist Mono")
    ax.annotate("", xy=(10.4, 2.05), xytext=(9.8, 2.05), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    ax.add_patch(FancyBboxPatch((10.5, 1.4), 2.2, 1.3, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=nbfig.BLUEVIOLET, edgecolor="none"))
    ax.text(11.6, 2.05, "TabPFN", ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", family="Geist Mono")
    nbfig.title(fig, "Late fusion: every signal becomes a column, then one model decides")
    _save(fig, "multimodal_stack.png")


def leakage():
    fig, ax = nbfig.fig(figsize=(9.5, 3.2))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.grid(False)
    ax.add_patch(FancyBboxPatch((0.4, 1.3), 6.6, 2.0, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor="#FFFFFF", edgecolor="#DDD", lw=1.5))
    ax.text(0.8, 2.85, "the report says:", fontsize=11, color=MUTED, family="Geist Mono")
    ax.text(0.8, 2.35, '"IMPRESSION: Cardiomegaly."', fontsize=15, color=INK, fontweight="bold", va="top")
    ax.text(0.8, 1.7, "...which is exactly what we're predicting.", fontsize=11.5, color=nbfig.DEEPPINK, va="top")
    ax.add_patch(FancyBboxPatch((7.3, 1.5), 4.3, 1.6, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=nbfig.DEEPPINK, edgecolor="none"))
    ax.text(9.45, 2.55, "corr ~ 0.96", ha="center", fontsize=20, fontweight="bold",
            color="white", family="Geist Mono")
    ax.text(9.45, 1.95, "the text feature\nIS the answer", ha="center", va="center", fontsize=11, color="white")
    nbfig.title(fig, "Target leakage: a feature that already knows the label")
    _save(fig, "leakage.png")


if __name__ == "__main__":
    multimodal_stack()
    leakage()
    print("day2 concept diagrams ->", OUT)
