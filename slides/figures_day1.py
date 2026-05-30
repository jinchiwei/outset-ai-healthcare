"""Generate brand-styled research figures for the Day 1 deck.

Uses the shared mpl_style helper (Geist fonts, brand palette) so figures match
the deck. Each conceptual figure carries an "(adapted from Author Year)"
attribution in its title. Output: slides/figures/*.png

Run:  python slides/figures_day1.py
"""
import sys
from pathlib import Path

# wire the shared matplotlib brand style
sys.path.insert(0, str(Path.home() / ".claude/skills/_shared"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

from mpl_style import apply_style, title, TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET, GOLD

apply_style()  # light mode, Geist
INK = "#14141C"
MUTED = "#555560"
PAPER = "#FAFAF7"

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 200


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def figtitle(fig, text, *, color=INK, y=1.04):
    fig.suptitle(text, fontsize=16, fontweight="bold", family="Geist Mono", color=color, y=y)


# --------------------------------------------------------------------------- #
# 1. Diabetic retinopathy screening: the deployment story
# --------------------------------------------------------------------------- #
def fig_dr_screening():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})

    # Left: stylized fundus with lesion markers
    axL.set_aspect("equal")
    axL.add_patch(plt.Circle((0.5, 0.5), 0.46, color="#3A1815"))      # retina
    axL.add_patch(plt.Circle((0.5, 0.5), 0.45, color="#C0633A"))      # fundus orange
    axL.add_patch(plt.Circle((0.68, 0.55), 0.09, color="#F2C879"))    # optic disc
    # vessels
    rng = np.random.RandomState(3)
    for ang in np.linspace(0, 2 * np.pi, 7, endpoint=False):
        x = 0.68 + np.cos(ang) * np.linspace(0, 0.34, 20)
        y = 0.55 + np.sin(ang) * np.linspace(0, 0.34, 20) + rng.randn(20) * 0.01
        axL.plot(x, y, color="#7A2418", lw=1.6, alpha=0.8)
    # lesions (hemorrhages / exudates) = the disease signal
    for _ in range(7):
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0.12, 0.4)
        axL.add_patch(plt.Circle((0.5 + np.cos(a) * r, 0.5 + np.sin(a) * r), 0.015,
                                 color="#6B0F0F"))
    for _ in range(5):
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0.12, 0.4)
        axL.add_patch(plt.Circle((0.5 + np.cos(a) * r, 0.5 + np.sin(a) * r), 0.013,
                                 color="#F4E03A"))
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
    axL.text(0.5, -0.04, "fundus photo\nred = hemorrhages, yellow = exudates",
             ha="center", va="top", fontsize=10, color=MUTED)

    # Right: the screening funnel / why it matters
    axR.axis("off")
    stages = [
        ("463M", "adults with diabetes\nworldwide", TURQUOISE),
        ("1 in 3", "develop diabetic\nretinopathy", AMBER),
        ("Preventable", "blindness if caught\nand referred early", DEEPPINK),
    ]
    for i, (big, small, c) in enumerate(stages):
        y = 0.78 - i * 0.32
        axR.add_patch(FancyBboxPatch((0.04, y - 0.11), 0.92, 0.24,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     facecolor=c, edgecolor="none"))
        tc = INK if c == AMBER else "white"
        axR.text(0.12, y, big, fontsize=26, fontweight="bold", va="center",
                 ha="left", color=tc, family="Geist Mono")
        axR.text(0.46, y, small, fontsize=12, va="center", ha="left", color=tc)
    axR.set_xlim(0, 1); axR.set_ylim(0, 1)

    figtitle(fig, "Diabetic retinopathy: a screening problem AI was built for", color=INK)
    fig.text(0.5, -0.02, "Automated DR screening was among the first medical AI deployed at scale "
             "(adapted from Gulshan et al. 2016, JAMA)",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
    save(fig, "intro_dr_screening.png")


# --------------------------------------------------------------------------- #
# 2. An image is a grid of numbers; color = three grids
# --------------------------------------------------------------------------- #
def fig_image_as_numbers():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    # tiny patch of an image
    rng = np.random.RandomState(7)
    patch = rng.randint(0, 256, (6, 6))
    ax = axes[0]
    ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
    for (i, j), v in np.ndenumerate(patch):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                color="white" if v < 128 else "black", family="Geist Mono")
    ax.set_xticks([]); ax.set_yticks([])
    title(ax, "Grayscale = one grid", color=INK)

    # RGB channels stacked
    ax = axes[1]
    ax.axis("off")
    for k, (c, lab) in enumerate([(DEEPPINK, "R"), (TURQUOISE, "G"), (BLUEVIOLET, "B")]):
        off = k * 0.16
        ax.add_patch(Rectangle((0.18 + off, 0.30 - off), 0.5, 0.5,
                               facecolor=c, edgecolor="white", lw=2, alpha=0.85))
        ax.text(0.18 + off + 0.46, 0.30 - off + 0.46, lab, fontsize=15,
                fontweight="bold", color="white", family="Geist Mono")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)
    title(ax, "Color = three grids", color=INK)

    # the model just sees the array
    ax = axes[2]
    ax.axis("off")
    ax.text(0.5, 0.62, "(224, 224, 3)", fontsize=22, fontweight="bold",
            ha="center", color=BLUEVIOLET, family="Geist Mono")
    ax.text(0.5, 0.42, "height x width x channels", fontsize=12, ha="center", color=MUTED)
    ax.text(0.5, 0.22, "150,528 numbers per image", fontsize=12, ha="center", color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    title(ax, "What the model sees", color=INK)

    fig.suptitle("An image is just numbers", fontsize=17, fontweight="bold",
                 family="Geist Mono", color=INK, y=1.04)
    save(fig, "concept_image_numbers.png")


# --------------------------------------------------------------------------- #
# 3. CNN: a filter slides across the image -> feature map
# --------------------------------------------------------------------------- #
def fig_cnn():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)

    # input grid
    def grid(x0, y0, n, cell, color, hl=None):
        for i in range(n):
            for j in range(n):
                fc = color if (hl and (i, j) in hl) else "white"
                ax.add_patch(Rectangle((x0 + j * cell, y0 + (n - 1 - i) * cell), cell, cell,
                                       facecolor=fc, edgecolor="#BBB", lw=0.8))
    grid(0.4, 0.7, 7, 0.5, TURQUOISE, hl={(i, j) for i in range(3) for j in range(3)})
    ax.text(2.15, 0.2, "input image", ha="center", fontsize=11, color=MUTED)

    # filter
    ax.add_patch(FancyBboxPatch((4.4, 2.0), 1.3, 1.3, boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=DEEPPINK, edgecolor="none"))
    ax.text(5.05, 2.65, "3x3\nfilter", ha="center", va="center", fontsize=11,
            color="white", fontweight="bold", family="Geist Mono")
    ax.annotate("", xy=(4.3, 2.65), xytext=(3.9, 2.65),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))

    # feature map
    grid(6.6, 1.1, 5, 0.5, AMBER, hl={(0, 0)})
    ax.text(7.85, 0.6, "feature map", ha="center", fontsize=11, color=MUTED)
    ax.annotate("", xy=(6.5, 2.65), xytext=(5.9, 2.65),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))

    ax.text(11.4, 2.65, "slide the\nsame filter\neverywhere", ha="center", va="center",
            fontsize=11, color=BLUEVIOLET, family="Geist Mono")

    figtitle(fig, "A CNN slides small filters across the image (adapted from LeCun et al. 1998)",
          color=INK)
    fig.text(0.5, -0.02, "The same filter scans the whole image, so the model learns to find a "
             "shape anywhere it appears. That is what 'seeing structure' means.",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
    save(fig, "concept_cnn.png")


# --------------------------------------------------------------------------- #
# 4. ResNet residual block
# --------------------------------------------------------------------------- #
def fig_resnet():
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    def block(x, label, color):
        ax.add_patch(FancyBboxPatch((x, 1.4), 1.5, 1.2,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + 0.75, 2.0, label, ha="center", va="center", fontsize=11,
                color="white", fontweight="bold", family="Geist Mono")

    ax.text(0.4, 2.0, "x", fontsize=18, ha="center", va="center", color=INK,
            family="Geist Mono", fontweight="bold")
    block(1.3, "conv\n+ReLU", TURQUOISE)
    block(3.6, "conv", DEEPPINK)
    # plus node
    ax.add_patch(plt.Circle((6.4, 2.0), 0.3, facecolor=AMBER, edgecolor="none"))
    ax.text(6.4, 2.0, "+", fontsize=20, ha="center", va="center", color=INK, fontweight="bold")
    ax.text(8.7, 2.0, "output", fontsize=12, ha="center", va="center", color=INK,
            family="Geist Mono")

    for x0, x1 in [(0.7, 1.3), (2.8, 3.6), (5.1, 6.1), (6.7, 8.0)]:
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    # skip connection
    ax.annotate("", xy=(6.4, 1.7), xytext=(0.4, 0.6),
                arrowprops=dict(arrowstyle="-|>", color=BLUEVIOLET, lw=2.5,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(3.4, 0.5, "skip connection: carry the input forward", fontsize=11,
            color=BLUEVIOLET, family="Geist Mono")

    figtitle(fig, "ResNet adds a shortcut so very deep networks still train "
          "(adapted from He et al. 2016)", color=INK)
    save(fig, "concept_resnet.png")


# --------------------------------------------------------------------------- #
# 5. Vision Transformer: patches -> tokens -> attention
# --------------------------------------------------------------------------- #
def fig_vit():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)

    # image split into patches
    n = 4
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((0.4 + j * 0.6, 0.9 + (n - 1 - i) * 0.6), 0.58, 0.58,
                                   facecolor=TURQUOISE, edgecolor="white", lw=1.5, alpha=0.85))
    ax.text(1.6, 0.5, "split into patches", ha="center", fontsize=11, color=MUTED)

    ax.annotate("", xy=(3.5, 2.1), xytext=(3.0, 2.1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))

    # patch embeddings as token vectors
    for k in range(5):
        x = 3.8 + k * 0.5
        ax.add_patch(Rectangle((x, 1.4), 0.34, 1.4, facecolor=DEEPPINK, edgecolor="white", lw=1))
    ax.text(5.0, 1.0, "patch embeddings\n(vectors)", ha="center", fontsize=11, color=MUTED)

    ax.annotate("", xy=(7.0, 2.1), xytext=(6.5, 2.1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))

    # attention block
    ax.add_patch(FancyBboxPatch((7.2, 1.3), 2.0, 1.6,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=BLUEVIOLET, edgecolor="none"))
    ax.text(8.2, 2.1, "attention\n(which patches\nmatter?)", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", family="Geist Mono")

    ax.annotate("", xy=(10.0, 2.1), xytext=(9.4, 2.1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.add_patch(FancyBboxPatch((10.1, 1.6), 1.7, 1.0,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=AMBER, edgecolor="none"))
    ax.text(10.95, 2.1, "refer?", ha="center", va="center", fontsize=13,
            color=INK, fontweight="bold", family="Geist Mono")

    figtitle(fig, "A Vision Transformer treats image patches like words "
          "(adapted from Dosovitskiy et al. 2021)", color=INK)
    save(fig, "concept_vit.png")


# --------------------------------------------------------------------------- #
# 6. The bridge: patches <-> words, same attention machinery
# --------------------------------------------------------------------------- #
def fig_bridge():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 4.6))

    def row(ax, items, color, embed_label, out_label):
        ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 2)
        for k, it in enumerate(items):
            x = 0.3 + k * 1.15
            ax.add_patch(FancyBboxPatch((x, 0.7), 1.0, 0.8,
                                        boxstyle="round,pad=0.02,rounding_size=0.06",
                                        facecolor=color, edgecolor="white", lw=1.2))
            tc = INK if color == AMBER else "white"
            ax.text(x + 0.5, 1.1, it, ha="center", va="center", fontsize=12,
                    color=tc, fontweight="bold", family="Geist Mono")
        ax.annotate("", xy=(6.4, 1.1), xytext=(5.9, 1.1),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
        ax.add_patch(FancyBboxPatch((6.6, 0.6), 2.0, 1.0,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=BLUEVIOLET, edgecolor="none"))
        ax.text(7.6, 1.1, "attention", ha="center", va="center", fontsize=12,
                color="white", fontweight="bold", family="Geist Mono")
        ax.annotate("", xy=(9.6, 1.1), xytext=(8.8, 1.1),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
        ax.text(10.8, 1.1, out_label, ha="center", va="center", fontsize=13,
                color=INK, fontweight="bold", family="Geist Mono")
        ax.text(0.3, 0.2, embed_label, fontsize=10, color=MUTED)

    row(ax1, ["[img]", "[img]", "[img]", "[img]"], TURQUOISE, "image patches", "DR grade")
    row(ax2, ["the", "eye", "is", "..."], AMBER, "words", "next word")

    ax1.text(11.7, 1.1, "ViT", fontsize=12, color=TURQUOISE, fontweight="bold",
             family="Geist Mono", ha="left", va="center")
    ax2.text(11.7, 1.1, "LLM", fontsize=12, color=DEEPPINK, fontweight="bold",
             family="Geist Mono", ha="left", va="center")

    fig.suptitle("Same machinery, different input: that bottom row is a language model",
                 fontsize=15, fontweight="bold", family="Geist Mono", color=INK, y=1.0)
    save(fig, "concept_bridge.png")


# --------------------------------------------------------------------------- #
# 7. The leaderboard we actually built
# --------------------------------------------------------------------------- #
def fig_ladder_results():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    models = ["logistic\nregression", "MLP", "CNN\n(scratch)", "ResNet50\n(transfer)", "ViT-Base\n(transfer)"]
    accs = [0.76, 0.77, 0.79, 0.87, 0.85]
    colors = [MUTED, MUTED, TURQUOISE, DEEPPINK, BLUEVIOLET]
    bars = ax.bar(range(len(models)), accs, color=colors, width=0.62)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.012, f"{a:.0%}",
                ha="center", fontsize=14, fontweight="bold", family="Geist Mono", color=INK)
    ax.axhline(0.5, ls="--", color=MUTED, lw=1)
    ax.text(4.4, 0.515, "coin flip", fontsize=10, color=MUTED, ha="right", va="bottom")
    # call out the transfer-learning jump in open space below the tall bars
    ax.text(1.5, 0.63, "transfer learning\nis the jump", ha="center", fontsize=12,
            color="#B8860B", fontweight="bold", family="Geist Mono")
    ax.annotate("", xy=(2.78, 0.84), xytext=(1.9, 0.66),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=3,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.4, 0.95); ax.set_ylabel("validation accuracy")
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["50%", "60%", "70%", "80%", "90%"])
    title(ax, "Same data, five models: the ladder you build in the lab", color=INK)
    save(fig, "result_ladder.png")


if __name__ == "__main__":
    fig_dr_screening()
    fig_image_as_numbers()
    fig_cnn()
    fig_resnet()
    fig_vit()
    fig_bridge()
    fig_ladder_results()
    print("all D1 figures written to", OUT)
