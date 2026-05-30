"""Day 3 (capstone) figures: the 3-day journey, the project options, the build
workflow, the presentation rubric, and where to go next. figbase -> canvas-matched.
Output: slides/figures/d3_*.png

Run:  python slides/figures_day3.py
"""
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

from figbase import (plt, np, save, figtitle, txt_on, INK, MUTED,
                     TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET)


# --------------------------------------------------------------------------- #
# The 3-day journey
# --------------------------------------------------------------------------- #
def fig_journey():
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    days = [("DAY 1", "The ladder", "logreg -> ViT on eye scans;\nend-to-end deep learning", TURQUOISE),
            ("DAY 2", "Multimodal", "image + text + data -> TabPFN;\nfeatures + foundation model", DEEPPINK),
            ("DAY 3", "Your capstone", "pick a problem and build it,\nstart to finish", AMBER)]
    for i, (day, name, desc, c) in enumerate(days):
        x = 0.5 + i * 3.95
        ax.add_patch(FancyBboxPatch((x, 0.8), 3.4, 2.6, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(x + 1.7, 2.95, day, ha="center", fontsize=13, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x + 1.7, 2.45, name, ha="center", fontsize=16, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x + 1.7, 1.55, desc, ha="center", va="center", fontsize=11, color=tc)
        if i < 2:
            ax.annotate("", xy=(x + 3.95, 2.1), xytext=(x + 3.4, 2.1),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.5))
    figtitle(fig, "Three afternoons, one arc")
    save(fig, "d3_journey.png")


# --------------------------------------------------------------------------- #
# The three capstone options
# --------------------------------------------------------------------------- #
def fig_options():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    opts = [("PNEUMONIA", "chest X-rays", "is there pneumonia?\nbinary, like Day 2", TURQUOISE),
            ("SKIN LESIONS", "dermatology photos", "7 types, including\nmelanoma", DEEPPINK),
            ("CHOOSE YOUR OWN", "any MedMNIST set", "retina, blood, pathology,\norgans...", AMBER)]
    for i, (name, data, task, c) in enumerate(opts):
        x = 0.5 + i * 3.85
        ax.add_patch(FancyBboxPatch((x, 0.5), 3.5, 3.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(x + 1.75, 3.15, name, ha="center", fontsize=15, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x + 1.75, 2.45, data, ha="center", fontsize=12, color=tc, style="italic")
        ax.text(x + 1.75, 1.55, task, ha="center", va="center", fontsize=12, color=tc)
    figtitle(fig, "Pick one (or pitch your own)")
    fig.text(0.5, -0.02, "All three use MedMNIST: pip-installable, downloads in seconds, no account "
             "needed. So you spend the time building, not fighting data.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d3_options.png")


# --------------------------------------------------------------------------- #
# The build workflow loop
# --------------------------------------------------------------------------- #
def fig_workflow():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    steps = [("LOAD", "get the data,\nlook at it", TURQUOISE),
             ("BASELINE", "simplest model\nthat runs", AMBER),
             ("IMPROVE", "one change\nat a time", DEEPPINK),
             ("EVALUATE", "honest test,\nfind a failure", BLUEVIOLET)]
    cx = [1.7, 4.6, 7.5, 10.4]
    for i, (name, desc, c) in enumerate(steps):
        x = cx[i]
        ax.add_patch(FancyBboxPatch((x - 1.1, 1.6), 2.2, 1.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(x, 2.85, name, ha="center", fontsize=14, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x, 2.1, desc, ha="center", va="center", fontsize=10.5, color=tc)
        if i < 3:
            ax.annotate("", xy=(cx[i + 1] - 1.15, 2.5), xytext=(x + 1.15, 2.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
    # loop-back arrow from improve/evaluate to baseline
    ax.annotate("", xy=(4.6, 1.5), xytext=(10.4, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2, connectionstyle="arc3,rad=0.25"))
    ax.text(7.5, 0.7, "repeat: each loop, one honest improvement", ha="center", fontsize=11,
            color=MUTED, family="Geist Mono")
    figtitle(fig, "The build loop: baseline first, then improve")
    save(fig, "d3_workflow.png")


# --------------------------------------------------------------------------- #
# The presentation rubric (5 points)
# --------------------------------------------------------------------------- #
def fig_rubric():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    pts = [("1", "Built it", "runs end to end, you can show it"),
           ("2", "Evaluated honestly", "held-out test, the right metric"),
           ("3", "Found a failure", "one case it got wrong, and why"),
           ("4", "Defended a choice", "each partner names one decision"),
           ("5", "Both contributed", "each can answer about the other's part")]
    colors = [TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET, TURQUOISE]
    for i, ((num, head, desc), c) in enumerate(zip(pts, colors)):
        y = 4.3 - i * 0.85
        ax.add_patch(Circle((0.9, y), 0.30, color=c))
        ax.text(0.9, y, num, ha="center", va="center", fontsize=18, fontweight="bold",
                color=txt_on(c), family="Geist Mono")
        ax.text(1.5, y + 0.08, head, fontsize=15, fontweight="bold", color=INK, family="Geist Mono", va="center")
        ax.text(5.2, y, desc, fontsize=12, color=MUTED, va="center")
    figtitle(fig, "How your 3-minute talk is judged")
    fig.text(0.5, -0.02, "Not the highest accuracy. Honesty and understanding win: a model you "
             "can explain beats a better number you cannot.", ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d3_rubric.png")


# --------------------------------------------------------------------------- #
# Where to go next
# --------------------------------------------------------------------------- #
def fig_whats_next():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    cards = [("KEEP BUILDING", "Kaggle competitions,\nfast.ai, your own ideas", TURQUOISE),
             ("GO DEEPER", "Andrew Ng's courses,\nthe MICCAI community", DEEPPINK),
             ("THE FIELD IS REAL", "medical AI is hiring;\nthis is a career, not a hobby", AMBER)]
    for i, (name, desc, c) in enumerate(cards):
        x = 0.5 + i * 3.85
        ax.add_patch(FancyBboxPatch((x, 0.7), 3.5, 2.7, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(x + 1.75, 2.75, name, ha="center", fontsize=15, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x + 1.75, 1.8, desc, ha="center", va="center", fontsize=12, color=tc)
    figtitle(fig, "Where to go from here")
    save(fig, "d3_whats_next.png")


if __name__ == "__main__":
    fig_journey()
    fig_options()
    fig_workflow()
    fig_rubric()
    fig_whats_next()
    print("Day 3 figures done")
