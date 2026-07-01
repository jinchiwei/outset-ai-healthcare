"""Bespoke illustrations for the facilitator handout deck: one visual per
interactive/unplugged activity, showing what the setup actually looks like.
Brand-styled via figbase. Output: slides/figures/fac_*.png

Run:  python slides/figures_facilitator.py
"""
import textwrap

from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch

from figbase import (plt, np, save, figtitle, txt_on, INK, MUTED,
                     TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET)

PAPER = "#FBFAF6"
LINE = "#E3E0D6"


# --------------------------------------------------------------------------- #
# small reusable primitives
# --------------------------------------------------------------------------- #
def _person(ax, x, y, s=1.0, color=INK):
    ax.add_patch(Circle((x, y + 0.42 * s), 0.13 * s, color=color, zorder=5))
    ax.plot([x, x], [y + 0.29 * s, y - 0.12 * s], color=color, lw=2.2 * s, zorder=5)
    ax.plot([x - 0.16 * s, x + 0.16 * s], [y + 0.16 * s, y + 0.16 * s], color=color, lw=2.2 * s, zorder=5)
    ax.plot([x, x - 0.13 * s], [y - 0.12 * s, y - 0.4 * s], color=color, lw=2.2 * s, zorder=5)
    ax.plot([x, x + 0.13 * s], [y - 0.12 * s, y - 0.4 * s], color=color, lw=2.2 * s, zorder=5)


def _card(ax, x, y, w, h, color, text, fs=10, tc=None, mono=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=color, edgecolor="none", zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or txt_on(color), family="Geist Mono" if mono else "Geist",
            fontweight="bold", zorder=5)


def _whiteboard(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor="white", edgecolor="#BBB", lw=2.5, zorder=1))


# --------------------------------------------------------------------------- #
# WIDGETS
# --------------------------------------------------------------------------- #
def fig_widget_augment():
    fig, (axU, axI) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    # slider panel
    axU.axis("off"); axU.set_xlim(0, 1); axU.set_ylim(0, 1)
    axU.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.04",
                                 facecolor=PAPER, edgecolor=LINE, lw=1.3))
    axU.text(0.5, 0.88, "the sliders (Colab / Jupyter)", ha="center", fontsize=11,
             color=INK, family="Geist Mono")
    knobs = [("rotate", 0.6, TURQUOISE), ("brightness", 0.35, AMBER), ("blur", 0.15, DEEPPINK),
             ("zoom", 0.5, BLUEVIOLET), ("h-flip", 0.8, TURQUOISE)]
    for i, (name, val, c) in enumerate(knobs):
        y = 0.72 - i * 0.13
        axU.text(0.10, y, name, fontsize=10, color=INK, va="center", family="Geist Mono")
        axU.plot([0.42, 0.92], [y, y], color=LINE, lw=3, solid_capstyle="round")
        axU.add_patch(Circle((0.42 + val * 0.5, y), 0.026, color=c, zorder=6))
    # image that responds
    axI.axis("off"); axI.set_xlim(0, 1); axI.set_ylim(0, 1)
    rng = np.random.RandomState(3)
    axI.add_patch(Circle((0.5, 0.52), 0.36, color="#7a1f10", zorder=2))         # fundus
    axI.add_patch(Circle((0.5, 0.52), 0.36, fill=False, ec="#3a0f08", lw=6, zorder=3))
    for _ in range(6):
        axI.add_patch(Circle((0.5 + rng.uniform(-0.25, 0.25), 0.52 + rng.uniform(-0.25, 0.25)),
                             rng.uniform(0.01, 0.03), color="#F2C14E", zorder=4))
    axI.annotate("drag a slider ->\nthis updates live", (0.14, 0.52), xytext=(-0.02, 0.9),
                 fontsize=10, family="Geist Mono", color=INK,
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    figtitle(fig, "Augmentation explorer: drag, and the eye changes instantly")
    fig.text(0.5, -0.02, "Day 1, notebook 0.7. No training -- pure instant feedback. "
             "Ask: which of these still looks like a real patient photo?",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_widget_augment.png")


def fig_widget_threshold():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rng = np.random.RandomState(0)
    healthy = np.clip(rng.beta(2, 5, 400), 0, 1)
    sick = np.clip(rng.beta(5, 2, 260), 0, 1)
    ax.hist(healthy, bins=22, alpha=0.7, color=TURQUOISE, label="healthy")
    ax.hist(sick, bins=22, alpha=0.7, color=DEEPPINK, label="has disease")
    thr = 0.5
    ax.axvline(thr, color=INK, lw=3, ls="--")
    ax.annotate("drag me", (thr, ax.get_ylim()[1] * 0.85), xytext=(thr + 0.16, ax.get_ylim()[1] * 0.9),
                fontsize=11, family="Geist Mono", color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(0.02, ax.get_ylim()[1] * 0.6, "everything left\n= 'not referable'", fontsize=9,
            color=MUTED, family="Geist Mono")
    ax.text(0.72, ax.get_ylim()[1] * 0.6, "everything right\n= 'referable'", fontsize=9,
            color=MUTED, family="Geist Mono")
    ax.set_xlabel("model's predicted probability of disease"); ax.set_yticks([]); ax.legend(loc="upper center")
    figtitle(fig, "Threshold slider: watch sensitivity vs specificity trade off")
    fig.text(0.5, -0.03, "Day 1, notebook 7.1. Slide the cutoff: lower it to catch more disease "
             "(more false alarms), raise it for the reverse. The ROC curve, in their hands.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_widget_threshold.png")


# --------------------------------------------------------------------------- #
# UNPLUGGED
# --------------------------------------------------------------------------- #
def fig_feature_ladder():
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    _whiteboard(ax, 0.3, 0.3, 11.4, 5.4)
    rungs = [("raw pixel numbers", TURQUOISE), ("average redness", TURQUOISE),
             ("count of dark spots", AMBER), ("bleeding near center?", AMBER),
             ("hand-drawn vessel map", DEEPPINK), ("borrowed CNN brain", BLUEVIOLET)]
    for i, (label, c) in enumerate(rungs):
        x = 1.1 + i * 1.75
        y = 0.9 + i * 0.62
        ax.plot([x - 0.2, x + 1.5], [y - 0.15, y - 0.15], color="#DDD", lw=2)  # step
        _card(ax, x, y, 1.7, 0.72, c, textwrap.fill(label, 12), fs=8.5)
    ax.annotate("more complex ->", (10.5, 1.2), xytext=(6.0, 0.7), fontsize=11,
                family="Geist Mono", color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8))
    ax.text(10.5, 5.0, "star your\nbest bet", ha="center", fontsize=10, color=DEEPPINK, family="Geist Mono")
    ax.text(10.55, 4.4, "*", ha="center", fontsize=34, color=AMBER, fontweight="bold")
    figtitle(fig, "The Feature Ladder: rank the magnets, bet on the winner")
    fig.text(0.5, -0.02, "Teams rank these ways-of-seeing-the-eye and bet which is most accurate. "
             "The reveal: it IS logreg->CNN->ResNet, and the borrowed brain wins.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_feature_ladder.png")


def fig_human_confusion():
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    quads = [(0, 5, TURQUOISE, "TRUE POSITIVE", "sick, called sick", 3),
             (5, 5, DEEPPINK, "FALSE POSITIVE", "healthy, called sick", 2),
             (0, 0, AMBER, "FALSE NEGATIVE", "sick, called healthy", 1),
             (5, 0, "#8AA0A8", "TRUE NEGATIVE", "healthy, called healthy", 3)]
    for x, y, c, name, sub, n in quads:
        ax.add_patch(Rectangle((x + 0.15, y + 0.15), 4.7, 4.7, facecolor=c, alpha=0.18,
                               edgecolor=c, lw=2.2))
        ax.text(x + 2.5, y + 4.4, name, ha="center", fontsize=11, fontweight="bold",
                color=INK, family="Geist Mono")
        ax.text(x + 2.5, y + 3.95, sub, ha="center", fontsize=8.5, color=MUTED, style="italic")
        for k in range(n):
            _person(ax, x + 1.2 + k * 1.1, y + 1.7, s=0.9, color=c if c != "#8AA0A8" else MUTED)
    ax.text(5, 9.4, "walk to your corner", ha="center", fontsize=12, color=INK, family="Geist Mono", fontweight="bold")
    figtitle(fig, "Human Confusion Matrix: four corners, real bodies")
    fig.text(0.5, -0.02, "A 'model' student guesses; everyone walks to their corner. Count bodies to get "
             "sensitivity & specificity. Then make the model cautious and re-sort.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_human_confusion.png")


def fig_marble_run():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    x = np.linspace(0.6, 8.5, 200)
    y = 0.11 * (x - 5.2) ** 2 + 0.7
    ax.plot(x, y, color=BLUEVIOLET, lw=6, solid_capstyle="round")
    # marble rolling down
    for i, xx in enumerate([1.1, 2.2, 3.4, 4.4, 5.15]):
        yy = 0.11 * (xx - 5.2) ** 2 + 0.7
        c = DEEPPINK
        a = 0.3 + 0.7 * (i / 4)
        ax.add_patch(Circle((xx, yy + 0.28), 0.24, color=c, alpha=a, zorder=5))
    ax.add_patch(Circle((1.1, 0.11 * (1.1 - 5.2) ** 2 + 0.7 + 0.28), 0.28, fill=False, ec=INK, lw=1.5, zorder=6))
    ax.annotate("start:\nbad model", (1.1, 3.1), xytext=(0.4, 5.0), fontsize=10, family="Geist Mono",
                color=INK, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.annotate("settles:\nleast error", (5.2, 0.98), xytext=(6.2, 2.4), fontsize=10, family="Geist Mono",
                color=INK, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    # too-steep inset: marble flies off
    ax.plot([9.2, 11.4], [4.6, 1.2], color=BLUEVIOLET, lw=5, solid_capstyle="round")
    ax.add_patch(Circle((11.7, 2.1), 0.22, color=DEEPPINK, zorder=5))
    ax.annotate("", (12.0, 3.0), xytext=(11.4, 1.4), arrowprops=dict(arrowstyle="-|>", color=DEEPPINK, lw=2))
    ax.text(10.3, 5.1, "tilt too steep =\nlearning rate too big", ha="center", fontsize=9,
            color=DEEPPINK, family="Geist Mono")
    figtitle(fig, "Gradient-Descent Marble Run: roll it to the valley")
    fig.text(0.5, -0.02, "Roll a marble down a curved track to the lowest point (least error). Tilt = the "
             "learning rate; too steep and it flies off instead of settling.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_marble_run.png")


def fig_overfitting():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax in (axA, axB):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # team A: memorize
    axA.text(0.5, 0.93, "TEAM A: memorize", ha="center", fontsize=12, fontweight="bold",
             color=DEEPPINK, family="Geist Mono")
    for i in range(3):
        _card(axA, 0.12 + i * 0.26, 0.55, 0.22, 0.26, "#ECEAE3", f"eye {i+1}\n= refer", fs=8, tc=INK)
    axA.text(0.5, 0.4, "learns the exact answers", ha="center", fontsize=9.5, color=MUTED, style="italic")
    axA.text(0.5, 0.2, "NEW eye ->  X", ha="center", fontsize=15, color=DEEPPINK, family="Geist Mono", fontweight="bold")
    # team B: rule
    axB.text(0.5, 0.93, "TEAM B: a rule", ha="center", fontsize=12, fontweight="bold",
             color=TURQUOISE, family="Geist Mono")
    _card(axB, 0.2, 0.55, 0.6, 0.26, TURQUOISE, "more dark spots\n-> refer", fs=10)
    axB.text(0.5, 0.4, "learns a general pattern", ha="center", fontsize=9.5, color=MUTED, style="italic")
    axB.text(0.5, 0.2, "NEW eye ->  check", ha="center", fontsize=15, color=TURQUOISE, family="Geist Mono", fontweight="bold")
    figtitle(fig, "Overfitting Showdown: memorize vs learn a rule")
    fig.text(0.5, -0.02, "Team A memorizes 10 eyes; Team B agrees a rule. Quiz both on NEW eyes. "
             "Team A face-plants -- overfitting, felt not lectured.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_overfitting.png")


def fig_word_tiles():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    tiles = [("The", TURQUOISE), ("heart", AMBER), ("is", TURQUOISE),
             ("en", DEEPPINK), ("larged", DEEPPINK), ("?", "#DDD")]
    x = 0.7
    for i, (t, c) in enumerate(tiles):
        w = 1.2 + 0.18 * len(t)
        tc = INK if c == "#DDD" else txt_on(c)
        _card(ax, x, 1.7, w, 1.1, c, t, fs=15, tc=tc)
        if t == "larged":
            ax.plot([x - 0.05, x - 0.05], [1.6, 2.9], color=INK, lw=1.5, ls=":")
        x += w + 0.28
    ax.annotate("one word, two tokens", (5.9, 1.6), xytext=(4.2, 0.7), fontsize=10,
                family="Geist Mono", color=DEEPPINK,
                arrowprops=dict(arrowstyle="-|>", color=DEEPPINK, lw=1.6))
    ax.text(x - 0.9, 3.4, "cover it: guess the next tile", ha="right", fontsize=10, color=INK, family="Geist Mono")
    figtitle(fig, "Word-Piece Tiles: text becomes tokens you can hold")
    fig.text(0.5, -0.02, "Chop a report into tiles; rare words split ('en'+'larged'). Then cover the last "
             "tile and guess it -- that guess IS what a language model does.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_word_tiles.png")


def fig_hallucination():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    _person(ax, 1.6, 2.0, s=1.8, color=BLUEVIOLET)
    ax.text(1.6, 0.5, "the confident\nnarrator", ha="center", fontsize=10, color=INK, family="Geist Mono")
    # speech bubble
    ax.add_patch(FancyBboxPatch((3.3, 1.8), 6.2, 2.6, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor=PAPER, edgecolor=DEEPPINK, lw=2.5))
    ax.add_patch(Polygon([(3.35, 2.9), (2.7, 2.5), (3.35, 2.4)], closed=True, facecolor=PAPER,
                         edgecolor=DEEPPINK, lw=2.5))
    ax.text(6.4, 3.7, '"Absolutely! The capital of', ha="center", fontsize=12, color=INK, style="italic")
    ax.text(6.4, 3.25, 'that province is Zendovia,', ha="center", fontsize=12, color=INK, style="italic")
    ax.text(6.4, 2.8, 'established in 1743."', ha="center", fontsize=12, color=INK, style="italic")
    ax.text(6.4, 2.25, "(none of this is real)", ha="center", fontsize=10, color=DEEPPINK, family="Geist Mono")
    ax.text(6.4, 4.75, "rule: never allowed to say 'I don't know'", ha="center", fontsize=10.5,
            color=INK, family="Geist Mono", fontweight="bold")
    figtitle(fig, "The Hallucination Game: fluent, confident, wrong")
    fig.text(0.5, -0.02, "A volunteer must answer every obscure question confidently. They start inventing -- "
             "exactly what an LLM does when it doesn't know. Use it, but verify.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_hallucination.png")


def fig_spot_cheat():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    cols = ["age", "sex", "smoker", "final_dx", "LABEL"]
    rows = [["71", "M", "yes", "disease", "1"], ["54", "F", "no", "healthy", "0"],
            ["63", "M", "no", "disease", "1"], ["48", "F", "yes", "healthy", "0"]]
    x0, y0, cw, rh = 1.0, 1.0, 2.0, 0.8
    for j, c in enumerate(cols):
        _card(ax, x0 + j * cw, y0 + 4 * rh, cw - 0.1, rh, INK, c, fs=10, tc="white")
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (3 - i) * rh), cw - 0.1, rh,
                                   facecolor=PAPER, edgecolor=LINE, lw=1))
            ax.text(x0 + j * cw + (cw - 0.1) / 2, y0 + (3 - i) * rh + rh / 2, v,
                    ha="center", va="center", fontsize=10, color=INK, family="Geist Mono")
    # circle the leaked column
    lx = x0 + 3 * cw - 0.06
    ax.add_patch(FancyBboxPatch((lx, y0 - 0.1), cw, 5 * rh + 0.2, boxstyle="round,pad=0.03,rounding_size=0.1",
                                fill=False, edgecolor=DEEPPINK, lw=3.5, zorder=6))
    ax.annotate("this IS the answer", (lx + cw / 2, y0 + 5 * rh + 0.1), xytext=(lx - 1.2, y0 + 5 * rh + 0.9),
                fontsize=11, color=DEEPPINK, family="Geist Mono", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=DEEPPINK, lw=2))
    figtitle(fig, "Spot-the-Cheat: race to find the leak")
    fig.text(0.5, -0.02, "One printed column is basically the label. First pair to circle it and explain why "
             "it's cheating wins. That's target leakage -- the day's key skill.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_spot_cheat.png")


def fig_regulator_cards():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    _whiteboard(ax, 0.3, 0.3, 11.4, 5.4)
    ax.text(6, 5.2, "rank them: top 3, then how to enforce each", ha="center", fontsize=11,
            color=INK, family="Geist Mono")
    prios = [("Safety", TURQUOISE), ("Fairness", DEEPPINK), ("Transparency", AMBER),
             ("Evidence", BLUEVIOLET), ("Privacy", TURQUOISE), ("Monitoring", DEEPPINK),
             ("Accountability", AMBER)]
    for i, (name, c) in enumerate(prios):
        col, row = i % 4, i // 4
        _card(ax, 1.0 + col * 2.6, 3.1 - row * 1.5, 2.3, 1.0, c, name, fs=10)
    ax.annotate("", (3.2, 2.6), xytext=(3.2, 3.1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    figtitle(fig, "Regulator Card-Sort: you make the rules")
    fig.text(0.5, -0.02, "In pairs, physically rank what a medical AI must prove, then name one way to enforce "
             "each. No answer key -- real US/EU/Asia regulators disagree.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_regulator_cards.png")


def fig_timing_map():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    days = [("DAY 1", TURQUOISE, ["Feature Ladder (open)", "Marble Run (training)",
                                  "Overfitting Showdown", "2 live widgets (eval)"]),
            ("DAY 2", DEEPPINK, ["Word-Piece Tiles", "Hallucination Game", "Spot-the-Cheat"]),
            ("DAY 3", AMBER, ["Regulator Card-Sort", "(before picking projects)"])]
    for i, (name, c, acts) in enumerate(days):
        x = 0.4 + i * 3.95
        ax.add_patch(FancyBboxPatch((x, 0.5), 3.6, 5.0, boxstyle="round,pad=0.03,rounding_size=0.05",
                                    facecolor=PAPER, edgecolor=LINE, lw=1.3))
        ax.add_patch(FancyBboxPatch((x, 4.75), 3.6, 0.75, boxstyle="round,pad=0.03,rounding_size=0.05",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + 1.8, 5.12, name, ha="center", va="center", fontsize=13, fontweight="bold",
                color=txt_on(c), family="Geist Mono")
        for k, a in enumerate(acts):
            ax.add_patch(Circle((x + 0.35, 4.2 - k * 0.75), 0.08, color=c))
            ax.text(x + 0.55, 4.2 - k * 0.75, a, fontsize=9.5, color=INK, va="center")
    figtitle(fig, "When to reach for what")
    fig.text(0.5, -0.02, "Two or three activities per day is plenty. Mix to taste and to your room's energy.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "fac_timing_map.png")


if __name__ == "__main__":
    fig_widget_augment()
    fig_widget_threshold()
    fig_feature_ladder()
    fig_human_confusion()
    fig_marble_run()
    fig_overfitting()
    fig_word_tiles()
    fig_hallucination()
    fig_spot_cheat()
    fig_regulator_cards()
    fig_timing_map()
    print("facilitator figures done")
