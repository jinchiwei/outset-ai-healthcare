"""G6 intro/background figures -- bespoke redraws of the estrogen-&-cognition literature (attributed).

  effect_flip       -- the single most important slide: just WATCHING women said estrogen protects
                       the brain; a COIN-FLIP trial said the opposite (adapted from Tang 1996 /
                       Zandi 2002 observational vs. Shumaker 2003 WHIMS randomized trial).
  confounding_triangle -- healthy-user bias in one picture: a hidden "already healthier / wealthier /
                       more active" factor feeds BOTH taking hormones AND better memory, so the
                       bottom link only LOOKS like cause (concept; ref Wharton 2009 / Vandenbroucke 2009).
  objective_vs_subjective -- the tests say "fine," the patient says "brain fog": a slope plot where
                       objective scores land in the normal band but self-reported fog is high
                       (adapted from Maki 2024, Harvard Health).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
np = sf.np
plt = sf.plt


def effect_flip():
    """Two studies, same drug, opposite answer. Observational risk points DOWN (looks protective),
    randomized-trial risk points UP (harm). We plot relative dementia risk vs. a risk = 1.0 line."""
    # Illustrative relative-risk values that match the DIRECTION each literature reported.
    labels = ["Just watching\n(observational)", "Coin-flip trial\n(randomized, WHIMS)"]
    rel_risk = [0.66, 2.05]          # <1 = looks protective (Tang/Zandi); >1 = harm (Shumaker 2003)
    colors = [sf.TURQUOISE, sf.DEEPPINK]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(labels, rel_risk, color=colors, edgecolor=sf.INK, linewidth=0.6, width=0.55)
    ax.axhline(1.0, color=sf.INK, lw=1.0, ls="--")                       # risk = 1.0 -> no effect
    ax.text(1.46, 1.02, "no effect", fontsize=9, color=sf.MUTED, va="bottom", ha="right")
    for b, v in zip(bars, rel_risk):                                    # print each value on its bar
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}x",
                ha="center", fontsize=11, family="Geist Mono", color=sf.INK)
    # A big curved arrow showing the sign flip from the left bar up to the right bar.
    ax.add_patch(FancyArrowPatch((0.15, 0.72), (0.85, 1.95), connectionstyle="arc3,rad=-0.35",
                                 arrowstyle="-|>", mutation_scale=22, color=sf.AMBER, lw=2.2))
    ax.text(0.46, 1.18, "same drug,\nopposite answer", ha="center", fontsize=10,
            family="Geist Mono", color="#C79A10")
    ax.text(0, 0.30, "hormones look\nPROTECTIVE", ha="center", fontsize=9, color="#0B6B63")
    ax.text(1, 0.30, "hormones look\nHARMFUL", ha="center", fontsize=9, color="#9B1060")
    ax.set_ylabel("dementia risk vs. non-users  (1.0 = no effect)")
    ax.set_ylim(0, 2.4)
    sf.title(ax, "When they finally flipped a coin, the benefit vanished")
    ax.text(0.5, -0.30, "adapted from Tang 1996 / Zandi 2002 (observational) vs. Shumaker 2003 WHIMS (randomized)",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED, transform=ax.transAxes)
    sf.save(fig, HERE, "intro_effect_flip")
    sf.save_raw(sf.pd.DataFrame({"study": ["observational", "randomized_trial"],
                                 "relative_risk": rel_risk}), HERE, "intro_effect_flip")


def confounding_triangle():
    """Healthy-user bias as a 3-node diagram. A hidden top factor points down to BOTH boxes;
    the bottom link is dashed and labeled 'looks like cause -- but isn't'."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    def box(cx, cy, w, h, text, face, edge, tcolor):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.08,rounding_size=0.18",
                                    facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                family="Geist Mono", color=tcolor)

    # Hidden common cause at the top (the villain).
    box(5.0, 5.8, 6.4, 1.35, "ALREADY HEALTHIER / WEALTHIER / MORE ACTIVE\n(the hidden third factor)",
        "#FDEBF4", sf.DEEPPINK, "#9B1060")
    # Two observed boxes at the bottom.
    box(2.3, 1.4, 3.6, 1.2, "Takes hormone\ntherapy", "#DCF6F2", sf.TURQUOISE, "#0B6B63")
    box(7.7, 1.4, 3.6, 1.2, "Better memory\nscore", "#DCF6F2", sf.TURQUOISE, "#0B6B63")

    # Arrows from the hidden cause DOWN to each bottom box (these are the REAL drivers).
    ax.add_patch(FancyArrowPatch((3.9, 5.15), (2.5, 2.05), arrowstyle="-|>",
                                 mutation_scale=20, color=sf.DEEPPINK, lw=2.0))
    ax.add_patch(FancyArrowPatch((6.1, 5.15), (7.5, 2.05), arrowstyle="-|>",
                                 mutation_scale=20, color=sf.DEEPPINK, lw=2.0))
    ax.text(2.7, 3.5, "really\ncauses", fontsize=9, color=sf.DEEPPINK, ha="center")
    ax.text(7.3, 3.5, "really\ncauses", fontsize=9, color=sf.DEEPPINK, ha="center")

    # The tempting but false bottom link.
    ax.add_patch(FancyArrowPatch((4.15, 1.4), (5.85, 1.4), arrowstyle="-|>", ls="dashed",
                                 mutation_scale=18, color=sf.MUTED, lw=1.6))
    ax.text(5.0, 1.95, "looks like cause -- but isn't", ha="center", fontsize=9.5,
            family="Geist Mono", color=sf.MUTED)

    ax.text(5.0, 6.85, "Healthy-user bias: a hidden factor fakes the link", ha="center",
            fontsize=13, family="Geist Mono", color=sf.INK)
    ax.text(5.0, 0.25, "concept -- adapted from Wharton 2009 / Vandenbroucke 2009",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_confounding_triangle")


def objective_vs_subjective():
    """Slope plot: objective test score (mostly 'normal' band) on the left, self-reported brain-fog
    severity on the right. Lines cross -- the tests say fine, the women say something's wrong."""
    rng = np.random.default_rng(0)                                   # fixed seed -> reproducible
    n = 14
    objective = rng.normal(0.72, 0.10, n).clip(0.45, 0.95)          # most land in the 'normal' band
    subjective = rng.normal(0.62, 0.16, n).clip(0.15, 0.95)        # self-reported fog: high & scattered

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xL, xR = 0.0, 1.0
    for o, s in zip(objective, subjective):                         # one connecting line per woman
        ax.plot([xL, xR], [o, s], color=sf.BLUEVIOLET, alpha=0.35, lw=1.3, zorder=1)
    ax.scatter([xL] * n, objective, s=55, color=sf.TURQUOISE, edgecolor=sf.INK,
               linewidth=0.5, zorder=3, label="objective test")
    ax.scatter([xR] * n, subjective, s=55, color=sf.DEEPPINK, edgecolor=sf.INK,
               linewidth=0.5, zorder=3, label="self-reported fog")
    # Shade the 'normal' band on the objective side.
    ax.axhspan(0.6, 1.0, xmin=0.0, xmax=0.12, color=sf.TURQUOISE, alpha=0.12, zorder=0)
    ax.text(xL, 1.02, "objective\ncognitive test", ha="center", fontsize=10,
            family="Geist Mono", color="#0B6B63")
    ax.text(xR, 1.02, "self-reported\n'brain fog'", ha="center", fontsize=10,
            family="Geist Mono", color="#9B1060")
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([])
    ax.set_ylabel("severity  (higher = worse fog / better test)")
    ax.annotate("tests say 'fine'", xy=(xL, 0.80), xytext=(-0.30, 0.30), fontsize=9,
                color=sf.MUTED, arrowprops=dict(arrowstyle="->", color=sf.MUTED, lw=1))
    ax.annotate("she says 'not fine'", xy=(xR, 0.72), xytext=(1.02, 0.22), fontsize=9,
                color=sf.MUTED, arrowprops=dict(arrowstyle="->", color=sf.MUTED, lw=1))
    sf.title(ax, "The tests can say 'normal' while the patient feels foggy")
    ax.text(0.5, -0.12, "adapted from Maki 2024, Harvard Health (menopause & brain fog)",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED, transform=ax.transAxes)
    sf.save(fig, HERE, "intro_objective_vs_subjective")
    sf.save_raw(sf.pd.DataFrame({"objective_score": objective, "subjective_fog": subjective}),
                HERE, "intro_objective_vs_subjective")


if __name__ == "__main__":
    effect_flip()
    confounding_triangle()
    objective_vs_subjective()
    print("G6 intro figures done")
