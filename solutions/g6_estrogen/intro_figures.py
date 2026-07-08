"""G6 intro/background figures -- bespoke redraws that VISUALLY depict the clinical problem of
estrogen, cognition, and a symptom a test can miss (all attributed). Theme "bone" via the shared
solfig helper, so every PNG drops straight onto a slide canvas.

  doctor_office          -- the clinical scene: a woman in the exam room describing "brain fog" while
                            the objective cognitive test scores her in the NORMAL band. The test says
                            "fine," she says "not fine." (concept -- Maki 2024 / Chen & Shafir 2025)
  effect_flip            -- the single most important slide: just WATCHING women said estrogen protects
                            the brain; a COIN-FLIP trial said the opposite (adapted from Tang 1996 /
                            Zandi 2002 observational vs. Shumaker 2003 WHIMS randomized trial).
  confounding_triangle   -- healthy-user bias in one picture: a hidden "already healthier / wealthier /
                            more active" factor feeds BOTH taking hormones AND better memory, so the
                            bottom link only LOOKS like cause (concept; Wharton 2009 / Vandenbroucke 2009).
  objective_vs_subjective-- the tests say "fine," the patient says "brain fog": a slope plot where
                            objective scores land in the normal band but self-reported fog is high
                            (adapted from Maki 2024, Harvard Health).
  prediction_vs_causation-- the model-choice figure: a black box may predict, but only an interpretable
                            model hands you the adjustable effect size a CAUSAL question needs
                            (concept; AUCs measured on our NHANES data).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Rectangle,  # shapes for the scenes
                                Circle, Polygon, Arc, Ellipse)

HERE = Path(__file__).resolve().parent
np = sf.np
plt = sf.plt

SKIN = "#E8C6A8"                                                  # a warm skin tone for the patient
GOWN = "#DCF6F2"                                                  # a pale exam-gown / clinic teal


def doctor_office():
    """The clinical scene: a seated woman describes 'brain fog'; the objective test reads NORMAL.
    The whole project starts here -- a symptom the number on the chart cannot see."""
    fig, ax = plt.subplots(figsize=(9.6, 5.4)); ax.axis("off")    # wide blank canvas
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.set_aspect("equal")

    ax.text(10, 10.5, "In the exam room: a symptom the test can't see", ha="center", fontsize=15,
            family="Geist Mono", color=sf.INK, fontweight="bold")                 # title

    # ---- LEFT: the patient, seated ----
    ax.add_patch(FancyBboxPatch((1.4, 1.4), 3.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor="#C9C4B6", edgecolor="none"))           # chair seat
    ax.add_patch(FancyBboxPatch((1.4, 2.3), 0.5, 2.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#C9C4B6", edgecolor="none"))           # chair back
    # torso (shoulders -> lap): a rounded trapezoid body
    ax.add_patch(Polygon([(2.2, 2.4), (4.2, 2.4), (3.9, 5.2), (2.5, 5.2)], closed=True,
                         facecolor=sf.DEEPPINK, edgecolor=sf.INK, linewidth=1.0, zorder=3))  # blouse
    ax.add_patch(Circle((3.2, 6.0), 0.78, facecolor=SKIN, edgecolor=sf.INK, linewidth=1.0, zorder=4))  # head
    ax.add_patch(Polygon([(2.35, 6.5), (2.35, 5.55), (4.05, 5.55), (4.05, 6.5)], closed=True,
                         facecolor="#5B3A2E", edgecolor="none", zorder=3))        # hair behind head
    ax.add_patch(Circle((3.2, 6.55), 0.82, facecolor="none", edgecolor="#5B3A2E", linewidth=3.2, zorder=3))  # hair top
    ax.text(3.2, 1.05, "the patient", ha="center", fontsize=10.5, family="Geist Mono", color=sf.INK)

    # her speech bubble (what the number can't capture)
    ax.add_patch(FancyBboxPatch((4.6, 6.4), 5.2, 3.0, boxstyle="round,pad=0.05,rounding_size=0.3",
                                facecolor="white", edgecolor=sf.DEEPPINK, linewidth=1.8, zorder=5))
    ax.add_patch(Polygon([(4.9, 6.6), (4.0, 6.0), (5.3, 6.7)], closed=True,
                         facecolor="white", edgecolor=sf.DEEPPINK, linewidth=1.8, zorder=5))  # tail
    ax.add_patch(Polygon([(4.95, 6.62), (4.25, 6.18), (5.25, 6.66)], closed=True,
                         facecolor="white", edgecolor="none", zorder=6))          # hide the tail seam
    ax.text(7.2, 8.75, "“I lose words mid-sentence.", ha="center", fontsize=12.5,
            family="Geist Mono", color=sf.DEEPPINK, zorder=7)
    ax.text(7.2, 7.95, "I walk into a room and", ha="center", fontsize=12.5,
            family="Geist Mono", color=sf.DEEPPINK, zorder=7)
    ax.text(7.2, 7.35, "forget why I'm there.”", ha="center", fontsize=12.5,
            family="Geist Mono", color=sf.DEEPPINK, zorder=7)
    ax.text(7.2, 6.75, "— real, daily brain fog", ha="center", fontsize=9.5,
            style="italic", color=sf.MUTED, zorder=7)

    # ---- RIGHT: the doctor's screen / chart, reading NORMAL ----
    ax.add_patch(FancyBboxPatch((11.6, 3.1), 7.4, 5.0, boxstyle="round,pad=0.04,rounding_size=0.25",
                                facecolor="white", edgecolor=sf.INK, linewidth=1.6, zorder=4))  # monitor
    ax.add_patch(FancyBboxPatch((11.6, 7.25), 7.4, 0.85, boxstyle="round,pad=0.04,rounding_size=0.25",
                                facecolor=sf.TURQUOISE, edgecolor="none", zorder=5))            # header bar
    ax.text(15.3, 7.66, "COGNITIVE  TEST", ha="center", va="center", fontsize=13,
            family="Geist Mono", color=sf.INK, fontweight="bold", zorder=6)
    # the score bar: a NORMAL band with a marker landing inside it
    bx, bw, by = 12.2, 6.2, 5.6
    ax.add_patch(Rectangle((bx, by), bw, 0.5, facecolor="#EDEAE1", edgecolor="#B8B4A6", linewidth=1.0, zorder=5))
    ax.add_patch(Rectangle((bx + bw * 0.45, by), bw * 0.55, 0.5, facecolor=sf.TURQUOISE, alpha=0.35,
                           edgecolor="none", zorder=5))                                          # NORMAL band
    ax.text(bx + bw * 0.72, by + 0.9, "normal range", ha="center", fontsize=9, family="Geist Mono",
            color="#0B6B63", zorder=6)
    mx = bx + bw * 0.62                                                                          # her score marker
    ax.add_patch(Polygon([(mx, by + 0.5), (mx - 0.18, by + 0.85), (mx + 0.18, by + 0.85)], closed=True,
                         facecolor=sf.INK, edgecolor="none", zorder=7))
    ax.text(bx, by - 0.45, "low", ha="left", fontsize=8.5, color=sf.MUTED, zorder=6)
    ax.text(bx + bw, by - 0.45, "high", ha="right", fontsize=8.5, color=sf.MUTED, zorder=6)
    # the verdict on the chart: a big NORMAL with a green check
    ax.add_patch(Circle((12.9, 4.15), 0.55, facecolor=sf.TURQUOISE, edgecolor="none", zorder=6))
    ax.plot([12.62, 12.85, 13.22], [4.15, 3.9, 4.5], color="white", lw=3.5, solid_capstyle="round", zorder=7)
    ax.text(13.7, 4.15, "result: NORMAL", ha="left", va="center", fontsize=14, family="Geist Mono",
            color=sf.INK, fontweight="bold", zorder=6)
    ax.text(15.3, 3.5, "“everything looks fine”", ha="center", fontsize=9.5, style="italic",
            color=sf.MUTED, zorder=6)
    ax.text(15.3, 1.02, "the objective test", ha="center", fontsize=10.5, family="Geist Mono", color=sf.INK)

    ax.text(10, 0.15, "concept -- Maki 2024 (menopause & brain fog) / Chen & Shafir 2025 (dismissed symptoms), Harvard Health",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED)                            # attribution
    sf.save(fig, HERE, "intro_doctor_office")


def effect_flip():
    """Two studies, same drug, opposite answer. Observational risk points DOWN (looks protective),
    randomized-trial risk points UP (harm). We plot relative dementia risk vs. a risk = 1.0 line."""
    labels = ["Just watching\n(observational)", "Coin-flip trial\n(randomized, WHIMS)"]
    rel_risk = [0.66, 2.05]          # <1 = looks protective (Tang/Zandi); >1 = harm (Shumaker 2003)
    colors = [sf.TURQUOISE, sf.DEEPPINK]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, rel_risk, color=colors, edgecolor=sf.INK, linewidth=0.6, width=0.55)
    ax.axhline(1.0, color=sf.INK, lw=1.0, ls="--")                       # risk = 1.0 -> no effect
    ax.text(1.46, 1.02, "no effect", fontsize=9, color=sf.MUTED, va="bottom", ha="right")
    for b, v in zip(bars, rel_risk):                                    # print each value on its bar
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}x",
                ha="center", fontsize=11, family="Geist Mono", color=sf.INK)
    ax.add_patch(FancyArrowPatch((0.15, 0.72), (0.85, 1.95), connectionstyle="arc3,rad=-0.35",
                                 arrowstyle="-|>", mutation_scale=22, color=sf.AMBER, lw=2.2))
    ax.text(0.46, 1.20, "same drug,\nopposite answer", ha="center", fontsize=10,
            family="Geist Mono", color="#C79A10")
    ax.text(0, 0.30, "hormones look\nPROTECTIVE", ha="center", fontsize=9, color="#0B6B63")
    ax.text(1, 0.30, "hormones look\nHARMFUL", ha="center", fontsize=9, color="#9B1060")
    ax.set_ylabel("dementia risk vs. non-users  (1.0 = no effect)")
    ax.set_ylim(0, 2.5)
    sf.title(ax, "When they finally flipped a coin, the benefit vanished")
    ax.text(0.5, -0.30, "adapted from Tang 1996 / Zandi 2002 (observational) vs. Shumaker 2003 WHIMS (randomized)",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED, transform=ax.transAxes)
    sf.save(fig, HERE, "intro_effect_flip")
    sf.save_raw(sf.pd.DataFrame({"study": ["observational", "randomized_trial"],
                                 "relative_risk": rel_risk}), HERE, "intro_effect_flip")


def confounding_triangle():
    """Healthy-user bias as a 3-node diagram. A hidden top factor points down to BOTH boxes;
    the bottom link is dashed and labeled 'looks like cause -- but isn't'."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)

    def box(cx, cy, w, h, text, face, edge, tcolor):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.08,rounding_size=0.18",
                                    facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
                family="Geist Mono", color=tcolor)

    box(5.0, 6.0, 6.6, 1.35, "ALREADY HEALTHIER / WEALTHIER / MORE ACTIVE\n(the hidden third factor)",
        "#FDEBF4", sf.DEEPPINK, "#9B1060")
    box(2.3, 2.15, 3.6, 1.2, "Takes hormone\ntherapy", "#DCF6F2", sf.TURQUOISE, "#0B6B63")
    box(7.7, 2.15, 3.6, 1.2, "Better memory\nscore", "#DCF6F2", sf.TURQUOISE, "#0B6B63")

    ax.add_patch(FancyArrowPatch((3.9, 5.35), (2.5, 2.8), arrowstyle="-|>",
                                 mutation_scale=20, color=sf.DEEPPINK, lw=2.0))
    ax.add_patch(FancyArrowPatch((6.1, 5.35), (7.5, 2.8), arrowstyle="-|>",
                                 mutation_scale=20, color=sf.DEEPPINK, lw=2.0))
    ax.text(2.55, 4.05, "really\ncauses", fontsize=9, color=sf.DEEPPINK, ha="center")
    ax.text(7.45, 4.05, "really\ncauses", fontsize=9, color=sf.DEEPPINK, ha="center")

    # the tempting but false bottom link -- label placed BELOW the boxes so nothing overlaps
    ax.add_patch(FancyArrowPatch((4.15, 2.15), (5.85, 2.15), arrowstyle="-|>", ls="dashed",
                                 mutation_scale=18, color=sf.MUTED, lw=1.6))
    ax.text(5.0, 0.85, "looks like cause -- but isn't", ha="center", fontsize=9.5,
            family="Geist Mono", color=sf.MUTED)

    ax.text(5.0, 7.15, "Healthy-user bias: a hidden factor fakes the link", ha="center",
            fontsize=13, family="Geist Mono", color=sf.INK)
    ax.text(5.0, 0.12, "concept -- adapted from Wharton 2009 / Vandenbroucke 2009",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_confounding_triangle")


def objective_vs_subjective():
    """Slope plot: objective test score (mostly 'normal' band) on the left, self-reported brain-fog
    severity on the right. Lines cross -- the tests say fine, the women say something's wrong."""
    rng = np.random.default_rng(0)                                   # fixed seed -> reproducible
    n = 14
    objective = rng.normal(0.72, 0.10, n).clip(0.45, 0.95)          # most land in the 'normal' band
    subjective = rng.normal(0.62, 0.16, n).clip(0.15, 0.95)        # self-reported fog: high & scattered

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xL, xR = 0.0, 1.0
    for o, s in zip(objective, subjective):                         # one connecting line per woman
        ax.plot([xL, xR], [o, s], color=sf.BLUEVIOLET, alpha=0.35, lw=1.3, zorder=1)
    ax.scatter([xL] * n, objective, s=55, color=sf.TURQUOISE, edgecolor=sf.INK,
               linewidth=0.5, zorder=3, label="objective test")
    ax.scatter([xR] * n, subjective, s=55, color=sf.DEEPPINK, edgecolor=sf.INK,
               linewidth=0.5, zorder=3, label="self-reported fog")
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


def prediction_vs_causation():
    """The model-choice figure. A black-box predictor and an interpretable one, side by side. Only the
    interpretable model hands you the adjustable effect size a CAUSAL question needs -- and here it
    does not even predict worse. AUCs are the real 5-fold values measured on our NHANES data."""
    fig, ax = plt.subplots(figsize=(9.8, 5.4)); ax.axis("off")
    ax.set_xlim(0, 20); ax.set_ylim(0, 11)

    ax.text(10, 10.5, "Prediction is not causation: which model can answer the question?",
            ha="center", fontsize=14.5, family="Geist Mono", color=sf.INK, fontweight="bold")

    # ---- LEFT panel: the black box ----
    lx = 0.8
    ax.add_patch(FancyBboxPatch((lx, 2.1), 8.4, 7.3, boxstyle="round,pad=0.05,rounding_size=0.25",
                                facecolor="#26262E", edgecolor=sf.INK, linewidth=1.4))
    ax.text(lx + 4.2, 8.85, "BLACK BOX", ha="center", fontsize=15, family="Geist Mono",
            color="white", fontweight="bold")
    ax.text(lx + 4.2, 8.25, "CatBoost / TabPFN / Random Forest", ha="center", fontsize=9.5,
            family="Geist Mono", color="#B8B4A6")
    # a padlock over the effect size -> you can't read it
    px, py = lx + 4.2, 6.55
    ax.add_patch(Arc((px, py + 0.55), 0.9, 1.1, theta1=0, theta2=180, color=sf.AMBER, lw=3.2))  # shackle
    ax.add_patch(FancyBboxPatch((px - 0.7, py - 0.55), 1.4, 1.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=sf.AMBER, edgecolor="none"))                           # lock body
    ax.add_patch(Circle((px, py), 0.13, facecolor="#26262E", edgecolor="none"))                 # keyhole
    ax.add_patch(Rectangle((px - 0.05, py - 0.28), 0.1, 0.22, facecolor="#26262E", edgecolor="none"))
    ax.text(px, py - 1.35, "estrogen effect = ???", ha="center", fontsize=12.5, family="Geist Mono",
            color="white", fontweight="bold")
    ax.text(px, py - 2.15, "locked inside -- can't read how it\nchanges when you adjust", ha="center",
            fontsize=9.5, color="#B8B4A6")
    ax.text(lx + 4.2, 2.55, "predicts:  AUC 0.76", ha="center", fontsize=11.5, family="Geist Mono",
            color=sf.TURQUOISE)

    # ---- RIGHT panel: the interpretable model (open / readable) ----
    rx = 10.8
    ax.add_patch(FancyBboxPatch((rx, 2.1), 8.4, 7.3, boxstyle="round,pad=0.05,rounding_size=0.25",
                                facecolor="white", edgecolor=sf.TURQUOISE, linewidth=2.2))
    ax.text(rx + 4.2, 8.85, "LOGISTIC REGRESSION", ha="center", fontsize=15, family="Geist Mono",
            color=sf.INK, fontweight="bold")
    ax.text(rx + 4.2, 8.25, "one readable equation -- our choice", ha="center", fontsize=9.5,
            family="Geist Mono", color="#0B6B63")
    ax.text(rx + 4.2, 6.95, "estrogen effect = -0.11", ha="center", fontsize=13,
            family="Geist Mono", color=sf.INK, fontweight="bold")
    # the readable, adjustable coefficient shown as a little slider
    sx0, sx1, sy = rx + 1.3, rx + 7.1, 6.1
    ax.plot([sx0, sx1], [sy, sy], color="#B8B4A6", lw=3, solid_capstyle="round")
    ax.plot([sx0, sx0 + (sx1 - sx0) * 0.62], [sy, sy], color=sf.TURQUOISE, lw=3, solid_capstyle="round")
    ax.add_patch(Circle((sx0 + (sx1 - sx0) * 0.62, sy), 0.22, facecolor=sf.DEEPPINK,
                        edgecolor=sf.INK, linewidth=0.8, zorder=4))
    ax.text(rx + 4.2, sy - 0.85, "and you can WATCH it shrink from", ha="center",
            fontsize=9.5, color=sf.MUTED)
    ax.text(rx + 4.2, sy - 1.35, "-0.33 as you add the confounders", ha="center",
            fontsize=9.5, color=sf.MUTED)
    ax.text(rx + 4.2, 2.55, "predicts:  AUC 0.78", ha="center", fontsize=11.5, family="Geist Mono",
            color="#0B6B63", fontweight="bold")

    # verdict band
    ax.text(10, 1.35, "A causal question needs a model you can READ -- and here it predicts just as well anyway.",
            ha="center", fontsize=11.5, family="Geist Mono", color=sf.DEEPPINK, fontweight="bold")
    ax.text(10, 0.55, "AUCs = 5-fold cross-validation on our NHANES sample (illustrative model-choice figure)",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_prediction_vs_causation")
    sf.save_raw(sf.pd.DataFrame({"model": ["black_box_catboost", "logistic_regression"],
                                 "cv_auc": [0.76, 0.78],
                                 "gives_adjustable_effect": [0, 1]}),
                HERE, "intro_prediction_vs_causation")


if __name__ == "__main__":
    doctor_office()                                              # the clinical scene
    effect_flip()                                                # the landmark WHIMS redraw
    confounding_triangle()                                       # healthy-user bias in one picture
    objective_vs_subjective()                                    # the test-vs-symptom gap
    prediction_vs_causation()                                    # why an interpretable model wins here
    print("G6 intro figures done")
