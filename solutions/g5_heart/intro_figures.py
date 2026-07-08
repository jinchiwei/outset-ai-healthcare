"""G5 intro/background figures -- bespoke redraws of the foundational heart-disease refs (attributed).

  risk_factor_stack   -- no single cause: a handful of modifiable risk factors stack up to most of
                         the risk of a first heart attack; abnormal cholesterol + smoking lead
                         (adapted from Yusuf et al. 2004, INTERHEART).
  symptom_presentation -- the "textbook" heart attack is the MALE picture; women more often feel it
                         as back pain, nausea, and shortness of breath than as classic chest pain
                         (concept from an AHA scientific statement, Mehta et al. 2016 / van Oosterhout 2020).
  dataset_sex_skew    -- our 303-patient Cleveland dataset is roughly 2:1 male, so the model sees far
                         more men than women (from the Cleveland Heart Disease data, Detrano et al. 1989).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
np = sf.np
plt = sf.plt


def risk_factor_stack():
    # Approximate odds ratios (risk multipliers) for a first heart attack, from the INTERHEART
    # case-control study. Abnormal blood lipids (which cholesterol is part of) and smoking lead.
    factors = ["Abnormal\ncholesterol", "Smoking", "Diabetes", "High blood\npressure", "Belly fat\n(obesity)"]
    odds = [3.25, 2.87, 2.37, 1.91, 1.62]                       # odds ratio: >1 means higher risk
    # cholesterol IS a column in our dataset, so highlight it; the rest in supporting colors.
    colors = [sf.DEEPPINK] + [sf.TURQUOISE, sf.GOLD, sf.BLUEVIOLET, "#B8B4A8"]
    y = np.arange(len(factors))[::-1]                           # top-to-bottom, biggest on top
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(y, odds, color=colors, edgecolor=sf.INK, linewidth=0.5, height=0.62)
    ax.axvline(1.0, ls="--", color=sf.MUTED, lw=1)             # 1.0 = no change in risk
    ax.text(1.03, -0.78, "1.0 = no change in risk", fontsize=8.5, color=sf.MUTED, ha="left", va="center")
    for yi, v in zip(y, odds):
        tag = "  (in our dataset)" if yi == y[0] else ""       # flag the one factor our data records
        ax.text(v + 0.05, yi, f"{v:.2f}x{tag}", va="center", ha="left", fontsize=10,
                family="Geist Mono", color=sf.DEEPPINK if tag else sf.INK)
    ax.set_yticks(y); ax.set_yticklabels(factors, fontsize=9.5)
    ax.set_ylim(-1.15, len(factors) - 0.35)
    ax.set_xlim(0, 4.5); ax.set_xlabel("how many times more likely a first heart attack becomes")
    sf.title(ax, "No single cause -- risk is a stack of factors")
    ax.text(0.5, -0.20, "adapted from Yusuf et al. 2004 (INTERHEART)", ha="center", fontsize=9,
            style="italic", color=sf.MUTED, transform=ax.transAxes)
    sf.save(fig, HERE, "intro_risk_factor_stack")
    sf.save_raw(sf.pd.DataFrame({"risk_factor": [f.replace(chr(10), " ") for f in factors],
                                 "odds_ratio": odds}), HERE, "intro_risk_factor_stack")


def _silhouette(ax, cx, color):
    """Draw a very simple gender-neutral body silhouette centred at x=cx (head + shoulders + torso)."""
    ax.add_patch(Circle((cx, 7.2), 0.55, color=color, ec=sf.INK, lw=0.5))               # head
    torso = FancyBboxPatch((cx - 0.85, 3.6), 1.7, 3.0, boxstyle="round,pad=0.02,rounding_size=0.4",
                           color=color, ec=sf.INK, lw=0.5)                              # torso
    ax.add_patch(torso)
    ax.add_patch(Rectangle((cx - 1.35, 5.7), 0.5, 0.35, color=color, ec=sf.INK, lw=0.5))  # arms
    ax.add_patch(Rectangle((cx + 0.85, 5.7), 0.5, 0.35, color=color, ec=sf.INK, lw=0.5))


def symptom_presentation():
    fig, ax = plt.subplots(figsize=(9.4, 4.8)); ax.axis("off")
    ax.set_xlim(-3.5, 18.5); ax.set_ylim(0, 9)
    men_cx, women_cx = 4.0, 11.0
    # --- MEN (left silhouette, labels to its LEFT) ---
    _silhouette(ax, men_cx, sf.TURQUOISE)
    ax.text(men_cx, 8.55, "MEN", ha="center", fontsize=14,
            family="Geist Mono", color=sf.INK, weight="bold")
    ax.text(men_cx, 8.0, "the 'textbook' picture", ha="center", fontsize=9.5,
            color=sf.MUTED, style="italic")
    men = [("Crushing chest pressure", 6.2), ("Pain down the left arm", 4.9)]
    for label, yy in men:
        ax.annotate(label, xy=(men_cx - 0.85, yy), xytext=(men_cx - 1.4, yy), fontsize=9.5,
                    va="center", ha="right", color=sf.INK,
                    arrowprops=dict(arrowstyle="-", color=sf.TURQUOISE, lw=1.3))
    # --- WOMEN (right silhouette, labels to its RIGHT) ---
    _silhouette(ax, women_cx, sf.DEEPPINK)
    ax.text(women_cx, 8.55, "WOMEN", ha="center", fontsize=14,
            family="Geist Mono", color=sf.INK, weight="bold")
    ax.text(women_cx, 8.0, "more often 'atypical'", ha="center", fontsize=9.5,
            color=sf.MUTED, style="italic")
    women = [("Shortness of breath", 6.7), ("Nausea / indigestion", 5.6),
             ("Pain in back, jaw or neck", 4.5), ("Unusual fatigue", 3.4)]
    for label, yy in women:
        ax.annotate(label, xy=(women_cx + 0.85, yy), xytext=(women_cx + 1.4, yy), fontsize=9.5,
                    va="center", ha="left", color=sf.INK,
                    arrowprops=dict(arrowstyle="-", color=sf.DEEPPINK, lw=1.3))
    ax.text(7.5, 1.55, "The 'classic' heart-attack symptoms are the MALE ones --\none reason women's heart attacks are more often missed.",
            ha="center", fontsize=10, color=sf.INK, style="italic")
    ax.text(7.5, 0.4, "concept from an AHA scientific statement (Mehta et al. 2016) and van Oosterhout et al. 2020",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_symptom_presentation")


def dataset_sex_skew():
    # Waffle pictograph: each square stands for ~3 patients; colour by sex. 201 male, 96 female.
    n_male, n_female = 201, 96
    per = 3                                                        # patients per square
    sq_m, sq_f = round(n_male / per), round(n_female / per)        # 67 male, 32 female squares
    total = sq_m + sq_f
    ncol = 11
    fig, ax = plt.subplots(figsize=(7.6, 4.2)); ax.axis("off")
    ax.set_xlim(-0.5, ncol + 0.5); ax.set_ylim(-2.2, (total // ncol) + 2)
    for i in range(total):
        r, c = divmod(i, ncol)
        color = sf.TURQUOISE if i < sq_m else sf.DEEPPINK         # men first, then women
        ax.add_patch(Rectangle((c, r), 0.86, 0.86, color=color, ec=sf.INK, lw=0.4))
    top = (total - 1) // ncol + 1
    ax.text(0, top + 1.0, "Who is in the data?  Each square = 3 patients", fontsize=11,
            family="Geist Mono", color=sf.INK, weight="bold")
    ax.add_patch(Rectangle((0, -1.5), 0.7, 0.7, color=sf.TURQUOISE, ec=sf.INK, lw=0.4))
    ax.text(0.9, -1.15, f"men -- {n_male} patients (about 2 of every 3)", fontsize=9.5, va="center", color=sf.INK)
    ax.add_patch(Rectangle((6.3, -1.5), 0.7, 0.7, color=sf.DEEPPINK, ec=sf.INK, lw=0.4))
    ax.text(7.2, -1.15, f"women -- {n_female}", fontsize=9.5, va="center", color=sf.INK)
    ax.text(ncol + 0.4, -1.95, "from the Cleveland Heart Disease data (Detrano et al. 1989)",
            ha="right", fontsize=8.5, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_dataset_sex_skew")
    sf.save_raw(sf.pd.DataFrame({"sex": ["male", "female"], "patients": [n_male, n_female]}),
                HERE, "intro_dataset_sex_skew")


if __name__ == "__main__":
    risk_factor_stack(); symptom_presentation(); dataset_sex_skew()
    print("G5 intro figures done")
