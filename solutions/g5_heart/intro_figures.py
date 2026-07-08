"""G5 intro/background figures -- bespoke redraws that VISUALLY depict the clinical problem of
predicting heart-disease risk from a routine checkup (all attributed). Theme "bone" via the shared
solfig helper, so every PNG drops straight onto a slide canvas.

  heart_risk_engine    -- the clinical thing: a patient's heart and its routine-checkup risk factors
                          (cholesterol, blood pressure, age, chest pain) feeding a single risk gauge
                          -- the job a cardiologist's risk calculator does (concept, cf. Wilson 1998)
  framingham_1998      -- redraw of the landmark: the Framingham Risk Score turns combined risk
                          factors into a 10-year risk of heart disease, and it is SEX-SPECIFIC --
                          men and women with the same factors sit on different curves
                          (adapted from Wilson et al. 1998, Circulation)
  risk_factor_stack    -- no single cause: a handful of modifiable risk factors stack up to most of
                          the risk of a first heart attack; abnormal cholesterol + smoking lead
                          (adapted from Yusuf et al. 2004, INTERHEART)
  symptom_presentation -- the "textbook" heart attack is the MALE picture; women more often feel it
                          as back pain, nausea, and shortness of breath than as classic chest pain
                          (concept from an AHA scientific statement, Mehta et al. 2016 / van Oosterhout 2020)
  dataset_sex_skew     -- our 303-patient Cleveland dataset is roughly 2:1 male, so the model sees far
                          more men than women (from the Cleveland Heart Disease data, Detrano et al. 1989)
"""
import sys                                                           # sys lets us extend the import path
from pathlib import Path                                             # Path builds file locations safely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))  # find the shared solfig helper
import solfig as sf                                                  # brand palette + save/save_raw + theme "bone"
from matplotlib.patches import (Circle, FancyArrowPatch,             # shapes for the schematics
                                FancyBboxPatch, Rectangle, Wedge, Polygon)

HERE = Path(__file__).resolve().parent                              # this folder (figures land in ./figures)
np = sf.np                                                          # re-use solfig's numpy
plt = sf.plt                                                        # re-use solfig's pyplot (already themed)


def _heart(ax, cx, cy, s, color, ec=sf.INK, lw=1.4, zorder=3, alpha=1.0):
    """Draw a filled anatomical-valentine heart centred at (cx, cy), scaled by s."""
    t = np.linspace(0, 2 * np.pi, 240)                             # parameter around the heart curve
    hx = 16 * np.sin(t) ** 3                                        # classic parametric heart, x
    hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)  # heart, y
    ax.add_patch(Polygon(np.column_stack([cx + s * hx, cy + s * hy]), closed=True,
                         facecolor=color, edgecolor=ec, linewidth=lw, zorder=zorder, alpha=alpha))


def heart_risk_engine():
    """The clinical scene: routine-checkup risk factors around a heart, feeding one risk gauge."""
    fig, ax = plt.subplots(figsize=(9.6, 5.3)); ax.axis("off")     # wide blank canvas
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.set_aspect("equal") # equal aspect so the heart + dial stay round

    # --- LEFT: four routine-checkup risk factors as labeled input chips ---
    factors = [("Cholesterol", sf.DEEPPINK), ("Blood pressure", sf.TURQUOISE),
               ("Age", sf.BLUEVIOLET), ("Chest-pain type", "#B8860B")]
    ys = np.linspace(8.4, 2.6, 4)                                  # stack the four chips down the left
    for (name, col), y in zip(factors, ys):
        ax.add_patch(FancyBboxPatch((0.55, y - 0.52), 4.2, 1.04, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    facecolor="#EDEAE1", edgecolor=col, linewidth=1.8))  # chip surface
        ax.add_patch(Circle((1.25, y), 0.2, facecolor=col, edgecolor="none"))            # a colored dot
        ax.text(1.8, y, name, ha="left", va="center", fontsize=11.5, family="Geist Mono",
                color=sf.INK)                                                             # the factor name
        # a converging arrow from each chip into the heart at the centre; curve OUTWARD (away from
        # the chip text) -- flip the arc sign for chips below the heart so the line never loops back
        # over its own label ("Chest-pain type")
        rad = 0.16 if y > 5.5 else -0.16
        ax.add_patch(FancyArrowPatch((4.95, y), (8.35, 5.5), arrowstyle="-|>", mutation_scale=13,
                                     color=col, lw=1.8, connectionstyle=f"arc3,rad={rad}", zorder=2, alpha=0.9))
    ax.text(2.65, 9.5, "routine checkup", ha="center", fontsize=10.5, family="Geist Mono",
            color=sf.INK, fontweight="bold")                                              # section label
    ax.text(2.65, 1.55, "...and 9 more measurements", ha="center", fontsize=9,
            color=sf.MUTED, style="italic")                                               # ties to the 13 features

    # --- CENTER: the patient's heart (the organ at risk) with a "combine" plus ---
    _heart(ax, 10.0, 5.7, 0.135, sf.DEEPPINK, zorder=4)           # the heart itself
    ax.text(10.0, 5.55, "+", ha="center", va="center", fontsize=30, family="Geist Mono",
            color="white", fontweight="bold", zorder=5)                                   # "combine the clues"
    ax.text(10.0, 2.9, "combine the clues", ha="center", fontsize=10, family="Geist Mono",
            color=sf.INK)                                                                 # what the centre does

    # --- one arrow from the heart to the risk gauge on the right ---
    ax.add_patch(FancyArrowPatch((11.9, 5.6), (13.7, 5.6), arrowstyle="-|>", mutation_scale=20,
                                 color=sf.INK, lw=2.4, zorder=4))

    # --- RIGHT: a semicircular risk gauge (low -> high), needle pointing to "elevated" ---
    gx, gy, gr = 16.4, 4.4, 2.6                                    # gauge centre + radius
    seg = [(180, 132, sf.TURQUOISE), (132, 84, sf.GOLD), (84, 36, "#FF7A45"), (36, 0, sf.DEEPPINK)]
    for a0, a1, col in seg:                                        # four colored arc segments, low to high risk
        ax.add_patch(Wedge((gx, gy), gr, a1, a0, width=0.85, facecolor=col, edgecolor="white", linewidth=1.2))
    ang = np.deg2rad(58)                                           # needle angle -> "elevated" zone
    ax.add_patch(FancyArrowPatch((gx, gy), (gx + (gr - 0.5) * np.cos(ang), gy + (gr - 0.5) * np.sin(ang)),
                                 arrowstyle="-|>", mutation_scale=16, color=sf.INK, lw=2.6, zorder=6))
    ax.add_patch(Circle((gx, gy), 0.22, facecolor=sf.INK, edgecolor="none", zorder=7))   # needle hub
    ax.text(gx, gy + gr + 0.25, "elevated risk", ha="center", va="bottom", fontsize=12,
            family="Geist Mono", color=sf.DEEPPINK, fontweight="bold", zorder=8)          # the read-out, above the arc
    ax.text(gx - gr + 0.2, gy - 0.55, "low", ha="center", fontsize=9, color=sf.MUTED, family="Geist Mono")
    ax.text(gx + gr - 0.2, gy - 0.55, "high", ha="center", fontsize=9, color=sf.MUTED, family="Geist Mono")
    ax.text(gx, gy - 1.15, "risk score", ha="center", fontsize=10.5, family="Geist Mono",
            color=sf.INK, fontweight="bold")

    ax.text(10.0, 10.5, "One risk score from many checkup clues", ha="center", fontsize=15.5,
            family="Geist Mono", color=sf.INK, fontweight="bold")                          # title
    ax.text(10.0, 0.35, "the job a cardiologist's risk calculator does  (concept, cf. Framingham -- Wilson 1998)",
            ha="center", fontsize=9, style="italic", color=sf.MUTED)                        # attribution
    sf.save(fig, HERE, "intro_heart_risk_engine")                  # -> figures/intro_heart_risk_engine.png


def framingham_1998():
    """Redraw the landmark Framingham Risk Score: combined factors -> 10-year risk, and it is SEX-SPECIFIC."""
    pts = np.linspace(0, 20, 120)                                  # combined "risk-factor burden" points
    men = 44 / (1 + np.exp(-0.34 * (pts - 11.5)))                  # men's 10-year risk curve (higher)
    women = 27 / (1 + np.exp(-0.32 * (pts - 14.0)))               # women's curve (lower for the same burden)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))                     # one panel with real axes
    ax.plot(pts, men, color=sf.TURQUOISE, lw=3, zorder=4, label="men")        # men in turquoise
    ax.plot(pts, women, color=sf.DEEPPINK, lw=3, zorder=4, label="women")     # women in deeppink
    ax.fill_between(pts, women, men, color=sf.MUTED, alpha=0.10)              # the sex gap, shaded

    xb = 13.0                                                      # a shared risk-factor burden to compare at
    mb = 44 / (1 + np.exp(-0.34 * (xb - 11.5)))                    # men's risk there
    wb = 27 / (1 + np.exp(-0.32 * (xb - 14.0)))                    # women's risk there
    ax.axvline(xb, color=sf.INK, ls="--", lw=1.2, zorder=2)                    # the comparison line
    ax.scatter([xb, xb], [mb, wb], s=55, color=[sf.TURQUOISE, sf.DEEPPINK],
               edgecolor=sf.INK, linewidth=0.7, zorder=6)                      # the two operating points
    ax.annotate(f"{mb:.0f}%", (xb, mb), (xb + 0.9, mb + 2.5), fontsize=11, family="Geist Mono",
                color="#17b5a3", fontweight="bold")
    ax.annotate(f"{wb:.0f}%", (xb, wb), (xb + 0.6, wb - 3.5), fontsize=11, family="Geist Mono",
                color=sf.DEEPPINK, fontweight="bold")
    ax.text(xb - 0.3, 40, "same risk factors,\ndifferent risk by sex", ha="right", va="top",
            fontsize=9.5, family="Geist Mono", color=sf.INK)

    ax.set_xlim(0, 20); ax.set_ylim(0, 46)                        # keep headroom for labels
    ax.set_xlabel("risk-factor burden  (age + cholesterol + blood pressure + smoking + diabetes)")
    ax.set_ylabel("predicted 10-year risk of heart disease (%)")
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    sf.title(ax, "A risk score is sex-specific by design")
    ax.text(10, -0.185, "adapted from Wilson et al. 1998 (Framingham Risk Score), Circulation",
            ha="center", fontsize=9, style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_framingham_1998")
    sf.save_raw(sf.pd.DataFrame({"risk_points": pts, "men_risk_pct": men, "women_risk_pct": women}),
                HERE, "intro_framingham_1998")


def risk_factor_stack():
    """No single cause: the biggest modifiable risk factors for a first heart attack, INTERHEART."""
    # Approximate odds ratios (risk multipliers) for a first heart attack, from the INTERHEART
    # case-control study. Abnormal blood lipids (which cholesterol is part of) and smoking lead.
    factors = ["Abnormal\ncholesterol", "Smoking", "Diabetes", "High blood\npressure", "Belly fat\n(obesity)"]
    odds = [3.25, 2.87, 2.37, 1.91, 1.62]                          # odds ratio: >1 means higher risk
    colors = [sf.DEEPPINK] + [sf.TURQUOISE, sf.GOLD, sf.BLUEVIOLET, "#B8B4A8"]  # highlight cholesterol (in our data)
    y = np.arange(len(factors))[::-1]                              # top-to-bottom, biggest on top
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(y, odds, color=colors, edgecolor=sf.INK, linewidth=0.5, height=0.62)
    ax.axvline(1.0, ls="--", color=sf.MUTED, lw=1)                # 1.0 = no change in risk
    ax.text(1.03, -0.78, "1.0 = no change in risk", fontsize=8.5, color=sf.MUTED, ha="left", va="center")
    for yi, v in zip(y, odds):
        tag = "  (in our dataset)" if yi == y[0] else ""          # flag the one factor our data records
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
    """Men vs women feel a heart attack differently -- the 'textbook' picture is the male one."""
    fig, ax = plt.subplots(figsize=(9.4, 4.8)); ax.axis("off")
    ax.set_xlim(-3.5, 18.5); ax.set_ylim(0, 9)
    men_cx, women_cx = 4.0, 11.0
    # --- MEN (left silhouette, labels to its LEFT) ---
    _silhouette(ax, men_cx, sf.TURQUOISE)
    ax.text(men_cx, 8.55, "MEN", ha="center", fontsize=14, family="Geist Mono", color=sf.INK, weight="bold")
    ax.text(men_cx, 8.0, "the 'textbook' picture", ha="center", fontsize=9.5, color=sf.MUTED, style="italic")
    men = [("Crushing chest pressure", 6.2), ("Pain down the left arm", 4.9)]
    for label, yy in men:
        ax.annotate(label, xy=(men_cx - 0.85, yy), xytext=(men_cx - 1.4, yy), fontsize=9.5,
                    va="center", ha="right", color=sf.INK,
                    arrowprops=dict(arrowstyle="-", color=sf.TURQUOISE, lw=1.3))
    # --- WOMEN (right silhouette, labels to its RIGHT) ---
    _silhouette(ax, women_cx, sf.DEEPPINK)
    ax.text(women_cx, 8.55, "WOMEN", ha="center", fontsize=14, family="Geist Mono", color=sf.INK, weight="bold")
    ax.text(women_cx, 8.0, "more often 'atypical'", ha="center", fontsize=9.5, color=sf.MUTED, style="italic")
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
    """Waffle pictograph of the ~2:1 male-to-female split among the 303 Cleveland patients."""
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
    heart_risk_engine()                                           # the clinical scene
    framingham_1998()                                             # the landmark redraw
    risk_factor_stack()                                           # INTERHEART: risk is a stack
    symptom_presentation()                                        # men vs women presentation
    dataset_sex_skew()                                            # the 2:1 male skew
    print("G5 intro figures done")
