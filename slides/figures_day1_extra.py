"""Extra Day 1 figures to flesh the lecture to a full hour: the ML mechanics
(data split, gradient descent, overfitting, augmentation) and the evaluation
block that was missing (confusion matrix, sensitivity vs specificity, the
operating-point threshold). Brand-styled. Output: slides/figures/*.png

Run:  python slides/figures_day1_extra.py
"""
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

from PIL import Image, ImageOps, ImageEnhance

from figbase import (plt, np, save, figtitle, txt_on, INK, MUTED, REALIMG,
                     TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET)


# --------------------------------------------------------------------------- #
# Augmentation: free data
# --------------------------------------------------------------------------- #
def fig_augmentation():
    # Real fundus photo, with REAL augmentations applied (not four copies of a schematic).
    base = Image.open(REALIMG / "fundus_normal.jpg").convert("RGB")
    s = min(base.size)
    base = base.crop(((base.width - s) // 2, (base.height - s) // 2,
                      (base.width + s) // 2, (base.height + s) // 2))
    variants = [("original", base, INK),
                ("flipped", ImageOps.mirror(base), TURQUOISE),
                ("rotated 15", base.rotate(15, expand=False), DEEPPINK),
                ("brighter", ImageEnhance.Brightness(base).enhance(1.6), AMBER)]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    for ax, (lab, im, c) in zip(axes, variants):
        ax.imshow(im)
        ax.set_title(lab, fontsize=13, family="Geist Mono", color=c, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    figtitle(fig, "Augmentation: free extra training data")
    fig.text(0.5, -0.04, "The same real fundus, actually flipped, rotated, and brightened. "
             "Still the same eye, but a fresh set of numbers, so the model memorizes less. "
             "(Image: public domain, Wikimedia Commons.)",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "concept_augmentation.png")


# --------------------------------------------------------------------------- #
# Train / validation / test split
# --------------------------------------------------------------------------- #
def fig_data_split():
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    parts = [("TRAIN", 0.70, TURQUOISE, "the model\nlearns from these"),
             ("VALIDATION", 0.15, AMBER, "tune choices,\nwatch overfitting"),
             ("TEST", 0.15, DEEPPINK, "touched once:\nthe honest grade")]
    x = 0.5
    total_w = 11.0
    for name, frac, c, desc in parts:
        w = total_w * frac
        ax.add_patch(FancyBboxPatch((x, 1.7), w - 0.08, 1.4, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor=c, edgecolor="none"))
        # name + % inside the colored block (always fits)
        nm = name if frac > 0.2 else name[:3]  # TRA/VAL/TES would be cryptic; keep full but smaller
        fs = 15 if frac > 0.2 else 11
        ax.text(x + w / 2, 2.6, name, ha="center", fontsize=fs, fontweight="bold",
                color=txt_on(c), family="Geist Mono")
        ax.text(x + w / 2, 2.1, f"{int(frac*100)}%", ha="center", fontsize=11,
                color=txt_on(c), family="Geist Mono")
        # description below, font sized to the block width so neighbors do not collide
        ax.text(x + w / 2, 1.5, desc, ha="center", va="top",
                fontsize=8.5 if frac < 0.2 else 10.5, color=MUTED)
        x += w
    figtitle(fig, "Split the data: learn, tune, then grade honestly")
    fig.text(0.5, -0.04, "The golden rule: never let the model train on the test set. "
             "Testing on data it already saw is cheating, and it is the #1 way to fool yourself.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "concept_data_split.png")


# --------------------------------------------------------------------------- #
# How learning works: gradient descent
# --------------------------------------------------------------------------- #
def _glossy_ball(ax, cx, cy, r, color, z=6):
    """A ball that actually looks 3D: shaded body + a white sheen highlight."""
    ax.add_patch(Circle((cx, cy), r, color=color, zorder=z, ec="none"))
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=INK, lw=1.0, alpha=0.25, zorder=z + 0.1))
    ax.add_patch(Circle((cx - r * 0.32, cy + r * 0.34), r * 0.34, color="white", alpha=0.65, zorder=z + 0.2))


def fig_gradient_descent():
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.linspace(0, 10, 500)
    # a real rolling landscape: a deep valley on the right, a little local dip + bump on the left
    y = (0.09 * (x - 6.6) ** 2          # the big bowl (global minimum near x=6.6)
         + 0.9 * np.exp(-((x - 2.3) / 1.1) ** 2)   # a hump the ball rolls over
         + 0.35 * np.sin(0.8 * x) + 1.0)

    ymin = y.min() - 0.5
    ax.set_xlim(0, 10); ax.set_ylim(ymin, y.max() + 1.7)

    # gradient-filled hill under the curve (light at the surface -> brand violet deep down)
    hill = LinearSegmentedColormap.from_list("hill", ["#EDE7FB", BLUEVIOLET])
    grad = np.linspace(1, 0, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[0, 10, ymin, y.max() + 1.7], origin="upper", aspect="auto",
                   cmap=hill, zorder=0, alpha=0.92)
    verts = list(zip(x, y)) + [(10, ymin), (0, ymin)]
    clip = Polygon(verts, closed=True, transform=ax.transData)
    im.set_clip_path(clip)
    ax.plot(x, y, color=BLUEVIOLET, lw=3, zorder=3)            # the hill surface line

    def yof(xx):
        return float(np.interp(xx, x, y))

    # the descending trajectory: ball starts high-left, rolls over the hump into the valley
    path = [0.7, 1.5, 3.2, 4.4, 5.3, 6.0, 6.45, 6.6]
    for i in range(len(path) - 1):
        x0, x1 = path[i], path[i + 1]
        ax.annotate("", xy=(x1, yof(x1) + 0.12), xytext=(x0, yof(x0) + 0.12),
                    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.4,
                                    connectionstyle="arc3,rad=0.08"), zorder=4)
    for xx in path[1:-1]:                                       # faded "motion" balls
        _glossy_ball(ax, xx, yof(xx) + 0.16, 0.12, DEEPPINK, z=4)
        ax.patches[-3].set_alpha(0.35)

    # start + end emphasised
    _glossy_ball(ax, path[0], yof(path[0]) + 0.18, 0.22, DEEPPINK, z=7)
    ax.annotate("START\nbad model, high error", (path[0], yof(path[0]) + 0.18),
                xytext=(1.5, y.max() + 1.25), fontsize=10.5, family="Geist Mono", color=INK, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    _glossy_ball(ax, path[-1], yof(path[-1]) + 0.2, 0.26, TURQUOISE, z=7)
    ax.annotate("MINIMUM\nbest the model can do", (path[-1], yof(path[-1]) + 0.2),
                xytext=(8.4, yof(path[-1]) + 1.6), fontsize=10.5, family="Geist Mono", color=INK, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))

    # learning-rate = step size inset (two arrows of different lengths), in the open top-center
    lx, ly = 3.7, y.max() + 1.15
    ax.annotate("", xy=(lx + 0.45, ly), xytext=(lx, ly), arrowprops=dict(arrowstyle="-|>", color=TURQUOISE, lw=3))
    ax.text(lx + 0.6, ly, "small step: slow but safe", fontsize=9, color=INK, va="center", family="Geist Mono")
    ax.annotate("", xy=(lx + 1.35, ly - 0.5), xytext=(lx, ly - 0.5), arrowprops=dict(arrowstyle="-|>", color=DEEPPINK, lw=3))
    ax.text(lx + 1.5, ly - 0.5, "big step = learning rate (overshoots)", fontsize=9, color=INK, va="center", family="Geist Mono")

    ax.set_xlabel("the model's millions of knobs (weights)", fontsize=12)
    ax.set_ylabel("error (how wrong it is)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    figtitle(fig, "Learning is a ball rolling downhill (gradient descent)")
    fig.text(0.5, -0.03, "Each training step nudges the weights downhill, toward less error. The step size is "
             "the learning rate: too big overshoots, too small crawls.", ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "concept_gradient_descent.png")


# --------------------------------------------------------------------------- #
# Overfitting: memorizing vs learning
# --------------------------------------------------------------------------- #
def fig_overfitting():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ep = np.linspace(1, 20, 100)
    train = 0.5 + 0.48 * (1 - np.exp(-ep / 4))
    val = 0.5 + 0.34 * (1 - np.exp(-ep / 3)) - 0.012 * np.clip(ep - 8, 0, None)
    ax.plot(ep, train, color=TURQUOISE, lw=3, label="training accuracy")
    ax.plot(ep, val, color=DEEPPINK, lw=3, label="validation accuracy")
    ax.axvline(8, ls="--", color=MUTED, lw=1.5)
    ax.text(8.2, 0.55, "past here: memorizing,\nnot learning", fontsize=11, color=MUTED, family="Geist Mono")
    ax.set_xlabel("training time (epochs)", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_ylim(0.45, 1.0); ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=12)
    figtitle(fig, "Overfitting: when the model memorizes instead of learning")
    fig.text(0.5, -0.03, "Training accuracy keeps climbing while validation stalls or drops. "
             "The model is memorizing the training eyes, not the disease. The validation curve catches it.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "concept_overfitting.png")


# --------------------------------------------------------------------------- #
# Accuracy can lie (class imbalance)
# --------------------------------------------------------------------------- #
def fig_accuracy_lies():
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    # 100 patients, ~10 with referable DR
    rng = np.random.RandomState(0)
    idx = set(rng.choice(100, 10, replace=False))
    for k in range(100):
        r, cc = divmod(k, 20)
        x = 0.6 + cc * 0.42
        y = 4.2 - r * 0.5
        col = DEEPPINK if k in idx else TURQUOISE
        ax.add_patch(Circle((x, y), 0.16, color=col))
    ax.text(0.6, 4.75, "100 patients screened    ● 90 healthy    ● 10 need referral",
            fontsize=11, color=INK, family="Geist Mono")
    ax.add_patch(FancyBboxPatch((9.3, 1.4), 2.5, 2.2, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=AMBER, edgecolor="none"))
    ax.text(10.55, 2.95, "90%", ha="center", va="center", fontsize=34, fontweight="bold", color=INK, family="Geist Mono")
    ax.text(10.55, 2.0, 'accuracy by\nsaying "healthy"\nto everyone', ha="center", va="center",
            fontsize=11, color=INK, family="Geist Mono")
    figtitle(fig, "Accuracy can lie: a useless model that looks great")
    fig.text(0.5, -0.03, "If only 10% have disease, a model that refers NOBODY is 90% accurate, "
             "and misses every single sick patient. Accuracy alone is a trap.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "eval_accuracy_lies.png")


# --------------------------------------------------------------------------- #
# Confusion matrix (clinical)
# --------------------------------------------------------------------------- #
def fig_confusion():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    cells = [((1, "TRUE POSITIVE", "correctly referred", TURQUOISE), (1, "FALSE NEGATIVE", "MISSED disease\n-> blindness risk", DEEPPINK)),
             ((0, "FALSE POSITIVE", "false alarm\n-> needless visit", AMBER), (0, "TRUE NEGATIVE", "correctly cleared", TURQUOISE))]
    coords = [(2.0, 5.2), (5.6, 5.2), (2.0, 1.4), (5.6, 1.4)]
    flat = [cells[0][0], cells[0][1], cells[1][0], cells[1][1]]
    for (x, y), (_, lab, desc, c) in zip(coords, flat):
        ax.add_patch(FancyBboxPatch((x, y), 3.3, 3.2, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=c, edgecolor="white", lw=2))
        tc = txt_on(c)
        ax.text(x + 1.65, y + 2.4, lab, ha="center", fontsize=13, fontweight="bold",
                color=tc, family="Geist Mono")
        ax.text(x + 1.65, y + 1.1, desc, ha="center", va="center", fontsize=11, color=tc)
    ax.text(3.65, 8.8, "model says REFER", ha="center", fontsize=11, color=INK, family="Geist Mono")
    ax.text(7.25, 8.8, "model says CLEAR", ha="center", fontsize=11, color=INK, family="Geist Mono")
    ax.text(1.6, 6.8, "really\nsick", ha="center", va="center", fontsize=11, color=INK, family="Geist Mono")
    ax.text(1.6, 3.0, "really\nhealthy", ha="center", va="center", fontsize=11, color=INK, family="Geist Mono")
    figtitle(fig, "The confusion matrix: four outcomes, not one number")
    fig.text(0.5, -0.02, "The two errors are NOT equal. A missed referral (top-right) can mean "
             "preventable blindness. A false alarm (bottom-left) is a wasted visit.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "eval_confusion.png")


# --------------------------------------------------------------------------- #
# Sensitivity vs specificity
# --------------------------------------------------------------------------- #
def fig_sens_spec():
    """100 real patients: show who gets caught, cleared, missed, and falsely alarmed."""
    fig, (axG, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # --- left: a population of 100 patients ---
    axG.axis("off"); axG.set_xlim(-0.6, 10); axG.set_ylim(-2.0, 9.8)
    axG.invert_yaxis()
    # 20 sick patients at fixed scattered positions; 2 of them MISSED; 9 healthy = false alarms
    sick = {3, 7, 11, 16, 22, 24, 31, 38, 40, 45, 52, 56, 61, 67, 70, 74, 81, 88, 93, 96}
    missed = {40, 74}                 # sick but model said "fine" -> the costly errors
    false_alarm = {5, 19, 28, 47, 59, 64, 83, 90, 99}   # healthy but model referred
    LIGHT = "#D9D6CC"
    for i in range(100):
        r, c = divmod(i, 10)
        if i in sick:
            axG.add_patch(Circle((c, r), 0.38, color=DEEPPINK, zorder=3))
            if i in missed:           # ring + slash = a missed case
                axG.add_patch(Circle((c, r), 0.46, fill=False, ec=INK, lw=2.6, zorder=4))
                axG.plot([c - 0.3, c + 0.3], [r + 0.3, r - 0.3], color=INK, lw=2.4, zorder=5)
        else:
            axG.add_patch(Circle((c, r), 0.38, color=LIGHT, zorder=3))
            if i in false_alarm:
                axG.add_patch(Circle((c, r), 0.46, fill=False, ec=AMBER, lw=2.6, zorder=4))
    axG.set_title("100 patients screened", fontsize=12, color=INK, family="Geist Mono")
    # mini legend above the grid (2x2)
    leg = [(DEEPPINK, "has disease"), (LIGHT, "healthy"), (INK, "MISSED (x)"), (AMBER, "false alarm (o)")]
    for k, (col, lab) in enumerate(leg):
        lx = (k % 2) * 5.0; ly = -1.65 + (k // 2) * 0.7
        axG.add_patch(Circle((lx, ly), 0.22, color=col if col != INK else "white",
                             ec=col if col in (INK, AMBER) else "none", lw=2.2))
        axG.text(lx + 0.45, ly, lab, fontsize=9.5, color=INK, va="center", family="Geist Mono")

    # --- right: the two numbers + the human cost ---
    axR.axis("off"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    axR.add_patch(FancyBboxPatch((0.03, 0.55), 0.94, 0.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 facecolor=DEEPPINK, edgecolor="none"))
    axR.text(0.5, 0.85, "SENSITIVITY = 90%", ha="center", fontsize=17, fontweight="bold",
             color="white", family="Geist Mono")
    axR.text(0.5, 0.66, "caught 18 of 20 sick.\nthe 2 we MISSED -> preventable blindness",
             ha="center", va="center", fontsize=11, color="white")
    axR.add_patch(FancyBboxPatch((0.03, 0.08), 0.94, 0.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 facecolor=TURQUOISE, edgecolor="none"))
    axR.text(0.5, 0.38, "SPECIFICITY = 89%", ha="center", fontsize=17, fontweight="bold",
             color=txt_on(TURQUOISE), family="Geist Mono")
    axR.text(0.5, 0.19, "cleared 71 of 80 healthy.\n9 false alarms -> a needless specialist visit",
             ha="center", va="center", fontsize=11, color=txt_on(TURQUOISE))
    figtitle(fig, "Sensitivity and specificity, in actual people")
    fig.text(0.5, -0.03, "A missed case (✕) is far costlier than a false alarm (○), so screening tools are "
             "tuned for high sensitivity, even at the price of more false alarms.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "eval_sens_spec.png")


# --------------------------------------------------------------------------- #
# ROC curve + thresholds
# --------------------------------------------------------------------------- #
def fig_roc():
    """A real ROC curve with three operating points and what moving the threshold does."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    # a realistic ROC: TPR vs FPR for a decent model (AUC ~ 0.9)
    fpr = np.linspace(0, 1, 200)
    tpr = fpr ** 0.28                     # concave, well above the diagonal
    axL.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.5, label="random guessing")
    axL.plot(fpr, tpr, color=BLUEVIOLET, lw=3.5, zorder=3)
    axL.fill_between(fpr, tpr, fpr, color=BLUEVIOLET, alpha=0.10)
    # three operating points along the curve
    ops = [(0.04, 0.04 ** 0.28, "strict\n(high specificity)", TURQUOISE, (0.07, -0.10)),
           (0.20, 0.20 ** 0.28, "balanced", AMBER, (0.06, 0.05)),
           (0.55, 0.55 ** 0.28, "screening\n(high sensitivity)", DEEPPINK, (0.02, 0.06))]
    for fx, ty, lab, c, (ox, oy) in ops:
        axL.add_patch(Circle((fx, ty), 0.022, color=c, zorder=6))
        axL.annotate(lab, (fx, ty), xytext=(fx + ox, ty + oy), fontsize=9.5,
                     family="Geist Mono", color=INK)
    axL.text(0.55, 0.18, "AUC ~ 0.90", fontsize=13, fontweight="bold", color=BLUEVIOLET, family="Geist Mono")
    axL.set_xlabel("false-alarm rate  (1 - specificity)", fontsize=11)
    axL.set_ylabel("catch rate  (sensitivity)", fontsize=11)
    axL.set_xlim(-0.02, 1.02); axL.set_ylim(-0.02, 1.05)
    axL.set_xticks([0, 0.5, 1]); axL.set_yticks([0, 0.5, 1]); axL.legend(loc="lower right", fontsize=9.5)
    axL.set_title("the ROC curve", fontsize=12, color=INK, family="Geist Mono")

    # right: how to read it
    axR.axis("off"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    rows = [("Every point is a threshold.", "Slide the cutoff and you move along the curve, trading "
             "catch rate against false alarms.", BLUEVIOLET),
            ("Up-and-left is better.", "The closer the curve hugs the top-left corner, the better the "
             "model. The dashed line is a coin flip.", TURQUOISE),
            ("AUC = one summary number.", "The area under the curve, 0.5 is random, 1.0 is perfect. "
             "It rates the model independent of any one threshold.", AMBER),
            ("Screening sits to the right.", "We accept more false alarms to push catch rate up; "
             "missing a patient is the worst error.", DEEPPINK)]
    import textwrap
    for i, (head, body, c) in enumerate(rows):
        y = 0.95 - i * 0.245
        axR.add_patch(Rectangle((0.02, y - 0.13), 0.05, 0.17, color=c))
        axR.text(0.12, y, head, fontsize=12, fontweight="bold", color=INK, family="Geist Mono", va="center")
        axR.text(0.12, y - 0.10, textwrap.fill(body, 42), fontsize=9.3, color=INK, va="top")
    figtitle(fig, "Reading an ROC curve")
    save(fig, "eval_roc.png")


# --------------------------------------------------------------------------- #
# The operating point / threshold
# --------------------------------------------------------------------------- #
def fig_threshold():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    t = np.linspace(0, 1, 100)
    sens = 1 - t ** 1.6
    spec = t ** 0.8
    ax.plot(t, sens, color=DEEPPINK, lw=3, label="sensitivity (catch disease)")
    ax.plot(t, spec, color=TURQUOISE, lw=3, label="specificity (avoid false alarms)")
    ax.axvline(0.32, ls="--", color=AMBER, lw=2.5)
    ax.text(0.34, 0.12, "screening picks a\nlow threshold:\ncatch more disease", fontsize=10.5,
            color="#B8860B", family="Geist Mono")
    ax.set_xlabel("decision threshold (how sure before we refer)", fontsize=12)
    ax.set_ylabel("rate", fontsize=12); ax.set_xticks([]); ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="center right", fontsize=11)
    figtitle(fig, "You choose the trade-off: the decision threshold")
    fig.text(0.5, -0.03, "Lower the bar for referral and you catch more disease but raise false alarms. "
             "There is no free lunch; the clinical context picks the point.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "eval_threshold.png")


if __name__ == "__main__":
    fig_augmentation()
    fig_data_split()
    fig_gradient_descent()
    fig_overfitting()
    fig_accuracy_lies()
    fig_confusion()
    fig_sens_spec()
    fig_roc()
    fig_threshold()
    print("extra figures done")
