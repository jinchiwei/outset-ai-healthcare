"""Extra Day 1 figures to flesh the lecture to a full hour: the ML mechanics
(data split, gradient descent, overfitting, augmentation) and the evaluation
block that was missing (confusion matrix, sensitivity vs specificity, the
operating-point threshold). Brand-styled. Output: slides/figures/*.png

Run:  python slides/figures_day1_extra.py
"""
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

from figbase import (plt, np, save, figtitle, txt_on, INK, MUTED,
                     TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET)


# --------------------------------------------------------------------------- #
# Augmentation: free data
# --------------------------------------------------------------------------- #
def _mini_fundus(ax, transform=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(Circle((0.5, 0.5), 0.42, color="#C0633A"))
    ax.add_patch(Circle((0.66, 0.55), 0.1, color="#F2C879"))
    rng = np.random.RandomState(2)
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        xs = 0.66 + np.cos(ang) * np.linspace(0, 0.32, 12)
        ys = 0.55 + np.sin(ang) * np.linspace(0, 0.32, 12)
        ax.plot(xs, ys, color="#7A2418", lw=1.2)


def fig_augmentation():
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    labels = ["original", "flipped", "rotated", "brighter"]
    colors = [INK, TURQUOISE, DEEPPINK, AMBER]
    for ax, lab, c in zip(axes, labels, colors):
        _mini_fundus(ax)
        ax.set_title(lab, fontsize=13, family="Geist Mono", color=c, fontweight="bold")
    figtitle(fig, "Augmentation: free extra training data")
    fig.text(0.5, -0.04, "Flip, rotate, or brighten the same eye and it is still that eye, "
             "but a fresh set of numbers. The model sees more variety and memorizes less.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "concept_augmentation.png")


# --------------------------------------------------------------------------- #
# Train / validation / test split
# --------------------------------------------------------------------------- #
def fig_data_split():
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    parts = [("TRAIN", 0.70, TURQUOISE, "the model learns from these"),
             ("VALIDATION", 0.15, AMBER, "tune choices, watch for overfitting"),
             ("TEST", 0.15, DEEPPINK, "touched once, the honest grade")]
    x = 0.5
    total_w = 11.0
    for name, frac, c, desc in parts:
        w = total_w * frac
        ax.add_patch(FancyBboxPatch((x, 1.6), w - 0.08, 1.4, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + w / 2, 2.5, name, ha="center", fontsize=15, fontweight="bold",
                color=txt_on(c), family="Geist Mono")
        ax.text(x + w / 2, 2.0, f"{int(frac*100)}%", ha="center", fontsize=12,
                color=txt_on(c), family="Geist Mono")
        ax.text(x + w / 2, 1.25, desc, ha="center", va="top", fontsize=9.5, color=MUTED)
        x += w
    figtitle(fig, "Split the data: learn, tune, then grade honestly")
    fig.text(0.5, -0.04, "The golden rule: never let the model train on the test set. "
             "Testing on data it already saw is cheating, and it is the #1 way to fool yourself.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "concept_data_split.png")


# --------------------------------------------------------------------------- #
# How learning works: gradient descent
# --------------------------------------------------------------------------- #
def fig_gradient_descent():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    x = np.linspace(-3, 3, 200)
    y = 0.5 * x ** 2 + 0.4
    ax.plot(x, y, color=BLUEVIOLET, lw=3)
    ax.set_ylim(-0.3, 6.2)
    pts = [(-2.6, "start: bad guesses"), (-1.4, ""), (-0.6, ""), (-0.1, "minimum: best the model can do")]
    for px, lab in pts:
        py = 0.5 * px ** 2 + 0.4
        ax.add_patch(Circle((px, py), 0.13, color=DEEPPINK, zorder=5))
        if lab:
            ax.annotate(lab, (px, py), xytext=(px, py + 1.1), fontsize=11, family="Geist Mono",
                        color=INK, ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    for i in range(len(pts) - 1):
        x0 = pts[i][0]; x1 = pts[i + 1][0]
        ax.annotate("", xy=(x1, 0.5 * x1 ** 2 + 0.4), xytext=(x0, 0.5 * x0 ** 2 + 0.4),
                    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.5))
    ax.set_xlabel("the model's knobs (millions of them)", fontsize=12)
    ax.set_ylabel("error (how wrong)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    figtitle(fig, "Learning = rolling downhill to less error (gradient descent)")
    fig.text(0.5, -0.03, "Each training step nudges the knobs in the direction that reduces error, "
             "a little at a time, like a ball settling into a valley.",
             ha="center", fontsize=10, color=MUTED, style="italic")
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
    ax.text(10.55, 3.1, "90%", ha="center", fontsize=34, fontweight="bold", color=INK, family="Geist Mono")
    ax.text(10.55, 2.2, 'accuracy by\nsaying "healthy"\nto everyone', ha="center", va="center",
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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, name, defn, q, c in [
        (axL, "SENSITIVITY", "of the truly sick,\nhow many did we catch?", "Miss a referral\n= preventable blindness", DEEPPINK),
        (axR, "SPECIFICITY", "of the truly healthy,\nhow many did we clear?", "False alarm\n= a needless specialist visit", TURQUOISE)]:
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(0.5, 0.76, name, ha="center", fontsize=20, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(0.5, 0.52, defn, ha="center", va="center", fontsize=13, color=tc)
        ax.text(0.5, 0.22, q, ha="center", va="center", fontsize=11, color=tc, style="italic")
    figtitle(fig, "The two numbers that matter in screening")
    fig.text(0.5, -0.03, "For screening we usually favor SENSITIVITY: better to send a few healthy "
             "people for a second look than to miss someone going blind.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "eval_sens_spec.png")


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
    fig_threshold()
    print("extra figures done")
