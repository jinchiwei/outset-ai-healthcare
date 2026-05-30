"""Clinical-background figures for the Day 1 deck (brand-styled).

These ground the technical ladder in the medicine: AI across specialties, what
diabetic retinopathy does to the eye, the screening access gap, the deployment
story, and the clinical severity scale. Output: slides/figures/clinical_*.png

Run:  python slides/figures_day1_clinical.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/_shared"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np

from mpl_style import apply_style, TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET

apply_style()
INK = "#14141C"
MUTED = "#555560"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 200

# Contrast rule: pick ink or white by the fill's luminance. Turquoise/amber and
# other light fills get ink text; deeppink/blueviolet/dark fills get white.
def _lum(hexc):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def txt_on(color):
    return INK if _lum(color) > 0.55 else "white"


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def figtitle(fig, text, y=1.03):
    fig.suptitle(text, fontsize=16, fontweight="bold", family="Geist Mono", color=INK, y=y)


# --------------------------------------------------------------------------- #
# 1. AI is already in medicine, across specialties
# --------------------------------------------------------------------------- #
def fig_ai_in_medicine():
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    cards = [("Radiology", "chest X-rays,\nCT, MRI", "find nodules,\nbleeds, fractures", TURQUOISE),
             ("Pathology", "biopsy slides", "grade tumors", DEEPPINK),
             ("Dermatology", "skin photos", "flag melanoma", AMBER),
             ("Cardiology", "ECG, echo", "catch arrhythmias", BLUEVIOLET),
             ("Ophthalmology", "retina photos", "screen for\nblindness", TURQUOISE)]
    n = len(cards); gap = 0.25; cw = (12 - (n - 1) * gap) / n
    for i, (spec, data, task, c) in enumerate(cards):
        x = i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, 0.3), cw, 3.2, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(x + cw / 2, 3.1, spec, ha="center", va="center", fontsize=13,
                fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x + cw / 2, 2.2, data, ha="center", va="center", fontsize=11, color=tc)
        ax.text(x + cw / 2, 1.0, task, ha="center", va="center", fontsize=11,
                color=tc, style="italic")
    figtitle(fig, "AI is already reading medical data across specialties")
    fig.text(0.5, -0.02, "The U.S. FDA has cleared well over 1,000 AI/ML-enabled medical "
             "devices, most in imaging. Ours today: the eye.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "clinical_ai_in_medicine.png")


# --------------------------------------------------------------------------- #
# 2. What diabetic retinopathy does to the eye (progression)
# --------------------------------------------------------------------------- #
def _fundus(ax, cx, cy, r, lesions=0, neo=False):
    ax.add_patch(Circle((cx, cy), r, color="#C0633A"))
    ax.add_patch(Circle((cx + r * 0.4, cy + r * 0.1), r * 0.2, color="#F2C879"))  # disc
    rng = np.random.RandomState(lesions + 1)
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        xs = cx + r * 0.4 + np.cos(ang) * np.linspace(0, r * 0.8, 12)
        ys = cy + r * 0.1 + np.sin(ang) * np.linspace(0, r * 0.8, 12)
        ax.plot(xs, ys, color="#7A2418", lw=1.1, alpha=0.8)
    for _ in range(lesions):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.2, 0.85) * r
        ax.add_patch(Circle((cx + np.cos(a) * rr, cy + np.sin(a) * rr), r * 0.05, color="#6B0F0F"))
    for _ in range(max(0, lesions - 2)):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.2, 0.85) * r
        ax.add_patch(Circle((cx + np.cos(a) * rr, cy + np.sin(a) * rr), r * 0.045, color="#F4E03A"))
    if neo:
        for _ in range(5):
            a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.3, 0.8) * r
            x0, y0 = cx + np.cos(a) * rr, cy + np.sin(a) * rr
            for _ in range(4):
                a2 = rng.uniform(0, 2 * np.pi)
                ax.plot([x0, x0 + np.cos(a2) * r * 0.15], [y0, y0 + np.sin(a2) * r * 0.15],
                        color="#C8102E", lw=1.4)


def fig_dr_progression():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.set_aspect("equal")
    stages = [("Healthy", "clear retina", 0, False, TURQUOISE),
              ("Early DR", "microaneurysms,\ntiny hemorrhages", 3, False, AMBER),
              ("Advanced DR", "more bleeding,\nlipid exudates", 6, False, DEEPPINK),
              ("Proliferative", "fragile new vessels\n-> blindness", 6, True, BLUEVIOLET)]
    for i, (name, desc, les, neo, c) in enumerate(stages):
        cx = 1.5 + i * 3.1
        _fundus(ax, cx, 2.4, 1.05, lesions=les, neo=neo)
        ax.text(cx, 0.95, name, ha="center", fontsize=13, fontweight="bold",
                color=c, family="Geist Mono")
        ax.text(cx, 0.35, desc, ha="center", va="top", fontsize=10, color=MUTED)
        if i < 3:
            ax.annotate("", xy=(cx + 1.55, 2.4), xytext=(cx + 1.05, 2.4),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    figtitle(fig, "What diabetic retinopathy does to the eye "
             "(adapted from the Intl. Clinical DR Severity Scale, Wilkinson 2003)")
    fig.text(0.5, -0.03, "Chronic high blood sugar damages the retina's tiny vessels. "
             "Caught early and referred, the blindness is preventable.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "clinical_dr_progression.png")


# --------------------------------------------------------------------------- #
# 3. The access gap
# --------------------------------------------------------------------------- #
def fig_access_gap():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    axL.axis("off"); axL.set_xlim(0, 10); axL.set_ylim(0, 10)
    rng = np.random.RandomState(1)
    for _ in range(220):
        axL.add_patch(Circle((rng.uniform(0, 10), rng.uniform(0, 8)), 0.12, color=TURQUOISE, alpha=0.8))
    for _ in range(6):
        axL.add_patch(Circle((rng.uniform(1, 9), rng.uniform(1, 7)), 0.28, color=DEEPPINK))
    axL.text(5, 9.2, "many patients, few specialists", ha="center", fontsize=12,
             color=INK, family="Geist Mono")
    axL.text(0.2, -0.3, "● ~463M adults with diabetes      ● too few eye doctors",
             fontsize=10, color=MUTED)

    axR.axis("off"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    facts = [("1 in 3", "people with diabetes\ndevelop retinopathy", AMBER),
             ("Leading", "cause of blindness in\nworking-age adults", DEEPPINK),
             ("Preventable", "with early screening\nand referral", TURQUOISE)]
    for i, (big, small, c) in enumerate(facts):
        y = 0.78 - i * 0.31
        axR.add_patch(FancyBboxPatch((0.02, y - 0.1), 0.96, 0.24,
                                     boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        axR.text(0.08, y, big, fontsize=20, fontweight="bold", va="center", color=tc, family="Geist Mono")
        axR.text(0.42, y, small, fontsize=10.5, va="center", color=tc)
    figtitle(fig, "A screening problem: too many eyes, too few specialists")
    save(fig, "clinical_access_gap.png")


# --------------------------------------------------------------------------- #
# 4. Deployment timeline
# --------------------------------------------------------------------------- #
def fig_deployment():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    ax.plot([0.5, 11.5], [2, 2], color=INK, lw=2)
    miles = [(1.5, "2016", "Gulshan et al., JAMA", "deep learning grades DR as\nwell as ophthalmologists", TURQUOISE),
             (5.0, "2018", "IDx-DR", "first FDA-authorized AUTONOMOUS\nAI diagnosis (no doctor in loop)", DEEPPINK),
             (8.5, "2019+", "Clinics in India & Thailand", "screening deployed where\nspecialists are scarce", AMBER)]
    for x, yr, title_, desc, c in miles:
        ax.add_patch(Circle((x, 2), 0.16, color=c, zorder=3))
        ax.text(x, 2.55, yr, ha="center", fontsize=14, fontweight="bold", color=c, family="Geist Mono")
        ax.text(x, 1.55, title_, ha="center", va="top", fontsize=12, fontweight="bold",
                color=INK, family="Geist Mono")
        ax.text(x, 1.05, desc, ha="center", va="top", fontsize=10, color=MUTED)
    figtitle(fig, "From research to the clinic, fast")
    save(fig, "clinical_deployment.png")


# --------------------------------------------------------------------------- #
# 5. The 0-4 severity scale + referable threshold
# --------------------------------------------------------------------------- #
def fig_severity_scale():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    grades = [("0", "None", "#7FB069"), ("1", "Mild", AMBER),
              ("2", "Moderate", "#E8943A"), ("3", "Severe", DEEPPINK), ("4", "Proliferative", "#8A1A1A")]
    bw = 2.2; gap = 0.12
    total = len(grades) * bw + (len(grades) - 1) * gap
    x0 = (12 - total) / 2
    for i, (g, name, c) in enumerate(grades):
        x = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, 1.5), bw, 1.4, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + bw / 2, 2.5, g, ha="center", fontsize=26, fontweight="bold", color=txt_on(c), family="Geist Mono")
        ax.text(x + bw / 2, 1.85, name, ha="center", fontsize=12, color=txt_on(c), family="Geist Mono")
    # brackets
    split = x0 + 2 * (bw + gap) - gap / 2
    ax.plot([x0, split - 0.1], [1.2, 1.2], color=MUTED, lw=2)
    ax.text((x0 + split) / 2, 0.75, "NOT referable", ha="center", fontsize=12,
            color=MUTED, family="Geist Mono")
    ax.plot([split + 0.1, x0 + total], [1.2, 1.2], color=DEEPPINK, lw=3)
    ax.text((split + x0 + total) / 2, 0.75, "REFERABLE -> see an ophthalmologist",
            ha="center", fontsize=12, color=DEEPPINK, fontweight="bold", family="Geist Mono")
    figtitle(fig, "Reading the retina: the 0 to 4 severity scale")
    fig.text(0.5, -0.03, "Our task all day is this one yes/no line: grade 2 or worse means refer.",
             ha="center", fontsize=10.5, color=MUTED, style="italic")
    save(fig, "clinical_severity_scale.png")


if __name__ == "__main__":
    fig_ai_in_medicine()
    fig_dr_progression()
    fig_access_gap()
    fig_deployment()
    fig_severity_scale()
    print("clinical figures done")
