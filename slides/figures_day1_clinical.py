"""Clinical-background figures for the Day 1 deck (brand-styled).

These ground the technical ladder in the medicine: AI across specialties, what
diabetic retinopathy does to the eye, the screening access gap, the deployment
story, and the clinical severity scale. Output: slides/figures/clinical_*.png

Run:  python slides/figures_day1_clinical.py
"""
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from PIL import Image

from figbase import (plt, np, save, figtitle, txt_on, INK, MUTED, REALIMG,
                     TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET)


def _sq(name):
    im = Image.open(REALIMG / name).convert("RGB")
    s = min(im.size)
    return im.crop(((im.width - s) // 2, (im.height - s) // 2,
                    (im.width + s) // 2, (im.height + s) // 2))


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
    # REAL fundus photos showing the progression (public domain, Wikimedia).
    stages = [("fundus_normal.jpg", "Healthy", "clear retina", TURQUOISE),
              ("fundus_dr.jpg", "Diabetic retinopathy", "hemorrhages, exudates", DEEPPINK),
              ("fundus_proliferative.jpg", "Proliferative", "new vessels -> blindness", BLUEVIOLET)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    for i, (ax, (fname, name, desc, c)) in enumerate(zip(axes, stages)):
        ax.imshow(_sq(fname))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(c); sp.set_linewidth(3)
        ax.set_title(name, fontsize=14, fontweight="bold", color=c, family="Geist Mono", pad=8)
        ax.set_xlabel(desc, fontsize=10.5, color=MUTED)
    figtitle(fig, "What diabetic retinopathy does to the eye", y=1.02)
    fig.text(0.5, -0.04, "Real fundus photographs (public domain, Wikimedia Commons). Chronic high "
             "blood sugar damages the retina's vessels; caught early and referred, the blindness is preventable.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "clinical_dr_progression.png")


# --------------------------------------------------------------------------- #
# 3. The access gap
# --------------------------------------------------------------------------- #
def fig_access_gap():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    axL.axis("off"); axL.set_xlim(0, 10); axL.set_ylim(-1.2, 10)
    rng = np.random.RandomState(1)
    for _ in range(220):
        axL.add_patch(Circle((rng.uniform(0, 10), rng.uniform(1.2, 9)), 0.12, color=TURQUOISE, alpha=0.85))
    for _ in range(6):
        axL.add_patch(Circle((rng.uniform(1, 9), rng.uniform(1.5, 8.5)), 0.30, color=DEEPPINK))
    axL.text(5, 9.6, "many patients, few specialists", ha="center", fontsize=12,
             color=INK, family="Geist Mono")
    # color-matched legend (dots match the cloud)
    axL.add_patch(Circle((0.5, 0.2), 0.16, color=TURQUOISE))
    axL.text(0.85, 0.2, "people who need screening", fontsize=10, color=INK, va="center")
    axL.add_patch(Circle((0.5, -0.7), 0.16, color=DEEPPINK))
    axL.text(0.85, -0.7, "the few eye specialists", fontsize=10, color=INK, va="center")

    axR.axis("off"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    facts = [("1 in 3", "people with diabetes develop retinopathy", AMBER),
             ("Leading", "cause of blindness in working-age adults", DEEPPINK),
             ("Preventable", "with early screening and referral", TURQUOISE)]
    bh = 0.30
    for i, (big, small, c) in enumerate(facts):
        y = 0.92 - i * 0.33
        axR.add_patch(FancyBboxPatch((0.02, y - bh), 0.96, bh,
                                     boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        axR.text(0.07, y - 0.09, big, fontsize=18, fontweight="bold", va="center", color=tc, family="Geist Mono")
        axR.text(0.07, y - 0.225, small, fontsize=9.5, va="center", color=tc)
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
