"""Day 3 (capstone) figures: the 3-day journey, the project options, the build
workflow, the presentation rubric, and where to go next. figbase -> canvas-matched.
Output: slides/figures/d3_*.png

Run:  python slides/figures_day3.py
"""
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from PIL import Image

from figbase import (plt, np, save, figtitle, txt_on, INK, MUTED, REALIMG,
                     TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET)


def _sq(name):
    im = Image.open(REALIMG / name).convert("RGB")
    s = min(im.size)
    return im.crop(((im.width - s) // 2, (im.height - s) // 2,
                    (im.width + s) // 2, (im.height + s) // 2))


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
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.4)
    opts = [("PNEUMONIA", "chest X-rays", "cxr_normal.png", TURQUOISE),
            ("SKIN LESIONS", "dermoscopy, incl. melanoma", "skin_melanoma.jpg", DEEPPINK),
            ("CHOOSE YOUR OWN", "any MedMNIST set", "histology_idc.jpg", AMBER)]
    cw = 3.5
    for i, (name, data, fname, c) in enumerate(opts):
        x = 0.5 + i * 3.85
        ax.add_patch(FancyBboxPatch((x, 0.4), cw, 3.7, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(x + cw / 2, 3.78, name, ha="center", fontsize=14, fontweight="bold", color=tc, family="Geist Mono")
        # real image thumbnail
        iw = 2.5
        ix = x + (cw - iw) / 2
        ax.imshow(_sq(fname), extent=[ix, ix + iw, 1.05, 1.05 + iw], aspect="auto", zorder=3)
        ax.text(x + cw / 2, 0.72, data, ha="center", fontsize=11, color=tc, style="italic")
    figtitle(fig, "Pick one (or pitch your own)", y=1.0)
    fig.text(0.5, -0.03, "All three use MedMNIST: pip-installable, downloads in seconds, no account "
             "needed. So you spend the time building, not fighting data. "
             "(Sample images: PD / CC-BY, Wikimedia.)",
             ha="center", fontsize=9.5, color=MUTED, style="italic")
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


# --------------------------------------------------------------------------- #
# STATE OF THE FIELD: existing AI-in-healthcare applications (lit review)
# --------------------------------------------------------------------------- #
def fig_dataset_menu():
    """A concrete menu of MedMNIST datasets students can drop into config.DATASET."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    items = [
        ("pneumoniamnist", "chest X-ray", "pneumonia? (binary)", TURQUOISE),
        ("retinamnist", "fundus photo", "retinopathy grade (5)", DEEPPINK),
        ("dermamnist", "dermoscopy", "skin lesion (7)", AMBER),
        ("breastmnist", "ultrasound", "malignant? (binary)", BLUEVIOLET),
        ("bloodmnist", "microscopy", "blood-cell type (8)", DEEPPINK),
        ("pathmnist", "histology", "colon tissue (9)", TURQUOISE),
        ("octmnist", "retinal OCT", "retinal disease (4)", BLUEVIOLET),
        ("organamnist", "abdominal CT", "organ (11)", AMBER),
    ]
    for i, (flag, modality, task, c) in enumerate(items):
        col, row = i % 4, i // 4
        x, y = 0.35 + col * 2.92, 3.0 - row * 2.55
        ax.add_patch(FancyBboxPatch((x, y), 2.7, 2.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#FBFAF6", edgecolor="#E3E0D6", lw=1.3))
        ax.add_patch(FancyBboxPatch((x, y + 1.62), 2.7, 0.58, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + 1.35, y + 1.9, flag, ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=txt_on(c), family="Geist Mono")
        ax.text(x + 1.35, y + 1.05, modality, ha="center", va="center", fontsize=11, color=INK)
        ax.text(x + 1.35, y + 0.45, task, ha="center", va="center", fontsize=10, color=c,
                family="Geist Mono", fontweight="bold")
    figtitle(fig, "The dataset menu: set DATASET in config.py")
    fig.text(0.5, -0.02, "All are MedMNIST 2D sets: pip install medmnist, downloads in seconds, no account. "
             "A dozen more exist, organ views, tissue, more. (medmnist.com)",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d3_dataset_menu.png")


def fig_field_map():
    """Where AI is already deployed across medicine, with real example systems."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    domains = [
        ("RADIOLOGY", "stroke & bleed triage,\nfracture, nodule flags", "Aidoc, Viz.ai", TURQUOISE),
        ("OPHTHALMOLOGY", "autonomous diabetic-\nretinopathy screening", "IDx-DR, 2018", DEEPPINK),
        ("PATHOLOGY", "first FDA-cleared AI\nin digital pathology", "Paige Prostate, 2021", AMBER),
        ("CARDIOLOGY", "atrial-fib alerts;\nlow-EF from an ECG", "Apple Watch; Mayo ECG", BLUEVIOLET),
        ("CLINICAL TEXT", "ambient scribes; note\nsummaries & coding", "Nuance DAX, Abridge", TURQUOISE),
        ("DRUG / GENOMICS", "protein structure for\ndrug discovery", "AlphaFold, 2021", DEEPPINK),
    ]
    for i, (name, what, ex, c) in enumerate(domains):
        col, row = i % 3, i // 3
        x, y = 0.4 + col * 3.85, 3.1 - row * 2.7
        ax.add_patch(FancyBboxPatch((x, y), 3.55, 2.35, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#FBFAF6", edgecolor="#E3E0D6", lw=1.3))
        ax.add_patch(FancyBboxPatch((x, y + 1.85), 3.55, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + 1.775, y + 2.1, name, ha="center", va="center", fontsize=12.5, fontweight="bold",
                color=txt_on(c), family="Geist Mono")
        ax.text(x + 1.775, y + 1.18, what, ha="center", va="center", fontsize=10.5, color=INK)
        ax.text(x + 1.775, y + 0.34, ex, ha="center", va="center", fontsize=10, color=c,
                family="Geist Mono", fontweight="bold")
    figtitle(fig, "AI is already in the clinic, across every specialty")
    fig.text(0.5, -0.02, "~1,000 AI/ML-enabled devices have FDA clearance (2024), the large majority in "
             "radiology. Examples are real, deployed systems.", ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d3_field_map.png")


def fig_flagships():
    """Four flagship deployed systems, each with what it does and a citation."""
    import textwrap
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    cards = [
        ("IDx-DR", "Autonomous AI that screens for diabetic retinopathy in a primary-care "
         "clinic, with no specialist in the loop. First FDA-authorized autonomous diagnostic.",
         "Abramoff et al., npj Digital Medicine 2018", TURQUOISE),
        ("Viz.ai LVO", "Scans emergency CT for large-vessel-occlusion stroke and pages the "
         "stroke team directly, cutting time-to-treatment by tens of minutes.",
         "FDA cleared 2018", DEEPPINK),
        ("Apple Watch", "Irregular-rhythm notifications surface possible atrial fibrillation "
         "on the wrist, at population scale.",
         "Apple Heart Study, Perez et al., NEJM 2019", AMBER),
        ("Med-PaLM", "A large language model answers medical questions at expert level, "
         "the first to pass the USMLE bar, pointing at clinical LLMs.",
         "Singhal et al., Nature 2023", BLUEVIOLET),
    ]
    for i, (name, what, cite, c) in enumerate(cards):
        y = 4.25 - i * 1.12
        ax.add_patch(Rectangle((0.4, y - 0.42), 0.12, 1.02, color=c))
        ax.text(0.72, y + 0.42, name, fontsize=14, fontweight="bold", color=INK,
                family="Geist Mono", va="center")
        ax.text(0.72, y - 0.02, textwrap.fill(what, 92), fontsize=10.2, color=INK, va="center")
        ax.text(0.72, y - 0.46, cite, fontsize=9, color=c, family="Geist Mono", va="center", fontweight="bold")
    figtitle(fig, "Four systems treating patients right now")
    save(fig, "d3_flagships.png")


def fig_frontier():
    """The shift from narrow single-task models to multimodal foundation models."""
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    ax.add_patch(FancyBboxPatch((0.4, 0.7), 4.4, 2.6, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor="#FBFAF6", edgecolor="#E3E0D6", lw=1.3))
    ax.text(2.6, 2.95, "YESTERDAY", ha="center", fontsize=12, fontweight="bold", color=MUTED, family="Geist Mono")
    ax.text(2.6, 2.2, "one model,\none narrow task", ha="center", va="center", fontsize=13, color=INK)
    ax.text(2.6, 1.25, "a detector per finding", ha="center", va="center", fontsize=10.5,
            color=TURQUOISE, family="Geist Mono")
    ax.annotate("", xy=(7.2, 2.0), xytext=(5.0, 2.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    ax.add_patch(FancyBboxPatch((7.3, 0.7), 4.3, 2.6, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=BLUEVIOLET, edgecolor="none"))
    ax.text(9.45, 2.95, "THE FRONTIER", ha="center", fontsize=12, fontweight="bold", color="white", family="Geist Mono")
    ax.text(9.45, 2.2, "one multimodal model:\nimages + text + signals", ha="center", va="center",
            fontsize=12.5, color="white")
    ax.text(9.45, 1.25, "Med-Gemini, GPT-4, foundation models", ha="center", va="center", fontsize=9.5,
            color="white", family="Geist Mono")
    figtitle(fig, "The frontier: generalist, multimodal medical AI")
    fig.text(0.5, -0.02, "The same shift you felt across Days 1-2: from narrow, hand-built models toward large "
             "pretrained models that take any signal. (Med-Gemini: Saab et al., 2024.)",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d3_frontier.png")


if __name__ == "__main__":
    fig_journey()
    fig_options()
    fig_workflow()
    fig_rubric()
    fig_whats_next()
    fig_field_map()
    fig_flagships()
    fig_frontier()
    fig_dataset_menu()
    print("Day 3 figures done")
