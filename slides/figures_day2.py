"""Day 2 figures: clinical (chest X-ray + reports), language models (tokens,
attention, hallucination), the multimodal stack (image-model vote + LLM + demographics
-> tabular -> TabPFN, late fusion), and the leakage evaluation. Brand-styled via figbase, so
backgrounds match the deck canvas. Output: slides/figures/d2_*.png

Run:  python slides/figures_day2.py
"""
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from PIL import Image

from figbase import (plt, np, save, figtitle, txt_on, INK, MUTED, REALIMG,
                     TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET)


# --------------------------------------------------------------------------- #
# CLINICAL 1: what a chest X-ray shows
# --------------------------------------------------------------------------- #
def fig_cxr_findings():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.1]})
    # REAL chest X-ray (CC0, Wikimedia)
    axL.imshow(Image.open(REALIMG / "cxr_normal.png").convert("L"), cmap="gray")
    axL.set_xticks([]); axL.set_yticks([])
    for sp in axL.spines.values():
        sp.set_visible(False)
    axL.set_title("a real chest X-ray", fontsize=11, color=INK, family="Geist Mono")

    axR.axis("off"); axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    findings = [("Cardiomegaly", "the heart is enlarged", DEEPPINK),
                ("Effusion", "fluid around the lung", TURQUOISE),
                ("Opacity / pneumonia", "a hazy patch of infection", AMBER),
                ("Pneumothorax", "collapsed lung", BLUEVIOLET)]
    for i, (name, desc, c) in enumerate(findings):
        y = 0.82 - i * 0.22
        axR.add_patch(Rectangle((0.02, y - 0.075), 0.05, 0.15, color=c))
        axR.text(0.12, y + 0.02, name, fontsize=14, fontweight="bold", color=INK, family="Geist Mono", va="center")
        axR.text(0.12, y - 0.05, desc, fontsize=11, color=MUTED, va="center")
    figtitle(fig, "One chest X-ray, many possible findings")
    fig.text(0.5, -0.02, "The chest X-ray is the most common imaging test in the world. A single "
             "image can show the heart, the lungs, and more. (Image: CC0, Wikimedia Commons.)",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_cxr_findings.png")


# --------------------------------------------------------------------------- #
# CLINICAL 2: the radiology report
# --------------------------------------------------------------------------- #
def fig_report():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.add_patch(FancyBboxPatch((0.4, 0.4), 7.0, 4.2, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor="white", edgecolor="#DDD", lw=1.5))
    ax.text(0.8, 4.2, "RADIOLOGY REPORT", fontsize=12, fontweight="bold", color=INK, family="Geist Mono")
    ax.text(0.8, 3.6, "FINDINGS:", fontsize=12, fontweight="bold", color=DEEPPINK, family="Geist Mono")
    ax.text(0.8, 3.1, "The heart is enlarged. There is a small\nleft pleural effusion. No pneumothorax.",
            fontsize=12, color=INK, va="top")
    ax.text(0.8, 1.9, "IMPRESSION:", fontsize=12, fontweight="bold", color=DEEPPINK, family="Geist Mono")
    ax.text(0.8, 1.4, "Cardiomegaly with small effusion.", fontsize=12, color=INK, va="top")
    # arrow to structured data
    ax.annotate("", xy=(8.6, 2.5), xytext=(7.6, 2.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    for i, (k, v, c) in enumerate([("cardiomegaly", "yes", DEEPPINK), ("effusion", "yes", TURQUOISE),
                                   ("pneumothorax", "no", BLUEVIOLET)]):
        y = 3.4 - i * 0.85
        ax.add_patch(FancyBboxPatch((8.8, y - 0.3), 2.8, 0.62, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=c, edgecolor="none"))
        ax.text(9.0, y, f"{k}: {v}", fontsize=12, fontweight="bold", color=txt_on(c),
                family="Geist Mono", va="center")
    figtitle(fig, "Every scan comes with text: the radiology report")
    fig.text(0.5, -0.02, "Free-text the radiologist wrote. An LLM can turn it into structured "
             "yes/no findings a model can use.", ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_report.png")


# --------------------------------------------------------------------------- #
# LLM 1: tokenization
# --------------------------------------------------------------------------- #
def fig_tokenization():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    toks = ["The", "heart", "is", "en", "larged", "."]
    x = 0.5
    for t in toks:
        w = 0.55 + 0.18 * len(t)
        ax.add_patch(FancyBboxPatch((x, 2.5), w, 0.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=TURQUOISE, edgecolor="white", lw=1.5))
        ax.text(x + w / 2, 2.9, t, ha="center", va="center", fontsize=13, color=txt_on(TURQUOISE),
                fontweight="bold", family="Geist Mono")
        ax.annotate("", xy=(x + w / 2, 1.7), xytext=(x + w / 2, 2.4),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
        ax.text(x + w / 2, 1.3, str(np.random.RandomState(len(t)).randint(1000, 9999)),
                ha="center", fontsize=11, color=BLUEVIOLET, family="Geist Mono")
        x += w + 0.25
    ax.text(0.5, 0.6, "text  ->  tokens (word-pieces)  ->  numbers", fontsize=12, color=INK, family="Geist Mono")
    figtitle(fig, "Tokenization: text becomes numbers, just like an image")
    save(fig, "d2_tokenization.png")


# --------------------------------------------------------------------------- #
# LLM 2: attention
# --------------------------------------------------------------------------- #
def fig_attention():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    words = ["the", "heart", "is", "enlarged"]
    weights = [0.1, 0.9, 0.2, 0.8]
    x = 1.5
    for wd, wt in zip(words, weights):
        size = 0.55 + wt * 0.85   # circle size = attention weight
        ax.add_patch(Circle((x, 2.9), size / 2, color=DEEPPINK, alpha=0.3 + 0.7 * wt))
        # word label sits BELOW the blob, so long words never overflow the circle
        ax.text(x, 1.75, wd, ha="center", va="center", fontsize=14,
                color=INK, family="Geist Mono", fontweight="bold")
        x += 2.7
    ax.text(6.5, 0.8, 'predicting the finding: attention focuses on "heart" and "enlarged"',
            ha="center", fontsize=12, color=INK, family="Geist Mono")
    figtitle(fig, "Attention: the model decides which words matter")
    fig.text(0.5, -0.02, "Bigger, bolder = more attention. The model learns that "
             '"heart" and "enlarged" carry the signal, "the" and "is" do not.',
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_attention.png")


# --------------------------------------------------------------------------- #
# LLM 3: hallucination
# --------------------------------------------------------------------------- #
def fig_hallucination():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    ax.add_patch(FancyBboxPatch((0.4, 1.2), 5.2, 2.4, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor="white", edgecolor="#DDD", lw=1.5))
    ax.text(0.7, 3.2, "you ask:", fontsize=11, color=MUTED, family="Geist Mono")
    ax.text(0.7, 2.7, "What medication did this\npatient take?", fontsize=13, color=INK, va="top")
    ax.text(0.7, 1.5, "(the report never says)", fontsize=10, color=MUTED, style="italic")
    ax.annotate("", xy=(6.4, 2.4), xytext=(5.7, 2.4), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    ax.add_patch(FancyBboxPatch((6.6, 1.2), 5.0, 2.4, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=DEEPPINK, edgecolor="none"))
    ax.text(6.9, 3.2, "it confidently answers:", fontsize=11, color="white", family="Geist Mono")
    ax.text(6.9, 2.6, '"The patient was given\n10mg of lisinopril daily."', fontsize=13,
            color="white", va="top", fontweight="bold")
    ax.text(6.9, 1.5, "...completely made up.", fontsize=11, color="white", style="italic")
    figtitle(fig, "Hallucination: confident, fluent, and wrong")
    fig.text(0.5, -0.02, "An LLM will fill a gap with plausible fiction. In medicine a confident "
             "wrong answer is the dangerous kind. Always verify.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_hallucination.png")


# --------------------------------------------------------------------------- #
# STACK 1: three signals
# --------------------------------------------------------------------------- #
def fig_three_signals():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    sigs = [("IMAGE", "the chest X-ray\npixels", TURQUOISE),
            ("TEXT", "the radiology\nreport", DEEPPINK),
            ("DEMOGRAPHICS", "age, sex,\nsmoking history", AMBER)]
    for i, (name, desc, c) in enumerate(sigs):
        x = 0.5 + i * 2.7
        ax.add_patch(FancyBboxPatch((x, 1.6), 2.3, 2.0, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + 1.15, 2.9, name, ha="center", fontsize=14, fontweight="bold",
                color=txt_on(c), family="Geist Mono")
        ax.text(x + 1.15, 2.1, desc, ha="center", va="center", fontsize=11, color=txt_on(c))
        ax.annotate("", xy=(9.2, 2.6), xytext=(x + 2.35, 2.6),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.add_patch(FancyBboxPatch((9.3, 1.9), 2.3, 1.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=BLUEVIOLET, edgecolor="none"))
    ax.text(10.45, 2.6, "one\npatient", ha="center", va="center", fontsize=14, fontweight="bold",
            color="white", family="Geist Mono")
    figtitle(fig, "Three signals about the same patient")
    save(fig, "d2_three_signals.png")


# --------------------------------------------------------------------------- #
# STACK 2: the image's vote (late fusion / stacking)
# --------------------------------------------------------------------------- #
def fig_stacking():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    # the X-ray
    ax.imshow(Image.open(REALIMG / "cxr_normal.png").convert("L"), cmap="gray",
              extent=[0.4, 2.6, 1.4, 3.6], aspect="auto", zorder=1)
    ax.text(1.5, 1.05, "the X-ray", ha="center", fontsize=11, color=MUTED)
    ax.annotate("", xy=(4.0, 2.5), xytext=(2.8, 2.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    # the trained image model (a box, like Day 1)
    ax.add_patch(FancyBboxPatch((4.0, 1.7), 3.4, 1.7, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=BLUEVIOLET, edgecolor="none"))
    ax.text(5.7, 2.9, "image model", ha="center", fontsize=14, fontweight="bold",
            color="white", family="Geist Mono")
    ax.text(5.7, 2.2, "transfer learning,\nlike Day 1", ha="center", va="center",
            fontsize=11, color="white")
    ax.annotate("", xy=(9.0, 2.5), xytext=(7.6, 2.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    # its single output: one probability
    ax.add_patch(FancyBboxPatch((9.1, 1.85), 2.5, 1.3, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=TURQUOISE, edgecolor="none"))
    ax.text(10.35, 2.72, "img_pred", ha="center", fontsize=13, fontweight="bold",
            color=txt_on(TURQUOISE), family="Geist Mono")
    ax.text(10.35, 2.18, "0.83", ha="center", fontsize=20, fontweight="bold",
            color=txt_on(TURQUOISE), family="Geist Mono")
    ax.text(10.35, 1.5, "one number", ha="center", fontsize=10, color=MUTED, style="italic")
    figtitle(fig, "Late fusion: the image model casts one vote")
    fig.text(0.5, -0.02, "Instead of 100 handcrafted numbers or a 512-d embedding, we feed the table the "
             "image model's own probability. Each modality votes; TabPFN combines the votes.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_stacking.png")


# --------------------------------------------------------------------------- #
# STACK 3: everything becomes a tabular row
# --------------------------------------------------------------------------- #
def fig_tabular_row():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    groups = [("image vote", 1, TURQUOISE), ("text features", 5, DEEPPINK),
              ("demographics", 3, AMBER)]
    x = 0.5
    for name, n, c in groups:
        for j in range(n):
            ax.add_patch(Rectangle((x, 2.0), 0.85, 0.85, facecolor=c, edgecolor="white", lw=2))
            x += 0.9
        ax.text(x - n * 0.45 - 0.45 * (n - 1) / n, 1.6, name, ha="center", fontsize=11,
                color=c if c != AMBER else "#B8860B", family="Geist Mono")
    ax.add_patch(Rectangle((x + 0.2, 2.0), 0.85, 0.85, facecolor=INK, edgecolor="white", lw=2))
    ax.text(x + 0.62, 2.42, "y", ha="center", va="center", fontsize=16, color="white",
            fontweight="bold", family="Geist Mono")
    ax.text(x + 0.62, 1.6, "label", ha="center", fontsize=11, color=INK, family="Geist Mono")
    figtitle(fig, "Everything becomes one tabular row")
    fig.text(0.5, -0.02, "Image -> numbers. Text -> numbers. Demographics -> numbers. Stack them "
             "side by side: one row per patient, then predict.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_tabular_row.png")


# --------------------------------------------------------------------------- #
# STACK 4: TabPFN
# --------------------------------------------------------------------------- #
def fig_tabpfn():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    ax.add_patch(FancyBboxPatch((0.5, 1.3), 3.2, 2.0, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=BLUEVIOLET, edgecolor="none"))
    ax.text(2.1, 2.7, "TabPFN", ha="center", fontsize=20, fontweight="bold", color="white", family="Geist Mono")
    ax.text(2.1, 2.0, "pretrained on millions\nof synthetic tables", ha="center", va="center",
            fontsize=11, color="white")
    steps = [("fit", "it just looks at\nyour examples", TURQUOISE),
             ("predict", "answers in\nseconds", AMBER)]
    for i, (name, desc, c) in enumerate(steps):
        x = 4.6 + i * 3.6
        ax.add_patch(FancyBboxPatch((x, 1.5), 3.0, 1.6, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        ax.text(x + 1.5, 2.6, name, ha="center", fontsize=16, fontweight="bold", color=txt_on(c), family="Geist Mono")
        ax.text(x + 1.5, 1.95, desc, ha="center", va="center", fontsize=11, color=txt_on(c))
    figtitle(fig, "TabPFN: a foundation model for tables")
    fig.text(0.5, -0.02, "No training loop, no tuning. The 2020s way to model a table: "
             "fit, then predict, in seconds.", ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_tabpfn.png")


# --------------------------------------------------------------------------- #
# RESULT: multimodal vs image-only
# --------------------------------------------------------------------------- #
def fig_result():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ["image +\ndemographics", "+ text\n(multimodal)"]
    accs = [0.72, 0.98]
    colors = [TURQUOISE, DEEPPINK]
    b = ax.bar(range(2), accs, color=colors, width=0.5)
    for bar, a in zip(b, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.015, f"{a:.0%}",
                ha="center", fontsize=18, fontweight="bold", family="Geist Mono", color=INK)
    ax.set_xticks(range(2)); ax.set_xticklabels(bars, fontsize=12)
    ax.set_ylim(0.4, 1.0); ax.set_yticks([0.5, 0.75, 1.0]); ax.set_yticklabels(["50%", "75%", "100%"])
    ax.set_ylabel("accuracy")
    figtitle(fig, "Adding the text features looks amazing")
    fig.text(0.5, -0.02, "Multimodal jumps from 72% to 98%. A huge gain... almost too huge. Why?",
             ha="center", fontsize=10.5, color=MUTED, style="italic")
    save(fig, "d2_result.png")


# --------------------------------------------------------------------------- #
# EVAL: the leakage reveal
# --------------------------------------------------------------------------- #
def fig_leakage():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    ax.add_patch(FancyBboxPatch((0.4, 1.5), 6.6, 2.0, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor="white", edgecolor="#DDD", lw=1.5))
    ax.text(0.8, 3.0, "the report says:", fontsize=11, color=MUTED, family="Geist Mono")
    ax.text(0.8, 2.5, '"IMPRESSION: Cardiomegaly."', fontsize=15, color=INK, fontweight="bold", va="top")
    ax.text(0.8, 1.85, "we are trying to predict... cardiomegaly.", fontsize=12, color=DEEPPINK, va="top")
    ax.add_patch(FancyBboxPatch((7.3, 1.7), 4.3, 1.6, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=DEEPPINK, edgecolor="none"))
    ax.text(9.45, 2.8, "corr = 0.96", ha="center", fontsize=22, fontweight="bold", color="white", family="Geist Mono")
    ax.text(9.45, 2.1, "the text feature basically\nIS the answer", ha="center", va="center",
            fontsize=11, color="white")
    figtitle(fig, "Target leakage: the report names the diagnosis")
    fig.text(0.5, -0.02, "The model is not predicting, it is reading the answer off the report. "
             "Impressive numbers, meaningless test. This is everywhere in clinical ML.",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "d2_leakage.png")


# --------------------------------------------------------------------------- #
# RECAP: two paradigms
# --------------------------------------------------------------------------- #
def fig_two_paradigms():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4)
    cols = [("DAY 1", "End-to-end deep learning", "raw image -> one big neural net -> answer",
             "learns its own features", TURQUOISE),
            ("DAY 2", "Combine signals + foundation model", "image vote + text + data -> one table -> TabPFN",
             "you assemble the signals", DEEPPINK)]
    for i, (day, name, how, note, c) in enumerate(cols):
        x = 0.5 + i * 5.9
        ax.add_patch(FancyBboxPatch((x, 0.5), 5.4, 3.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="none"))
        tc = txt_on(c)
        ax.text(x + 0.35, 3.2, day, fontsize=12, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x + 0.35, 2.75, name, fontsize=15, fontweight="bold", color=tc, family="Geist Mono")
        ax.text(x + 0.35, 1.95, how, fontsize=11, color=tc, va="center")
        ax.text(x + 0.35, 1.0, note, fontsize=12, color=tc, style="italic", va="center")
    figtitle(fig, "Two paradigms, both in your toolkit now")
    save(fig, "d2_two_paradigms.png")


# --------------------------------------------------------------------------- #
# PANEL FIGURES: replace sparse 3-text-card slides with a 3-panel diagram.
# Each panel = colored header band + a small drawn icon + one line of body text.
# --------------------------------------------------------------------------- #
def _card(ax, label, color, body):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="#FBFAF6", edgecolor="#E3E0D6", lw=1.3))
    ax.add_patch(FancyBboxPatch((0.03, 0.80), 0.94, 0.17, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=color, edgecolor="none"))
    ax.text(0.5, 0.885, label, ha="center", va="center", fontsize=12, fontweight="bold",
            color=txt_on(color), family="Geist Mono")
    ax.text(0.5, 0.135, body, ha="center", va="center", fontsize=10.3, color=INK, wrap=True)


def _box(ax, x, y, w, h, color, text="", tc=None, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=color, edgecolor="none"))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or txt_on(color), family="Geist Mono", fontweight="bold")


def _arrow(ax, x0, y0, x1, y1, lw=2.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=lw))


def _panels(fname, title, builders, caption=""):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    for ax, b in zip(axes, builders):
        b(ax)
    figtitle(fig, title)
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, fname)


def fig_recap_days():
    def p1(ax):
        _card(ax, "DAY 1", TURQUOISE, "one image, one end-to-end\nnetwork, the eye")
        _box(ax, 0.16, 0.46, 0.20, 0.16, BLUEVIOLET, "eye")
        _arrow(ax, 0.40, 0.54, 0.56, 0.54)
        _box(ax, 0.60, 0.46, 0.24, 0.16, TURQUOISE, "network")
    def p2(ax):
        _card(ax, "DAY 2", DEEPPINK, "image + text + patient\ndata, on the chest")
        for k, (lab, c) in enumerate([("img", TURQUOISE), ("txt", DEEPPINK), ("age", AMBER)]):
            _box(ax, 0.10, 0.66 - k * 0.16, 0.20, 0.12, c, lab, fs=8)
            _arrow(ax, 0.32, 0.72 - k * 0.16, 0.58, 0.54, lw=1.4)
        _box(ax, 0.60, 0.46, 0.26, 0.16, BLUEVIOLET, "one model")
    def p3(ax):
        _card(ax, "THE THROUGHLINE", AMBER, "medicine is multimodal:\nscan AND notes AND history")
        ax.text(0.5, 0.55, "+", ha="center", va="center", fontsize=40, color=DEEPPINK, fontweight="bold")
        for ang in (0.20, 0.50, 0.80):
            _arrow(ax, ang, 0.74, 0.5, 0.60, lw=1.4)
    _panels("d2_recap_days.png", "Yesterday one image; today the whole patient", [p1, p2, p3])


def fig_llm_does():
    def p1(ax):
        _card(ax, "THE OBJECTIVE", TURQUOISE, "predict the next\nword-piece, over and over")
        _box(ax, 0.08, 0.50, 0.40, 0.14, "#ECEAE3", "the heart is", tc=INK)
        _box(ax, 0.54, 0.50, 0.34, 0.14, DEEPPINK, "enlarged?")
    def p2(ax):
        _card(ax, "WHAT IT BUYS", AMBER, "summarize, extract, answer,\nstructure messy text")
        for k, t in enumerate(["summarize", "extract", "answer"]):
            ax.text(0.20, 0.68 - k * 0.13, "+", ha="center", va="center", fontsize=14,
                    color=AMBER, fontweight="bold")
            ax.text(0.28, 0.68 - k * 0.13, t, ha="left", va="center", fontsize=11,
                    color=INK, family="Geist Mono")
    def p3(ax):
        _card(ax, "THE CATCH", DEEPPINK, "plausible is not the\nsame as correct")
        ax.text(0.27, 0.55, "plausible", ha="center", va="center", fontsize=11, color=INK, family="Geist Mono")
        ax.text(0.5, 0.55, "≠", ha="center", va="center", fontsize=34, color=DEEPPINK, fontweight="bold")
        ax.text(0.74, 0.55, "true", ha="center", va="center", fontsize=11, color=INK, family="Geist Mono")
    _panels("d2_llm_does.png", "What a language model actually does", [p1, p2, p3])


def fig_oof():
    def p1(ax):
        _card(ax, "THE TRAP", DEEPPINK, "overconfident on data\nit trained on = cheating")
        _box(ax, 0.30, 0.46, 0.40, 0.18, DEEPPINK, "100%", fs=15)
        ax.annotate("", xy=(0.30, 0.55), xytext=(0.20, 0.70),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                    connectionstyle="arc3,rad=-0.5"))
    def p2(ax):
        _card(ax, "THE FIX", TURQUOISE, "score each patient with a\nmodel that never saw them")
        for k in range(5):
            c = "#ECEAE3" if k == 2 else TURQUOISE
            _box(ax, 0.10 + k * 0.16, 0.50, 0.13, 0.14, c, "test" if k == 2 else "", tc=INK, fs=8)
        ax.text(0.10 + 2 * 0.16 + 0.065, 0.42, "held out", ha="center", fontsize=8, color=MUTED, family="Geist Mono")
    def p3(ax):
        _card(ax, "WHY IT MATTERS", AMBER, "never let a model\ngrade itself")
        ax.add_patch(Circle((0.5, 0.55), 0.14, fill=False, edgecolor=DEEPPINK, lw=3))
        ax.plot([0.40, 0.60], [0.65, 0.45], color=DEEPPINK, lw=3)
        ax.text(0.5, 0.55, "self", ha="center", va="center", fontsize=10, color=INK, family="Geist Mono")
    _panels("d2_oof.png", "Out-of-fold: never let a model grade itself", [p1, p2, p3])


def fig_text_to_num():
    def p1(ax):
        _card(ax, "THE PROMPT", TURQUOISE, '"Read this report.\nReturn findings as JSON."')
        ax.add_patch(FancyBboxPatch((0.16, 0.44), 0.32, 0.22, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor="white", edgecolor="#CCC", lw=1.3))
        for yy in (0.60, 0.55, 0.50):
            ax.plot([0.20, 0.44], [yy, yy], color=MUTED, lw=1.2)
        _arrow(ax, 0.52, 0.55, 0.70, 0.55)
        ax.text(0.80, 0.55, "LLM", ha="center", va="center", fontsize=10, color=BLUEVIOLET, family="Geist Mono", fontweight="bold")
    def p2(ax):
        _card(ax, "THE OUTPUT", DEEPPINK, "a handful of yes/no flags,\nnow just numbers")
        ax.text(0.5, 0.55, "{ cardiomegaly: 1,\n  effusion: 1,\n  pneumothorax: 0 }",
                ha="center", va="center", fontsize=9.5, color=INK, family="Geist Mono")
    def p3(ax):
        _card(ax, "PRE-CACHED", AMBER, "load saved answers; one\nlive call is the demo")
        ax.add_patch(FancyBboxPatch((0.34, 0.46), 0.32, 0.20, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=AMBER, edgecolor="none"))
        ax.text(0.5, 0.56, "cache", ha="center", va="center", fontsize=11, color=txt_on(AMBER),
                family="Geist Mono", fontweight="bold")
    _panels("d2_text_to_num.png", "Text becomes numbers: the LLM extracts findings", [p1, p2, p3])


def fig_fair_test():
    def p1(ax):
        _card(ax, "NO PEEKING", DEEPPINK, "a feature that encodes the\nanswer is cheating")
        ax.add_patch(Circle((0.5, 0.55), 0.13, fill=False, edgecolor=DEEPPINK, lw=3))
        ax.add_patch(Circle((0.5, 0.55), 0.04, facecolor=DEEPPINK, edgecolor="none"))
        ax.plot([0.39, 0.61], [0.66, 0.44], color=DEEPPINK, lw=3)
        ax.text(0.5, 0.33, "label", ha="center", fontsize=9, color=MUTED, family="Geist Mono")
    def p2(ax):
        _card(ax, "INPUTS COME FIRST", TURQUOISE, "use signals that exist\nbefore the label is known")
        ax.plot([0.12, 0.88], [0.55, 0.55], color=MUTED, lw=1.5)
        _box(ax, 0.12, 0.58, 0.22, 0.10, TURQUOISE, "pixels", fs=8)
        _box(ax, 0.12, 0.42, 0.22, 0.10, AMBER, "demo", fs=8)
        _box(ax, 0.64, 0.49, 0.24, 0.12, DEEPPINK, "diagnosis", fs=8)
        ax.plot([0.5, 0.5], [0.45, 0.65], color=INK, lw=1.2, ls=":")
    def p3(ax):
        _card(ax, "REPORT HONESTLY", AMBER, "show leaked and de-leaked\nnumbers side by side")
        ax.add_patch(FancyBboxPatch((0.22, 0.40), 0.16, 0.30, boxstyle="square,pad=0", facecolor=DEEPPINK, edgecolor="none"))
        ax.add_patch(FancyBboxPatch((0.58, 0.40), 0.16, 0.18, boxstyle="square,pad=0", facecolor=TURQUOISE, edgecolor="none"))
        ax.text(0.30, 0.74, "98%", ha="center", fontsize=10, color=DEEPPINK, family="Geist Mono", fontweight="bold")
        ax.text(0.66, 0.62, "72%", ha="center", fontsize=10, color=INK, family="Geist Mono", fontweight="bold")
    _panels("d2_fair_test.png", "What a fair test looks like", [p1, p2, p3])


def fig_text_sensitive():
    def p1(ax):
        _card(ax, "HIDDEN IDENTIFIERS", TURQUOISE, "names, dates, places;\n'de-identified' often isn't")
        ax.add_patch(FancyBboxPatch((0.22, 0.42), 0.46, 0.26, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor="white", edgecolor="#CCC", lw=1.3))
        for k, yy in enumerate((0.62, 0.55, 0.48)):
            ax.plot([0.27, 0.63], [yy, yy], color=(DEEPPINK if k == 0 else MUTED), lw=1.6 if k == 0 else 1.0)
    def p2(ax):
        _card(ax, "THE MODEL CAN LEAK IT", DEEPPINK, "LLMs can repeat training\ntext verbatim")
        _box(ax, 0.30, 0.48, 0.40, 0.16, DEEPPINK, "“...notes...”", fs=11)
        ax.annotate("", xy=(0.70, 0.42), xytext=(0.30, 0.42),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, connectionstyle="arc3,rad=0.4"))
    def p3(ax):
        _card(ax, "MINIMIZE + PROTECT", AMBER, "use the least data; guard it\nlike the patient record it is")
        ax.add_patch(FancyBboxPatch((0.40, 0.46), 0.20, 0.16, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=AMBER, edgecolor="none"))
        ax.add_patch(Circle((0.5, 0.62), 0.07, fill=False, edgecolor=AMBER, lw=3.5))
        ax.text(0.5, 0.53, "lock", ha="center", va="center", fontsize=9, color=txt_on(AMBER), family="Geist Mono", fontweight="bold")
    _panels("d2_text_sensitive.png", "Clinical text is even more sensitive than pixels", [p1, p2, p3])


if __name__ == "__main__":
    fig_recap_days()
    fig_llm_does()
    fig_oof()
    fig_text_to_num()
    fig_fair_test()
    fig_text_sensitive()
    fig_cxr_findings()
    fig_report()
    fig_tokenization()
    fig_attention()
    fig_hallucination()
    fig_three_signals()
    fig_stacking()
    fig_tabular_row()
    fig_tabpfn()
    fig_result()
    fig_leakage()
    fig_two_paradigms()
    print("Day 2 figures done")
