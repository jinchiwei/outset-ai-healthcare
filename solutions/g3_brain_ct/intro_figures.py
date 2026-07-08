"""G3 intro/background figures -- bespoke redraws of the head-CT + datasheets literature (attributed).

  datasheet_checklist  -- what a dataset SHOULD record vs. what this one actually has
                          (concept from Gebru et al. 2021 / Tripathi et al. 2023)
  triage_queue         -- how a hemorrhage-triage AI reorders the radiologist's reading queue
                          (adapted from Chilamkurthy et al. 2018 / Seyam et al. 2022)
  unauditable_bias     -- you cannot audit a bias the data never records: empty bars for the
                          groups with no metadata (concept from Seyyed-Kalantari 2021 / Yang 2024)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
np = sf.np
plt = sf.plt


def _checkbox(ax, x, y, checked, color):
    """Draw a small square checkbox at (x, y); filled + check-mark if `checked`, else a greyed box."""
    s = 0.42                                                        # box side length (data units)
    if checked:
        ax.add_patch(Rectangle((x, y), s, s, facecolor=color, edgecolor=color, lw=1.4))
        ax.plot([x + 0.08, x + 0.18, x + 0.35], [y + 0.20, y + 0.09, y + 0.33],
                color=sf.CANVAS, lw=2.2, solid_capstyle="round")   # a check-mark in canvas color
    else:
        ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor="#B8B6AC",
                               lw=1.4, linestyle=(0, (3, 2))))      # empty, dashed = "not recorded"


def datasheet_checklist():
    # Left = the questions a proper datasheet asks; right = what our brain-CT set actually ships.
    asks = ["Age", "Sex", "Race / ethnicity", "Scanner / site", "Label source", "Patient consent"]
    has = [("CT image", True), ("Stroke / normal label", True), ("Age", False), ("Sex", False),
           ("Race / ethnicity", False), ("Scanner / site", False)]
    fig, ax = plt.subplots(figsize=(9.2, 4.8)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)

    ax.text(3.0, 7.4, "What a datasheet asks for", ha="center", fontsize=12.5,
            family="Geist Mono", fontweight="bold", color=sf.TURQUOISE)
    ax.text(9.0, 7.4, "What our brain-CT dataset has", ha="center", fontsize=12.5,
            family="Geist Mono", fontweight="bold", color=sf.DEEPPINK)

    for i, label in enumerate(asks):                               # left column: every box checked
        y = 6.3 - i * 0.85
        _checkbox(ax, 0.6, y, True, sf.TURQUOISE)
        ax.text(1.25, y + 0.09, label, fontsize=11.5, color=sf.INK)
    for i, (label, ok) in enumerate(has):                          # right column: only image + label
        y = 6.3 - i * 0.85
        _checkbox(ax, 6.6, y, ok, sf.DEEPPINK)
        ax.text(7.25, y + 0.09, label, fontsize=11.5,
                color=sf.INK if ok else sf.MUTED)

    ax.axvline(6.0, color="#D8D6CC", lw=1.2)                        # divider between the two columns
    ax.text(9.0, 0.55, "Four of the six questions have no answer -- so we cannot check who the model fails.",
            ha="center", fontsize=9.5, style="italic", color=sf.MUTED)
    ax.text(6.0, -0.15, "concept from Gebru et al. 2021 (Datasheets for Datasets) / Tripathi et al. 2023",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_datasheet_checklist")
    sf.save_raw(sf.pd.DataFrame({"field": [h[0] for h in has], "recorded": [h[1] for h in has]}),
                HERE, "intro_datasheet_checklist")


def triage_queue():
    # A stack of scans in ARRIVAL order (left) vs. AI-SORTED order (right); the bleed jumps to the top.
    rng = np.random.default_rng(1)
    arrival = ["normal", "normal", "bleed", "normal", "normal", "bleed", "normal"]
    order = list(range(len(arrival)))
    # AI-sorted: bleeds first (keep their arrival order), then normals.
    ai = sorted(order, key=lambda i: (arrival[i] != "bleed", i))
    fig, ax = plt.subplots(figsize=(8.6, 5.0)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 9)

    def draw_stack(x0, seq, title, sorted_flag):
        ax.text(x0 + 1.0, 8.4, title, ha="center", fontsize=12, family="Geist Mono",
                fontweight="bold", color=sf.INK)
        for row, i in enumerate(seq):
            y = 7.1 - row * 1.0
            is_bleed = arrival[i] == "bleed"
            face = sf.DEEPPINK if is_bleed else "#D8D6CC"
            ax.add_patch(FancyBboxPatch((x0, y), 2.0, 0.78, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=face, edgecolor=sf.INK, lw=0.8, alpha=0.95))
            txt = "BLEED" if is_bleed else "normal"
            ax.text(x0 + 1.0, y + 0.39, txt, ha="center", va="center", fontsize=9,
                    family="Geist Mono", color=sf.CANVAS if is_bleed else sf.INK)
            if sorted_flag and row == 0:                           # highlight the top of the sorted queue
                ax.add_patch(FancyArrowPatch((x0 - 0.7, y + 0.39), (x0 - 0.05, y + 0.39),
                                             arrowstyle="-|>", mutation_scale=18, color=sf.AMBER, lw=2.4))
                ax.text(x0 - 0.75, y + 0.39, "read\nfirst", ha="right", va="center", fontsize=9,
                        family="Geist Mono", fontweight="bold", color=sf.AMBER)

    draw_stack(2.4, order, "Arrival order", False)
    draw_stack(8.0, ai, "After AI triage", True)
    ax.add_patch(FancyArrowPatch((5.0, 4.4), (7.4, 4.4), arrowstyle="-|>", mutation_scale=22,
                                 color=sf.TURQUOISE, lw=2.4))
    ax.text(6.2, 4.9, "AI flags likely\nbleeds", ha="center", fontsize=9.5, family="Geist Mono",
            color=sf.TURQUOISE)
    ax.text(6.0, 0.35, "Real triage tools move likely bleeds to the top of the list, cutting time-to-treatment.",
            ha="center", fontsize=9.5, style="italic", color=sf.MUTED)
    ax.text(6.0, -0.25, "adapted from Chilamkurthy et al. 2018 / Seyam et al. 2022",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_triage_queue")
    sf.save_raw(sf.pd.DataFrame({"arrival_row": order, "label": arrival,
                                 "ai_sorted_position": [ai.index(i) for i in order]}),
                HERE, "intro_triage_queue")


def unauditable_bias():
    # A fairness bar chart where the groups the dataset never records are blank outlines: unmeasurable.
    groups = ["overall", "younger", "older", "female", "male", "scanner B"]
    acc = [0.71, None, None, None, None, None]                     # only "overall" is measurable here
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(groups))
    for i, (g, a) in enumerate(zip(groups, acc)):
        if a is not None:
            ax.bar(i, a, width=0.62, color=sf.GOLD, edgecolor=sf.INK, linewidth=0.8)
            ax.text(i, a + 0.03, f"{a:.2f}", ha="center", fontsize=10, family="Geist Mono", color=sf.INK)
        else:
            ax.bar(i, 0.85, width=0.62, facecolor="none", edgecolor=sf.DEEPPINK, linewidth=1.6,
                   linestyle=(0, (4, 3)))                          # dashed empty = "no data"
            ax.text(i, 0.42, "no data\ncannot\nmeasure", ha="center", va="center", fontsize=8.5,
                    family="Geist Mono", color=sf.DEEPPINK)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylabel("accuracy for this group"); ax.set_ylim(0, 1.02)
    ax.axhline(0.71, ls="--", color=sf.MUTED, lw=1)
    sf.title(ax, "You can't audit a bias you can't see")
    ax.text(len(groups) / 2 - 0.5, -0.30,
            "If age / sex / race / scanner were recorded, we could fill these bars in and compare.",
            ha="center", fontsize=9, style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    ax.text(len(groups) / 2 - 0.5, -0.40, "concept from Seyyed-Kalantari et al. 2021 / Yang et al. 2024",
            ha="center", fontsize=8.5, style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_unauditable_bias")
    sf.save_raw(sf.pd.DataFrame({"group": groups, "accuracy": [a if a is not None else np.nan for a in acc],
                                 "measurable": [a is not None for a in acc]}),
                HERE, "intro_unauditable_bias")


if __name__ == "__main__":
    datasheet_checklist(); triage_queue(); unauditable_bias()
    print("G3 intro figures done")
