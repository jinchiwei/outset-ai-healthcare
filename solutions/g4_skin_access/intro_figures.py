"""G4 intro/background figures -- bespoke redraws of the skin-tone-bias literature (attributed).

The equity lens: skin-cancer AI is pitched to reach people with little dermatology access, but
the data it learns from is skewed. These three background figures set that up for HS students.

  dataset_skew        -- most training images are light skin; almost none are dark
                         (adapted from Wen 2022 / Groh 2021 -- Fitzpatrick 17k)
  accuracy_by_group   -- ONE model, very different accuracy by skin group; the "average" hides it
                         (adapted from Daneshjou 2022 / Groh 2021)
  proxy_gone_wrong    -- why a fair-looking metric can still be unfair: the algorithm optimized a
                         proxy (cost / "looks typical"), not the thing we cared about (who is sick)
                         (concept from Obermeyer 2019)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
np = sf.np
plt = sf.plt

# A light -> dark Fitzpatrick skin-tone ramp (types I..VI). Here the COLOR *is* the data,
# so we use real skin-tone swatches instead of the brand palette -- the visual and the label agree.
FITZ = ["#F1DEC6", "#E7C39C", "#D3A273", "#B67C4E", "#8B5A34", "#573421"]
FITZ_LABELS = ["I", "II", "III", "IV", "V", "VI"]


def dataset_skew():
    # Illustrative image counts by Fitzpatrick type: tall on the light end, almost nothing on the dark.
    # Numbers echo the finding in Wen 2022 (most public skin images are light skin) and Groh 2021.
    counts = [2100, 3400, 2800, 900, 260, 70]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bars = ax.bar(FITZ_LABELS, counts, color=FITZ, edgecolor=sf.INK, linewidth=0.6)
    # annotate each bar with its count so the drop-off is unmistakable
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 60, f"{c:,}", ha="center",
                fontsize=9, family="Geist Mono", color=sf.INK)
    # bracket the dark-skin end to name the gap
    ax.annotate("", xy=(5.4, 500), xytext=(3.6, 500),
                arrowprops=dict(arrowstyle="-", color=sf.DEEPPINK, lw=1.5))
    ax.text(4.5, 1250, "few dark-skin\nimages", ha="center", fontsize=10,
            color=sf.DEEPPINK, family="Geist Mono")
    ax.set_xlabel("Fitzpatrick skin type  (I = lightest  ->  VI = darkest)")
    ax.set_ylabel("training images")
    ax.set_ylim(0, 3800)
    sf.title(ax, "The AI mostly practiced on light skin")
    ax.text(2.5, -0.30, "adapted from Wen et al. 2022 / Groh et al. 2021",
            ha="center", fontsize=9, style="italic", color=sf.MUTED,
            transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_dataset_skew")
    sf.save_raw(sf.pd.DataFrame({"fitzpatrick": FITZ_LABELS, "n_images": counts}),
                HERE, "intro_dataset_skew")


def accuracy_by_group():
    # ONE model, evaluated on two skin groups. Illustrative accuracies in the spirit of Daneshjou 2022,
    # which found top models drop sharply on darker skin. The dashed line is the single "average"
    # number a report would show -- it sits between the groups and hides the gap.
    groups = ["light skin\n(types I-III)", "dark skin\n(types V-VI)"]
    acc = [0.88, 0.63]
    pts = [FITZ[1], FITZ[4]]                     # tie each point's color to its skin group
    overall = 0.85                              # weighted toward light skin (there are far more of them)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = [1, 0]
    # dumbbell: a connecting line then the two colored endpoints
    ax.plot([acc[1], acc[0]], [0, 1], color="#C7C2B4", lw=3, zorder=1)
    for xi, yi, col in zip(acc, y, pts):
        ax.scatter([xi], [yi], s=520, color=col, edgecolor=sf.INK, linewidth=1.0, zorder=3)
        ax.text(xi, yi + 0.16, f"{xi:.0%}", ha="center", fontsize=13,
                family="Geist Mono", fontweight="bold", color=sf.INK)
    # the deceptive "overall average" line
    ax.axvline(overall, color=sf.DEEPPINK, ls="--", lw=2, zorder=2)
    ax.text(overall - 0.015, -0.32, f"reported average {overall:.0%}",
            va="bottom", ha="right", fontsize=10, color=sf.DEEPPINK, family="Geist Mono")
    ax.set_yticks(y); ax.set_yticklabels(groups, fontsize=11)
    ax.set_xlim(0.5, 1.0); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("accuracy on held-out test images")
    sf.title(ax, "Same model -- one number hides who it fails")
    ax.text(0.75, -0.34, "adapted from Daneshjou et al. 2022 / Groh et al. 2021",
            ha="center", fontsize=9, style="italic", color=sf.MUTED,
            transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_accuracy_by_group")
    sf.save_raw(sf.pd.DataFrame({
        "group": ["light skin (I-III)", "dark skin (V-VI)", "reported average"],
        "accuracy": [acc[0], acc[1], overall]}),
        HERE, "intro_accuracy_by_group")


def proxy_gone_wrong():
    # A label-driven schematic (no data) teaching WHY a fair-looking metric can be unfair:
    # the algorithm scored well on a proxy it optimized, not on the thing we actually cared about.
    fig, ax = plt.subplots(figsize=(8.2, 4.0)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, face, edge, text, tcol, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tcol, family="Geist Mono")

    # top track: what we WANTED
    box(0.3, 4.2, 3.0, 1.2, "#EAF7F5", sf.TURQUOISE, "What we WANTED\nto predict", sf.INK)
    box(6.7, 4.2, 3.0, 1.2, "#EAF7F5", sf.TURQUOISE, "who is actually\nsick", sf.INK)
    ax.add_patch(FancyArrowPatch((3.4, 4.8), (6.6, 4.8), arrowstyle="-|>",
                                 mutation_scale=20, color=sf.TURQUOISE, lw=2.2))
    # bottom track: what the algorithm ACTUALLY did
    box(0.3, 0.6, 3.0, 1.2, "#FDE7F1", sf.DEEPPINK, "What the algorithm\nACTUALLY learned", sf.INK)
    box(6.7, 0.6, 3.0, 1.2, "#FDE7F1", sf.DEEPPINK, "who costs money /\nwho looks 'typical'", sf.INK)
    ax.add_patch(FancyArrowPatch((3.4, 1.2), (6.6, 1.2), arrowstyle="-|>",
                                 mutation_scale=20, color=sf.DEEPPINK, lw=2.2))
    # the gap between the two tracks -- where bias sneaks in (in the clear left gutter)
    ax.annotate("", xy=(1.8, 4.0), xytext=(1.8, 2.0),
                arrowprops=dict(arrowstyle="<->", color=sf.MUTED, lw=1.6))
    box(3.55, 2.55, 2.9, 0.9, "#F0C840", sf.AMBER, "the GAP = bias", sf.INK, fs=11)
    ax.text(5.0, 5.75, "A model can score great on the wrong target", ha="center",
            fontsize=12.5, family="Geist Mono", fontweight="bold", color=sf.INK)
    ax.text(5.0, 0.12, "concept from Obermeyer et al. 2019", ha="center",
            fontsize=9, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_proxy_gone_wrong")
    sf.save_raw({
        "wanted": ["what we wanted to predict", "who is actually sick"],
        "actual": ["what the algorithm actually learned", "who costs money / who looks typical"],
        "gap": "bias sneaks in between the goal and the proxy"},
        HERE, "intro_proxy_gone_wrong")


if __name__ == "__main__":
    dataset_skew(); accuracy_by_group(); proxy_gone_wrong()
    print("G4 intro figures done")
