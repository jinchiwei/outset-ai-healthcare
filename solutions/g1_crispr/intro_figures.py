"""G1 intro/background figures -- bespoke redraws of foundational CRISPR papers (attributed).

  guide_anatomy       -- what the 20 letters ARE: guide + PAM + cut site   (adapted from Hsu 2014)
  seed_importance     -- not all positions count equally; the seed near the PAM matters most
                         (adapted from Zheng 2017 / Doench 2016)
  predicted_vs_measured -- a model can capture signal without being perfect (adapted from Doench 2016)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
np = sf.np
plt = sf.plt


def guide_anatomy():
    fig, ax = plt.subplots(figsize=(8.5, 3.6)); ax.axis("off")
    ax.set_xlim(0, 20); ax.set_ylim(0, 6)
    # two DNA strands
    ax.add_patch(Rectangle((0.5, 3.6), 18.5, 0.55, color="#D8D6CC"))
    ax.add_patch(Rectangle((0.5, 2.4), 18.5, 0.55, color="#D8D6CC"))
    # protospacer (the 20-nt guide target)
    ax.add_patch(Rectangle((5, 3.6), 9, 0.55, color=sf.TURQUOISE))
    ax.add_patch(Rectangle((5, 2.4), 9, 0.55, color=sf.TURQUOISE, alpha=0.5))
    ax.text(9.5, 4.55, "the 20-letter guide target (protospacer)", ha="center", fontsize=11,
            family="Geist Mono", color=sf.INK)
    # PAM
    ax.add_patch(Rectangle((14, 3.6), 1.6, 0.55, color=sf.DEEPPINK))
    ax.text(14.8, 4.35, "PAM\n(NGG)", ha="center", fontsize=9, family="Geist Mono", color=sf.DEEPPINK)
    # cut site ~3bp upstream of PAM
    ax.add_patch(FancyArrowPatch((12.5, 5.2), (12.5, 4.25), arrowstyle="-|>", mutation_scale=18,
                                 color=sf.AMBER, lw=2))
    ax.text(12.5, 5.45, "Cas9 cuts here", ha="center", fontsize=10, family="Geist Mono", color=sf.INK)
    # seed bracket
    ax.annotate("", xy=(14, 2.1), xytext=(10.5, 2.1),
                arrowprops=dict(arrowstyle="-", color=sf.BLUEVIOLET, lw=1.5))
    ax.text(12.25, 1.55, "the 'seed': letters next to the PAM matter most", ha="center", fontsize=10,
            color=sf.BLUEVIOLET)
    ax.text(9.5, 0.7, "adapted from Hsu, Lander & Zhang 2014", ha="center", fontsize=9,
            style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_guide_anatomy")


def seed_importance():
    pos = np.arange(1, 21)
    imp = 0.25 + 0.75 / (1 + np.exp(-(pos - 14) / 1.6))          # rises toward the PAM (position 20)
    colors = [sf.DEEPPINK if p >= 13 else sf.TURQUOISE for p in pos]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(pos, imp, color=colors, edgecolor=sf.INK, linewidth=0.4)
    ax.annotate("", xy=(20.4, 1.06), xytext=(12.6, 1.06),
                arrowprops=dict(arrowstyle="-", color=sf.DEEPPINK, lw=1.5))
    ax.text(16.5, 1.12, "seed / core region", ha="center", fontsize=10, color=sf.DEEPPINK)
    ax.set_xlabel("position along the 20-letter guide  (20 = next to the PAM / cut site)")
    ax.set_ylabel("how much this position matters")
    ax.set_ylim(0, 1.25); ax.set_xticks([1, 5, 10, 15, 20])
    sf.title(ax, "Not all 20 letters count equally")
    ax.text(10.5, -0.42, "adapted from Zheng et al. 2017 / Doench et al. 2016", ha="center",
            fontsize=9, style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_seed_importance")
    sf.save_raw(sf.pd.DataFrame({"position": pos, "importance": imp}), HERE, "intro_seed_importance")


def predicted_vs_measured():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 220)
    y = np.clip(0.15 + 0.6 * x + rng.normal(0, 0.16, 220), 0, 1)     # real but loose trend
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(x, y, s=12, alpha=0.4, color=sf.TURQUOISE)
    m, b = np.polyfit(x, y, 1)
    ax.plot([0, 1], [b, m + b], color=sf.DEEPPINK, lw=2)
    ax.set_xlabel("model-predicted efficiency"); ax.set_ylabel("measured efficiency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    sf.title(ax, "The trend is real but loose")
    ax.text(0.5, -0.16, "adapted from Doench et al. 2016", ha="center", fontsize=9,
            style="italic", color=sf.MUTED, transform=ax.transAxes)
    sf.save(fig, HERE, "intro_predicted_vs_measured")


if __name__ == "__main__":
    guide_anatomy(); seed_importance(); predicted_vs_measured()
    print("G1 intro figures done")
