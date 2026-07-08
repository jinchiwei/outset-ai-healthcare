"""G2 intro/background figures -- bespoke redraws of foundational skin-cancer-AI papers (attributed).

  sensitivity_tradeoff -- the screening dial: sliding the decision threshold trades false alarms
                          for missed cancers                     (concept adapted from Trevethan 2017)
  cost_of_a_miss       -- a 2x2 of the two possible mistakes, and why they are NOT equally bad
                          (adapted from Esteva 2017; Haenssle 2018)
  transfer_learning    -- borrow a brain: reuse a net trained on everyday photos, fine-tune on
                          ~10,000 skin images                    (adapted from Kim 2022)

Theme "bone" via the shared solfig helper, so every PNG drops straight onto a slide.
"""
import sys                                                        # sys lets us extend the import path
from pathlib import Path                                          # Path builds file locations safely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))  # find the shared solfig helper
import solfig as sf                                               # brand palette + save/save_raw + theme "bone"
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch  # shapes for the schematics

HERE = Path(__file__).resolve().parent                           # this folder (figures land in ./figures)
np = sf.np                                                        # re-use solfig's numpy
plt = sf.plt                                                      # re-use solfig's pyplot (already themed)


def sensitivity_tradeoff():
    """Two overlapping score distributions + a movable threshold -- the core screening tension."""
    x = np.linspace(0, 10, 400)                                  # the model's "cancer score" axis, low -> high
    healthy = np.exp(-0.5 * ((x - 3.4) / 1.15) ** 2)             # harmless spots cluster at LOW scores
    cancer = np.exp(-0.5 * ((x - 6.6) / 1.15) ** 2)             # melanomas cluster at HIGH scores
    thr = 5.4                                                     # where we draw the "call it cancer" line

    fig, ax = plt.subplots(figsize=(8.2, 4.0))                   # one wide panel
    ax.plot(x, healthy, color=sf.TURQUOISE, lw=2)               # curve for healthy spots
    ax.plot(x, cancer, color=sf.DEEPPINK, lw=2)                 # curve for melanomas
    ax.fill_between(x, healthy, where=(x < thr), color=sf.TURQUOISE, alpha=0.18)  # healthy correctly cleared
    ax.fill_between(x, cancer, where=(x >= thr), color=sf.DEEPPINK, alpha=0.18)   # cancers correctly caught
    # the two mistakes, shaded darker so they stand out:
    ax.fill_between(x, cancer, where=(x < thr), color=sf.DEEPPINK, alpha=0.55)    # cancer BELOW the line = MISSED
    ax.fill_between(x, healthy, where=(x >= thr), color=sf.GOLD, alpha=0.60)      # healthy ABOVE the line = false alarm

    ax.axvline(thr, color=sf.INK, ls="--", lw=1.6)              # the decision threshold
    ax.text(thr, 1.12, "decision threshold", ha="center", fontsize=10, family="Geist Mono", color=sf.INK)
    ax.text(4.35, 0.14, "missed\nmelanomas", ha="center", fontsize=9, color=sf.DEEPPINK, family="Geist Mono")
    ax.text(6.35, 0.14, "false\nalarms", ha="center", fontsize=9, color="#B8860B", family="Geist Mono")
    ax.text(2.6, 0.92, "healthy spots", color=sf.TURQUOISE, fontsize=10, family="Geist Mono")
    ax.text(6.9, 0.92, "melanomas", color=sf.DEEPPINK, fontsize=10, family="Geist Mono")
    # arrow: slide the threshold LEFT to catch more cancers (higher sensitivity), at the cost of more false alarms
    ax.add_patch(FancyArrowPatch((thr - 0.15, 0.62), (thr - 1.7, 0.62), arrowstyle="-|>",
                                 mutation_scale=16, color=sf.BLUEVIOLET, lw=2))
    ax.text(thr - 0.95, 0.68, "slide left = catch more cancers", ha="center", fontsize=9,
            color=sf.BLUEVIOLET, family="Geist Mono")

    ax.set_xlabel("the model's cancer score   (low  ->  high)")  # x-axis meaning in plain words
    ax.set_yticks([])                                            # the height is just "how many spots"; hide ticks
    ax.set_ylim(0, 1.25); ax.set_xlim(0, 10)                     # leave headroom for labels
    sf.title(ax, "Screening tunes one dial: sensitivity vs specificity")  # brand title
    ax.text(5, -0.20, "concept adapted from Trevethan 2017", ha="center", fontsize=9,
            style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())  # attribution
    sf.save(fig, HERE, "intro_sensitivity_tradeoff")            # -> figures/intro_sensitivity_tradeoff.png
    sf.save_raw(sf.pd.DataFrame({"score": x, "healthy": healthy, "cancer": cancer}),
                HERE, "intro_sensitivity_tradeoff")             # persist the curves behind the figure


def cost_of_a_miss():
    """A 2x2 truth-vs-guess grid where one wrong cell is deadly and the other is merely annoying."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2)); ax.axis("off")   # blank canvas, no axes
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)                       # simple 10x10 drawing space

    cells = {                                                    # (label, subtitle, fill, text-color) per quadrant
        (1, 1): ("caught it", "correct -- refer on", sf.TURQUOISE, sf.INK),      # true mel, said mel
        (1, 0): ("MISSED CANCER", "the deadly mistake", sf.DEEPPINK, sf.WHITE_RGB if False else "white"),  # true mel, said not
        (0, 1): ("false alarm", "just a check-up visit", sf.GOLD, sf.INK),                # true not, said mel
        (0, 0): ("all clear", "correct", "#D8D6CC", sf.INK),                              # true not, said not
    }
    # grid geometry: columns = model says (mel / not), rows = truth (mel / not)
    xs = {1: 5.1, 0: 2.0}                                        # left x of the "says melanoma" / "says not" columns
    ys = {1: 5.1, 0: 2.0}                                        # bottom y of the "truly melanoma" / "truly not" rows
    w = h = 2.9                                                  # each cell is a square
    for (truth, guess), (label, sub, fill, tcol) in cells.items():
        x0, y0 = xs[guess], ys[truth]                           # place the cell
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fill, edgecolor=sf.INK, linewidth=1.2))  # the colored tile
        ax.text(x0 + w / 2, y0 + h / 2 + 0.35, label, ha="center", va="center", fontsize=12,
                family="Geist Mono", color=tcol, fontweight="bold")                    # big label
        ax.text(x0 + w / 2, y0 + h / 2 - 0.5, sub, ha="center", va="center", fontsize=9,
                color=tcol)                                                             # small subtitle

    # axis labels around the grid
    ax.text(xs[1] + w / 2, 8.35, "says: melanoma", ha="center", fontsize=9.5, family="Geist Mono", color=sf.INK)
    ax.text(xs[0] + w / 2, 8.35, "says: not", ha="center", fontsize=9.5, family="Geist Mono", color=sf.INK)
    ax.text(1.35, ys[1] + h / 2, "truly\nmelanoma", ha="center", va="center", rotation=90,
            fontsize=10, family="Geist Mono", color=sf.INK)
    ax.text(1.35, ys[0] + h / 2, "truly\nharmless", ha="center", va="center", rotation=90,
            fontsize=10, family="Geist Mono", color=sf.INK)
    ax.text(5, 9.4, "The two mistakes are not equally bad", ha="center", fontsize=14,
            family="Geist Mono", color=sf.INK, fontweight="bold")                        # title
    ax.text(5, 0.6, "adapted from Esteva 2017; Haenssle 2018", ha="center", fontsize=9,
            style="italic", color=sf.MUTED)                                              # attribution
    sf.save(fig, HERE, "intro_cost_of_a_miss")                  # -> figures/intro_cost_of_a_miss.png


def transfer_learning():
    """A left-to-right pipeline: pretrain on everyday photos -> fine-tune on skin -> melanoma vs not."""
    fig, ax = plt.subplots(figsize=(9.0, 3.4)); ax.axis("off")   # wide blank canvas
    ax.set_xlim(0, 18); ax.set_ylim(0, 6)                        # drawing space

    def box(x, w, color, top, sub, fill_alpha=0.16):
        """Draw one rounded stage box with a bold title line and a smaller subtitle line."""
        ax.add_patch(FancyBboxPatch((x, 1.9), w, 2.2, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    facecolor=color, edgecolor=color, linewidth=2, alpha=fill_alpha))  # tinted fill
        ax.add_patch(FancyBboxPatch((x, 1.9), w, 2.2, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    facecolor="none", edgecolor=color, linewidth=2))                    # crisp outline
        ax.text(x + w / 2, 3.35, top, ha="center", va="center", fontsize=10.5, family="Geist Mono",
                color=sf.INK, fontweight="bold")                # stage title
        ax.text(x + w / 2, 2.55, sub, ha="center", va="center", fontsize=8.5, color=sf.MUTED)  # stage detail

    box(0.4, 6.2, sf.TURQUOISE, "pre-trained on millions\nof everyday photos", "cats, cars, faces")
    box(7.6, 4.6, sf.DEEPPINK, "fine-tune on ~10,000\nskin images", "HAM10000 dermatoscopy")
    box(13.6, 4.0, sf.BLUEVIOLET, "melanoma\nvs. not", "a screening call")

    def arrow(x0, x1, label):
        """Draw an arrow between two stages, with an optional label placed ABOVE the boxes."""
        ax.add_patch(FancyArrowPatch((x0, 3.0), (x1, 3.0), arrowstyle="-|>", mutation_scale=18,
                                     color=sf.INK, lw=1.8))
        if label:                                               # label sits above the box tops (y=4.1) to avoid overlap
            ax.text((x0 + x1) / 2, 4.55, label, ha="center", va="center", fontsize=8,
                    family="Geist Mono", color=sf.BLUEVIOLET)

    arrow(6.7, 7.5, "reuse\nfeatures")                          # what carries over from photos to skin
    arrow(12.3, 13.5, "")                                        # into the final decision

    ax.text(9, 5.35, "Borrow a brain: transfer learning", ha="center", fontsize=14,
            family="Geist Mono", color=sf.INK, fontweight="bold")   # title
    ax.text(9, 0.85, "The reused knowledge is why a model can learn skin cancer from thousands, not millions, of photos",
            ha="center", fontsize=9, color=sf.INK)                  # the takeaway sentence
    ax.text(9, 0.25, "adapted from Kim et al. 2022", ha="center", fontsize=9,
            style="italic", color=sf.MUTED)                         # attribution
    sf.save(fig, HERE, "intro_transfer_learning")               # -> figures/intro_transfer_learning.png


if __name__ == "__main__":
    sensitivity_tradeoff()                                       # build the screening-dial figure
    cost_of_a_miss()                                             # build the 2x2 cost-of-a-miss figure
    transfer_learning()                                          # build the transfer-learning pipeline figure
    print("G2 intro figures done")                              # confirm success
