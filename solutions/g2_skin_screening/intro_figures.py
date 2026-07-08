"""G2 intro/background figures -- bespoke redraws that VISUALLY depict the clinical problem
of melanoma screening (all attributed). Theme "bone" via the shared solfig helper, so every
PNG drops straight onto a slide canvas.

  dermatoscope         -- the clinical thing: a dermatoscope examining a pigmented lesion, with the
                          7 HAM10000 lesion types labeled around it and MELANOMA highlighted
                          (lesion types from the HAM10000 dataset; Tschandl 2018)
  cost_of_a_miss       -- "why a miss is deadly": a tilted balance where a missed melanoma vastly
                          outweighs a false alarm -- the asymmetry that drives screening
                          (adapted from Esteva 2017; Haenssle 2018)
  esteva_2017          -- redraw of the landmark result: one network matched 21 board-certified
                          dermatologists on the sensitivity/specificity plane
                          (adapted from Esteva et al. 2017, Nature)
  sensitivity_tradeoff -- the screening dial: sliding the decision threshold trades false alarms for
                          missed cancers                          (concept adapted from Trevethan 2017)
  transfer_learning    -- borrow a brain: reuse a net trained on everyday photos, fine-tune on
                          ~10,000 skin images                     (adapted from Kim 2022)
"""
import sys                                                        # sys lets us extend the import path
from pathlib import Path                                          # Path builds file locations safely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))  # find the shared solfig helper
import solfig as sf                                               # brand palette + save/save_raw + theme "bone"
from matplotlib.patches import (FancyArrowPatch, Rectangle,       # shapes for the schematics
                                FancyBboxPatch, Circle, Polygon, Wedge)

HERE = Path(__file__).resolve().parent                           # this folder (figures land in ./figures)
np = sf.np                                                        # re-use solfig's numpy
plt = sf.plt                                                      # re-use solfig's pyplot (already themed)

SKIN = "#E8C6A8"                                                  # a warm skin tone for the lesion backdrop
LESION = "#5B3A2E"                                                # a dark irregular pigmented lesion


def dermatoscope():
    """A dermatoscope hovering over a pigmented skin lesion, ringed by the 7 HAM10000 lesion types."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4)); ax.axis("off")   # wide blank canvas
    ax.set_xlim(0, 18); ax.set_ylim(0, 11); ax.set_aspect("equal")  # equal aspect so circles stay round

    # --- the skin patch under examination (a rounded square of skin) ---
    ax.add_patch(FancyBboxPatch((6.2, 2.6), 5.6, 5.6, boxstyle="round,pad=0.02,rounding_size=0.5",
                                facecolor=SKIN, edgecolor="#C89B78", linewidth=1.5))  # skin backdrop
    # an irregular, asymmetric dark lesion (melanoma looks irregular -- the "ugly duckling")
    ang = np.linspace(0, 2 * np.pi, 13)                          # 12 vertices around a circle
    rad = 1.55 + 0.55 * np.sin(3.1 * ang) + 0.3 * np.cos(5 * ang)  # wobble the radius -> irregular border
    lx = 9.0 + rad * np.cos(ang); ly = 5.4 + rad * np.sin(ang)  # lesion outline coordinates
    ax.add_patch(Polygon(np.column_stack([lx, ly]), closed=True, facecolor=LESION,
                         edgecolor="#2E1C15", linewidth=1.5, zorder=3))              # the lesion blob
    # a couple of darker patches inside -> uneven color, another melanoma warning sign
    ax.add_patch(Circle((8.4, 5.8), 0.55, facecolor="#241009", edgecolor="none", alpha=0.8, zorder=4))
    ax.add_patch(Circle((9.7, 4.9), 0.42, facecolor="#3A2018", edgecolor="none", alpha=0.9, zorder=4))

    # --- the dermatoscope: a lit lens ring hovering over the lesion ---
    ax.add_patch(Circle((9.0, 5.4), 2.55, facecolor="none", edgecolor=sf.INK, linewidth=3.0, zorder=6))   # scope barrel
    ax.add_patch(Circle((9.0, 5.4), 2.30, facecolor="none", edgecolor=sf.TURQUOISE, linewidth=2.2, zorder=6))  # lit rim (polarized light)
    ax.add_patch(Circle((9.0, 5.4), 2.62, facecolor="none", edgecolor="#B8B4A6", linewidth=1.0, zorder=6))
    # the eyepiece / handle rising to the upper-right
    ax.add_patch(FancyBboxPatch((10.7, 6.9), 2.9, 1.0, boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=sf.INK, edgecolor="none", zorder=5))       # scope body
    ax.add_patch(FancyArrowPatch((10.9, 7.0), (11.3, 6.4), arrowstyle="-", color=sf.INK, lw=6, zorder=5))
    ax.text(12.15, 7.4, "dermatoscope", ha="center", va="center", fontsize=10.5, family="Geist Mono",
            color="white", zorder=7, fontweight="bold")          # label on the scope body
    # little light glints on the lens rim to say "illuminated"
    for a in np.linspace(0.2, 2 * np.pi, 9):
        ax.add_patch(Circle((9.0 + 2.30 * np.cos(a), 5.4 + 2.30 * np.sin(a)), 0.05,
                            facecolor=sf.AMBER, edgecolor="none", zorder=7))

    # --- the 7 HAM10000 lesion types arranged around the scene; melanoma highlighted ---
    types = [                                                    # (short label, is_the_dangerous_one)
        ("actinic keratoses", False), ("basal cell carcinoma", False),
        ("benign keratosis", False), ("dermatofibroma", False),
        ("MELANOMA", True), ("melanocytic nevi", False), ("vascular lesions", False),
    ]
    ys = np.linspace(8.6, 1.6, 7)                                # stack the 7 labels down the left edge
    for (name, danger), y in zip(types, ys):
        col = sf.DEEPPINK if danger else "#8A8578"              # melanoma pops in deeppink; others muted
        ax.add_patch(Circle((0.75, y), 0.16, facecolor=col, edgecolor="none"))       # a bullet dot
        ax.text(1.15, y, name, ha="left", va="center", fontsize=12 if danger else 11,
                family="Geist Mono", color=col, fontweight="bold" if danger else "normal")
    ax.text(0.55, 9.35, "the 7 lesion types:", ha="left", fontsize=10.5,
            family="Geist Mono", color=sf.INK, fontweight="bold")                     # section label

    # an arrow from the MELANOMA label to the lesion, saying "this is the one we must not miss"
    ax.add_patch(FancyArrowPatch((4.5, ys[4]), (7.1, 5.4), arrowstyle="-|>", mutation_scale=18,
                                 color=sf.DEEPPINK, lw=2.2, connectionstyle="arc3,rad=-0.25", zorder=8))
    ax.text(5.6, 3.55, "the deadly one", ha="center", fontsize=10, family="Geist Mono",
            color=sf.DEEPPINK, style="italic")

    ax.text(9.0, 10.55, "Screening a skin spot: is it melanoma?", ha="center", fontsize=15,
            family="Geist Mono", color=sf.INK, fontweight="bold")                     # title
    ax.text(12.9, 1.05, "lesion types from HAM10000 (Tschandl 2018)", ha="center", fontsize=9,
            style="italic", color=sf.MUTED)                                           # attribution
    sf.save(fig, HERE, "intro_dermatoscope")                     # -> figures/intro_dermatoscope.png


def cost_of_a_miss():
    """A tilted balance: a missed melanoma outweighs a false alarm. The asymmetry that drives screening."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2)); ax.axis("off")   # blank canvas
    ax.set_xlim(0, 18); ax.set_ylim(0, 11); ax.set_aspect("equal")

    # --- the balance: a fulcrum triangle with a beam tilted heavy toward the "missed cancer" pan ---
    fx, fy = 9.0, 4.2                                            # fulcrum apex position
    ax.add_patch(Polygon([(fx - 0.9, 1.9), (fx + 0.9, 1.9), (fx, fy)], closed=True,
                         facecolor=sf.INK, edgecolor="none"))     # the fulcrum triangle
    ax.add_patch(Rectangle((fx - 1.9, 1.55), 3.8, 0.35, facecolor="#3A3A44", edgecolor="none"))  # base
    theta = np.deg2rad(15)                                       # tilt: left pan (missed cancer) drops
    half = 5.6                                                   # half the beam length
    x0, y0 = fx - half * np.cos(theta), fy - half * np.sin(theta)   # left (heavy) end -> lower
    x1, y1 = fx + half * np.cos(theta), fy + half * np.sin(theta)   # right (light) end -> higher
    ax.plot([x0, x1], [y0, y1], color=sf.INK, lw=5, solid_capstyle="round", zorder=4)  # the beam
    ax.add_patch(Circle((fx, fy), 0.22, facecolor=sf.AMBER, edgecolor=sf.INK, lw=1.2, zorder=6))  # pivot

    # heavy pan: a MISSED MELANOMA (big deeppink disc)
    ax.add_patch(Circle((x0, y0), 1.7, facecolor=sf.DEEPPINK, edgecolor=sf.INK, lw=1.4, zorder=5))
    ax.text(x0, y0 + 0.35, "MISSED", ha="center", va="center", fontsize=13, family="Geist Mono",
            color="white", fontweight="bold", zorder=6)
    ax.text(x0, y0 - 0.45, "MELANOMA", ha="center", va="center", fontsize=13, family="Geist Mono",
            color="white", fontweight="bold", zorder=6)
    # light pan: a FALSE ALARM (small amber disc)
    ax.add_patch(Circle((x1, y1), 0.95, facecolor=sf.GOLD, edgecolor=sf.INK, lw=1.2, zorder=5))
    ax.text(x1, y1, "false\nalarm", ha="center", va="center", fontsize=10.5, family="Geist Mono",
            color=sf.INK, zorder=6)

    # consequence captions under each pan -- bbox auto-sizes to the text so nothing spills out
    ax.text(x0, 0.7, "cancer keeps growing -- can be fatal", ha="center", va="center",
            fontsize=10, color=sf.DEEPPINK, family="Geist Mono",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F4DCE6", edgecolor=sf.DEEPPINK, linewidth=1.2))
    ax.text(x1, 9.3, "just one extra check-up", ha="center", va="center",
            fontsize=10, color="#8A6D0B", family="Geist Mono",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F6EBCB", edgecolor="#B8860B", linewidth=1.2))

    ax.text(9.0, 10.55, "The two mistakes are not equal", ha="center", fontsize=15,
            family="Geist Mono", color=sf.INK, fontweight="bold")
    ax.text(9.0, 10.0, "so a screening tool leans toward catching cancers, even at the cost of false alarms",
            ha="center", fontsize=10.5, color=sf.INK)
    ax.text(9.0, -0.35, "adapted from Esteva 2017; Haenssle 2018", ha="center", fontsize=9,
            style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_cost_of_a_miss")                   # -> figures/intro_cost_of_a_miss.png


def esteva_2017():
    """Redraw the landmark Esteva result: a CNN matched 21 dermatologists on the sens/spec plane."""
    fig, ax = plt.subplots(figsize=(7.4, 5.4))                   # one panel with real axes
    # the CNN's operating curve (a smooth ROC in sensitivity vs specificity space)
    spec = np.linspace(1, 0, 200)                               # specificity axis, high -> low (x reversed)
    sens = 1 - (1 - spec) ** 2.1                                # a strong concave curve (good classifier)
    ax.plot(spec, sens, color=sf.TURQUOISE, lw=3, zorder=4, label="the network (CNN)")

    # 21 board-certified dermatologists, scattered as points mostly ON or just under the curve
    rng = np.random.default_rng(7)                              # fixed seed -> reproducible scatter
    dsp = rng.uniform(0.62, 0.95, 21)                          # each dermatologist's specificity
    dsn = (1 - (1 - dsp) ** 2.1) - rng.uniform(0.0, 0.14, 21)  # their sensitivity, mostly below the curve
    ax.scatter(dsp, dsn, s=42, color=sf.MUTED, edgecolor=sf.INK, linewidth=0.5, zorder=5,
               label="21 dermatologists")
    # the dermatologists' average, in deeppink -- the CNN's curve passes above it
    ax.scatter([dsp.mean()], [dsn.mean()], s=150, marker="*", color=sf.DEEPPINK,
               edgecolor=sf.INK, linewidth=0.8, zorder=6, label="their average")

    ax.set_xlim(1.02, 0.5); ax.set_ylim(0.5, 1.02)             # zoom to the interesting corner; x reversed
    ax.set_xlabel("specificity  (correctly clearing healthy spots)")
    ax.set_ylabel("sensitivity  (catching real cancers)")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    sf.title(ax, "One network matched 21 dermatologists")
    ax.text(0.635, 0.965, "trained on\n129,450 clinical images", ha="center", fontsize=9.5,
            family="Geist Mono", color=sf.INK, transform=ax.transData)
    ax.text(0.76, -0.155, "adapted from Esteva et al. 2017, Nature", ha="center", fontsize=9,
            style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_esteva_2017")                      # -> figures/intro_esteva_2017.png
    sf.save_raw(sf.pd.DataFrame({"spec": dsp, "sens": dsn}), HERE, "intro_esteva_2017")  # the 21 points


def sensitivity_tradeoff():
    """Two overlapping score distributions + a movable threshold -- the core screening tension."""
    x = np.linspace(0, 10, 400)                                  # the model's "cancer score" axis, low -> high
    healthy = np.exp(-0.5 * ((x - 3.4) / 1.15) ** 2)             # harmless spots cluster at LOW scores
    cancer = np.exp(-0.5 * ((x - 6.6) / 1.15) ** 2)             # melanomas cluster at HIGH scores
    thr = 5.4                                                     # where we draw the "call it cancer" line

    fig, ax = plt.subplots(figsize=(8.2, 4.4))                   # one wide panel
    ax.plot(x, healthy, color=sf.TURQUOISE, lw=2)               # curve for healthy spots
    ax.plot(x, cancer, color=sf.DEEPPINK, lw=2)                 # curve for melanomas
    ax.fill_between(x, healthy, where=(x < thr), color=sf.TURQUOISE, alpha=0.18)  # healthy correctly cleared
    ax.fill_between(x, cancer, where=(x >= thr), color=sf.DEEPPINK, alpha=0.18)   # cancers correctly caught
    ax.fill_between(x, cancer, where=(x < thr), color=sf.DEEPPINK, alpha=0.55)    # cancer BELOW the line = MISSED
    ax.fill_between(x, healthy, where=(x >= thr), color=sf.GOLD, alpha=0.60)      # healthy ABOVE the line = false alarm

    ax.axvline(thr, color=sf.INK, ls="--", lw=1.6)              # the decision threshold
    ax.text(thr, 1.12, "decision threshold", ha="center", fontsize=10, family="Geist Mono", color=sf.INK)
    ax.text(4.35, 0.14, "missed\nmelanomas", ha="center", fontsize=9, color=sf.DEEPPINK, family="Geist Mono")
    ax.text(6.35, 0.14, "false\nalarms", ha="center", fontsize=9, color="#B8860B", family="Geist Mono")
    ax.text(2.6, 0.92, "healthy spots", color=sf.TURQUOISE, fontsize=10, family="Geist Mono")
    ax.text(6.9, 0.92, "melanomas", color=sf.DEEPPINK, fontsize=10, family="Geist Mono")
    ax.add_patch(FancyArrowPatch((thr - 0.15, 0.62), (thr - 1.7, 0.62), arrowstyle="-|>",
                                 mutation_scale=16, color=sf.BLUEVIOLET, lw=2))
    ax.text(thr - 0.95, 0.68, "slide left = catch more cancers", ha="center", fontsize=9,
            color=sf.BLUEVIOLET, family="Geist Mono")

    ax.set_xlabel("the model's cancer score   (low  ->  high)")  # x-axis meaning in plain words
    ax.set_yticks([])                                            # the height is just "how many spots"; hide ticks
    ax.set_ylim(0, 1.25); ax.set_xlim(0, 10)                     # leave headroom for labels
    sf.title(ax, "Screening tunes one dial: sensitivity vs specificity")
    ax.text(5, -0.20, "concept adapted from Trevethan 2017", ha="center", fontsize=9,
            style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_sensitivity_tradeoff")
    sf.save_raw(sf.pd.DataFrame({"score": x, "healthy": healthy, "cancer": cancer}),
                HERE, "intro_sensitivity_tradeoff")


def transfer_learning():
    """A left-to-right pipeline: pretrain on everyday photos -> fine-tune on skin -> melanoma vs not."""
    fig, ax = plt.subplots(figsize=(9.0, 3.4)); ax.axis("off")   # wide blank canvas
    ax.set_xlim(0, 18); ax.set_ylim(0, 6)                        # drawing space

    def box(x, w, color, top, sub):
        """Draw one rounded stage box with a bold title line and a smaller subtitle line."""
        ax.add_patch(FancyBboxPatch((x, 1.9), w, 2.2, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    facecolor=color, edgecolor=color, linewidth=2, alpha=0.16))
        ax.add_patch(FancyBboxPatch((x, 1.9), w, 2.2, boxstyle="round,pad=0.03,rounding_size=0.15",
                                    facecolor="none", edgecolor=color, linewidth=2))
        ax.text(x + w / 2, 3.35, top, ha="center", va="center", fontsize=10.5, family="Geist Mono",
                color=sf.INK, fontweight="bold")
        ax.text(x + w / 2, 2.55, sub, ha="center", va="center", fontsize=8.5, color=sf.MUTED)

    box(0.4, 6.2, sf.TURQUOISE, "pre-trained on millions\nof everyday photos", "cats, cars, faces")
    box(7.6, 4.6, sf.DEEPPINK, "fine-tune on ~10,000\nskin images", "HAM10000 dermatoscopy")
    box(13.6, 4.0, sf.BLUEVIOLET, "melanoma\nvs. not", "a screening call")

    def arrow(x0, x1, label):
        ax.add_patch(FancyArrowPatch((x0, 3.0), (x1, 3.0), arrowstyle="-|>", mutation_scale=18,
                                     color=sf.INK, lw=1.8))
        if label:
            ax.text((x0 + x1) / 2, 4.55, label, ha="center", va="center", fontsize=8,
                    family="Geist Mono", color=sf.BLUEVIOLET)

    arrow(6.7, 7.5, "reuse\nfeatures")
    arrow(12.3, 13.5, "")

    ax.text(9, 5.35, "Borrow a brain: transfer learning", ha="center", fontsize=14,
            family="Geist Mono", color=sf.INK, fontweight="bold")
    ax.text(9, 0.85, "The reused knowledge is why a model can learn skin cancer from thousands, not millions, of photos",
            ha="center", fontsize=9, color=sf.INK)
    ax.text(9, 0.25, "adapted from Kim et al. 2022", ha="center", fontsize=9,
            style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_transfer_learning")


if __name__ == "__main__":
    dermatoscope()                                               # the clinical scene
    cost_of_a_miss()                                             # the deadly-asymmetry balance
    esteva_2017()                                                # the landmark redraw
    sensitivity_tradeoff()                                       # the screening dial
    transfer_learning()                                          # the transfer-learning pipeline
    print("G2 intro figures done")                              # confirm success
