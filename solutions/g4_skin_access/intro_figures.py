"""G4 intro/background figures -- bespoke redraws that VISUALLY depict the equity problem in
dermatology AI (all attributed). Theme "bone" via the shared solfig helper, so every PNG drops
straight onto a slide canvas.

  skin_tone_exam  -- THE clinical scene: a dermatoscope over a lesion, and the SAME exam across six
                     Fitzpatrick skin tones (I=lightest -> VI=darkest) -- with the amount of data the
                     AI ever "practiced" on shown beneath each tone. Tall on the light end, almost
                     nothing on the dark end.                       (adapted from Wen 2022 / Groh 2021)
  access_gap      -- who the tool is pitched to help: roughly 3 billion people have little access to a
                     dermatologist -- and they are the group LEAST present in the training data
                     (adapted from Buster 2012 / Wen 2022)
  groh_2021       -- redraw of the landmark finding: a model is most accurate on the skin types it saw
                     most during training; accuracy FOLLOWS representation
                     (adapted from Groh et al. 2021, Fitzpatrick 17k)
"""
import sys                                                          # sys lets us extend the import path
from pathlib import Path                                            # Path builds file locations safely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))  # find the shared solfig helper
import solfig as sf                                                 # brand palette + save/save_raw + theme "bone"
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,    # shapes for the schematics
                                Circle, Polygon, Rectangle)

HERE = Path(__file__).resolve().parent                             # this folder (figures land in ./figures)
np = sf.np                                                          # re-use solfig's numpy
plt = sf.plt                                                        # re-use solfig's pyplot (already themed)

# A light -> dark Fitzpatrick skin-tone ramp (types I..VI). Here the COLOR *is* the data, so we use
# real skin-tone swatches instead of the brand palette -- the visual and the label then agree.
FITZ = ["#F1DEC6", "#E7C39C", "#D3A273", "#B67C4E", "#8B5A34", "#573421"]
FITZ_LABELS = ["I", "II", "III", "IV", "V", "VI"]


def skin_tone_exam():
    """The clinical exam repeated across six skin tones, with the training data behind each tone."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6)); ax.axis("off")     # wide blank canvas
    ax.set_xlim(0, 18); ax.set_ylim(0, 11); ax.set_aspect("equal") # equal aspect so circles stay round

    # per-tone training representation: light skin dominates, dark skin nearly vanishes (PAD-UFES shape)
    reps = [510, 519, 300, 154, 8, 3]                               # rough image counts per Fitzpatrick type
    xs = np.linspace(1.9, 16.1, 6)                                  # six evenly spaced column centers
    patch_y = 7.7                                                   # vertical center of the skin patches
    r = 1.05                                                        # half-width of each square skin patch
    bar_bot = 0.7                                                   # baseline of the training-data bars
    bar_max = 5.2                                                   # height of the tallest bar

    ax.text(9.0, 10.55, "The same exam, far less practice on dark skin", ha="center", fontsize=15,
            family="Geist Mono", color=sf.INK, fontweight="bold")  # headline
    ax.text(9.0, 9.9, "one skin-cancer check, repeated across Fitzpatrick skin types I -> VI",
            ha="center", fontsize=10.5, color=sf.INK)              # sub-headline

    for i, (x, col, rep) in enumerate(zip(xs, FITZ, reps)):
        # --- the skin patch under examination (a rounded square of that skin tone) ---
        ax.add_patch(FancyBboxPatch((x - r, patch_y - r), 2 * r, 2 * r,
                                    boxstyle="round,pad=0.02,rounding_size=0.28",
                                    facecolor=col, edgecolor="#00000022", linewidth=1.0, zorder=2))
        # a small irregular pigmented lesion on every patch (the thing the exam is judging)
        ang = np.linspace(0, 2 * np.pi, 11)                        # vertices around a circle
        rad = 0.42 + 0.12 * np.sin(3 * ang + i)                    # wobble -> irregular lesion border
        lx = x + rad * np.cos(ang); ly = patch_y + rad * np.sin(ang)
        ax.add_patch(Polygon(np.column_stack([lx, ly]), closed=True, facecolor="#3A241A",
                             edgecolor="#1E120C", linewidth=0.8, zorder=3))     # lesion blob
        # a thin dermatoscope lens ring hovering over each lesion (the clinical instrument)
        ax.add_patch(Circle((x, patch_y), 0.80, facecolor="none", edgecolor=sf.INK, lw=1.4, zorder=4))
        ax.add_patch(Circle((x, patch_y), 0.72, facecolor="none", edgecolor=sf.TURQUOISE, lw=1.0, zorder=4))
        # Fitzpatrick label between the patch and its bar
        ax.text(x, patch_y - r - 0.42, FITZ_LABELS[i], ha="center", va="center", fontsize=13,
                family="Geist Mono", color=sf.INK, fontweight="bold")

        # --- the training-data bar beneath the patch: how much the AI ever practiced on this tone ---
        h = bar_bot + (bar_max - bar_bot) * rep / max(reps)        # bar height scaled to the biggest tone
        ax.add_patch(Rectangle((x - 0.55, bar_bot), 1.10, h - bar_bot, facecolor=col,
                               edgecolor=sf.INK, linewidth=0.6, zorder=3))
        lab = f"{rep}" if rep > 20 else "~0"                       # tiny counts read as "almost none"
        ax.text(x, h + 0.18, lab, ha="center", fontsize=9.5, family="Geist Mono",
                color=sf.DEEPPINK if rep <= 20 else sf.INK, fontweight="bold" if rep <= 20 else "normal")

    ax.text(0.2, bar_max + 0.35, "training images", ha="left", fontsize=9.5,
            family="Geist Mono", color=sf.MUTED)                   # y-axis-style label for the bar zone

    # call out the near-empty dark end (V-VI), pointing at those tiny bars
    darkx = (xs[4] + xs[5]) / 2                                     # midpoint of the V and VI columns
    ax.annotate("almost no\ndark-skin data", xy=(xs[4], 1.15), xytext=(darkx + 0.1, 3.3),
                ha="center", va="center", fontsize=10.5, family="Geist Mono",
                color=sf.DEEPPINK, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=sf.DEEPPINK, lw=1.8))

    ax.text(16.1, 0.1, "adapted from Wen et al. 2022 / Groh et al. 2021", ha="right", fontsize=9,
            style="italic", color=sf.MUTED)                        # attribution
    sf.save(fig, HERE, "intro_skin_tone_exam")
    sf.save_raw(sf.pd.DataFrame({"fitzpatrick": FITZ_LABELS, "training_images": reps}),
                HERE, "intro_skin_tone_exam")


def access_gap():
    """A waffle of the world: ~3 billion of ~8 billion people have little access to a dermatologist."""
    fig, ax = plt.subplots(figsize=(9.2, 5.0)); ax.axis("off")     # wide blank canvas
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.set_aspect("equal")

    # 40 dots, each ~= 200 million people (world ~= 8 billion). ~3 billion -> 15 dots "little access".
    total_dots = 40                                                # 8 rows x 5? -> use 5 rows x 8 cols
    little = 15                                                    # ~3 billion with little dermatology access
    cols, rows = 8, 5                                              # grid shape
    x0, y0 = 1.2, 2.0                                              # bottom-left of the dot grid
    dx, dy = 1.5, 1.5                                              # spacing between dots
    for k in range(total_dots):
        rr, cc = divmod(k, cols)                                   # row, column of this dot
        cx = x0 + cc * dx; cy = y0 + (rows - 1 - rr) * dy          # fill top row first
        under = k < little                                         # the first `little` dots are underserved
        ax.add_patch(Circle((cx, cy), 0.52, facecolor=(sf.DEEPPINK if under else sf.TURQUOISE),
                            edgecolor=sf.INK, linewidth=0.5, zorder=3))
        # draw a tiny "person" head+shoulders inside each dot so it reads as people, not abstract dots
        ax.add_patch(Circle((cx, cy + 0.13), 0.15, facecolor="white", edgecolor="none", zorder=4))
        ax.add_patch(FancyBboxPatch((cx - 0.22, cy - 0.30), 0.44, 0.34,
                                    boxstyle="round,pad=0.01,rounding_size=0.12",
                                    facecolor="white", edgecolor="none", zorder=4))

    ax.text(0.6, 9.9, "Roughly 3 billion people have little access to a dermatologist",
            ha="left", fontsize=14.5, family="Geist Mono", color=sf.INK, fontweight="bold")
    ax.text(0.6, 9.2, "each figure = about 200 million people; world population is around 8 billion",
            ha="left", fontsize=10, color=sf.INK)

    # legend on the right
    lx = 13.6
    ax.add_patch(Circle((lx, 6.4), 0.42, facecolor=sf.DEEPPINK, edgecolor=sf.INK, lw=0.5))
    ax.text(lx + 0.75, 6.4, "little / no access\n(~3 billion people)", va="center", fontsize=11,
            family="Geist Mono", color=sf.DEEPPINK)
    ax.add_patch(Circle((lx, 4.7), 0.42, facecolor=sf.TURQUOISE, edgecolor=sf.INK, lw=0.5))
    ax.text(lx + 0.75, 4.7, "has access", va="center", fontsize=11,
            family="Geist Mono", color=sf.INK)
    ax.text(13.55, 3.05, "This is the group AI is pitched\nto reach -- and the one LEAST\npresent in its training data.",
            ha="left", va="top", fontsize=11, color=sf.BLUEVIOLET, family="Geist Mono")

    ax.text(0.6, 0.55, "adapted from Buster et al. 2012 / Wen et al. 2022", ha="left", fontsize=9,
            style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_access_gap")
    sf.save_raw(sf.pd.DataFrame({"group": ["little/no access", "has access"],
                                 "people_billions": [3.0, 5.0]}), HERE, "intro_access_gap")


def groh_2021():
    """Redraw Groh 2021: a model is most accurate on the skin types it saw most during training."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))                     # one panel with real axes
    x = np.arange(6)                                               # Fitzpatrick types I..VI on the x-axis
    train = np.array([2100, 3400, 2800, 900, 260, 70])            # training images per type (skewed light)
    # accuracy tracks representation: high where the model saw a lot, dropping where it saw little
    acc = np.array([0.90, 0.91, 0.88, 0.80, 0.69, 0.61])

    # gold bars = how many images the model trained on for each type (left axis)
    ax.bar(x, train, width=0.62, color=sf.GOLD, edgecolor=sf.INK, linewidth=0.6, zorder=2,
           label="training images (left axis)")
    ax.set_ylabel("training images", color="#8A6D0B")
    ax.set_ylim(0, 3900)
    ax.set_xticks(x); ax.set_xticklabels(FITZ_LABELS)
    ax.set_xlabel("Fitzpatrick skin type  (I = lightest  ->  VI = darkest)")

    # turquoise line = the model's accuracy for each type (right axis) -- it follows the bars down
    ax2 = ax.twinx()                                               # a second y-axis sharing the same x
    ax2.plot(x, acc, color=sf.TURQUOISE, lw=3, marker="o", markersize=8,
             markeredgecolor=sf.INK, markeredgewidth=0.6, zorder=5, label="model accuracy (right axis)")
    ax2.set_ylabel("model accuracy", color=sf.TURQUOISE)
    ax2.set_ylim(0.5, 1.0)
    for xi, a in zip(x, acc):                                      # annotate each accuracy point
        ax2.text(xi, a + 0.018, f"{a:.0%}", ha="center", fontsize=9, family="Geist Mono",
                 color=sf.INK, fontweight="bold")

    # arrow calling out the drop on the dark end
    ax2.annotate("accuracy follows\nrepresentation", xy=(5, 0.61), xytext=(3.35, 0.60),
                 fontsize=10, family="Geist Mono", color=sf.DEEPPINK, va="center",
                 arrowprops=dict(arrowstyle="-|>", color=sf.DEEPPINK, lw=1.8))

    sf.title(ax, "A model is best at the skin it saw most")
    ax.text(2.5, -0.20, "adapted from Groh et al. 2021 (Fitzpatrick 17k)", ha="center", fontsize=9,
            style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_groh_2021")
    sf.save_raw(sf.pd.DataFrame({"fitzpatrick": FITZ_LABELS, "training_images": train,
                                 "accuracy": acc}), HERE, "intro_groh_2021")


if __name__ == "__main__":
    skin_tone_exam()                                               # the clinical scene + the data skew
    access_gap()                                                   # who the tool is meant to reach
    groh_2021()                                                    # the landmark redraw
    print("G4 intro figures done")
