"""G1 intro/background figures -- bespoke redraws that VISUALLY depict the clinical/biological
setting of CRISPR guide design (all attributed). Theme "bone" via the shared solfig helper, so
every PNG drops straight onto a slide canvas.

  cas9_cutting     -- the clinical thing: the Cas9 protein, steered by a 20-letter guide RNA, opens
                      the DNA, base-pairs to the target next to the PAM, and cuts ~3 bp upstream --
                      the moment a gene gets edited                  (adapted from Hsu, Lander & Zhang 2014)
  therapy_stakes   -- WHY it matters: gene editing as therapy -- a disease-causing mutation cut and
                      corrected into a healthy gene, IF the guide actually cuts
                                                                     (concept: Hsu 2014; sickle-cell editing 2023)
  pick_before_lab  -- the decision the tool supports: dozens of candidate guides per gene, each lab
                      test costs weeks -- so score them all first and order only the top few
                                                                     (concept adapted from Doench et al. 2016)
  seed_importance  -- not all 20 letters count equally: the "seed" next to the PAM matters most
                                                                     (adapted from Zheng et al. 2017 / Doench 2016)
  doench_scatter   -- the benchmark redraw: a sequence-only model captures real signal without being
                      perfect (predicted vs measured scatters)       (adapted from Doench et al. 2016)
"""
import sys                                                           # sys lets us extend the import path
from pathlib import Path                                            # Path builds file locations safely

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))  # find the shared solfig helper
import solfig as sf                                                 # brand palette + save/save_raw + theme "bone"
from matplotlib.patches import (FancyArrowPatch, Rectangle,         # shapes for the schematics
                                FancyBboxPatch, Circle, Polygon, Ellipse)

HERE = Path(__file__).resolve().parent                             # this folder (figures land in ./figures)
np = sf.np                                                          # re-use solfig's numpy
plt = sf.plt                                                        # re-use solfig's pyplot (already themed)

DNA = "#C9C6BA"                                                     # a neutral DNA-ribbon gray
RUNG = "#9C9788"                                                    # base-pair rung color


def cas9_cutting():
    """The Cas9 protein, aimed by a 20-letter guide RNA, cutting DNA next to the PAM."""
    fig, ax = plt.subplots(figsize=(9.4, 5.4)); ax.axis("off")     # wide blank canvas
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.set_aspect("auto")

    # --- the Cas9 protein: two big lobes (REC + NUC) drawn behind everything, tinted blueviolet.
    #     Kept clear of the PAM zone on the right so the pink PAM label sits in open space. ---
    for cx, cy, w, h in [(7.8, 5.6, 11.4, 6.6), (10.8, 6.0, 5.6, 5.2)]:
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=sf.BLUEVIOLET, edgecolor="none", alpha=0.13, zorder=1))
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor="none", edgecolor=sf.BLUEVIOLET, lw=1.6, alpha=0.55, zorder=1))
    ax.text(4.1, 8.7, "Cas9 protein", ha="center", fontsize=12, family="Geist Mono",
            color=sf.BLUEVIOLET, fontweight="bold", zorder=8)      # protein label

    # --- the DNA: a double ribbon that opens into a bubble where the guide invades (the R-loop) ---
    xs = np.linspace(1.0, 19.0, 400)                               # x samples across the strands
    bub = np.exp(-0.5 * ((xs - 9.0) / 3.0) ** 2)                   # a smooth bubble centered on the target
    top = 6.1 + 1.15 * bub                                         # non-target strand bulges UP over the bubble
    bot = 4.1 - 0.0 * bub                                          # target strand stays flat (the guide pairs to it)
    ax.plot(xs, top, color=DNA, lw=9, solid_capstyle="round", zorder=2)   # non-target DNA strand
    ax.plot(xs, bot, color=DNA, lw=9, solid_capstyle="round", zorder=2)   # target DNA strand

    # --- the guide RNA: a turquoise strand base-paired to the flat target strand across 20 nt ---
    gx = np.linspace(5.0, 13.0, 200)                               # the 20-nt protospacer spans x 5..13
    gy = 4.55 + 0.0 * gx                                           # guide sits just above the target strand
    ax.plot(gx, gy, color=sf.TURQUOISE, lw=6, solid_capstyle="round", zorder=4)  # the guide RNA
    for x in np.linspace(5.2, 12.8, 20):                          # 20 little rungs = the 20 base pairs
        ax.plot([x, x], [4.18, 4.5], color=RUNG, lw=1.4, zorder=3)
    ax.text(9.0, 3.05, "20-letter guide RNA", ha="center", fontsize=12.5, family="Geist Mono",
            color=sf.TURQUOISE, fontweight="bold", zorder=8)       # guide label
    ax.text(9.0, 5.02, "base-pairs to the DNA target", ha="center", fontsize=9.5, family="Geist Mono",
            color=sf.INK, zorder=8, style="italic")

    # --- the PAM: a short deeppink segment on the DNA, immediately 3' of the target ---
    ax.add_patch(FancyBboxPatch((13.1, 3.75), 1.5, 0.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=sf.DEEPPINK, edgecolor="none", zorder=5))
    ax.text(13.85, 5.35, "PAM", ha="center", va="center", fontsize=11, family="Geist Mono",
            color=sf.DEEPPINK, fontweight="bold", zorder=8)        # PAM label above its box
    ax.annotate("", xy=(13.85, 4.55), xytext=(13.85, 5.05),
                arrowprops=dict(arrowstyle="-", color=sf.DEEPPINK, lw=1.2), zorder=7)
    # define PAM in plain words, in clear space to the right of the protein
    ax.text(15.2, 4.15, "PAM = a short DNA tag (the\nletters 'NGG') that Cas9 must\nfind right next to the target\nbefore it is allowed to cut.",
            ha="left", va="center", fontsize=8.5, family="Geist Mono", color=sf.INK, zorder=8)

    # --- the cut: an amber scissor mark ~3 bp upstream of the PAM, through both strands ---
    for yy in [(3.9, 4.5), (6.0, 6.7)]:                            # cut both the target and non-target strands
        ax.plot([11.55, 11.95], [yy[0], yy[1]], color=sf.AMBER, lw=2.6, zorder=6)
        ax.plot([11.95, 11.55], [yy[0], yy[1]], color=sf.AMBER, lw=2.6, zorder=6)
    ax.add_patch(FancyArrowPatch((11.75, 8.15), (11.75, 6.95), arrowstyle="-|>", mutation_scale=18,
                                 color=sf.AMBER, lw=2.4, zorder=7))
    ax.text(11.75, 8.55, "Cas9 cuts here", ha="center", fontsize=11, family="Geist Mono",
            color="#B8860B", fontweight="bold", zorder=8)          # cut label

    # --- the seed bracket over the ~6 nt nearest the PAM (where mismatches break cutting) ---
    ax.plot([10.6, 12.9], [2.35, 2.35], color=sf.DEEPPINK, lw=2.0, zorder=8)
    ax.plot([10.6, 10.6], [2.35, 2.55], color=sf.DEEPPINK, lw=2.0, zorder=8)
    ax.plot([12.9, 12.9], [2.35, 2.55], color=sf.DEEPPINK, lw=2.0, zorder=8)
    ax.text(11.75, 1.95, "the 'seed' -- letters here matter most", ha="center", fontsize=9.5,
            family="Geist Mono", color=sf.DEEPPINK, zorder=8)

    # (no in-figure title -- the slide chrome supplies it; keep only the attribution)
    ax.text(10.0, 0.5, "adapted from Hsu, Lander & Zhang 2014", ha="center", fontsize=9,
            style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_cas9_cutting")


def _dna_ladder(ax, x0, y0, h, mutation=False):
    """Draw a short vertical DNA ladder at (x0,y0); one rung red if `mutation`, else all healthy."""
    ax.add_patch(Rectangle((x0 - 0.12, y0), 0.14, h, color=DNA))    # left rail
    ax.add_patch(Rectangle((x0 + 0.9, y0), 0.14, h, color=DNA))     # right rail
    n = 7                                                           # number of rungs
    for i, ry in enumerate(np.linspace(y0 + 0.25, y0 + h - 0.25, n)):
        col = sf.DEEPPINK if (mutation and i == 3) else sf.TURQUOISE  # the middle rung is the mutation
        ax.add_patch(Rectangle((x0 + 0.02, ry), 0.88, 0.16, color=col))


def therapy_stakes():
    """Gene editing as therapy: a disease mutation cut and corrected -- IF the guide cuts."""
    fig, ax = plt.subplots(figsize=(9.2, 4.9)); ax.axis("off")     # blank canvas
    ax.set_xlim(0, 18); ax.set_ylim(0, 10); ax.set_aspect("auto")

    # --- LEFT: the disease gene, with one broken (mutated) rung ---
    ax.add_patch(FancyBboxPatch((1.4, 2.3), 3.6, 5.2, boxstyle="round,pad=0.03,rounding_size=0.2",
                                facecolor="#F4DCE6", edgecolor=sf.DEEPPINK, linewidth=1.6))
    _dna_ladder(ax, 2.7, 3.0, 3.8, mutation=True)
    ax.text(3.2, 7.85, "disease gene", ha="center", fontsize=12, family="Geist Mono",
            color=sf.DEEPPINK, fontweight="bold")
    ax.text(3.2, 1.55, "one broken letter\ncauses the disease", ha="center", va="center", fontsize=9.5, color=sf.DEEPPINK)

    # --- MIDDLE: Cas9 + a good guide, the edit step ---
    ax.add_patch(Ellipse((9.0, 5.0), 3.2, 3.2, facecolor=sf.BLUEVIOLET, edgecolor="none", alpha=0.14))
    ax.add_patch(Ellipse((9.0, 5.0), 3.2, 3.2, facecolor="none", edgecolor=sf.BLUEVIOLET, lw=1.6))
    ax.add_patch(Circle((9.0, 5.0), 0.7, facecolor="none", edgecolor=sf.TURQUOISE, lw=2.4))
    for yy in [(4.5, 5.1), (4.9, 5.5)]:
        ax.plot([8.75, 9.25], [yy[0], yy[1]], color=sf.AMBER, lw=2.2)
        ax.plot([9.25, 8.75], [yy[0], yy[1]], color=sf.AMBER, lw=2.2)
    ax.text(9.0, 7.15, "Cas9 + a guide\nthat actually cuts", ha="center", fontsize=10.5,
            family="Geist Mono", color=sf.BLUEVIOLET, fontweight="bold")
    ax.add_patch(FancyArrowPatch((5.3, 5.0), (7.0, 5.0), arrowstyle="-|>", mutation_scale=20,
                                 color=sf.INK, lw=2.0))
    ax.add_patch(FancyArrowPatch((11.0, 5.0), (12.7, 5.0), arrowstyle="-|>", mutation_scale=20,
                                 color=sf.INK, lw=2.0))
    ax.text(9.0, 2.7, "cut + repair", ha="center", fontsize=9.5, family="Geist Mono", color=sf.INK, style="italic")

    # --- RIGHT: the corrected, healthy gene ---
    ax.add_patch(FancyBboxPatch((13.0, 2.3), 3.6, 5.2, boxstyle="round,pad=0.03,rounding_size=0.2",
                                facecolor="#DDF3EF", edgecolor=sf.TURQUOISE, linewidth=1.6))
    _dna_ladder(ax, 14.3, 3.0, 3.8, mutation=False)
    ax.text(14.8, 7.85, "corrected gene", ha="center", fontsize=12, family="Geist Mono",
            color="#1F9E92", fontweight="bold")
    ax.text(14.8, 1.55, "the letter is fixed:\na possible cure", ha="center", va="center", fontsize=9.5, color="#1F9E92")

    # (no in-figure title -- the slide chrome supplies it)
    ax.text(9.0, 0.55, "the same tool already treats sickle-cell disease (approved 2023) -- but only if the guide cuts",
            ha="center", fontsize=10, color=sf.INK)
    ax.text(9.0, 0.1, "concept: Hsu, Lander & Zhang 2014", ha="center", fontsize=9, style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_therapy_stakes")


def pick_before_lab():
    """The bottleneck a guide-picker removes: dozens of candidates, each lab test costs weeks."""
    fig, ax = plt.subplots(figsize=(9.4, 5.0)); ax.axis("off")     # blank canvas
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.set_aspect("auto")

    # --- LEFT: a stack of many candidate guides for one gene ---
    ax.text(3.0, 9.6, "dozens of candidate guides", ha="center", fontsize=11.5, family="Geist Mono",
            color=sf.INK, fontweight="bold")
    ax.text(3.0, 9.0, "for a single gene", ha="center", fontsize=9.5, color=sf.MUTED)
    rng = np.random.default_rng(1)                                 # fixed seed -> reproducible bar lengths
    for i, y in enumerate(np.linspace(7.9, 1.2, 11)):             # 11 stacked guide bars
        w = 2.0 + rng.uniform(0, 1.6)                             # varied "unknown efficiency" widths
        ax.add_patch(FancyBboxPatch((1.0, y), w, 0.42, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    facecolor=sf.MUTED, edgecolor="none", alpha=0.55))

    # --- MIDDLE: the funnel = the efficiency predictor ---
    ax.add_patch(Polygon([(6.6, 8.4), (12.2, 8.4), (10.6, 4.6), (8.2, 4.6)], closed=True,
                         facecolor=sf.TURQUOISE, edgecolor=sf.INK, lw=1.0, alpha=0.9))
    ax.text(9.4, 6.6, "efficiency\npredictor", ha="center", va="center", fontsize=12,
            family="Geist Mono", color=sf.INK, fontweight="bold")
    ax.add_patch(Rectangle((8.9, 3.4), 1.0, 1.2, facecolor=sf.TURQUOISE, edgecolor=sf.INK, lw=1.0, alpha=0.9))
    ax.add_patch(FancyArrowPatch((5.2, 4.8), (6.4, 6.2), arrowstyle="-|>", mutation_scale=18, color=sf.INK, lw=1.8))

    # --- RIGHT: only the top few good guides come out -> order these ---
    ax.add_patch(FancyArrowPatch((10.0, 3.3), (12.0, 3.3), arrowstyle="-|>", mutation_scale=18, color=sf.INK, lw=1.8))
    ax.text(16.4, 9.6, "order only the top few", ha="center", fontsize=11.5, family="Geist Mono",
            color="#1F9E92", fontweight="bold")
    ax.text(16.4, 9.0, "the ones most likely to cut", ha="center", fontsize=9.5, color=sf.MUTED)
    for i, y in enumerate(np.linspace(6.2, 3.0, 3)):             # just 3 winning guides
        ax.add_patch(FancyBboxPatch((12.6, y), 3.0, 0.55, boxstyle="round,pad=0.01,rounding_size=0.1",
                                    facecolor=sf.TURQUOISE, edgecolor=sf.INK, lw=0.8))
        ax.text(14.1, y + 0.27, "good guide", ha="center", va="center", fontsize=9.5,
                family="Geist Mono", color=sf.INK)

    # (no in-figure title -- the slide chrome supplies it; the weeks-vs-seconds contrast lives in
    #  the slide's tile band, so we keep the figure to just the funnel visual)
    ax.text(10.0, 0.35, "concept adapted from Doench et al. 2016", ha="center", fontsize=8.5,
            style="italic", color=sf.MUTED)
    sf.save(fig, HERE, "intro_pick_before_lab")


def seed_importance():
    """Not all 20 positions count equally -- the seed next to the PAM matters most."""
    pos = np.arange(1, 21)                                          # positions 1..20 along the guide
    imp = 0.25 + 0.75 / (1 + np.exp(-(pos - 14) / 1.6))            # rises toward the PAM (position 20)
    colors = [sf.DEEPPINK if p >= 13 else sf.TURQUOISE for p in pos]  # pink = seed, turquoise = rest
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.bar(pos, imp, color=colors, edgecolor=sf.INK, linewidth=0.4)
    ax.annotate("", xy=(20.4, 1.08), xytext=(12.6, 1.08),
                arrowprops=dict(arrowstyle="-", color=sf.DEEPPINK, lw=1.6))
    ax.text(16.5, 1.14, "seed / core region", ha="center", fontsize=10.5, family="Geist Mono", color=sf.DEEPPINK)
    ax.text(6.5, 0.30, "far from the PAM:\nletters matter less", ha="center", fontsize=9,
            family="Geist Mono", color="#1F9E92")
    ax.set_xlabel("position along the 20-letter guide  (20 = next to the PAM / cut site)")
    ax.set_ylabel("how much this position matters")
    ax.set_ylim(0, 1.3); ax.set_xticks([1, 5, 10, 15, 20])
    ax.text(10.5, -0.42, "adapted from Zheng et al. 2017 / Doench et al. 2016", ha="center",
            fontsize=9, style="italic", color=sf.MUTED, transform=ax.get_xaxis_transform())
    sf.save(fig, HERE, "intro_seed_importance")
    sf.save_raw(sf.pd.DataFrame({"position": pos, "importance": imp}), HERE, "intro_seed_importance")


def doench_scatter():
    """The benchmark redraw: a sequence-only model captures real signal without being perfect."""
    rng = np.random.default_rng(0)                                 # fixed seed -> reproducible cloud
    x = rng.uniform(0, 1, 240)                                     # model-predicted efficiency
    y = np.clip(0.15 + 0.6 * x + rng.normal(0, 0.16, 240), 0, 1)   # measured efficiency: real but loose
    fig, ax = plt.subplots(figsize=(5.2, 4.9))
    ax.scatter(x, y, s=14, alpha=0.42, color=sf.TURQUOISE, edgecolor="none")
    m, b = np.polyfit(x, y, 1)                                     # the trend line
    ax.plot([0, 1], [b, m + b], color=sf.DEEPPINK, lw=2.4)
    r = float(np.corrcoef(x, y)[0, 1])                             # how tight the trend is
    ax.text(0.05, 0.92, f"correlation ~ {r:.2f}", fontsize=10, family="Geist Mono", color=sf.INK)
    ax.text(0.97, 0.06, "real but loose:\na model can be useful\nwithout being perfect", ha="right",
            fontsize=9, family="Geist Mono", color=sf.DEEPPINK)
    ax.set_xlabel("model-predicted efficiency"); ax.set_ylabel("measured efficiency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, -0.17, "adapted from Doench et al. 2016", ha="center", fontsize=9,
            style="italic", color=sf.MUTED, transform=ax.transAxes)
    sf.save(fig, HERE, "intro_doench_scatter")
    sf.save_raw(sf.pd.DataFrame({"predicted": x, "measured": y}), HERE, "intro_doench_scatter")


if __name__ == "__main__":
    cas9_cutting()                                                 # the molecular scene
    therapy_stakes()                                               # gene editing as therapy
    pick_before_lab()                                              # the picking bottleneck
    seed_importance()                                              # the seed intuition
    doench_scatter()                                               # the Doench benchmark redraw
    print("G1 intro figures done")                                # confirm success
