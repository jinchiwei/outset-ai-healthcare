"""Regenerate the two RESULT figures for G4 straight from results.json + the dataset's own tone
counts -- no retraining needed. run_experiment.py produces these when the model is (re)trained; this
script redraws them from the saved numbers so the honest story is on disk without a GPU.

Two figures, and why each is drawn the way it is:

  tone_distribution   -- THE headline. All three skin-tone groups PAD-UFES-20 records, including the
                         V-VI dark group that has only 11 patients. You cannot validate a screen for a
                         group of 11, so the tiny bar is called out in deeppink as the real crisis.
  equity_before_after -- honest fairness result. Across the tones we CAN measure the screen is already
                         roughly fair (AUC gap ~0.01). A tone-balanced retrain did not change that --
                         and it cannot, because rebalancing can't invent dark-skin patients that the
                         dataset does not contain. The fix for dark skin is collecting data, not resampling.
"""
import json                                                        # to read results.json
import sys                                                         # to extend the import path
from pathlib import Path                                           # safe file locations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))  # find solfig
import solfig as sf                                                # brand palette + save/save_raw + theme "bone"

HERE = Path(__file__).resolve().parent                            # this folder
np = sf.np; plt = sf.plt                                           # re-use solfig's numpy + themed pyplot
R = json.loads((HERE / "results.json").read_text())               # the canonical saved numbers

# PAD-UFES-20 records the Fitzpatrick skin type of every patient. Grouped into three tone bands, the
# patient counts are these -- the light band dwarfs the rest, and the dark band is almost empty.
TONE_COUNTS = {"I-II (light)": 1029, "III-IV (medium)": 454, "V-VI (dark)": 11}
# skin-tone swatch colors so the bar color agrees with the label it carries
TONE_FILL = {"I-II (light)": "#E7C39C", "III-IV (medium)": "#B67C4E", "V-VI (dark)": "#573421"}


def tone_distribution():
    """Three bars: how many patients of each skin tone the dataset actually contains. 11 is the crisis."""
    groups = list(TONE_COUNTS)                                     # the three tone bands, light -> dark
    counts = [TONE_COUNTS[g] for g in groups]                      # their patient counts: 1029, 454, 11
    fig, ax = plt.subplots(figsize=(7.4, 4.2))                     # one wide panel
    bars = ax.bar(groups, counts, color=[TONE_FILL[g] for g in groups],
                  edgecolor=sf.INK, linewidth=0.6, zorder=2)       # one skin-toned bar per group
    for b, c in zip(bars, counts):                                 # label each bar with its exact count
        crisis = c < 50                                            # the dark group is the tiny, alarming one
        ax.text(b.get_x() + b.get_width() / 2, c + 14, f"{c:,}", ha="center",
                fontsize=13 if crisis else 11, family="Geist Mono",
                color=sf.DEEPPINK if crisis else sf.INK, fontweight="bold")
    # call out the dark bar: 11 patients is far too few to validate a screen on
    # arrow lands on the LEFT edge/top of the tiny dark bar, clear of the centered "11" count label
    ax.annotate("only 11 patients --\nyou cannot validate\na screen on this",
                xy=(1.72, 11), xytext=(1.35, 430), fontsize=10.5, family="Geist Mono",
                color=sf.DEEPPINK, va="center",
                arrowprops=dict(arrowstyle="-|>", color=sf.DEEPPINK, lw=1.8))
    ax.set_ylabel("patients in the dataset")
    ax.set_ylim(0, 1140)
    sf.title(ax, "The real crisis: 11 dark-skin patients out of 1,494")
    sf.save(fig, HERE, "tone_distribution")
    sf.save_raw(sf.pd.Series(TONE_COUNTS, name="patients"), HERE, "tone_distribution")


def equity_before_after():
    """Per-tone AUC where we can measure it: already fair, and a rebalance retrain did not change that."""
    b = R["auc_by_tone_baseline"]; m = R["auc_by_tone_mitigated"]  # AUC per tone, baseline vs rebalanced
    groups = [g for g in ["I-II (light)", "III-IV (medium)"] if g in b and g in m]  # the measurable tones
    x = np.arange(len(groups)); w = 0.38                           # bar x-positions and width
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - w / 2, [b[g] for g in groups], w, label="baseline screen", color=sf.MUTED,
           edgecolor=sf.INK, linewidth=0.5, zorder=2)
    ax.bar(x + w / 2, [m[g] for g in groups], w, label="after tone-balanced retrain", color=sf.TURQUOISE,
           edgecolor=sf.INK, linewidth=0.5, zorder=2)
    ax.axhline(0.8, ls="--", color=sf.DEEPPINK, lw=1.2, zorder=1)  # the 0.8 "working screen" line
    ax.text(len(groups) - 0.5, 0.808, "AUC 0.8", ha="right", fontsize=9, color=sf.DEEPPINK,
            family="Geist Mono")
    # the gap between the two measurable tones is tiny -- annotate it as "already fair"
    ax.text(0.5, 0.965, f"gap only {R['gap_baseline']:.02f} AUC -- already fair where we can measure",
            ha="center", fontsize=10.5, family="Geist Mono", color=sf.INK, fontweight="bold")
    # a third, empty slot naming the group we CANNOT score -- the honest limit
    ax.text(2.15, 0.66, "V-VI (dark)\nn = 11\ncannot compute", ha="center", va="center", fontsize=10,
            family="Geist Mono", color=sf.DEEPPINK)
    ax.add_patch(plt.Rectangle((1.75, 0.5), 0.8, 0.5, facecolor="#F4DCE6", edgecolor=sf.DEEPPINK,
                               linewidth=1.0, linestyle="--", zorder=0))
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10)
    ax.set_xlim(-0.6, 2.7)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC for this skin-tone group")
    ax.legend(fontsize=9, loc="lower left")
    sf.title(ax, "Fair where measurable -- rebalancing can't reach dark skin")
    sf.save(fig, HERE, "equity_before_after")
    sf.save_raw(sf.pd.DataFrame({"tone": groups, "baseline": [b[g] for g in groups],
                                 "after_rebalance": [m[g] for g in groups]}), HERE, "equity_before_after")


if __name__ == "__main__":
    tone_distribution()                                            # the 1029 / 454 / 11 headline
    equity_before_after()                                          # already fair; the fix is data, not resampling
    print("G4 result figures regenerated")
