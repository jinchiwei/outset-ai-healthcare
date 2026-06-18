"""Day 3 panel figures: fill the sparse 3-card slides with brand-styled diagrams.

Each panel = colored header band + a big brand-colored step number + caption text,
so the capstone/process slides read as intentional diagrams instead of text on
whitespace. Reuses the Day 2 card primitives. Output: slides/figures/d3_panel_*.png

Run:  python slides/figures_day3_panels.py
"""
from matplotlib.patches import FancyBboxPatch, Circle  # noqa: F401

from figbase import plt, save, figtitle, txt_on, INK, MUTED  # noqa: F401
from figbase import TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET  # noqa: F401

_COL = {"turquoise": TURQUOISE, "deeppink": DEEPPINK, "amber": AMBER, "blueviolet": BLUEVIOLET}


def _card(ax, label, color, body):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="#FBFAF6", edgecolor="#E3E0D6", lw=1.3))
    ax.add_patch(FancyBboxPatch((0.03, 0.80), 0.94, 0.17, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=color, edgecolor="none"))
    # auto-shrink so nothing spills past the card (mono header ~18 chars at full size,
    # body lines ~25 chars; scale down past that).
    hfs = min(12.5, 12.5 * 18 / max(len(label), 18))
    longest = max((len(ln) for ln in body.split("\n")), default=1)
    bfs = min(10.6, 10.6 * 25 / max(longest, 25))
    ax.text(0.5, 0.885, label, ha="center", va="center", fontsize=hfs, fontweight="bold",
            color=txt_on(color), family="Geist Mono")
    ax.text(0.5, 0.165, body, ha="center", va="center", fontsize=bfs, color=INK)


def _panels(fname, title, items, badges=None):
    """items: list of (label, color_name, body). badges: per-panel big glyph (default 1..n)."""
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.05 * n + 0.6, 4.7))
    if n == 1:
        axes = [axes]
    for k, (ax, (label, cname, body)) in enumerate(zip(axes, items)):
        color = _COL[cname]
        _card(ax, label, color, body)
        badge = badges[k] if badges else str(k + 1)
        ax.add_patch(Circle((0.5, 0.52), 0.155, facecolor=color, edgecolor="none", alpha=0.16))
        ax.text(0.5, 0.52, badge, ha="center", va="center", fontsize=44 if len(str(badge)) == 1 else 30,
                color=color, fontweight="bold", family="Geist Mono")
    figtitle(fig, title)
    save(fig, fname)


def toolkit():
    _panels("d3_panel_toolkit.png", "Everything in your toolkit now",
            [("MODELS", "turquoise", "logreg, MLP, CNN, transfer\nlearning, ViT, TabPFN"),
             ("METHOD", "deeppink", "train/val/test, gradient descent,\naugmentation, overfitting"),
             ("JUDGMENT", "amber", "sensitivity vs specificity,\nconfusion matrices, leakage, fairness")])


def capstone():
    _panels("d3_panel_capstone.png", "The capstone format",
            [("PAIRS", "turquoise", "two people, one project;\nboth build, both present"),
             ("BUILD SPRINT", "amber", "~90 minutes; check-ins\nat 3:15 and 4:00"),
             ("PRESENT", "deeppink", "three minutes per pair\nat 4:30; show what you built")])


def how_to_choose():
    _panels("d3_panel_choose.png", "How to choose your project",
            [("CLOSEST TO WHAT YOU KNOW", "turquoise", "pneumonia is the most like\nDay 2: safe and satisfying"),
             ("MOST INTERESTING TO YOU", "deeppink", "melanoma feels high-stakes\nand real; follow curiosity"),
             ("MOST ROOM TO EXPLORE", "amber", "MedMNIST has a dozen sets;\npick a weird one")])


def step1():
    _panels("d3_panel_step1.png", "Step 1: get a baseline running",
            [("MAKE IT RUN", "turquoise", "end to end, even if the\naccuracy is terrible"),
             ("GET A NUMBER", "amber", "that first number is\nyour baseline to beat"),
             ("ONLY NOW, IMPROVE", "deeppink", "a change you cannot measure\nis one you cannot trust")])


def step2():
    _panels("d3_panel_step2.png", "Step 2: improve, one change at a time",
            [("ONE CHANGE", "turquoise", "isolate it, re-run,\ncompare to baseline"),
             ("KEEP WHAT HELPS", "deeppink", "did the number move\nbeyond noise?"),
             ("LOG IT", "amber", "'augmentation: +3 points'\nis half your presentation")])


def step3():
    _panels("d3_panel_step3.png", "Step 3: debug like an engineer",
            [("READ THE ERROR", "turquoise", "the message usually says\nexactly what is wrong"),
             ("PRINT SHAPES", "amber", "most bugs are a shape\nmismatch; print(x.shape)"),
             ("THEN ASK CLAUDE", "deeppink", "paste error + code, then\nunderstand the fix")])


def good_eval():
    _panels("d3_panel_eval.png", "What makes a good evaluation",
            [("HELD-OUT TEST", "turquoise", "never seen in training;\nthe only honest grade"),
             ("THE RIGHT METRIC", "deeppink", "for screening, a missed case\nbeats a false alarm"),
             ("LOOK AT FAILURES", "amber", "the wrong cases are the best\nslide in your talk")])


def three_min():
    _panels("d3_panel_talk.png", "The three-minute talk",
            [("SHOW IT", "turquoise", "run the model live, or\nshow one clear result"),
             ("ONE FINDING", "amber", "the single most interesting\nthing you learned"),
             ("ONE LIMITATION", "deeppink", "what it gets wrong, or what\nyou'd do with more time")])


def breaks():
    _panels("d3_panel_breaks.png", "Where it still breaks",
            [("DATASET SHIFT", "turquoise", "a model trained at one\nhospital can fail at another"),
             ("THE WRONG METRIC", "deeppink", "high accuracy can still mean\nmost real cases are missed"),
             ("LEAKAGE, AGAIN", "amber", "the Day 2 trap is everywhere;\nhonest eval is the antidote")],
            badges=["~", "%", "!"])


if __name__ == "__main__":
    toolkit()
    breaks()
    capstone()
    how_to_choose()
    step1()
    step2()
    step3()
    good_eval()
    three_min()
    print("Day 3 panel figures done")
