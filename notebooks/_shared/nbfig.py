"""nbfig -- Colab-safe branded matplotlib for the course notebooks.

A tiny, dependency-free (matplotlib + numpy only) mirror of the superstack
`build-figure` skill's brand, so inline notebook plots look like the slide
figures even on Colab, where the skill isn't installed. Same palette, same warm
"bone" canvas, monospace titles (Geist Mono if present, else DejaVu fallback).

    import nbfig
    nbfig.use()                          # apply the brand style (inline plots adopt it)
    fig, ax = nbfig.fig(figsize=(6, 3))
    ax.plot(loss, color=nbfig.TURQUOISE)
    nbfig.show(fig, "Training loss")     # titled, tight, displayed

Author-time concept diagrams are made with the build-figure skill and committed
as PNGs; this is only for the live, in-notebook plots (sample images, training
curves, confusion matrices, predictions).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np  # noqa: F401  (re-exported for convenience)

# brand palette (matches _shared/mpl_style.py)
TURQUOISE = "#40E0D0"
DEEPPINK = "#FF1493"
AMBER = "#F0C840"
BLUEVIOLET = "#8A2BE2"
GOLD = "gold"
PALETTE = [TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET]

CANVAS = "#F6F4EE"   # bone
INK = "#14141C"
MUTED = "#555560"
RULE = "#D8D4C8"

_MONO = ["Geist Mono", "DejaVu Sans Mono", "monospace"]
_SANS = ["Geist", "DejaVu Sans", "sans-serif"]


_FONTS_REGISTERED = False


def _register_fonts():
    """Register the bundled Geist / Geist Mono TTFs so plots look branded even on
    Colab (where the fonts aren't installed). Also quiets matplotlib's per-figure
    'Font family Geist not found' warnings. Idempotent; safe if fonts are missing."""
    global _FONTS_REGISTERED
    # always silence the font-lookup warning, even if the TTFs aren't present
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if _FONTS_REGISTERED:
        return
    import os
    from matplotlib import font_manager
    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    if os.path.isdir(fonts_dir):
        for fn in os.listdir(fonts_dir):
            if fn.lower().endswith((".ttf", ".otf")):
                try:
                    font_manager.fontManager.addfont(os.path.join(fonts_dir, fn))
                except Exception:
                    pass
    _FONTS_REGISTERED = True


def use():
    """Apply the brand matplotlib defaults. Inline plots afterward adopt the canvas."""
    _register_fonts()
    # Force the inline backend under Jupyter/Colab so figures render in the
    # notebook instead of popping a native window (which on macOS shows a
    # "save the photo" dialog). No-op outside IPython.
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass
    mpl.rcParams.update({
        "font.family": _SANS,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": RULE,
        "axes.facecolor": CANVAS, "figure.facecolor": CANVAS, "savefig.facecolor": CANVAS,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.grid": True, "grid.color": RULE, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def fig(*args, **kwargs):
    """plt.subplots with the figure facecolor set to the bone canvas."""
    f, ax = plt.subplots(*args, **kwargs)
    f.set_facecolor(CANVAS)
    return f, ax


def title(fig, text):
    """Monospace suptitle in ink."""
    fig.suptitle(text, fontsize=15, fontweight="bold", family=_MONO, color=INK, y=1.02)


def show(fig, text=None):
    """Optionally title, then display inline (canvas-matched)."""
    if text:
        title(fig, text)
    fig.set_facecolor(CANVAS)
    plt.show()


def txt_on(fill_hex: str) -> str:
    """High-contrast text color (black/white) for text on a brand fill."""
    h = fill_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.5 else "white"


def palette(n: int):
    """n brand colors, cycling."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


class _Plot:
    """Wraps a figure so a chained `.show()` displays it inline via plt.show()
    instead of fig.show(), which warns 'FigureCanvasAgg is non-interactive'
    under the inline backend (whose canvas is Agg-based)."""
    def __init__(self, fig):
        self.fig = fig

    def show(self):
        plt.show()
        return None

    def __repr__(self):   # keep bare use from printing an object repr
        return ""


def confusion(y, pred, labels, normalize=True, text="Confusion matrix"):
    """Branded confusion-matrix heatmap. Returns the figure."""
    import numpy as _np
    y, pred = _np.asarray(y), _np.asarray(pred)
    k = len(labels)
    m = _np.zeros((k, k), float)
    for t, q in zip(y, pred):
        m[int(t), int(q)] += 1
    shown = m / m.sum(1, keepdims=True).clip(min=1) if normalize else m
    f, ax = fig(figsize=(4.6, 4.0))
    ax.grid(False)
    im = ax.imshow(shown, cmap="BuPu", vmin=0, vmax=shown.max() or 1)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=10, rotation=90, va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(k):
        for j in range(k):
            cell = f"{shown[i, j]:.0%}" if normalize else f"{int(m[i, j])}"
            ax.text(j, i, cell, ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if shown[i, j] > (shown.max() or 1) * 0.55 else INK,
                    family=_MONO)
    title(f, text)
    return _Plot(f)


def scatter_line(x, y, m=None, b=None, labels=False, text=None):
    """Branded scatter of points + optional best-fit line y=m*x+b + optional (x,y) labels.

    For the intro notebook: nbfig.scatter_line(x, y, *np.polyfit(x, y, 1), labels=True).show()
    """
    import numpy as _np
    x = _np.asarray(x, float); y = _np.asarray(y, float)
    f, ax = fig(figsize=(5, 4))
    if m is not None:
        xs = _np.linspace(x.min(), x.max(), 2)
        ax.plot(xs, m * xs + b, color=DEEPPINK, lw=2.5, zorder=2)
    ax.scatter(x, y, color=TURQUOISE, s=80, zorder=3, edgecolor=INK, linewidth=0.6)
    if labels:
        for xi, yi in zip(x, y):
            ax.annotate(f"({xi:g}, {yi:g})", (xi, yi), textcoords="offset points",
                        xytext=(7, 6), fontsize=8, color=MUTED, family=_MONO)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if text:
        title(f, text)
    return _Plot(f)


def boundary(model, X, y, text=None, feature_names=("x", "y")):
    """Branded 2-class decision boundary for any fitted sklearn model with .predict."""
    import numpy as _np
    X = _np.asarray(X, float); y = _np.asarray(y).astype(int)
    pad = 0.4 * (X.max(0) - X.min(0)).clip(min=1e-6)
    xx, yy = _np.meshgrid(
        _np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], 200),
        _np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], 200))
    Z = model.predict(_np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    f, ax = fig(figsize=(5, 4)); ax.grid(False)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[TURQUOISE, DEEPPINK], alpha=0.22)
    ax.scatter(X[:, 0], X[:, 1], c=[DEEPPINK if v else TURQUOISE for v in y],
               s=55, edgecolor=INK, linewidth=0.5, zorder=3)
    ax.set_xlabel(feature_names[0]); ax.set_ylabel(feature_names[1])
    if text:
        title(f, text)
    return _Plot(f)


def two_clusters(n=6, seed=0, healthy_center=(4.0, 1.5), diseased_center=(1.5, 4.0), spread=0.7, round_to=1):
    """A few HEALTHY patients (turquoise) and a few DISEASED patients (deeppink) for the intro,
    each a point (x, y) = two measurements. Returns (healthy, diseased), each an (n, 2) array.
    Kept small (n=6) with rounded coords so the coordinates fit in a table."""
    import numpy as _np
    g = _np.random.default_rng(seed)
    healthy = g.normal(healthy_center, spread, (n, 2))
    diseased = g.normal(diseased_center, spread, (n, 2))
    if round_to is not None:
        healthy = _np.round(healthy, round_to); diseased = _np.round(diseased, round_to)
    return healthy, diseased


def xor_clusters(n=5, seed=0, diseased_centers=((1.0, 1.0), (6.0, 6.0)),
                 healthy_centers=((1.0, 6.0), (6.0, 1.0)), spread=0.6, round_to=1):
    """An XOR-shaped problem: TWO diseased blobs and TWO healthy blobs at opposite corners, so
    no single line can separate them. Returns (healthy, diseased), each a (2*n, 2) array."""
    import numpy as _np
    g = _np.random.default_rng(seed)

    def blobs(centers):
        out = _np.vstack([g.normal(c, spread, (n, 2)) for c in centers])
        return _np.round(out, round_to) if round_to is not None else out

    healthy = blobs(healthy_centers); diseased = blobs(diseased_centers)
    return healthy, diseased


def _scatter_groups(ax, healthy, diseased, lo, hi, s=80):
    ax.scatter(healthy[:, 0], healthy[:, 1], color=TURQUOISE, s=s, edgecolor=INK, linewidth=0.5, label="healthy", zorder=3)
    ax.scatter(diseased[:, 0], diseased[:, 1], color=DEEPPINK, s=s, edgecolor=INK, linewidth=0.5, label="diseased", zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, color=RULE, alpha=0.3, linewidth=0.5); ax.legend(loc="upper right")


def show_points(healthy, diseased, text=None, table=True):
    """Show the healthy & diseased patients as a labeled scatter. With table=True (default) also
    print a coordinate table beside it -- the 'look at where the points sit' step (x across, y up).
    Pass table=False when there are too many points for a tidy table."""
    import numpy as _np
    healthy = _np.asarray(healthy, float); diseased = _np.asarray(diseased, float)
    lo = min(healthy.min(), diseased.min()) - 0.5
    hi = max(healthy.max(), diseased.max()) + 0.5
    if not table:
        f, axs = fig(figsize=(5, 4)); axs.grid(False)
        _scatter_groups(axs, healthy, diseased, lo, hi)
        title(f, text or "our patients")
        return _Plot(f)
    f = plt.figure(figsize=(7.8, 3.8)); f.set_facecolor(CANVAS)
    axs = f.add_axes([0.07, 0.13, 0.48, 0.75]); axs.set_facecolor(CANVAS)
    axt = f.add_axes([0.60, 0.02, 0.38, 0.94]); axt.axis("off")
    _scatter_groups(axs, healthy, diseased, lo, hi)
    rows, row_colors = [], []
    for px, py in healthy:
        rows.append(["healthy", f"{px:.1f}", f"{py:.1f}"]); row_colors.append(TURQUOISE)
    for px, py in diseased:
        rows.append(["diseased", f"{px:.1f}", f"{py:.1f}"]); row_colors.append(DEEPPINK)
    tbl = axt.table(cellText=rows, colLabels=["patient", "x", "y"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE); cell.set_facecolor(CANVAS)
        t = cell.get_text(); t.set_family(_MONO)
        if r == 0:
            t.set_fontweight("bold"); t.set_color(INK)
        elif c == 0:
            t.set_color(row_colors[r - 1]); t.set_fontweight("bold")
        else:
            t.set_color(INK)
    title(f, text or "our patients and their measurements")
    return _Plot(f)


def xgrid(healthy, diseased, n=50):
    """The x-values to write your line over. Feed it the two groups; you then write
    `y = m*x + b` yourself and hand (x, y) to show_line. For the intro's 'draw a line'."""
    import numpy as _np
    healthy = _np.asarray(healthy, float); diseased = _np.asarray(diseased, float)
    lo = min(healthy[:, 0].min(), diseased[:, 0].min()) - 1
    hi = max(healthy[:, 0].max(), diseased[:, 0].max()) + 1
    return _np.linspace(lo, hi, n)


def show_line(healthy, diseased, x, y, text=None):
    """Draw the healthy & diseased patients and the line the student wrote as `y = m*x + b`
    (passed in as the arrays x and y), then report how many points it separates.

    The slope/height are recovered from the line's endpoints so scoring works for any
    y the student writes -- sloped (y = 1*x + 0) or flat (y = 0*x + 3.5)."""
    import numpy as _np
    healthy = _np.asarray(healthy, float); diseased = _np.asarray(diseased, float)
    x = _np.asarray(x, float); y = _np.asarray(y, float)
    # recover m, b from the line the student built, so we can score every point
    m = (y[-1] - y[0]) / (x[-1] - x[0])
    b = y[0] - m * x[0]
    lo = min(healthy.min(), diseased.min()) - 0.5
    hi = max(healthy.max(), diseased.max()) + 0.5
    f, ax = fig(figsize=(5, 4)); ax.grid(False)
    ends = _np.array([lo, hi])
    ax.plot(ends, m * ends + b, color=INK, lw=2, zorder=2)
    ax.scatter(healthy[:, 0], healthy[:, 1], color=TURQUOISE, s=45, edgecolor=INK, linewidth=0.4, label="healthy")
    ax.scatter(diseased[:, 0], diseased[:, 1], color=DEEPPINK, s=45, edgecolor=INK, linewidth=0.4, label="diseased")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right")
    # count each group strictly above / below the line; a point ON the line counts as
    # neither, so it's wrong under both possible label assignments.
    dis_above = int((diseased[:, 1] > m * diseased[:, 0] + b).sum())
    dis_below = int((diseased[:, 1] < m * diseased[:, 0] + b).sum())
    hlth_above = int((healthy[:, 1] > m * healthy[:, 0] + b).sum())
    hlth_below = int((healthy[:, 1] < m * healthy[:, 0] + b).sum())
    total = len(healthy) + len(diseased)
    correct = max(dis_above + hlth_below, dis_below + hlth_above)
    title(f, text or f"your line separates {correct}/{total} patients")
    return _Plot(f)


def _line_mb(x, y):
    """Recover (m, b) of y = m*x + b from the endpoints of the arrays x, y."""
    m = (y[-1] - y[0]) / (x[-1] - x[0])
    return m, y[0] - m * x[0]


def show_band(healthy, diseased, x, y1, y2, text=None):
    """Draw TWO lines (each written by the student as y = m*x + b) and shade the band between
    them, then score by 'diseased inside the band, healthy outside'. For the XOR task, where no
    single line works but a stripe holding one group does."""
    import numpy as _np
    healthy = _np.asarray(healthy, float); diseased = _np.asarray(diseased, float)
    x = _np.asarray(x, float)
    m1, b1 = _line_mb(x, _np.asarray(y1, float))
    m2, b2 = _line_mb(x, _np.asarray(y2, float))
    lo = min(healthy.min(), diseased.min()) - 0.5
    hi = max(healthy.max(), diseased.max()) + 0.5
    f, ax = fig(figsize=(5, 4.2)); ax.grid(False)
    ends = _np.array([lo, hi])
    L1 = m1 * ends + b1; L2 = m2 * ends + b2
    ax.fill_between(ends, L1, L2, color=AMBER, alpha=0.18, zorder=1)
    ax.plot(ends, L1, color=INK, lw=2, zorder=2); ax.plot(ends, L2, color=INK, lw=2, zorder=2)
    _scatter_groups(ax, healthy, diseased, lo, hi, s=55)

    def inside(P):
        a = m1 * P[:, 0] + b1; c = m2 * P[:, 0] + b2
        return (P[:, 1] > _np.minimum(a, c)) & (P[:, 1] < _np.maximum(a, c))

    d_in = int(inside(diseased).sum()); d_out = len(diseased) - d_in
    h_in = int(inside(healthy).sum()); h_out = len(healthy) - h_in
    total = len(healthy) + len(diseased)
    correct = max(d_in + h_out, d_out + h_in)
    title(f, text or f"your band separates {correct}/{total} patients")
    return _Plot(f)


def sample_digit():
    """A real MNIST-style '5' as a 28x28 array of brightness values (0 = black, 255 = white)."""
    import os
    import numpy as _np
    return _np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_digit.npy"))


def sample_retina():
    """One APTOS-2019 retina (fundus) photo -- the same dataset Day 1 uses -- as a 96x96 RGB
    array, values 0-255. Shape (96, 96, 3)."""
    import os
    import numpy as _np
    return _np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_retina.npy"))


def sample_retina_vessels():
    """Precomputed blood-vessel segmentation for the sample retina: a 96x96 True/False mask
    (True where the pixel is a vessel)."""
    import os
    import numpy as _np
    return _np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_retina_vessels.npy"))


def sample_retina_disc():
    """Precomputed optic-disc (optic nerve head) segmentation for the sample retina: a 96x96
    True/False mask (True where the optic disc is)."""
    import os
    import numpy as _np
    return _np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_retina_disc.npy"))


def show_segmentation(image, vessels, disc, text=None):
    """Show the retina, the vessel mask, the optic-disc mask, and a color overlay of both
    (vessels in turquoise, optic disc in deeppink)."""
    import numpy as _np
    image = _np.asarray(image)
    v = _np.asarray(vessels).astype(bool)
    d = _np.asarray(disc).astype(bool)
    over = image.copy()
    over[v] = [64, 224, 208]     # turquoise vessels
    over[d] = [255, 20, 147]     # deeppink optic disc
    return show_images([
        ("retina", image),
        ("vessels", (v * 255).astype("uint8")),
        ("optic disc", (d * 255).astype("uint8")),
        ("overlay", over),
    ], text=text or "segmentation: vessels (turquoise) + optic disc (pink)")


def show_images(items, text=None):
    """Show a row of images side by side, each with a title. `items` is a list of (title, image).
    Grayscale (2D) images use a gray colormap; color (3D) images show in color."""
    import numpy as _np
    n = len(items)
    f, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.5)); f.set_facecolor(CANVAS)
    if n == 1:
        axes = [axes]
    for ax, (t, img) in zip(axes, items):
        img = _np.asarray(img)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(_np.clip(img, 0, 255).astype("uint8"))
        ax.set_title(t, fontsize=9.5, family=_MONO, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(CANVAS)
    if text:
        title(f, text)
    return _Plot(f)


def mail_letter_diagram(text=None):
    """A teaching picture for Part 0: a function is like mailing a letter. Three inputs (stamp,
    receiver address, sender address) go in, the function does the work, and an output comes out."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    f, ax = fig(figsize=(9.6, 3.8)); ax.axis("off"); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(-0.7, 6.2)

    def box(x, y, w, h, label, fill, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                                    linewidth=1.5, edgecolor=INK, facecolor=fill))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, family=_MONO, color=txt_on(fill))

    inputs = ["stamp", "receiver\naddress", "sender\naddress"]
    ys = [4.35, 2.55, 0.75]
    for lab, y in zip(inputs, ys):
        box(0.35, y, 2.5, 1.3, lab, TURQUOISE)
    box(4.15, 1.5, 2.5, 3.0, "mail_letter(\n  stamp,\n  receiver,\n  sender)", AMBER)
    box(7.55, 2.4, 2.1, 1.3, "letter on\nits way!", DEEPPINK)

    for y in ys:
        ax.add_patch(FancyArrowPatch((2.85, y + 0.65), (4.15, 3.0), arrowstyle="-|>",
                                     mutation_scale=13, color=INK, lw=1.3))
    ax.add_patch(FancyArrowPatch((6.65, 3.0), (7.55, 3.05), arrowstyle="-|>",
                                 mutation_scale=15, color=INK, lw=1.6))

    ax.text(1.6, -0.5, "inputs (the arguments)", ha="center", fontsize=9, color=MUTED, family=_MONO)
    ax.text(5.4, 0.9, "the function does the work", ha="center", fontsize=9, color=MUTED, family=_MONO)
    ax.text(8.6, 1.9, "output (return value)", ha="center", fontsize=9, color=MUTED, family=_MONO)
    title(f, text or "a function is like mailing a letter")
    return _Plot(f)


def sample_feature_maps():
    """Precomputed feature maps from a PRETRAINED ResNet50 run on the sample retina, at three
    depths. Returns (early, middle, late), each a (4, 96, 96) uint8 stack of 4 channels."""
    import os
    import numpy as _np
    d = os.path.dirname(os.path.abspath(__file__))
    return (_np.load(os.path.join(d, "sample_feat_early.npy")),
            _np.load(os.path.join(d, "sample_feat_middle.npy")),
            _np.load(os.path.join(d, "sample_feat_late.npy")))


def show_feature_maps(early, middle, late, text=None):
    """Show what a trained network 'sees' at three depths: one row of channels per depth
    (early = fine edges, middle = textures, late = coarse/abstract)."""
    rows = [("early layer\n(fine edges)", early),
            ("middle layer\n(textures)", middle),
            ("late layer\n(coarse, abstract)", late)]
    k = early.shape[0]
    f, axes = plt.subplots(3, k, figsize=(2.0 * k, 6.4)); f.set_facecolor(CANVAS)
    for r, (label, stack) in enumerate(rows):
        for c in range(k):
            ax = axes[r, c]
            ax.imshow(stack[c], cmap="magma"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor(CANVAS)
            if c == 0:
                ax.set_ylabel(label, fontsize=9.5, family=_MONO, color=INK,
                              rotation=0, ha="right", va="center", labelpad=18)
    title(f, text or "what a trained network sees, layer by layer")
    return _Plot(f)


def mlp_diagram(text=None):
    """A branded picture of a small MLP: columns of neurons (input -> hidden -> output) fully
    connected. Ties Part 3 (one neuron = one line) to Day 1's stacked layers."""
    from matplotlib.patches import Circle
    import numpy as _np
    f, ax = fig(figsize=(8.6, 4.3)); ax.axis("off"); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 6.4)
    layers = [4, 5, 2]; xs = [1.5, 5.0, 8.5]
    colors = [TURQUOISE, AMBER, DEEPPINK]
    coords = [[(x, y) for y in _np.linspace(1.2, 4.8, n)] for n, x in zip(layers, xs)]
    for a, b in [(0, 1), (1, 2)]:                         # fully-connected wires
        for (x1, y1) in coords[a]:
            for (x2, y2) in coords[b]:
                ax.plot([x1, x2], [y1, y2], color=RULE, lw=0.8, zorder=1)
    labels = ["input\n(pixels)", "hidden layer\n(neurons)", "output\n(yes / no)"]
    for col, c, x, lab in zip(coords, colors, xs, labels):
        for (px, py) in col:
            ax.add_patch(Circle((px, py), 0.28, facecolor=c, edgecolor=INK, lw=1.2, zorder=3))
        ax.text(x, 0.2, lab, ha="center", fontsize=9, color=MUTED, family=_MONO)
    ax.text(5, 5.9, "each wire is one  y = m*x + b  (a neuron, from Part 3)",
            ha="center", fontsize=9.5, color=INK, family=_MONO)
    title(f, text or "an MLP = layers of neurons, stacked")
    return _Plot(f)


def loss_explainer(text=None):
    """A bespoke two-panel figure to narrate the loss. LEFT: the sigmoid, which turns 'how far
    above the line' into a probability p. RIGHT: the penalty -log(chance given to the true label),
    which is ~0 when confidently right and huge when confidently wrong. The loss is its average."""
    import numpy as _np
    f, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.2)); f.set_facecolor(CANVAS)
    for ax in (axL, axR):
        ax.set_facecolor(CANVAS); ax.grid(True, color=RULE, alpha=0.3, linewidth=0.5)

    z = _np.linspace(-6, 6, 300); s = 1.0 / (1.0 + _np.exp(-z))
    axL.plot(z, s, color=TURQUOISE, lw=2.8, zorder=3)
    axL.axhline(0.5, color=RULE, lw=1, ls="--"); axL.axvline(0, color=RULE, lw=1, ls="--")
    axL.set_ylim(-0.05, 1.10); axL.set_xlim(-6, 6)
    axL.set_xlabel("the line's raw score:   z = y - (m*x + b)")
    axL.set_ylabel("p  =  sigmoid(z)")
    axL.annotate("z very negative\n-> p goes to 0", (-5.7, 0.12), fontsize=8, color=MUTED, family=_MONO)
    axL.annotate("z very positive\n-> p goes to 1", (1.4, 0.84), fontsize=8, color=MUTED, family=_MONO)
    axL.set_title("sigmoid squashes any score into a 0-to-1 probability", fontsize=8.5)

    q = _np.linspace(0.008, 1.0, 300)
    axR.plot(q, -_np.log(q), color=DEEPPINK, lw=2.8, zorder=3)
    axR.set_ylim(0, 5); axR.set_xlim(0, 1)
    axR.set_xlabel("q  =  chance we gave the TRUE answer")
    axR.set_ylabel("penalty  =  -log(q)")
    axR.annotate("sure and RIGHT\n-> tiny penalty", (0.55, 0.35), fontsize=8, color=MUTED, family=_MONO)
    axR.annotate("sure and WRONG\n-> huge penalty", (0.06, 3.7), fontsize=8, color=MUTED, family=_MONO)
    axR.set_title("penalty is small only when we back the truth confidently", fontsize=8.5)

    title(f, text or "how the loss is calculated, one patient at a time")
    f.text(0.5, -0.01,
           "Plain terms: for each patient, take the chance we gave the TRUE answer; penalize being confidently wrong; average over all N = the loss.",
           ha="center", fontsize=8.5, color=INK, family=_SANS)
    return _Plot(f)


def show_learning(healthy, diseased, lines, losses, text=None):
    """Draw the boundary line at each epoch (gray = start, pink = learned) beside the loss curve.
    The training loop lives in the notebook; it hands us `lines` (a list of (m, b) recorded per
    epoch) and `losses` (the loss per epoch). This function only draws them."""
    import numpy as _np
    H = _np.asarray(healthy, float); D = _np.asarray(diseased, float)
    lo = min(H.min(), D.min()) - 0.5
    hi = max(H.max(), D.max()) + 0.5
    f, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.0)); f.set_facecolor(CANVAS)
    for ax in (axL, axR):
        ax.set_facecolor(CANVAS)
    axL.grid(False)
    ends = _np.array([lo, hi]); n = len(lines)
    snap = _np.unique(_np.linspace(0, n - 1, 8).astype(int))
    for i, e in enumerate(snap):
        m, b = lines[e]
        frac = i / (len(snap) - 1)
        col = DEEPPINK if e == n - 1 else str(round(0.78 - 0.62 * frac, 3))  # light gray -> dark
        axL.plot(ends, m * ends + b, color=col, lw=(2.6 if e == n - 1 else 1.2), zorder=2 + i)
    _scatter_groups(axL, H, D, lo, hi, s=45)
    axL.set_title("the line each epoch  (gray = start, pink = final)", fontsize=9.5)
    axR.plot(range(len(losses)), losses, color=DEEPPINK, lw=2.2)
    axR.set_xlabel("epoch"); axR.set_ylabel("loss (how wrong it is)")
    axR.set_ylim(0, max(losses) * 1.05)
    axR.grid(True, color=RULE, alpha=0.3, linewidth=0.5)
    axR.set_title("loss each epoch  (we want it to drop)", fontsize=9.5)
    title(f, text or "the computer learns m and b by gradient descent")
    return _Plot(f)


def learning_curve(history, text="Learning curve"):
    """Plot val accuracy + train loss per epoch from train_model's history.

    history is a list of (epoch, val_acc, train_loss) tuples.
    """
    ep = [h[0] for h in history]
    acc = [h[1] for h in history]
    loss = [h[2] for h in history] if len(history[0]) > 2 else None
    f, ax = fig(figsize=(6.4, 3.4))
    ax.plot(ep, acc, "-o", color=TURQUOISE, lw=2.2, label="val accuracy")
    ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy", color=INK)
    ax.set_ylim(0, 1)
    if loss is not None:
        ax2 = ax.twinx(); ax2.grid(False)
        ax2.plot(ep, loss, "--o", color=DEEPPINK, lw=2.0, label="train loss")
        ax2.set_ylabel("train loss", color=DEEPPINK)
        ax2.tick_params(axis="y", labelcolor=DEEPPINK)
    title(f, text)
    return _Plot(f)
