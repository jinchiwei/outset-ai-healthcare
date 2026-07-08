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


def two_clusters(n=30, seed=0, blue_center=(4.0, 1.5), pink_center=(1.5, 4.0), spread=0.7):
    """Two blobs for the intro: a BLUE cluster and a PINK cluster. Returns (blue, pink),
    each an (n, 2) array of points. Change the centers to move the groups around."""
    import numpy as _np
    g = _np.random.default_rng(seed)
    blue = g.normal(blue_center, spread, (n, 2))
    pink = g.normal(pink_center, spread, (n, 2))
    return blue, pink


def show_line(blue, pink, m, b, text=None):
    """Draw the blue & pink clusters and the line y = m*x + b, then report how many points
    the line separates (whichever side each color falls on). For the intro's 'pick a line'."""
    import numpy as _np
    blue = _np.asarray(blue, float); pink = _np.asarray(pink, float)
    lo = min(blue.min(), pink.min()) - 0.5
    hi = max(blue.max(), pink.max()) + 0.5
    f, ax = fig(figsize=(5, 4)); ax.grid(False)
    xs = _np.array([lo, hi])
    ax.plot(xs, m * xs + b, color=INK, lw=2, zorder=2)
    ax.scatter(blue[:, 0], blue[:, 1], color=TURQUOISE, s=45, edgecolor=INK, linewidth=0.4, label="blue")
    ax.scatter(pink[:, 0], pink[:, 1], color=DEEPPINK, s=45, edgecolor=INK, linewidth=0.4, label="pink")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right")
    pink_above = int((pink[:, 1] > m * pink[:, 0] + b).sum())
    blue_below = int((blue[:, 1] < m * blue[:, 0] + b).sum())
    total = len(blue) + len(pink)
    correct = max(pink_above + blue_below, total - pink_above - blue_below)
    title(f, text or f"your line separates {correct}/{total} points")
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
