"""Shared base for every Day-N figure script in this repo.

ONE place that sets the matplotlib brand style and saves figures, so the figure
background ALWAYS matches the deck's slide canvas (no stark white rectangle) and
nobody can hardcode facecolor="white" again. Import from here:

    from figbase import plt, np, save, figtitle, txt_on, INK, MUTED, CANVAS
    from figbase import TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET

`save()` writes with the theme canvas facecolor and runs the overflow check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/_shared"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401  (re-exported)
import numpy as np  # noqa: E402,F401  (re-exported)

# Stable, long-standing mpl_style API (required).
from mpl_style import (  # noqa: E402,F401  (title re-exported)
    apply_style, theme_colors, title,
    TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET, GOLD,
)

# Newer helpers — import defensively so the figures still build on a machine
# whose superstack is a little behind (older ~/.claude/skills/_shared/mpl_style.py).
try:
    from mpl_style import text_on_brand_fill  # noqa: E402
except ImportError:
    def text_on_brand_fill(fill_hex):
        """Fallback: WCAG-luminance pick of black/white text on a fill."""
        h = fill_hex.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.5 else "white"

try:
    from mpl_style import warn_text_overflow  # noqa: E402
except ImportError:
    def warn_text_overflow(fig, *, source=""):  # no-op on older superstack
        return None

try:
    from mpl_style import warn_box_padding  # noqa: E402
except ImportError:
    def warn_box_padding(fig, *, source=""):  # no-op on older superstack
        return None

# The deck theme. MUST match the theme frozen in the .md.layout.json sidecar
# (currently "bone"). If the deck theme changes, change this too.
THEME = "bone"
apply_style(THEME)
CANVAS = theme_colors(THEME).canvas

INK = "#14141C"
MUTED = "#555560"
PAPER = "#FAFAF7"
DPI = 200

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Real, openly-licensed medical images (PD/CC0/CC-BY) for slides where photographic
# realism is the payload. See realimg/CREDITS.md for sources + licenses.
REALIMG = Path(__file__).resolve().parent / "realimg"

# canonical contrast helper (ink on turquoise/amber, white on deeppink/blueviolet)
txt_on = text_on_brand_fill


def save(fig, name):
    """Save a figure with the deck canvas as background, then warn on overflow."""
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor=CANVAS)
    warn_text_overflow(fig, source=name)
    warn_box_padding(fig, source=name)
    plt.close(fig)
    print("wrote", name)


def figtitle(fig, text, *, color=INK, y=1.03):
    fig.suptitle(text, fontsize=16, fontweight="bold", family="Geist Mono", color=color, y=y)
