"""Shared base for every Day-N figure script in this repo.

Thin course wrapper over the standalone **build-figure** skill (`brandfig`), so the
deck figures and the notebook figures share ONE foundation: the brand style, the
theme-matched canvas (no stark white rectangle on the bone slide), and the save-time
QA (text overflow + box padding). Import from here:

    from figbase import plt, np, save, figtitle, txt_on, INK, MUTED, CANVAS
    from figbase import TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET

`save()` writes with the theme canvas facecolor and runs the QA checks.

If the build-figure skill isn't installed (a machine a little behind), we fall back
to the shared `_shared/mpl_style.py` directly, so figures still build.
"""
import sys
from pathlib import Path

# The build-figure skill (brandfig) + its _shared dependency.
for _p in (Path.home() / ".claude/skills/build-figure",
           Path.home() / ".claude/skills/_shared"):
    sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401  (re-exported)
import numpy as np  # noqa: E402,F401  (re-exported)

# The deck theme. MUST match the theme frozen in the .md.layout.json sidecar.
THEME = "bone"

# Course text constants (kept stable so figures don't shift if a theme palette
# is retuned upstream): slightly darker ink + a readable muted than the raw theme.
INK = "#14141C"
MUTED = "#555560"
PAPER = "#FAFAF7"
DPI = 200

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Real, openly-licensed medical images (PD/CC0/CC-BY) for slides where photographic
# realism is the payload. See realimg/CREDITS.md for sources + licenses.
REALIMG = Path(__file__).resolve().parent / "realimg"

try:
    import brandfig as _bf  # the build-figure skill

    _bf.use(THEME)
    CANVAS = _bf.canvas(THEME)
    txt_on = _bf.txt_on
    from brandfig import TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET, GOLD  # noqa: F401
    from mpl_style import title  # noqa: F401  (ax-title helper, re-exported)

    def save(fig, name):
        """Save with the deck canvas as background, then run QA (overflow + box padding)."""
        _bf.save(fig, OUT / name, theme=THEME, dpi=DPI, source=name)
        plt.close(fig)
        print("wrote", name)

except ImportError:
    # Fallback: drive the shared style module directly (pre-build-figure behavior).
    from mpl_style import (  # noqa: E402,F401
        apply_style, theme_colors, title,
        TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET, GOLD,
    )
    try:
        from mpl_style import text_on_brand_fill as txt_on
    except ImportError:
        def txt_on(fill_hex):
            h = fill_hex.lstrip("#")
            r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
            return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.5 else "white"

    def _noop(fig, *, source=""):
        return None

    try:
        from mpl_style import warn_text_overflow
    except ImportError:
        warn_text_overflow = _noop
    try:
        from mpl_style import warn_box_padding
    except ImportError:
        warn_box_padding = _noop

    apply_style(THEME)
    CANVAS = theme_colors(THEME).canvas

    def save(fig, name):
        fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor=CANVAS)
        warn_text_overflow(fig, source=name)
        warn_box_padding(fig, source=name)
        plt.close(fig)
        print("wrote", name)


# canonical contrast helper (re-exported name used across figure scripts)
def figtitle(fig, text, *, color=INK, y=1.03):
    fig.suptitle(text, fontsize=16, fontweight="bold", family="Geist Mono", color=color, y=y)
