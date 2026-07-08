"""Shared plotting + persistence for the Day-3 SOLUTION (answer-key) experiments.

Brand-locked to Jinchi's palette and the deck theme ("bone"), so every figure a solution
deck embeds matches the slide canvas. Also enforces the standing rule: whenever we save a
figure, we save the raw numbers behind it next to the PNG, so it can be re-themed later.

Usage in solutions/<group>/run_experiment.py:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
    import solfig as sf
    HERE = Path(__file__).resolve().parent
    fig, ax = sf.plt.subplots(...)
    sf.save(fig, HERE, "onehot_vs_kmer")            # -> figures/onehot_vs_kmer.png
    sf.save_raw(df, HERE, "onehot_vs_kmer")         # -> figures/raw/onehot_vs_kmer.csv
    sf.save_results({...}, HERE)                     # -> results.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/_shared"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401  (re-exported)
import numpy as np  # noqa: E402,F401  (re-exported)
import pandas as pd  # noqa: E402,F401  (re-exported)

from mpl_style import (  # noqa: E402,F401
    apply_style, theme_colors, title,
    TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET, GOLD,
)

THEME = "bone"
apply_style(THEME)
CANVAS = theme_colors(THEME).canvas
INK = "#14141C"
MUTED = "#555560"
DPI = 200

# Brand palette in priority order, for categorical series (lines/outlines/text -> amber;
# solid fills -> gold, per the branding rule).
PALETTE = [TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET]
FILL = [TURQUOISE, DEEPPINK, GOLD, BLUEVIOLET]


def _figs(base):
    d = Path(base) / "figures"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d


def save(fig, base, name):
    """Save a PNG with the deck canvas as background so it drops onto a slide seamlessly."""
    d = _figs(base)
    fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor=CANVAS)
    plt.close(fig)
    print("wrote figure", name)


def save_raw(obj, base, name):
    """Persist the numbers behind a figure. DataFrame/Series -> csv; dict/list -> json."""
    d = _figs(base) / "raw"
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.to_csv(d / f"{name}.csv", index=isinstance(obj, pd.Series))
    else:
        (d / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
    print("wrote raw", name)


def save_results(obj, base):
    """Write the deck's headline numbers to results.json (the notebook + deck read this)."""
    (Path(base) / "results.json").write_text(json.dumps(obj, indent=2, default=float))
    print("wrote results.json")


def shap_bar(model, X, feature_names, base, name, headline, max_feats=12):
    """SHAP feature-importance bar: mean |SHAP value| per feature (which inputs drive the model)."""
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                                  # (n, features, classes)
        sv = sv[..., -1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:max_feats]
    names = [feature_names[i] for i in order][::-1]
    vals = imp[order][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.35 * len(names) + 1.2))
    ax.barh(names, vals, color=TURQUOISE, edgecolor=INK, linewidth=0.5)
    ax.set_xlabel("mean |SHAP| (how much this feature moves the prediction)")
    title(ax, headline)
    save(fig, base, name)
    save_raw(pd.DataFrame({"feature": names[::-1], "importance": vals[::-1]}), base, name)
    return dict(zip(names[::-1], vals[::-1]))
