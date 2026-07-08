"""G1 CRISPR -- run the whole worked project to completion, save branded figures + raw data.

Produces (in figures/ and figures/raw/):
  gc_vs_efficiency   -- the one-number baseline (GC content vs cutting efficiency)
  representation_auc -- THE result: one-hot (keeps order) vs k-mer (ignores order), across 4 models
  regression_fit     -- predicting the exact 0-1 efficiency, one-hot + Random Forest
  position_importance-- where along the guide the model relies (recovers the seed near the PAM)
and results.json with the headline numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks/day3_capstone"))
import solfig as sf
import capstone_seq as cs
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
np = sf.np


def main():
    df, meta = cs.load_guides()
    results = {"n_guides": int(len(df)), "n_genes": int(df["gene"].nunique()),
               "high_eff_frac": float(df["high_efficiency"].mean())}

    # 1. GC-content baseline ----------------------------------------------------
    r_gc = float(np.corrcoef(df["gc_content"], df["efficiency"])[0, 1])
    fig, ax = sf.plt.subplots(figsize=(5, 4))
    ax.scatter(df["gc_content"], df["efficiency"], s=7, alpha=0.3, color=sf.BLUEVIOLET)
    ax.set_xlabel("GC content of the guide"); ax.set_ylabel("cutting efficiency")
    sf.title(ax, f"One number is not enough  (correlation {r_gc:+.2f})")
    sf.save(fig, HERE, "gc_vs_efficiency")
    sf.save_raw(df[["gc_content", "efficiency"]], HERE, "gc_vs_efficiency")
    results["gc_corr"] = r_gc

    # 2. Representation battle: one-hot vs k-mer x 4 models ----------------------
    rows = []
    for mode in ["onehot", "kmer"]:
        X, _ = cs.featurize(df, mode, "guide20")
        y = df["high_efficiency"].astype(int).values
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        for name, mk in cs.make_classifiers().items():
            m = mk(); m.fit(Xa, ya)
            auc = roc_auc_score(yb, m.predict_proba(Xb)[:, 1])
            acc = float((np.asarray(m.predict(Xb)).ravel() == yb).mean())
            rows.append({"model": name, "representation": mode, "auc": auc, "accuracy": acc})
    battle = sf.pd.DataFrame(rows)

    models = list(cs.make_classifiers())
    x = np.arange(len(models)); w = 0.38
    fig, ax = sf.plt.subplots(figsize=(8, 4.2))
    for i, (mode, col) in enumerate([("onehot", sf.TURQUOISE), ("kmer", sf.GOLD)]):
        vals = [battle[(battle.model == mdl) & (battle.representation == mode)].auc.iloc[0] for mdl in models]
        ax.bar(x + (i - 0.5) * w, vals, w, label=("one-hot (keeps order)" if mode == "onehot" else "k-mer (ignores order)"),
               color=col, edgecolor=sf.INK, linewidth=0.6)
    ax.axhline(0.5, ls="--", color=sf.MUTED, lw=1)
    ax.text(len(models) - 0.5, 0.51, "coin flip", fontsize=8, color=sf.MUTED, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC  (higher = better)"); ax.set_ylim(0.45, max(battle.auc) + 0.05)
    ax.legend(fontsize=9, loc="upper left")
    sf.title(ax, "Keeping the letter order wins")
    sf.save(fig, HERE, "representation_auc")
    sf.save_raw(battle, HERE, "representation_auc")
    best = battle.sort_values("auc", ascending=False).iloc[0]
    results["best_auc"] = float(best.auc)
    results["best_model"] = best.model
    results["best_representation"] = best.representation
    oh = battle[battle.representation == "onehot"].auc.mean()
    km = battle[battle.representation == "kmer"].auc.mean()
    results["onehot_mean_auc"] = float(oh); results["kmer_mean_auc"] = float(km)

    # 3. Regression: predict the exact efficiency -------------------------------
    X, _ = cs.featurize(df, "onehot", "guide20")
    yv = df["efficiency"].astype(float).values
    Xa, Xb, ya, yb = train_test_split(X, yv, test_size=0.25, random_state=0)
    m = cs.make_regressors()["Random Forest"](); m.fit(Xa, ya)
    pred = m.predict(Xb); r2 = float(r2_score(yb, pred)); corr = float(np.corrcoef(yb, pred)[0, 1])
    fig, ax = sf.plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(yb, pred, s=9, alpha=0.4, color=sf.TURQUOISE)
    ax.plot([0, 1], [0, 1], "--", color=sf.MUTED, lw=1)
    ax.set_xlabel("true efficiency"); ax.set_ylabel("predicted efficiency")
    sf.title(ax, f"Predicting the exact score  (R² {r2:.2f}, r {corr:.2f})")
    sf.save(fig, HERE, "regression_fit")
    sf.save_raw(sf.pd.DataFrame({"true": yb, "pred": pred}), HERE, "regression_fit")
    results["reg_r2"] = r2; results["reg_corr"] = corr

    # 4. Position importance (the biology reveal) -------------------------------
    from sklearn.linear_model import LogisticRegression
    Xo, _ = cs.featurize(df, "onehot", "guide20")
    yo = df["high_efficiency"].astype(int).values
    lr = LogisticRegression(max_iter=2000).fit(Xo, yo)
    L = 20
    imp = np.abs(lr.coef_[0]).reshape(L, 4).sum(axis=1)
    top = int(np.argmax(imp)) + 1
    colors = [sf.DEEPPINK if p + 1 >= L - 4 else sf.TURQUOISE for p in range(L)]
    fig, ax = sf.plt.subplots(figsize=(8, 3.6))
    ax.bar(range(1, L + 1), imp, color=colors, edgecolor=sf.INK, linewidth=0.4)
    ax.set_xlabel("position along the 20-letter guide  (right end = next to the PAM / cut site)")
    ax.set_ylabel("importance")
    ax.annotate("the seed region\n(next to the PAM)", xy=(L - 1.5, imp[-3:].max()),
                xytext=(L - 7, imp.max() * 0.9), fontsize=9, color=sf.DEEPPINK,
                arrowprops=dict(arrowstyle="->", color=sf.DEEPPINK))
    sf.title(ax, f"The model rediscovered CRISPR biology  (peak at position {top})")
    sf.save(fig, HERE, "position_importance")
    sf.save_raw(sf.pd.DataFrame({"position": range(1, L + 1), "importance": imp}), HERE, "position_importance")
    results["top_position"] = top

    sf.save_results(results, HERE)
    print("\nG1 DONE:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in results.items()})


if __name__ == "__main__":
    main()
