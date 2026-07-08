"""G5 heart disease from a checkup -- worked to completion. Priority: fairness across sex.

Compares four models, asks whether cholesterol alone carries the signal (feature ablation), and
audits accuracy separately for women and men. Tabular = the model is easy; the lessons are about
which clues matter and who the model works for.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks/day3_capstone"))
import solfig as sf
import capstone_tabular as ct
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
np = sf.np


def _acc(df, features, model):
    X = df[features].astype(float).values; y = df[ct_target].astype(int).values
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    m = ct.make_models()[model](); m.fit(Xa, ya)
    return float((np.asarray(m.predict(Xb)).ravel() == yb).mean())


def main():
    global ct_target
    df, meta = ct.load_heart(); ct_target = meta["target"]
    feats = meta["features"]
    results = {"n": int(len(df)), "positive": meta["positive"]}

    # 1. model comparison
    models = list(ct.make_models())
    accs = [_acc(df, feats, m) for m in models]
    fig, ax = sf.plt.subplots(figsize=(6.4, 3.4))
    ax.bar(models, accs, color=[sf.TURQUOISE, sf.DEEPPINK, sf.GOLD, sf.BLUEVIOLET], edgecolor=sf.INK, linewidth=0.5)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, family="Geist Mono")
    base = max(df[ct_target].mean(), 1 - df[ct_target].mean())
    ax.axhline(base, ls="--", color=sf.MUTED, lw=1, label=f"guess-majority {base:.2f}")
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
    sf.plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    sf.title(ax, "Four models, similar scores -- the data sets the ceiling")
    sf.save(fig, HERE, "model_comparison")
    sf.save_raw(sf.pd.DataFrame({"model": models, "accuracy": accs}), HERE, "model_comparison")
    results["model_acc"] = dict(zip(models, accs)); results["best_acc"] = max(accs)

    # 2. feature ablation: does cholesterol alone carry it?
    setups = {"all 13 features": feats, "cholesterol only": ["chol"], "everything except chol": [f for f in feats if f != "chol"]}
    av = {k: _acc(df, v, "Logistic Regression") for k, v in setups.items()}
    fig, ax = sf.plt.subplots(figsize=(5.6, 3.4))
    ax.bar(list(av), list(av.values()), color=[sf.TURQUOISE, sf.GOLD, sf.BLUEVIOLET], edgecolor=sf.INK, linewidth=0.5)
    for i, v in enumerate(av.values()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, family="Geist Mono")
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy")
    sf.plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
    sf.title(ax, "Cholesterol alone is a weak predictor")
    sf.save(fig, HERE, "feature_ablation")
    sf.save_raw(sf.pd.Series(av, name="accuracy"), HERE, "feature_ablation")
    results["ablation"] = av

    # 3. fairness by sex
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.25, random_state=0, stratify=df[ct_target])
    m = ct.make_models()["Logistic Regression"]()
    m.fit(df.iloc[tr][feats].astype(float).values, df.iloc[tr][ct_target].astype(int).values)
    sub = df.iloc[te]; pred = np.asarray(m.predict(sub[feats].astype(float).values)).ravel()
    truth = sub[ct_target].astype(int).values
    gmap = meta["group_names"]
    groups = sorted(sub[meta["group"]].dropna().unique())
    gaccs = [float((pred[sub[meta["group"]].values == g] == truth[sub[meta["group"]].values == g]).mean()) for g in groups]
    gn = [gmap.get(g, str(g)) for g in groups]
    fig, ax = sf.plt.subplots(figsize=(4.2, 3.4))
    ax.bar(gn, gaccs, color=[sf.DEEPPINK, sf.TURQUOISE], edgecolor=sf.INK, linewidth=0.5)
    for i, v in enumerate(gaccs):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, family="Geist Mono")
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy")
    sf.title(ax, "Accurate overall -- but not equally for women and men")
    sf.save(fig, HERE, "fairness_by_sex")
    sf.save_raw(sf.pd.DataFrame({"group": gn, "accuracy": gaccs}), HERE, "fairness_by_sex")
    results["fairness_by_sex"] = dict(zip(gn, gaccs)); results["fairness_gap"] = float(max(gaccs) - min(gaccs))
    # sex balance of the cohort
    results["sex_balance"] = {gmap.get(g, str(g)): int((df[meta["group"]] == g).sum()) for g in groups}

    sf.save_results(results, HERE)
    print("\nG5 DONE:", {k: v for k, v in results.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
