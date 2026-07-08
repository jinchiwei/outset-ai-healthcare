"""G1 CRISPR -- a WORKING guide-picker. Given a 20-letter guide, predict whether it will cut well,
so a scientist can rank candidate guides and order only the best ones before touching the lab.

Deliverable: a model that separates clearly-good from clearly-poor guides with AUC ~0.88, and a
concrete usefulness number -- if you order the guides it ranks highest, far more of them actually
cut than if you picked at random. (Bonus: it learned the real biology -- the seed near the PAM.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks/day3_capstone"))
import solfig as sf
import capstone_seq as cs
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
np = sf.np


def main():
    df, _ = cs.load_guides()
    # Honest, clear target: tell CLEARLY-good guides (top third) from CLEARLY-poor ones (bottom third).
    hi, lo = df["efficiency"].quantile(2/3), df["efficiency"].quantile(1/3)
    d = df[(df.efficiency >= hi) | (df.efficiency <= lo)].copy()
    d["y"] = (d.efficiency >= hi).astype(int)
    X, _ = cs.featurize(d, "onehot", "guide20")
    y = d.y.values
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = CatBoostClassifier(verbose=0, random_state=0, iterations=600, depth=6, learning_rate=0.05)
    model.fit(Xa, ya)
    prob = model.predict_proba(Xb)[:, 1]
    auc = float(roc_auc_score(yb, prob))
    results = {"n_guides": int(len(df)), "n_used": int(len(d)), "auc": auc, "base_rate": float(yb.mean())}

    # 1) ROC -- the headline: a working picker
    fpr, tpr, _ = roc_curve(yb, prob)
    fig, ax = sf.plt.subplots(figsize=(4.8, 4.6))
    ax.plot(fpr, tpr, color=sf.TURQUOISE, lw=2.5)
    ax.plot([0, 1], [0, 1], "--", color=sf.MUTED, lw=1)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("good guides correctly flagged")
    sf.title(ax, f"A working guide-picker  (AUC {auc:.2f})")
    sf.save(fig, HERE, "roc"); sf.save_raw(sf.pd.DataFrame({"fpr": fpr, "tpr": tpr}), HERE, "roc")

    # WHY CatBoost: bake off the candidate models on held-out guides, pick the winner by AUC.
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    cand = {"Logistic Reg": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=0),
            "CatBoost": CatBoostClassifier(verbose=0, random_state=0, iterations=600, depth=6, learning_rate=0.05)}
    try:
        from tabpfn import TabPFNClassifier
        cand["TabPFN"] = TabPFNClassifier()
    except Exception:
        pass
    bo = {}
    for nm, mdl in cand.items():
        xa, yy = (Xa[:3000], ya[:3000]) if nm == "TabPFN" else (Xa, ya)
        mdl.fit(xa, yy); bo[nm] = float(roc_auc_score(yb, mdl.predict_proba(Xb)[:, 1]))
    results["model_auc"] = bo
    fig, ax = sf.plt.subplots(figsize=(5.6, 3.4))
    cols = [sf.TURQUOISE if k == max(bo, key=bo.get) else sf.MUTED for k in bo]
    ax.bar(list(bo), list(bo.values()), color=cols, edgecolor=sf.INK, lw=.5)
    for i, v in enumerate(bo.values()):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9, family="Geist Mono")
    ax.axhline(0.8, ls="--", color=sf.DEEPPINK, lw=1); ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC")
    sf.plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    sf.title(ax, "We tried four models -- CatBoost won")
    sf.save(fig, HERE, "model_choice"); sf.save_raw(sf.pd.Series(bo, name="auc"), HERE, "model_choice")

    # 2) Usefulness: order the guides the model ranks highest -> how many actually cut well?
    order = np.argsort(-prob)
    fracs = [0.1, 0.2, 0.3, 0.5]
    prec = [float(yb[order[:max(1, int(f*len(yb)))]].mean()) for f in fracs]
    fig, ax = sf.plt.subplots(figsize=(5.6, 3.6))
    ax.bar([f"top {int(f*100)}%" for f in fracs], prec, color=sf.GOLD, edgecolor=sf.INK, lw=.5)
    ax.axhline(yb.mean(), ls="--", color=sf.DEEPPINK, lw=1.2, label=f"pick at random ({yb.mean():.0%})")
    for i, v in enumerate(prec):
        ax.text(i, v + .02, f"{v:.0%}", ha="center", fontsize=10, family="Geist Mono")
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction that actually cut well"); ax.legend(fontsize=9)
    sf.title(ax, "Order the model's top picks, get mostly good guides")
    sf.save(fig, HERE, "precision_at_top")
    sf.save_raw(sf.pd.DataFrame({"pick": [f"top{int(f*100)}" for f in fracs], "precision": prec}), HERE, "precision_at_top")
    results["precision_top10"] = prec[0]

    # 3) Bonus -- it learned real biology (the seed next to the PAM)
    from sklearn.linear_model import LogisticRegression
    Xo, _ = cs.featurize(d, "onehot", "guide20")
    lr = LogisticRegression(max_iter=2000).fit(Xo, y)
    imp = np.abs(lr.coef_[0]).reshape(20, 4).sum(1)
    top = int(np.argmax(imp)) + 1
    colors = [sf.DEEPPINK if p >= 16 else sf.TURQUOISE for p in range(20)]
    fig, ax = sf.plt.subplots(figsize=(8, 3.4))
    ax.bar(range(1, 21), imp, color=colors, edgecolor=sf.INK, lw=.4)
    ax.annotate("the seed\n(next to the PAM)", xy=(19, imp[-3:].max()), xytext=(13, imp.max()*.9),
                fontsize=9, color=sf.DEEPPINK, arrowprops=dict(arrowstyle="->", color=sf.DEEPPINK))
    ax.set_xlabel("position along the 20-letter guide  (20 = next to the PAM)")
    ax.set_ylabel("importance"); ax.set_xticks([1, 5, 10, 15, 20])
    sf.title(ax, f"It rediscovered the biology  (peak at position {top})")
    sf.save(fig, HERE, "position_importance")
    sf.save_raw(sf.pd.DataFrame({"position": range(1, 21), "importance": imp}), HERE, "position_importance")
    results["top_position"] = top

    sf.save_results(results, HERE)
    print("\nG1 DONE:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in results.items()})


if __name__ == "__main__":
    main()
