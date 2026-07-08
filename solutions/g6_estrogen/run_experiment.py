"""G6 estrogen, cognition & the patient's account -- worked to completion.
Priority: confounding vs causation, plus the subjective-vs-objective gap.

Shows the trap in three moves: (1) the RAW picture -- estrogen users score better; (2) the honest
picture -- once you control for age, education, and income, most of that effect evaporates
(healthy-user bias); (3) fairness across groups. The shrink from crude to adjusted IS the lesson.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks/day3_capstone"))
import solfig as sf
import capstone_tabular as ct
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
np = sf.np


def main():
    df, meta = ct.load_estrogen()
    results = {"n": int(len(df)), "cohort": "women 60+, NHANES 2013-14"}

    # 1. RAW picture: mean digit-symbol score by estrogen use (users look BETTER)
    means = df.groupby("used_estrogen")["cognition_score"].mean()
    gn = ["never used", "used estrogen"]
    fig, ax = sf.plt.subplots(figsize=(4.4, 3.4))
    ax.bar(gn, [means.get(0, np.nan), means.get(1, np.nan)], color=[sf.MUTED, sf.TURQUOISE], edgecolor=sf.INK, linewidth=0.5)
    for i, v in enumerate([means.get(0), means.get(1)]):
        ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=10, family="Geist Mono")
    ax.set_ylabel("memory-test score (higher = better)")
    sf.title(ax, "The tempting headline: 'estrogen users think better'")
    sf.save(fig, HERE, "raw_means")
    sf.save_raw(means.rename("mean_score"), HERE, "raw_means")
    results["raw_mean_user"] = float(means.get(1)); results["raw_mean_nonuser"] = float(means.get(0))

    # 2. THE lesson: crude vs adjusted effect (confounding)
    y = df["low_cognition"].astype(int).values

    def coef(cols):
        X = StandardScaler().fit_transform(df[cols].astype(float).values)
        return LogisticRegression(max_iter=2000).fit(X, y).coef_[0][cols.index("used_estrogen")]

    crude = float(coef(["used_estrogen"]))
    adjusted = float(coef(["used_estrogen", "age", "education", "income_ratio"]))
    shrink = 0.0 if crude == 0 else (1 - adjusted / crude) * 100
    fig, ax = sf.plt.subplots(figsize=(5.4, 3.6))
    ax.bar(["crude\n(estrogen alone)", "adjusted\n(+ age, education, income)"], [crude, adjusted],
           color=[sf.DEEPPINK, sf.TURQUOISE], edgecolor=sf.INK, linewidth=0.6)
    ax.axhline(0, color=sf.INK, lw=0.8)
    # value labels centered INSIDE each bar (avoids colliding with the x-axis category labels)
    for i, (v, tc) in enumerate([(crude, "white"), (adjusted, sf.INK)]):
        ax.text(i, v * 0.5, f"{v:+.2f}", ha="center", va="center", fontsize=13,
                family="Geist Mono", color=tc, fontweight="bold")
    ax.set_ylim(min(crude * 1.18, -0.05), 0.02)          # headroom below; clamp top just above 0
    ax.set_ylabel("effect on low-cognition risk (log-odds)")
    sf.title(ax, f"{shrink:.0f}% of the effect was confounding, not cause")
    sf.save(fig, HERE, "confounding")
    sf.save_raw({"crude": crude, "adjusted": adjusted, "shrink_pct": shrink}, HERE, "confounding")
    results.update(crude=crude, adjusted=adjusted, shrink_pct=float(shrink))

    # 3. fairness / can a model even predict this?
    feats = meta["features"]
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.25, random_state=0, stratify=df["low_cognition"])
    m = ct.make_models()["Logistic Regression"]()
    m.fit(df.iloc[tr][feats].astype(float).values, df.iloc[tr]["low_cognition"].astype(int).values)
    sub = df.iloc[te]; pred = np.asarray(m.predict(sub[feats].astype(float).values)).ravel()
    truth = sub["low_cognition"].astype(int).values
    groups = sorted(sub["used_estrogen"].unique())
    gaccs = [float((pred[sub["used_estrogen"].values == g] == truth[sub["used_estrogen"].values == g]).mean()) for g in groups]
    labels = [gn[int(g)] for g in groups]
    fig, ax = sf.plt.subplots(figsize=(4.2, 3.4))
    ax.bar(labels, gaccs, color=[sf.MUTED, sf.TURQUOISE], edgecolor=sf.INK, linewidth=0.5)
    for i, v in enumerate(gaccs):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, family="Geist Mono")
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy")
    sf.title(ax, "How well the model predicts, by group")
    sf.save(fig, HERE, "fairness_by_group")
    sf.save_raw(sf.pd.DataFrame({"group": labels, "accuracy": gaccs}), HERE, "fairness_by_group")
    results["fairness_by_group"] = dict(zip(labels, gaccs))

    # Feature importance from the SAME interpretable model we used for the causal claim:
    # standardized logistic-regression coefficients. Age/education/income dominate; estrogen is
    # tiny -- the confounding, made visual. (An interpretable model is REQUIRED for a causal
    # question: only it lets you read how the estrogen effect shrinks once you adjust.)
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    Xs = _SS().fit_transform(df[feats].astype(float).values)
    yv = df["low_cognition"].astype(int).values
    lr = _LR(max_iter=2000).fit(Xs, yv)
    imp = dict(sorted(zip(feats, np.abs(lr.coef_[0])), key=lambda t: -t[1]))
    names = list(imp)[::-1]
    fig, ax = sf.plt.subplots(figsize=(6, 3.2))
    ax.barh(names, [imp[n] for n in names],
            color=[sf.DEEPPINK if n == "used_estrogen" else sf.TURQUOISE for n in names], edgecolor=sf.INK, lw=.5)
    ax.set_xlabel("standardized effect on low-cognition risk (|coefficient|)")
    sf.title(ax, "What actually predicts cognition -- not estrogen")
    sf.save(fig, HERE, "feature_importance"); sf.save_raw(sf.pd.Series(imp, name="abs_coef"), HERE, "feature_importance")
    results["auc"] = float(roc_auc_score(yv, cross_val_predict(
        _LR(max_iter=2000), Xs, yv, cv=5, method="predict_proba")[:, 1]))
    results["model"] = "Logistic Regression (interpretable -- required for the causal claim)"

    sf.save_results(results, HERE)
    print("\nG6 DONE:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in results.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
