"""Shared helpers for the Day 3 TABULAR capstone kits.

The image kits (capstone_common.py) cover pixel tasks. Some groups have TABULAR data --
rows of numbers per patient -- which is the Day 2 world: pick features, pick a model, and
the interesting questions are about confounding and fairness, not architecture.

Datasets wired in:
  heart      -- UCI Heart Disease (id=45): predict heart disease from clinical features
                incl cholesterol. 303 patients.
  estrogen   -- NHANES 2013-14, women 60+: estrogen/HRT use vs measured cognition, with the
                confounders (age, education, income) that make "does estrogen cause brain
                fog?" a genuinely hard question.

Same model family as Day 2: Logistic Regression / Random Forest / CatBoost / TabPFN.
"""
from __future__ import annotations
import io
import urllib.request

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def load_heart():
    """UCI Heart Disease. Returns (df, meta). Target 'disease' = any disease (severity>0)."""
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=45)
    df = d.data.features.copy()
    df["disease"] = (d.data.targets.iloc[:, 0] > 0).astype(int)   # 0-4 severity -> binary
    df = df.dropna().reset_index(drop=True)
    meta = dict(
        name="UCI Heart Disease",
        target="disease",
        features=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        group="sex", group_names={0: "female", 1: "male"},
        positive="heart disease",
    )
    return df, meta


def _grab_xpt(name):
    base = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/"
    req = urllib.request.Request(base + name + ".xpt", headers={"User-Agent": "Mozilla/5.0"})
    return pd.read_sas(io.BytesIO(urllib.request.urlopen(req).read()), format="xport")


def load_estrogen():
    """NHANES 2013-14, women 60+: estrogen use + confounders -> cognition. Returns (df, meta).

    cognition_score = digit-symbol substitution (higher = better). low_cognition = bottom half.
    used_estrogen from RHQ420 (ever used female hormones). years_estrogen from RHQ540 (duration).
    """
    demo = _grab_xpt("DEMO_H")[["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", "INDFMPIR"]]
    rhq = _grab_xpt("RHQ_H")[["SEQN", "RHQ420", "RHQ540"]]
    cfq = _grab_xpt("CFQ_H")[["SEQN", "CFDDS"]]
    m = demo.merge(rhq, on="SEQN").merge(cfq, on="SEQN")
    w = m[(m.RIAGENDR == 2) & (m.RIDAGEYR >= 60)].copy()
    w = w[w.RHQ420.isin([1, 2])]                                   # 1=used, 2=never
    w["used_estrogen"] = (w.RHQ420 == 1).astype(int)
    w["years_estrogen"] = w.RHQ540.where(w.RHQ540 < 8000, 0).fillna(0)   # junk/"refused" codes -> 0
    w = w.rename(columns={"RIDAGEYR": "age", "DMDEDUC2": "education",
                          "INDFMPIR": "income_ratio", "CFDDS": "cognition_score"})
    w = w.dropna(subset=["cognition_score", "age", "education", "income_ratio"]).reset_index(drop=True)
    w["low_cognition"] = (w.cognition_score < w.cognition_score.median()).astype(int)
    df = w[["age", "education", "income_ratio", "used_estrogen", "years_estrogen",
            "cognition_score", "low_cognition"]].copy()
    meta = dict(
        name="NHANES 2013-14 (women 60+)",
        target="low_cognition",
        features=["age", "education", "income_ratio", "used_estrogen", "years_estrogen"],
        group="used_estrogen", group_names={0: "never used estrogen", 1: "used estrogen"},
        positive="low cognition score",
    )
    return df, meta


def load_stroke():
    """Stroke Prediction dataset (5,110 patients). Returns (df, meta).

    Unlike a brain-CT image set, this one RECORDS demographics (sex, age, health history), so you
    can audit whether a stroke-risk model works equally for women and men. target 'stroke' (1 = had
    a stroke); group 'gender'. Loads the committed copy at datasets/stroke.csv.
    """
    from pathlib import Path
    from sklearn.preprocessing import LabelEncoder
    p = Path(__file__).resolve().parents[2] / "datasets" / "stroke.csv"
    raw = (pd.read_csv(p) if p.exists() else
           pd.read_csv("https://raw.githubusercontent.com/jinchiwei/outset-ai-healthcare/main/datasets/stroke.csv"))
    raw = raw.dropna(subset=["stroke"])
    raw = raw[raw["gender"].isin(["Male", "Female"])].reset_index(drop=True)
    df = raw.drop(columns=[c for c in ["id"] if c in raw]).copy()
    for c in df.select_dtypes("object").columns:                    # text -> numbers (gender: F=0, M=1)
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    df = df.fillna(df.median(numeric_only=True))
    features = [c for c in df.columns if c != "stroke"]
    meta = dict(
        name="Stroke Prediction (records sex, age, history)",
        target="stroke", features=features,
        group="gender", group_names={0: "female", 1: "male"},
        positive="stroke",
    )
    return df, meta


DATASETS = {"heart": load_heart, "estrogen": load_estrogen, "stroke": load_stroke}


# --------------------------------------------------------------------------- #
# Models (same family as Day 2)
# --------------------------------------------------------------------------- #
def make_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from catboost import CatBoostClassifier
    from tabpfn import TabPFNClassifier
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
        "Random Forest":       lambda: RandomForestClassifier(n_estimators=200, random_state=0),
        "CatBoost":            lambda: CatBoostClassifier(verbose=0, random_state=0),
        "TabPFN (foundation)": lambda: TabPFNClassifier(),
    }


def _split(df, features, target, test_size):
    from sklearn.model_selection import train_test_split
    X = df[features].astype(float).values
    y = df[target].astype(int).values
    return train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)


def fit_eval(df, features, target, model="Logistic Regression", test_size=0.25):
    """Train one model, print TEST accuracy, draw a confusion matrix. Returns accuracy."""
    import matplotlib.pyplot as plt
    Xa, Xb, ya, yb = _split(df, features, target, test_size)
    m = make_models()[model]()
    m.fit(Xa, ya)
    pred = np.asarray(m.predict(Xb)).ravel()
    acc = float((pred == yb).mean())
    base = max(np.mean(yb), 1 - np.mean(yb))
    print(f"features: {features}")
    print(f"model: {model}   ->   TEST accuracy = {acc:.3f}   (guess-majority baseline = {base:.3f})")
    M = np.zeros((2, 2), int)
    for t, q in zip(yb, pred):
        M[t, q] += 1
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, M[i, j], ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no", "yes"]); ax.set_xlabel("predicted")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"]); ax.set_ylabel("true")
    ax.set_title(f"{model}: {acc:.3f}")
    plt.tight_layout(); plt.show()
    return acc


def audit_by_group(df, features, target, group, group_names=None, model="Logistic Regression", test_size=0.25):
    """FAIRNESS: train once, then report TEST accuracy separately for each subgroup."""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=0, stratify=df[target])
    m = make_models()[model]()
    m.fit(df.iloc[tr][features].astype(float).values, df.iloc[tr][target].astype(int).values)
    sub = df.iloc[te]
    pred = np.asarray(m.predict(sub[features].astype(float).values)).ravel()
    truth = sub[target].astype(int).values
    groups = sorted(sub[group].dropna().unique())
    names = [(group_names or {}).get(g, str(g)) for g in groups]
    accs = [float((pred[sub[group].values == g] == truth[sub[group].values == g]).mean()) for g in groups]
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(groups), 3))
    ax.bar(names, accs, color=["#40E0D0", "#FF1493", "#F0C840", "#8A2BE2"][:len(groups)])
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy"); ax.set_title(f"accuracy by {group}")
    plt.tight_layout(); plt.show()
    print("The model can be accurate overall yet unequal across groups. Is that acceptable here?")
    return dict(zip(names, accs))


def association(df, feature, target, controls):
    """CONFOUNDING: the effect of `feature` on `target`, alone vs controlled for `controls`.

    Reports the standardized logistic-regression coefficient of `feature` in two models. If it
    shrinks toward 0 once you add the controls, the crude effect was (partly) confounding.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = df[target].astype(int).values

    def coef(cols):
        X = StandardScaler().fit_transform(df[cols].astype(float).values)
        lr = LogisticRegression(max_iter=2000).fit(X, y)
        return lr.coef_[0][cols.index(feature)]

    crude = coef([feature])
    adjusted = coef([feature] + list(controls))
    print(f"effect of '{feature}' on '{target}' (log-odds, standardized):")
    print(f"  crude (alone):                 {crude:+.3f}")
    print(f"  adjusted (+ {', '.join(controls)}): {adjusted:+.3f}")
    shrink = 0 if crude == 0 else (1 - adjusted / crude) * 100
    print(f"  -> controlling for confounders changed the effect by {shrink:+.0f}%.")
    print("  A big shrink means the crude association was mostly the confounders, not the cause.")
    return dict(crude=float(crude), adjusted=float(adjusted))
