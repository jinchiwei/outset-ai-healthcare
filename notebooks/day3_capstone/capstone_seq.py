"""Shared helpers for the Day 3 SEQUENCE capstone kit (Group 1: CRISPR guide design).

The image kits (capstone_common.py) cover pixels; the tabular kits (capstone_tabular.py)
cover rows of numbers. This one covers TEXT -- a DNA sequence -- which is the interesting
new idea: a computer cannot multiply the letter "A", so the first job is always to turn a
sequence into numbers. HOW you turn it into numbers matters more than which model you pick.

Task: given a 20-nucleotide guide RNA, will CRISPR-Cas9 cut efficiently at that site?
Data: Doench et al. 2016 (Nature Biotechnology), the canonical guide-efficiency benchmark,
5310 guides across 17 genes. Each row is a 30-mer of genomic context laid out as:

    [4 nt upstream][20 nt guide / protospacer][NGG PAM][3 nt downstream]
     positions 1-4     positions 5-24          25-27      28-30

  efficiency        -- 0..1, how well this guide cut (higher = better). A regression target.
  high_efficiency   -- 1 if this guide is in the top tier of cutters, else 0. Classification.

Two ways to featurize a sequence, and the whole lesson lives in the gap between them:
  "onehot"  -- one column per (position, base): position-AWARE. Learns that the bases next to
               the PAM (the "seed") matter more than bases far away.
  "kmer"    -- just count how many A/C/G/T and how many of each pair: position-BLIND. Throws
               the ordering away. Usually worse -- proof that WHERE a base sits matters.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

_CSV = Path(__file__).resolve().parents[2] / "datasets" / "crispr_guides.csv"
_URL = ("https://raw.githubusercontent.com/jinchiwei/outset-ai-healthcare/"
        "main/datasets/crispr_guides.csv")
_BASES = "ACGT"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_guides():
    """Load the Doench 2016 guide-efficiency table. Returns (df, meta).

    Adds two derived columns: `guide20` (the 20-nt protospacer) and `gc_content`
    (fraction G or C in the guide -- a classic, human-readable predictor).
    """
    df = pd.read_csv(_CSV) if _CSV.exists() else pd.read_csv(_URL)
    df["guide20"] = df["guide30"].str[4:24]                       # protospacer only
    df["gc_content"] = df["guide20"].apply(lambda s: (s.count("G") + s.count("C")) / len(s))
    df = df.reset_index(drop=True)
    meta = dict(
        name="Doench 2016 CRISPR-Cas9 guide efficiency",
        seq_col="guide30",
        class_target="high_efficiency", class_positive="high-efficiency guide",
        reg_target="efficiency",
        citation="Doench et al., Nature Biotechnology 2016 (Azimuth training set).",
    )
    return df, meta


# --------------------------------------------------------------------------- #
# Featurization: sequence text -> a table of numbers
# --------------------------------------------------------------------------- #
def _onehot(seqs):
    """(N sequences of equal length L) -> (N, L*4) one-hot. Position-aware."""
    idx = {b: i for i, b in enumerate(_BASES)}
    L = len(seqs.iloc[0])
    X = np.zeros((len(seqs), L * 4), dtype=np.float32)
    for r, s in enumerate(seqs):
        for pos, base in enumerate(s):
            if base in idx:
                X[r, pos * 4 + idx[base]] = 1.0
    names = [f"p{pos+1}_{b}" for pos in range(L) for b in _BASES]
    return X, names


def _kmer(seqs):
    """(N sequences) -> single + dinucleotide composition + GC. Position-BLIND (order lost)."""
    singles = list(_BASES)
    dinucs = [a + b for a in _BASES for b in _BASES]
    cols = singles + dinucs + ["gc"]
    X = np.zeros((len(seqs), len(cols)), dtype=np.float32)
    for r, s in enumerate(seqs):
        L = len(s)
        for i, b in enumerate(singles):
            X[r, i] = s.count(b) / L
        for j, d in enumerate(dinucs):
            X[r, len(singles) + j] = sum(s[k:k+2] == d for k in range(L - 1)) / (L - 1)
        X[r, -1] = (s.count("G") + s.count("C")) / L
    return X, cols


def featurize(df, mode="onehot", seq_col="guide20"):
    """Turn the sequence column into a numeric matrix. Returns (X, feature_names).

    mode="onehot" -> position-aware one-hot (best); mode="kmer" -> position-blind counts.
    seq_col="guide20" (the 20-nt guide) or "guide30" (with genomic context).
    """
    seqs = df[seq_col].astype(str)
    if mode == "onehot":
        return _onehot(seqs)
    if mode == "kmer":
        return _kmer(seqs)
    raise ValueError("mode must be 'onehot' or 'kmer'")


# --------------------------------------------------------------------------- #
# Models -- sklearn + CatBoost, no GPU needed (a guide is a short string)
# --------------------------------------------------------------------------- #
def make_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from catboost import CatBoostClassifier
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
        "Random Forest":       lambda: RandomForestClassifier(n_estimators=300, random_state=0),
        "Neural Net (MLP)":    lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=0),
        "CatBoost":            lambda: CatBoostClassifier(verbose=0, random_state=0),
    }


def make_regressors():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from catboost import CatBoostRegressor
    return {
        "Ridge (linear)":   lambda: Ridge(),
        "Random Forest":    lambda: RandomForestRegressor(n_estimators=300, random_state=0),
        "Neural Net (MLP)": lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=800, random_state=0),
        "CatBoost":         lambda: CatBoostRegressor(verbose=0, random_state=0),
    }


def _split(X, y, test_size=0.25, stratify=None):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=0, stratify=stratify)


def fit_eval_class(df, mode="onehot", seq_col="guide20", model="Logistic Regression",
                   target="high_efficiency", test_size=0.25):
    """Classify high- vs low-efficiency guides. Prints accuracy + AUC, draws a confusion matrix."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    X, _ = featurize(df, mode, seq_col)
    y = df[target].astype(int).values
    Xa, Xb, ya, yb = _split(X, y, test_size, stratify=y)
    m = make_classifiers()[model]()
    m.fit(Xa, ya)
    pred = np.asarray(m.predict(Xb)).ravel()
    acc = float((pred == yb).mean())
    base = max(np.mean(yb), 1 - np.mean(yb))
    try:
        proba = m.predict_proba(Xb)[:, 1]
        auc = roc_auc_score(yb, proba)
    except Exception:
        auc = float("nan")
    print(f"representation: {mode} on {seq_col}   |   model: {model}")
    print(f"TEST accuracy = {acc:.3f}   AUC = {auc:.3f}   (guess-majority baseline = {base:.3f})")
    M = np.zeros((2, 2), int)
    for t, q in zip(yb, pred):
        M[t, q] += 1
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, M[i, j], ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low", "high"]); ax.set_xlabel("predicted")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["low", "high"]); ax.set_ylabel("true efficiency")
    ax.set_title(f"{model}: acc {acc:.3f}")
    plt.tight_layout(); plt.show()
    return dict(accuracy=acc, auc=float(auc))


def fit_eval_reg(df, mode="onehot", seq_col="guide20", model="Random Forest",
                 target="efficiency", test_size=0.25):
    """Predict the continuous efficiency (0..1). Prints R^2 + correlation, draws pred-vs-true."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    X, _ = featurize(df, mode, seq_col)
    y = df[target].astype(float).values
    Xa, Xb, ya, yb = _split(X, y, test_size)
    m = make_regressors()[model]()
    m.fit(Xa, ya)
    pred = np.asarray(m.predict(Xb)).ravel()
    r2 = r2_score(yb, pred)
    r = float(np.corrcoef(yb, pred)[0, 1])
    print(f"representation: {mode} on {seq_col}   |   model: {model}")
    print(f"TEST R^2 = {r2:.3f}   correlation = {r:.3f}   (R^2=0 means no better than the mean)")
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.scatter(yb, pred, s=8, alpha=0.4, color="#40E0D0")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("true efficiency"); ax.set_ylabel("predicted efficiency")
    ax.set_title(f"{model}: R^2 {r2:.3f}")
    plt.tight_layout(); plt.show()
    return dict(r2=float(r2), corr=r)


# --------------------------------------------------------------------------- #
# Interpretability: WHERE in the guide does the model look?
# --------------------------------------------------------------------------- #
def position_importance(df, seq_col="guide20", target="high_efficiency"):
    """Train a linear model on one-hot features, then plot how much each POSITION matters.

    Biology payoff: the bases closest to the PAM (the "seed", the right-hand end of the guide)
    should carry more weight than the far end. If your plot shows that, your model rediscovered
    real CRISPR biology from data alone.
    """
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    X, names = featurize(df, "onehot", seq_col)
    y = df[target].astype(int).values
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    L = len(df[seq_col].iloc[0])
    coef = np.abs(lr.coef_[0]).reshape(L, 4).sum(axis=1)          # importance per position
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(1, L + 1), coef, color="#FF1493")
    ax.set_xlabel(f"position along the {L}-nt guide  (right end = next to the PAM / cut site)")
    ax.set_ylabel("importance")
    ax.set_title("which positions the model relies on")
    plt.tight_layout(); plt.show()
    top = int(np.argmax(coef)) + 1
    print(f"Most important position: {top} of {L}. Is it near the PAM end (the seed region)?")
    return coef


def gc_vs_efficiency(df):
    """The simplest possible predictor: does GC content track efficiency? A one-feature baseline."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.scatter(df["gc_content"], df["efficiency"], s=6, alpha=0.3, color="#8A2BE2")
    r = float(np.corrcoef(df["gc_content"], df["efficiency"])[0, 1])
    ax.set_xlabel("GC content of the guide"); ax.set_ylabel("efficiency")
    ax.set_title(f"GC vs efficiency (correlation {r:.2f})")
    plt.tight_layout(); plt.show()
    print(f"correlation = {r:.2f}. GC content alone explains a little, but not everything --")
    print("that's why the model reads the whole sequence, not just one number.")
    return r
