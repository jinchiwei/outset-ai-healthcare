"""G3, part 2 -- the fairness audit the brain-CT model COULDN'T get.

The CT dataset records no demographics, so we cannot check who the stroke detector fails. This
tabular stroke dataset DOES record sex, age, and health history -- so here we (1) build a working
stroke-risk predictor (AUC ~0.81), (2) audit how well it catches strokes in women vs men at a
sensitivity-first operating point, and (3) FIX an unequal miss-rate with group-specific thresholds.
Recording demographics is what makes fairness auditable -- and fixable.

Data: a local cache of MLLab-TS/healthcare_dataset_stroke at datasets/stroke.csv (5,110 patients).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "datasets" / "stroke.csv"          # local cache of the HF stroke dataset
np = sf.np


def main():
    # ---- load + encode the tabular stroke set (rows of numbers per patient) ----
    df = sf.pd.read_csv(CSV).dropna(subset=["stroke"])              # drop rows with no label
    df = df[df.gender.isin(["Male", "Female"])].reset_index(drop=True)   # keep the two recorded sexes
    y = df.stroke.astype(int).values                               # 1 = had a stroke, 0 = did not
    sex = df.gender.values                                         # the sex column we will AUDIT by
    feat = df.drop(columns=[c for c in ["id", "stroke"] if c in df]).copy()  # features = everything but id + label
    for c in feat.select_dtypes("object"):                         # turn text columns (smoking, work) into numbers
        feat[c] = LabelEncoder().fit_transform(feat[c].astype(str))
    feat = feat.fillna(feat.median())                              # fill missing bmi etc. with the column median

    # ---- split, then train a class-weighted CatBoost (a tree we can later explain with SHAP) ----
    Xtr, Xte, ytr, yte, str_tr, str_te = train_test_split(
        feat.values, y, sex, test_size=0.3, random_state=0, stratify=y)
    model = CatBoostClassifier(verbose=0, random_state=0, auto_class_weights="Balanced", iterations=400)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]                             # predicted stroke probability on held-out patients
    auc = float(roc_auc_score(yte, p))
    results = {"dataset": "Stroke Prediction (records sex/age/history)", "n": int(len(df)),
               "stroke_rate": float(y.mean()), "auc": auc}

    # 1) working predictor: ROC
    fpr, tpr, _ = roc_curve(yte, p)
    fig, ax = sf.plt.subplots(figsize=(4.8, 4.6))
    ax.plot(fpr, tpr, color=sf.TURQUOISE, lw=2.5); ax.plot([0, 1], [0, 1], "--", color=sf.MUTED, lw=1)
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("strokes caught")
    sf.title(ax, f"A working stroke-risk model  (AUC {auc:.2f})")
    sf.save(fig, HERE, "stroke_roc"); sf.save_raw(sf.pd.DataFrame({"fpr": fpr, "tpr": tpr}), HERE, "stroke_roc")

    # WHY this model: bake off the tabular candidates. All clear 0.80; we DEPLOY CatBoost because it is a
    # tree we can read with SHAP and audit for fairness. (A tight race is itself the teaching point.)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    cand = {"Logistic Reg": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0),
            "CatBoost": CatBoostClassifier(verbose=0, random_state=0, auto_class_weights="Balanced", iterations=400)}
    try:
        from tabpfn import TabPFNClassifier
        cand["TabPFN"] = TabPFNClassifier()
    except Exception:
        pass
    bo = {}
    for nm, mdl in cand.items():
        xa, ya2 = (Xtr[:3000], ytr[:3000]) if nm == "TabPFN" else (Xtr, ytr)
        mdl.fit(xa, ya2); bo[nm] = float(roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1]))
    results["model_auc"] = bo
    PICK = "CatBoost"                                              # our deployed model (explainable tree)
    fig, ax = sf.plt.subplots(figsize=(5.8, 3.6))
    ax.bar(list(bo), list(bo.values()),
           color=[sf.TURQUOISE if k == PICK else sf.MUTED for k in bo], edgecolor=sf.INK, lw=.5)
    for i, v in enumerate(bo.values()):
        ax.text(i, v + .012, f"{v:.2f}", ha="center", fontsize=9, family="Geist Mono")
    ax.axhline(0.8, ls="--", color=sf.DEEPPINK, lw=1); ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC")
    sf.plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    sf.title(ax, "Four models all clear 0.80 -- we deploy CatBoost")
    sf.save(fig, HERE, "stroke_model_choice"); sf.save_raw(sf.pd.Series(bo, name="auc"), HERE, "stroke_model_choice")

    # 2) AUDIT: at a sensitivity-first operating point (catch ~80% of strokes overall), do we catch
    #    women's strokes as often as men's? A SHARED threshold can hide a real gap.
    def recall(mask, t):
        m = mask & (yte == 1)
        return float(((p >= t)[m]).mean()) if m.any() else np.nan
    target_overall = 0.80                                          # a screening tool would aim to catch most strokes
    t0 = float(np.quantile(p[yte == 1], 1 - target_overall))       # shared threshold that catches ~80% overall
    rec0 = {g: recall(str_te == g, t0) for g in ["Female", "Male"]}

    # 3) FIX (equal opportunity): give each group its OWN threshold so both catch the same share of strokes.
    def group_thresh(g, tgt):
        m = (str_te == g) & (yte == 1)
        if not m.any():
            return t0
        return float(np.quantile(p[m], 1 - tgt))                   # threshold that catches `tgt` of THIS group's strokes
    target = max(rec0.values())                                    # equalize UP to the better-served group's rate
    rec1 = {g: recall(str_te == g, group_thresh(g, target)) for g in ["Female", "Male"]}
    results.update(operating_point_overall_recall=target_overall, shared_threshold=t0,
                   recall_by_sex_before=rec0, recall_by_sex_after=rec1,
                   gap_before=float(abs(rec0["Female"] - rec0["Male"])),
                   gap_after=float(abs(rec1["Female"] - rec1["Male"])))

    x = np.arange(2); w = 0.38
    fig, ax = sf.plt.subplots(figsize=(6, 3.9))
    ax.bar(x - w/2, [rec0["Female"], rec0["Male"]], w, label="one shared threshold", color=sf.MUTED, edgecolor=sf.INK, lw=.5)
    ax.bar(x + w/2, [rec1["Female"], rec1["Male"]], w, label="group-aware threshold (fix)", color=sf.TURQUOISE, edgecolor=sf.INK, lw=.5)
    for i, g in enumerate(["Female", "Male"]):
        ax.text(i - w/2, rec0[g] + .02, f"{rec0[g]:.0%}", ha="center", fontsize=9, family="Geist Mono")
        ax.text(i + w/2, rec1[g] + .02, f"{rec1[g]:.0%}", ha="center", fontsize=9, family="Geist Mono")
    ax.set_xticks(x); ax.set_xticklabels(["women", "men"]); ax.set_ylim(0, 1.12)
    ax.set_ylabel("strokes caught (sensitivity)"); ax.legend(fontsize=9, loc="upper left")
    sf.title(ax, "Auditing -- and equalizing -- who gets caught")
    sf.save(fig, HERE, "stroke_fairness_fix")
    sf.save_raw({"before": rec0, "after": rec1}, HERE, "stroke_fairness_fix")

    # SHAP: which recorded facts drive the stroke-risk call? (age should dominate.)
    sf.shap_bar(model, Xte, list(feat.columns), HERE, "stroke_shap",
                "What drives the stroke-risk prediction (SHAP)")

    import json
    (HERE / "results_tabular.json").write_text(json.dumps(results, indent=2, default=float))
    print("wrote results_tabular.json")
    print("\nG3-tabular DONE:", {k: v for k, v in results.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
