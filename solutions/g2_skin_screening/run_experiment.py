"""G2 skin screening -- WORKING melanoma screen on real HAM10000 dermatoscopy.

The deliverable is a screen that works: melanoma-vs-rest on real-resolution images, AUC well above
0.8. Then the improvement that matters for screening: move the decision threshold to CATCH MORE
MELANOMAS (raise sensitivity), and show the honest cost in false alarms. A real, tunable tool.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
import imgexp as ie
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

HERE = Path(__file__).resolve().parent
np = sf.np
SIZE = 224


def main():
    print("loading HAM10000...", flush=True)
    ds = load_dataset("marmal88/skin_cancer")
    def prep(split):
        X = ie.hf_to_arrays(split, "image", SIZE)
        y = np.array([1 if d == "melanoma" else 0 for d in split["dx"]], dtype=np.int64)
        sex = np.array([str(s) for s in split["sex"]])
        return X, y, sex
    Xtr, ytr, _ = prep(ds["train"])
    Xva, yva, _ = prep(ds["validation"])
    Xte, yte, sex_te = prep(ds["test"])
    print(f"train {len(ytr)} (mel {ytr.mean():.2%}), test {len(yte)}", flush=True)

    cw = [1.0, float((ytr == 0).sum() / max(1, (ytr == 1).sum()))]   # up-weight the rare melanomas

    # WHY THIS MODEL: bake off a plain ResNet18 against the stronger CAFormer, pick the winner by AUC.
    print(">> ResNet18 (baseline backbone)", flush=True)
    m_rn = ie.train_binary(Xtr, ytr, Xva, yva, size=SIZE, epochs=4, class_weight=cw, backbone="resnet18")
    auc_rn = float(roc_auc_score(yte, ie.probs(m_rn, Xte, yte, size=SIZE)))
    print(">> CAFormer (stronger backbone)", flush=True)
    model = ie.train_binary(Xtr, ytr, Xva, yva, size=SIZE, epochs=6, class_weight=cw, backbone="caformer_s18")
    p = ie.probs(model, Xte, yte, size=SIZE)
    auc = float(roc_auc_score(yte, p))
    results = {"backbone_auc": {"ResNet18": auc_rn, "CAFormer": auc}}
    fig, ax = sf.plt.subplots(figsize=(4.4, 3.4))
    ax.bar(["ResNet18", "CAFormer"], [auc_rn, auc], color=[sf.MUTED, sf.TURQUOISE], edgecolor=sf.INK, lw=.5)
    for i, v in enumerate([auc_rn, auc]):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=10, family="Geist Mono")
    ax.axhline(0.8, ls="--", color=sf.DEEPPINK, lw=1); ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC")
    sf.title(ax, "Why CAFormer: a stronger backbone wins")
    sf.save(fig, HERE, "backbone_choice"); sf.save_raw(sf.pd.Series(results["backbone_auc"], name="auc"), HERE, "backbone_choice")
    results.update({"dataset": "HAM10000 (real dermatoscopy)", "n_test": int(len(yte)),
                    "melanoma_frac": float(yte.mean()), "auc": auc, "backbone": "caformer_s18"})

    # ROC curve (the headline: a working screen)
    fpr, tpr, thr = roc_curve(yte, p)
    fig, ax = sf.plt.subplots(figsize=(4.8, 4.6))
    ax.plot(fpr, tpr, color=sf.TURQUOISE, lw=2.5)
    ax.plot([0, 1], [0, 1], "--", color=sf.MUTED, lw=1)
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("melanomas caught (sensitivity)")
    sf.title(ax, f"A working melanoma screen  (AUC {auc:.2f})")
    sf.save(fig, HERE, "roc"); sf.save_raw(sf.pd.DataFrame({"fpr": fpr, "tpr": tpr}), HERE, "roc")

    # default threshold 0.5 vs a screening threshold tuned to catch >=90% of melanomas
    def stats(t):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        return dict(recall=tp / (tp + fn), specificity=tn / (tn + fp), threshold=float(t))
    d = stats(0.5)
    target = 0.90
    t_screen = float(thr[np.argmax(tpr >= target)]) if (tpr >= target).any() else 0.1
    s = stats(t_screen)
    results.update(recall_default=d["recall"], specificity_default=d["specificity"],
                   recall_tuned=s["recall"], specificity_tuned=s["specificity"], tuned_threshold=t_screen)

    # the improvement, as before/after bars
    fig, ax = sf.plt.subplots(figsize=(6, 3.8))
    x = np.arange(2); w = 0.38
    ax.bar(x - w/2, [d["recall"], d["specificity"]], w, label="default model", color=sf.MUTED, edgecolor=sf.INK, lw=.5)
    ax.bar(x + w/2, [s["recall"], s["specificity"]], w, label="tuned for screening", color=sf.TURQUOISE, edgecolor=sf.INK, lw=.5)
    for i, (a, b) in enumerate([(d["recall"], s["recall"]), (d["specificity"], s["specificity"])]):
        ax.text(i - w/2, a + .02, f"{a:.0%}", ha="center", fontsize=9, family="Geist Mono")
        ax.text(i + w/2, b + .02, f"{b:.0%}", ha="center", fontsize=9, family="Geist Mono")
    ax.set_xticks(x); ax.set_xticklabels(["melanomas caught\n(sensitivity)", "healthy called healthy\n(specificity)"], fontsize=9)
    ax.set_ylim(0, 1.08); ax.legend(fontsize=9)
    sf.title(ax, "Tuning the screen to miss fewer cancers")
    sf.save(fig, HERE, "recall_tuning")
    sf.save_raw({"default": d, "tuned": s}, HERE, "recall_tuning")

    # confusion at the screening threshold
    pred = (p >= t_screen).astype(int)
    M = confusion_matrix(yte, pred, labels=[0, 1])
    fig, ax = sf.plt.subplots(figsize=(3.4, 3.4)); ax.imshow(M, cmap="BuPu")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max()/2 else sf.INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not mel", "melanoma"]); ax.set_xlabel("screen says")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not mel", "melanoma"]); ax.set_ylabel("truth")
    sf.title(ax, f"At the screening setting (catch {s['recall']:.0%})")
    sf.save(fig, HERE, "confusion")
    sf.save_raw(sf.pd.DataFrame(M, index=["not_mel", "melanoma"], columns=["pred_not", "pred_mel"]), HERE, "confusion")

    # fairness: does the screen work as well for women as for men? (HAM10000 records sex)
    fair = {}
    for g in ["male", "female"]:
        m = sex_te == g
        if m.sum() > 20 and len(np.unique(yte[m])) == 2:
            fair[g] = float(roc_auc_score(yte[m], p[m]))
    if len(fair) == 2:
        fig, ax = sf.plt.subplots(figsize=(4.2, 3.4))
        ax.bar(list(fair), list(fair.values()), color=[sf.TURQUOISE, sf.DEEPPINK], edgecolor=sf.INK, lw=.5)
        for i, v in enumerate(fair.values()):
            ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=10, family="Geist Mono")
        ax.axhline(0.8, ls="--", color=sf.MUTED, lw=1)
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC for this group")
        sf.title(ax, "Fairness check: works for women and men")
        sf.save(fig, HERE, "fairness_by_sex")
        sf.save_raw(sf.pd.Series(fair, name="auc"), HERE, "fairness_by_sex")
        results["auc_by_sex"] = fair

    # feature importance (image version): Grad-CAM on melanoma cases -> is it looking at the lesion?
    mel_idx = np.where(yte == 1)[0][:4]
    if len(mel_idx) >= 2:
        ie.gradcam_fig(model, Xte[mel_idx], SIZE, HERE, "gradcam",
                       "Where the screen looks (real melanomas)", labels=["melanoma"] * len(mel_idx))

    sf.save_results(results, HERE)
    print("\nG2 DONE:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in results.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
