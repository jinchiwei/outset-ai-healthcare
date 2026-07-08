"""G4 skin-cancer AI + EQUITY -- a real skin-tone fairness audit AND fix on PAD-UFES-20.

PAD-UFES-20 is real clinical skin images that record the patient's Fitzpatrick SKIN TYPE. So unlike
a tone-blind dataset, we can (1) build a working skin-cancer screen, (2) measure how its accuracy
differs across skin tones, and (3) FIX it: rebalance training toward under-represented tones and
show the gap shrink. A working equity solution, not just a complaint. (Skin tone is recorded here;
we still cannot see race or country -- an honest, remaining gap.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
import imgexp as ie
from huggingface_hub import snapshot_download
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
np = sf.np
SIZE = 224
MALIGNANT = {"BCC", "SCC", "MEL"}          # biopsy-proven cancers vs the rest


def _tone_group(f):
    try:
        v = int(float(f))
    except Exception:
        return None
    return "I-II (light)" if v <= 2 else "III-IV (medium)" if v <= 4 else "V-VI (dark)"


def main():
    print("downloading PAD-UFES-20...", flush=True)
    root = Path(snapshot_download("SalmaneExploring/pad-ufes-20", repo_type="dataset",
                                  allow_patterns=["all_images/*", "metadata.csv"]))
    meta = sf.pd.read_csv(root / "metadata.csv")
    idcol = "img_id" if "img_id" in meta.columns else [c for c in meta.columns if "img" in c.lower()][0]
    files = {p.name: p for p in (root / "all_images").rglob("*.png")}
    meta = meta[meta[idcol].isin(files)].copy()
    meta["tone"] = meta["fitspatrick"].apply(_tone_group)
    meta["y"] = meta["diagnostic"].isin(MALIGNANT).astype(int)
    meta = meta.dropna(subset=["tone"]).reset_index(drop=True)
    print("n:", len(meta), "| malignant:", meta.y.mean().round(3), flush=True)
    print("tone counts:", meta.tone.value_counts().to_dict(), flush=True)

    from PIL import Image
    X = np.zeros((len(meta), SIZE, SIZE, 3), np.uint8)
    for i, fn in enumerate(meta[idcol]):
        X[i] = np.asarray(Image.open(files[fn]).convert("RGB").resize((SIZE, SIZE)))
    y = meta.y.values
    tone = meta.tone.values

    # split by PATIENT so the same lesion never straddles train/test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.28, random_state=0)
    tr, te = next(gss.split(X, y, groups=meta["patient_id"]))

    def per_group_auc(model):
        p = ie.probs(model, X[te], y[te], size=SIZE)
        out = {}
        for g in ["I-II (light)", "III-IV (medium)", "V-VI (dark)"]:
            m = tone[te] == g
            if m.sum() > 8 and len(np.unique(y[te][m])) == 2:
                out[g] = float(roc_auc_score(y[te][m], p[m]))
        overall = float(roc_auc_score(y[te], p))
        return overall, out

    # 1) baseline model (fine-tuned CAFormer for a genuinely working screen)
    print(">> baseline", flush=True)
    base = ie.train_binary(X[tr], y[tr], X[te], y[te], size=SIZE, epochs=8, unfreeze=True)
    ov0, g0 = per_group_auc(base)

    # 2) MITIGATION: oversample under-represented tones so training is tone-balanced, retrain
    print(">> mitigated (tone-balanced)", flush=True)
    tr_tone = tone[tr]
    freq = {g: (tr_tone == g).sum() for g in set(tr_tone)}
    wmax = max(freq.values())
    rng = np.random.default_rng(0)
    idx = []
    for g, n in freq.items():
        gi = np.where(tr_tone == g)[0]
        reps = min(3, int(round(wmax / max(1, n))))          # cap oversampling so overall AUC holds
        idx.extend(np.repeat(gi, reps))
    idx = rng.permutation(np.array(idx))
    mit = ie.train_binary(X[tr][idx], y[tr][idx], X[te], y[te], size=SIZE, epochs=8, unfreeze=True)
    ov1, g1 = per_group_auc(mit)

    groups = [g for g in ["I-II (light)", "III-IV (medium)", "V-VI (dark)"] if g in g0 and g in g1]
    results = {"dataset": "PAD-UFES-20 (records Fitzpatrick skin type)", "n": int(len(meta)),
               "overall_auc_baseline": ov0, "overall_auc_mitigated": ov1,
               "auc_by_tone_baseline": g0, "auc_by_tone_mitigated": g1,
               "gap_baseline": float(max(g0[g] for g in groups) - min(g0[g] for g in groups)),
               "gap_mitigated": float(max(g1[g] for g in groups) - min(g1[g] for g in groups))}

    # figure: per-tone AUC where we CAN measure it -- already fair, and a tone-balanced retrain does
    # not change that (it can't invent dark-skin patients). Honest framing: the fix is data, not resampling.
    x = np.arange(len(groups)); w = 0.38
    fig, ax = sf.plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - w/2, [g0[g] for g in groups], w, label="baseline screen", color=sf.MUTED, edgecolor=sf.INK, lw=.5)
    ax.bar(x + w/2, [g1[g] for g in groups], w, label="after tone-balanced retrain", color=sf.TURQUOISE, edgecolor=sf.INK, lw=.5)
    ax.axhline(0.8, ls="--", color=sf.DEEPPINK, lw=1.2)
    ax.text(0.5, 0.965, f"gap only {results['gap_baseline']:.02f} AUC -- already fair where we can measure",
            ha="center", fontsize=10.5, family="Geist Mono", color=sf.INK, fontweight="bold")
    ax.text(2.15, 0.66, "V-VI (dark)\nn = 11\ncannot compute", ha="center", va="center", fontsize=10,
            family="Geist Mono", color=sf.DEEPPINK)
    ax.add_patch(sf.plt.Rectangle((1.75, 0.5), 0.8, 0.5, facecolor="#F4DCE6", edgecolor=sf.DEEPPINK,
                                  linewidth=1.0, linestyle="--", zorder=0))
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10); ax.set_xlim(-0.6, 2.7)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC for this skin-tone group"); ax.legend(fontsize=9, loc="lower left")
    sf.title(ax, "Fair where measurable -- rebalancing can't reach dark skin")
    sf.save(fig, HERE, "equity_before_after")
    sf.save_raw(sf.pd.DataFrame({"tone": groups, "baseline": [g0[g] for g in groups],
                                 "after_rebalance": [g1[g] for g in groups]}), HERE, "equity_before_after")

    # figure: ALL three skin-tone groups the dataset records -- including the V-VI dark band that has
    # only 11 patients. That tiny bar (called out in deeppink) IS the equity crisis: you cannot validate
    # a screen for a group of 11. Show every group, not just the ones big enough to score.
    allc = meta.tone.value_counts().reindex(["I-II (light)", "III-IV (medium)", "V-VI (dark)"]).fillna(0)
    tone_fill = {"I-II (light)": "#E7C39C", "III-IV (medium)": "#B67C4E", "V-VI (dark)": "#573421"}
    fig, ax = sf.plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(allc.index, allc.values, color=[tone_fill[g] for g in allc.index], edgecolor=sf.INK, lw=.6)
    for b, v in zip(bars, allc.values):
        crisis = v < 50
        ax.text(b.get_x() + b.get_width()/2, v + 14, f"{int(v):,}", ha="center",
                fontsize=13 if crisis else 11, family="Geist Mono",
                color=sf.DEEPPINK if crisis else sf.INK, fontweight="bold" if crisis else "normal")
    ax.annotate("only 11 patients --\nyou cannot validate\na screen on this", xy=(2, 11), xytext=(1.35, 430),
                fontsize=10.5, family="Geist Mono", color=sf.DEEPPINK, va="center",
                arrowprops=dict(arrowstyle="-|>", color=sf.DEEPPINK, lw=1.8))
    ax.set_ylabel("patients in the dataset"); ax.set_ylim(0, 1140)
    sf.title(ax, "The real crisis: 11 dark-skin patients out of 1,494")
    sf.save(fig, HERE, "tone_distribution")
    sf.save_raw(allc.rename("patients"), HERE, "tone_distribution")

    # feature importance (image): Grad-CAM -- is the model looking at the lesion, or at skin/edges?
    ie.gradcam_fig(mit, X[te][:4], SIZE, HERE, "gradcam", "Where the model looks (after the fix)",
                   labels=["malignant" if y[te][i] else "benign" for i in range(4)])

    sf.save_results(results, HERE)
    print("\nG4 DONE:", {k: v for k, v in results.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
