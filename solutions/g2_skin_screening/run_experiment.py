"""G2 skin-cancer screening -- worked to completion. Priority: sensitivity (never miss melanoma).

Trains three ways on DermaMNIST to make the 'data/pretraining > model' point, then audits the
best model the way a regulator would: confusion, per-class fairness (melanoma highlighted),
melanoma recall (the number that matters for screening), and the mistakes it makes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
import soltrain as st
import capstone_common as cc

HERE = Path(__file__).resolve().parent
FLAG = "dermamnist"
np = sf.np


def main():
    names = cc.class_names(FLAG)
    mel = next(i for i, n in enumerate(names) if "melanoma" in n.lower())
    short = [n.split()[0].rstrip(",")[:10] for n in names]         # tidy labels for plots
    results = {"dataset": FLAG, "n_classes": len(names), "melanoma_index": mel}

    tr, va, te, ncls, _ = cc.get_loaders(FLAG, size=64)
    tr_aug, _, _, _, _ = cc.get_loaders(FLAG, size=64, augment=True)

    def run(pretrained, loader, epochs=4, unfreeze=False):
        m = cc.make_model(ncls, backbone="resnet18", pretrained=pretrained, unfreeze_backbone=unfreeze)
        m = cc.train(m, loader, va, epochs=epochs, lr=(1e-4 if unfreeze else 1e-3), device=st.DEVICE)
        y, p, _ = st.get_preds(m, te)
        return m, float((y == p).mean())

    print(">> scratch"); _, acc_scratch = run(False, tr)
    print(">> pretrained baseline"); m_base, acc_base = run(True, tr)
    print(">> pretrained + augment (improved)"); m_best, acc_best = run(True, tr_aug, epochs=5)
    results.update(scratch_acc=acc_scratch, baseline_acc=acc_base, improved_acc=acc_best)

    # data > model: three bars
    fig, ax = sf.plt.subplots(figsize=(5.6, 3.6))
    labels = ["from scratch\n(blank brain)", "pretrained\n(borrowed eyes)", "pretrained\n+ augment"]
    vals = [acc_scratch, acc_base, acc_best]
    ax.bar(labels, vals, color=[sf.MUTED, sf.TURQUOISE, sf.GOLD], edgecolor=sf.INK, linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, family="Geist Mono")
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy")
    sf.title(ax, "Better starting point beats a fancier model")
    sf.save(fig, HERE, "data_beats_model")
    sf.save_raw(sf.pd.DataFrame({"setup": ["scratch", "pretrained", "pretrained+augment"], "accuracy": vals}),
                HERE, "data_beats_model")

    # audits on the best model
    y, p, prob = st.get_preds(m_best, te)
    st.fig_confusion(y, p, short, HERE, "confusion", f"Test accuracy {acc_best:.2f} -- but look off the diagonal")
    perclass = st.fig_perclass(y, p, short, HERE, "per_class_fairness",
                               "Screening lives or dies on the rare classes", highlight=mel)
    st.fig_failures(te, m_best, short, HERE, "failures", "The cases it got wrong")

    mel_recall = float((p[y == mel] == mel).mean()) if (y == mel).any() else 0.0
    results["melanoma_recall"] = mel_recall
    results["per_class_acc"] = perclass
    sf.save_results(results, HERE)
    print("\nG2 DONE:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in results.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
