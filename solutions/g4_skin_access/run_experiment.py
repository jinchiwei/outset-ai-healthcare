"""G4 skin cancer AI + access/equity -- worked to completion. Priority: fairness and access.

Same DermaMNIST model as the screening group, but the analysis is about EQUITY: does the model
work evenly across conditions, and what does the dataset hide about WHO it was trained on? Produces
a per-class equity audit, a 'confidence gap' view, and a bespoke figure on the skin-tone data gap.
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
    short = [n.split()[0].rstrip(",")[:10] for n in names]
    results = {"dataset": FLAG, "n_classes": len(names)}

    tr, va, te, ncls, _ = cc.get_loaders(FLAG, size=64, augment=True)
    m = cc.make_model(ncls, backbone="resnet18", pretrained=True)
    m = cc.train(m, tr, va, epochs=5, lr=1e-3, device=st.DEVICE)
    y, p, prob = st.get_preds(m, te)
    acc = float((y == p).mean())
    results["accuracy"] = acc

    perclass = st.fig_perclass(y, p, short, HERE, "equity_per_class",
                               "'Accurate overall' can still fail a whole condition")
    results["per_class_acc"] = perclass
    results["worst_class"] = min(perclass, key=perclass.get)
    results["best_class"] = max(perclass, key=perclass.get)
    results["equity_gap"] = float(max(perclass.values()) - min(perclass.values()))

    # counts per class in the TRAIN set: rare conditions = the model sees them least
    ytr = np.concatenate([cc._targets(yb).numpy() for _, yb in tr])
    counts = [int((ytr == c).sum()) for c in range(len(names))]
    fig, ax = sf.plt.subplots(figsize=(7, 3.4))
    ax.bar(short, counts, color=sf.TURQUOISE, edgecolor=sf.INK, linewidth=0.5)
    ax.set_ylabel("training images");
    sf.title(ax, "The model sees some conditions far more than others")
    sf.plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    sf.save(fig, HERE, "train_imbalance")
    sf.save_raw(sf.pd.DataFrame({"class": short, "train_images": counts}), HERE, "train_imbalance")

    # bespoke: the skin-tone data gap (what the dataset does NOT record)
    fig, ax = sf.plt.subplots(figsize=(6.6, 3.8))
    ax.axis("off")
    ax.text(0.5, 0.95, "Who is in the training data?", ha="center", fontsize=13,
            family="Geist Mono", fontweight="bold", color=sf.INK, transform=ax.transAxes)
    ax.text(0.5, 0.80, "DermaMNIST records the lesion type -- but not the patient's skin tone,\n"
                       "age, sex, or country. Dermatology datasets skew heavily toward lighter skin.",
            ha="center", fontsize=10.5, color=sf.INK, transform=ax.transAxes)
    ax.text(0.5, 0.52, "So we literally cannot check:", ha="center", fontsize=11,
            family="Geist Mono", color=sf.DEEPPINK, transform=ax.transAxes)
    for i, q in enumerate(["Does it work as well on darker skin?",
                           "Who was it never trained to see?",
                           "Who gets helped -- and who gets missed?"]):
        ax.text(0.5, 0.40 - i * 0.11, f"✗  {q}", ha="center", fontsize=11,
                color=sf.DEEPPINK, transform=ax.transAxes)
    sf.save(fig, HERE, "skin_tone_gap")
    sf.save_raw({"recorded": ["lesion type"], "missing": ["skin tone", "age", "sex", "country"]},
                HERE, "skin_tone_gap")

    sf.save_results(results, HERE)
    print("\nG4 DONE:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in results.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
