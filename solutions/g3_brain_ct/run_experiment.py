"""G3 head trauma / stroke on brain CT -- worked to completion. Priority: fairness you CAN'T measure.

Trains a stroke-vs-normal classifier, then audits it: confusion, Grad-CAM (is it looking at brain
or cheating off the skull?), robustness to noise, AND a deliberately-blank 'what the dataset does
not tell us' figure -- the missing demographics that make a fairness audit impossible. That gap is
the project's whole point.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
import solfig as sf
import soltrain as st
import capstone_common as cc

HERE = Path(__file__).resolve().parent
FLAG = "brainct"
np = sf.np


def main():
    names = cc.class_names(FLAG)                                    # ["normal", "stroke"]
    stroke = names.index("stroke")
    results = {"dataset": FLAG, "classes": names}

    tr, va, te, ncls, _ = cc.get_loaders(FLAG, size=64)
    tr_aug, _, _, _, _ = cc.get_loaders(FLAG, size=64, augment=True)

    m = cc.make_model(ncls, backbone="resnet18", pretrained=True)
    m = cc.train(m, tr, va, epochs=5, lr=1e-3, device=st.DEVICE)
    y, p, prob = st.get_preds(m, te)
    acc = float((y == p).mean())
    sens = float((p[y == stroke] == stroke).mean())                # catch-the-stroke rate
    spec = float((p[y != stroke] != stroke).mean())
    results.update(accuracy=acc, sensitivity=sens, specificity=spec)

    st.fig_confusion(y, p, names, HERE, "confusion", f"Stroke detector: test accuracy {acc:.2f}")
    st.fig_gradcam(te, m, names, HERE, "gradcam", "Is it reading the brain, or cheating off the edges?")
    st.fig_noise(te, m, HERE, "noise_robustness", "A real ER scanner is messier than clean training data")

    # The point of the project: a blank where the demographics should be.
    fields = ["age", "sex", "race / ethnicity", "scanner model", "hospital / site", "time since injury"]
    fig, ax = sf.plt.subplots(figsize=(6.4, 3.8))
    ax.axis("off")
    ax.text(0.5, 0.96, "What this dataset tells us about each patient", ha="center",
            fontsize=13, family="Geist Mono", fontweight="bold", color=sf.INK, transform=ax.transAxes)
    ax.text(0.02, 0.82, "GIVEN:", fontsize=11, family="Geist Mono", color=sf.TURQUOISE, transform=ax.transAxes)
    ax.text(0.20, 0.82, "the CT image   +   normal / stroke label", fontsize=11, color=sf.INK, transform=ax.transAxes)
    ax.text(0.02, 0.66, "MISSING:", fontsize=11, family="Geist Mono", color=sf.DEEPPINK, transform=ax.transAxes)
    for i, f in enumerate(fields):
        row, col = divmod(i, 2)
        ax.text(0.22 + col * 0.40, 0.66 - row * 0.13, f"✗ {f}", fontsize=11,
                color=sf.DEEPPINK, transform=ax.transAxes)
    ax.text(0.5, 0.06, "You cannot audit fairness across groups the data never records.",
            ha="center", fontsize=10.5, style="italic", color=sf.MUTED, transform=ax.transAxes)
    sf.save(fig, HERE, "missing_metadata")
    sf.save_raw({"given": ["image", "normal/stroke label"], "missing": fields}, HERE, "missing_metadata")

    sf.save_results(results, HERE)
    print("\nG3 DONE:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in results.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
