"""Grade the saved model on the HELD-OUT TEST set -- the only honest number.

    python evaluate.py     (run train.py first to create the checkpoint)

Prints overall test accuracy, a confusion matrix, and per-class accuracy, and saves a picture
of cases the model got WRONG to `failures.png`. Looking at failures (not just the score) is
both the most instructive thing you can do and the most compelling slide in your talk.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capstone_common as cc  # noqa: E402

import config  # noqa: E402
from data import get_data  # noqa: E402
from model import build_model  # noqa: E402

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")


def confusion_matrix(y, pred, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        m[t, p] += 1
    return m


def main():
    train_loader, val_loader, test_loader, n_classes, task = get_data()

    model = build_model(n_classes)
    if not os.path.exists(config.CHECKPOINT):
        sys.exit(f"no checkpoint at {config.CHECKPOINT} -- run `python train.py` first.")
    model.load_state_dict(torch.load(config.CHECKPOINT, map_location=device))

    res = cc.evaluate(model.to(device), test_loader, device)
    y, pred = res["y"], res["pred"]
    print(f"\nTEST accuracy: {res['accuracy']:.3f}   (n={len(y)}, {n_classes} classes)")

    cm = confusion_matrix(y, pred, n_classes)
    print("\nconfusion matrix (rows = truth, cols = prediction):")
    print(cm)
    print("\nper-class accuracy:")
    for c in range(n_classes):
        total = cm[c].sum()
        print(f"  class {c}: {cm[c, c] / total:.3f}" if total else f"  class {c}: (none)")

    # TODO (metric): for screening, a missed positive is worse than a false alarm.
    # Compute sensitivity/recall for the positive class, not just accuracy.

    # Save a few FAILURES for your talk.
    _save_failures(test_loader, y, pred)
    print("\nwrote failures.png  -- look at these before you present.")


def _save_failures(loader, y, pred, k=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrong = np.where(y != pred)[0]
    if len(wrong) == 0:
        print("no misclassified test cases (!) -- double-check for leakage.")
        return
    # pull the actual images back out of the test set in order
    imgs = []
    for xb, _ in loader:
        imgs.append(xb)
    imgs = torch.cat(imgs)

    pick = wrong[:k]
    fig, axes = plt.subplots(1, len(pick), figsize=(2 * len(pick), 2.4))
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, pick):
        img = imgs[idx].permute(1, 2, 0).numpy()
        ax.imshow((img - img.min()) / (img.max() - img.min() + 1e-8))
        ax.set_title(f"true {y[idx]} / pred {pred[idx]}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("failures.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
