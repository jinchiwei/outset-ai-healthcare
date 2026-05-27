"""Build the Day 3 capstone starter kits.

Each kit gets a runnable baseline notebook (NOT TODO-blank -- capstone is open-ended;
students extend the baseline with Claude) plus a README. Generated from one template
parameterized per dataset.

Run:  python scripts/build_capstone.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, md, new_nb, save  # noqa: E402

KITS = [
    {
        "folder": "pneumonia",
        "flag": "pneumoniamnist",
        "title": "Pneumonia detection (chest X-ray)",
        "blurb": "Binary: does this chest X-ray show pneumonia? Connects straight to what you saw on Days 1-2.",
        "levelup": "Swap in the full-resolution RSNA Pneumonia dataset (Kaggle). Real 1024x1024 images, bounding boxes, and a real class imbalance to wrestle with.",
    },
    {
        "folder": "skin",
        "flag": "dermamnist",
        "title": "Skin lesion triage",
        "blurb": "7-class: classify dermatoscopic images of skin lesions (including melanoma).",
        "levelup": "Swap in the full HAM10000 dataset (Kaggle), then try running your model on a phone photo of a mole. Watch how it behaves on data that looks nothing like training.",
    },
    {
        "folder": "choose",
        "flag": "retinamnist",
        "title": "Choose your own (any MedMNIST)",
        "blurb": "Pick any MedMNIST dataset and build on it. Defaults to RetinaMNIST (diabetic retinopathy, like Day 1).",
        "levelup": "Try a harder MedMNIST (pathmnist, organamnist, bloodmnist), or combine two datasets. Or bring your own images.",
    },
]


def starter_notebook(kit):
    nb = new_nb()
    nb.cells = [
        md(f"""
# Capstone: {kit['title']}

{kit['blurb']}

This notebook gives you a **working baseline**. Your job is to make it better and
understand what you did. Look for the `# IMPROVE ME` markers. Use Claude freely as your
pair programmer, but be ready to explain every change you make.

**Remember the rubric:** build it, evaluate it honestly, find a failure mode, defend one
design decision, and make sure both partners can explain the work.
"""),
        code("""
# Setup: on Colab, grab the course files. Locally this is a no-op.
import os, sys
if not os.path.exists("../../capstone_common.py"):
    os.system("git clone -q https://github.com/jinchiwei/outset-ai-healthcare.git")
    os.chdir("outset-ai-healthcare/notebooks/day3_capstone/starter_kits/%s")
sys.path.insert(0, "../..")            # day3_capstone (for capstone_common)
sys.path.insert(0, "../../../_shared") # colab_setup
import colab_setup
colab_setup.ensure(*colab_setup.DAY1, "medmnist")
""" % kit["folder"]),
        code(f"""
import torch
sys.path.insert(0, "../..")
import capstone_common as cc

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders("{kit['flag']}", size=64)
print("classes:", n_classes, "| task:", task)
"""),
        md("## A look at the data"),
        code("""
import matplotlib.pyplot as plt
imgs, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
for ax, im, lb in zip(axes, imgs, labels):
    ax.imshow(im[0], cmap="gray"); ax.set_title(int(lb)); ax.axis("off")
plt.tight_layout()
"""),
        md("## Baseline: transfer learning (frozen ResNet18 + new head)"),
        code("""
model = cc.make_baseline(n_classes)
model = cc.train(model, train_loader, val_loader, epochs=3, lr=1e-3, device=device)

test_acc = cc.evaluate(model, test_loader, device=device)["accuracy"]
print(f"\\nbaseline TEST accuracy: {test_acc:.3f}")
"""),
        md("""
## # IMPROVE ME

The baseline freezes the whole ResNet and trains only the last layer for 3 epochs.
Real, easy wins to try (pick a few, with Claude's help):

- **Train longer**: more epochs.
- **Unfreeze the backbone**: let more of the ResNet learn (lower the learning rate if you do).
- **Augment**: add flips/rotations in the transform (see `capstone_common.get_loaders`).
- **Bigger input**: try `size=128` or `size=224`.
- **Class weighting**: if the classes are imbalanced, weight the loss.
- **Different model**: try resnet50, or a timm model.

After each change, re-check TEST accuracy. Did it actually help, or did you just overfit?
"""),
        md("""
## Find a failure mode

Pull some test images your model got **wrong** and look at them. Is there a pattern?
Would a doctor have gotten them wrong too? This is rubric point 3, and often the most
interesting thing you'll present.
"""),
        code("""
import numpy as np
res = cc.evaluate(model, test_loader, device=device)
wrong = np.where(res["y"] != res["pred"])[0][:6]
print("indices the model got wrong:", wrong.tolist())
# IMPROVE ME: show these images with their true vs predicted labels, and write down
# what you notice.
"""),
    ]
    return nb


README_TMPL = """# Capstone kit: {title}

{blurb}

## What's here
- `starter.ipynb` -- a runnable baseline (transfer-learning ResNet18). Open it in Colab,
  run top to bottom, then improve it.

## Dataset
Uses **{flag}** from MedMNIST (pip-installable, downloads in seconds, no account needed).

## Your goal
Beat the baseline and understand how. The rubric rewards honest evaluation and finding a
failure mode, not just a higher number.

## Level up (for ambitious pairs)
{levelup}
"""


def build():
    base = ROOT / "notebooks/day3_capstone/starter_kits"
    for kit in KITS:
        folder = base / kit["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        save(starter_notebook(kit), folder / "starter.ipynb")
        (folder / "README.md").write_text(README_TMPL.format(**kit))
        print(f"wrote {kit['folder']}/ (starter.ipynb + README.md)")


if __name__ == "__main__":
    build()
