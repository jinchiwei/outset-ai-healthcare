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
from medmnist import INFO

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

FLAG = "{kit['flag']}"    # change this to any MedMNIST dataset (see project_options.md)
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders(FLAG, size=64)
CLASS_NAMES = list(INFO[FLAG]["label"].values())
print("classes:", n_classes, "->", CLASS_NAMES, "| task:", task)
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
## Build your own model (interactive)

Same idea as Day 2's model builder: **pick your options, hit Run, read the TEST accuracy.**
Change **one** thing at a time and watch what actually moves the number.

- **backbone** -- the network architecture (resnet18 is small/fast; resnet50, densenet, etc. are bigger).
- **pretrained** -- start from ImageNet weights (on) or from scratch (off). *This usually matters a lot.*
- **unfreeze backbone** -- train the whole network (on) vs just a new head (off). More power, more overfitting risk.
- **augment** -- add flips/rotations to training images.
- **epochs** -- how long to train.

Log every result (`"resnet50 + augment: 0.86"`). That log is half your presentation.
"""),
        code("""
from ipywidgets import interact_manual, Dropdown, Checkbox, IntSlider

def build_model(backbone="resnet18", pretrained=True, unfreeze_backbone=False, augment=False, epochs=3):
    global model, train_loader, val_loader, test_loader
    # augment changes the TRAIN transform, so rebuild the loaders each time
    train_loader, val_loader, test_loader, n_cls, _ = cc.get_loaders(FLAG, size=64, augment=augment)
    model = cc.make_model(n_cls, backbone=backbone, pretrained=pretrained, unfreeze_backbone=unfreeze_backbone)
    model = cc.train(model, train_loader, val_loader,
                     epochs=epochs, lr=(1e-4 if unfreeze_backbone else 1e-3), device=device)
    acc = cc.evaluate(model, test_loader, device=device)["accuracy"]
    print(f"\\n>>> TEST accuracy = {acc:.3f}   "
          f"[{backbone}, pretrained={pretrained}, unfreeze={unfreeze_backbone}, augment={augment}, {epochs}ep]")

try:
    interact_manual(build_model,
        backbone=Dropdown(options=cc.BACKBONES, value="resnet18", description="backbone"),
        pretrained=Checkbox(value=True, description="pretrained"),
        unfreeze_backbone=Checkbox(value=False, description="unfreeze"),
        augment=Checkbox(value=False, description="augment"),
        epochs=IntSlider(value=3, min=1, max=10, description="epochs"))
except ImportError:
    build_model()   # no widgets -> just train the default
"""),
        md("""
### Other ideas (not in the dropdown -- do these with Claude)
- **Bigger input**: `cc.get_loaders(FLAG, size=128)` (or 224). Slower, sometimes better.
- **Class weighting**: if the classes are imbalanced, weight the loss toward the rare one.
- **Different learning rate / optimizer**: ask Claude what to try and why.

After each change, re-check TEST accuracy. Did it help, or did you just overfit the training set?
"""),
        md("""
## The regulator's toolkit: audit your own model

Yesterday you decided the *rules* medical AI must follow. Now hold **your** model to them.
Pick a priority below and run the audit on your trained model. A high accuracy is not enough
if it fails one of these.

- **Safety/Evidence** -- an honest confusion matrix on the held-out test set.
- **Fairness** -- is accuracy even *across classes*, or does it quietly fail one?
- **Transparency** -- Grad-CAM: is the model looking at the right part of the image?
- **Monitoring** -- does accuracy survive noisier images (like it'll meet after deployment)?
- **Failure analysis** -- the actual cases it got wrong (rubric point 3).

*(Train a model first -- run the baseline or the builder above -- then pick a tool.)*
"""),
        code("""
from ipywidgets import interact_manual, Dropdown

def audit(tool):
    if "model" not in globals():
        print("Train a model first (run the baseline or the builder above)."); return
    cc.REGULATOR_TOOLS[tool](model, test_loader, device=device, class_names=CLASS_NAMES)

try:
    interact_manual(audit, tool=Dropdown(options=list(cc.REGULATOR_TOOLS),
                                          description="priority", style={"description_width": "initial"}))
except ImportError:
    for name in cc.REGULATOR_TOOLS:      # no widgets -> run them all
        print("\\n==", name); audit(name)
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
