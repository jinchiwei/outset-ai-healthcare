"""Build the Day 1 ladder notebooks from a single source.

Emits two notebooks from one definition:
  notebooks/day1_ladder/day1.ipynb           (with # TODO blanks -- the lab)
  notebooks/day1_ladder/day1_solution.ipynb  (filled in -- instructor reference)

The lab and solution never drift because they come from the same cell list:
code cells with blanks are declared once via nbutil.code_with_todos(solution, blanks).

Run:  python scripts/build_day1.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, code_with_todos, md, new_nb, save  # noqa: E402

# Each entry is either a finished cell (md/code) or a (solution_cell, lab_cell)
# pair produced by code_with_todos. We unzip into two cell lists at the end.
PAIRS = []  # list of (solution_cell, lab_cell)


def both(cell):
    """A cell that is identical in solution and lab."""
    PAIRS.append((cell, cell))


def todo(solution_src, blanks):
    PAIRS.append(code_with_todos(solution_src, blanks))


# --------------------------------------------------------------------------- #
both(md("""
# Day 1 -- From pixels to vision transformers

Today we build **five models** on the same medical images and watch each one beat
the last: logistic regression -> MLP -> CNN -> ResNet -> Vision Transformer.

The dataset is **APTOS-2019**: color photos of the back of the eye (fundus), graded
0-4 for diabetic retinopathy. This was the first AI screening tool deployed at scale
in real clinics.

**How this works:** fill in the `# TODO` lines. Stuck on one? Ask Claude -- then make
sure you understand what it gives you before moving on. The point isn't to finish
fastest, it's to understand why each model does better than the one before.
"""))

both(code("""
# Setup: on Colab, grab the course files. Locally (already in the repo) this is a no-op.
import os, sys
if not os.path.exists("common.py"):
    os.system("git clone -q https://github.com/jinchiwei/outset-ai-healthcare.git")
    os.chdir("outset-ai-healthcare/notebooks/day1_ladder")
sys.path.insert(0, ".")
sys.path.insert(0, "../_shared")
import colab_setup
colab_setup.ensure()
colab_setup.gpu_check()
"""))

both(code("""
import torch
import common

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
results = {}   # we'll collect each model's validation accuracy here
"""))

# ---- Step 0 -------------------------------------------------------------- #
both(md("""
## Step 0 -- What is an image, really?

Before we model anything: an image is just a grid of numbers. A color image is
*three* grids stacked (red, green, blue). Let's look.
"""))

both(code("""
train_loader, val_loader = common.get_loaders(size=224, batch_size=32)
images, labels = next(iter(train_loader))
print("one batch:", images.shape, "  <- (batch, channels, height, width)")

common.show_rgb_split(images[0])
common.show_pixel_histogram(images[0])
"""))

both(md("""
Augmentation = small random changes (flip, rotate, brighten) so the model sees more
variety and doesn't memorize. Here's the same eye, augmented a few ways:
"""))

both(code("""
import torchvision.transforms as T
# turn one (de-normalized) training image back into a PIL image to show augmentations
sample_pil = T.ToPILImage()(common._denorm(images[0]))
common.show_augmentations(sample_pil)
"""))

# ---- Step 1: logreg ------------------------------------------------------ #
both(md("""
## Step 1 -- Logistic regression

The simplest classifier there is. We flatten each image into one long row of numbers
and fit a linear model. We use small 64x64 images here so it runs fast.

**Predict first:** there are 5 grades. Random guessing would be ~20%. How well do you
think a linear model on raw pixels will do? Write your guess.
"""))

todo(
    """
from sklearn.linear_model import LogisticRegression

tr64, va64 = common.get_loaders(size=64, batch_size=64)
Xtr, ytr = common.flatten_for_classical(tr64)
Xva, yva = common.flatten_for_classical(va64)
print("flattened features per image:", Xtr.shape[1])

clf = LogisticRegression(max_iter=500, solver="saga")
clf.fit(Xtr, ytr)

acc = (clf.predict(Xva) == yva).mean()
results["logreg"] = acc
print(f"logistic regression val accuracy: {acc:.3f}")
""",
    [
        ("clf = LogisticRegression", "make a LogisticRegression with max_iter=500 and solver='saga'"),
        ("clf.fit(Xtr, ytr)", "fit the classifier on the training features Xtr, ytr"),
    ],
)

both(md("""
*How close was your guess?* Logistic regression treats every pixel as independent and
can only draw straight-line boundaries. It has no idea that neighboring pixels form
shapes. Hold that thought.
"""))

# ---- Step 2: MLP --------------------------------------------------------- #
both(md("""
## Step 2 -- Multi-layer perceptron (a small neural net)

Same flattened pixels, but now we stack layers with non-linearities so the model can
bend its decision boundaries. Still no notion of space, though.
"""))

todo(
    """
mlp = common.make_mlp(in_features=3 * 64 * 64)
history = common.train_model(mlp, tr64, va64, epochs=5, lr=1e-3, device=device)
results["mlp"] = history[-1][1]
print(f"mlp val accuracy: {history[-1][1]:.3f}")
""",
    [
        ("mlp = common.make_mlp", "build an MLP with common.make_mlp; in_features is 3*64*64 (the flattened size)"),
        ("history = common.train_model", "train it: common.train_model(mlp, tr64, va64, epochs=5, lr=1e-3, device=device)"),
    ],
)

both(md("""
*Did it beat logreg by much?* Usually only a little. More layers help, but the model
still can't see that pixels next to each other belong together. That's the next jump.
"""))

# ---- Step 3: CNN --------------------------------------------------------- #
both(md("""
## Step 3 -- Convolutional neural network (from scratch)

A CNN slides small filters across the image, so it *does* understand spatial structure
-- edges, blobs, textures. Now we use the full 224x224 images.

**Predict:** how much will accuracy jump versus the MLP?
"""))

todo(
    """
tr224, va224 = common.get_loaders(size=224, batch_size=32)

cnn = common.make_small_cnn()
history = common.train_model(cnn, tr224, va224, epochs=15, lr=1e-3, device=device)
results["cnn"] = history[-1][1]
print(f"cnn val accuracy: {history[-1][1]:.3f}")

common.show_first_layer_filters(cnn)
""",
    [
        ("cnn = common.make_small_cnn()", "build the CNN with common.make_small_cnn()"),
        ("history = common.train_model(cnn", "train it for 15 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

both(md("""
Look at the first-layer filters above. Even this tiny CNN learned little edge and
color detectors on its own. *That* is what convolutions buy you over a plain MLP.
"""))

# ---- Step 4: ResNet ------------------------------------------------------ #
both(md("""
## Step 4 -- ResNet50 (transfer learning)

Instead of learning from scratch, we take a 50-layer network already trained on a
million everyday photos (ImageNet) and reuse its "vision." We freeze it and train just
a new final layer for our 5 eye grades. This is the single biggest practical trick in
modern computer vision.
"""))

todo(
    """
resnet = common.make_resnet50(pretrained=True)
history = common.train_model(resnet, tr224, va224, epochs=3, lr=1e-3, device=device)
results["resnet"] = history[-1][1]
print(f"resnet val accuracy: {history[-1][1]:.3f}")

# where is it looking?
imgs, _ = next(iter(va224))
cam, predicted = common.gradcam(resnet, imgs[0], device=device)
print("Grad-CAM heatmap shape:", cam.shape, "predicted grade:", predicted)
""",
    [
        ("resnet = common.make_resnet50", "build a pretrained ResNet50 with common.make_resnet50(pretrained=True)"),
        ("history = common.train_model(resnet", "finetune for 3 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

both(md("""
*Why did pretraining help so much with so little training?* The network already knew
how to see edges, textures, and shapes. We only had to teach it which combinations
mean "diabetic retinopathy."
"""))

# ---- Step 5: ViT --------------------------------------------------------- #
both(md("""
## Step 5 -- Vision Transformer

A ViT chops the image into patches, turns each patch into a vector, and lets the
patches "pay attention" to each other. Same idea as the language models you've used --
just patches instead of words. (More on that tomorrow.)
"""))

todo(
    """
vit = common.make_vit_base(pretrained=True)
history = common.train_model(vit, tr224, va224, epochs=5, lr=1e-3, device=device)
results["vit"] = history[-1][1]
print(f"vit val accuracy: {history[-1][1]:.3f}")
""",
    [
        ("vit = common.make_vit_base", "build a pretrained ViT with common.make_vit_base(pretrained=True)"),
        ("history = common.train_model(vit", "train the head for 5 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

# ---- Leaderboard --------------------------------------------------------- #
both(md("## The leaderboard you just built"))

both(code("""
import matplotlib.pyplot as plt
names = list(results.keys())
accs = [results[n] for n in names]
plt.figure(figsize=(7, 4))
bars = plt.bar(names, accs, color="#40E0D0")
plt.ylabel("validation accuracy")
plt.title("Same data, five models")
for b, a in zip(bars, accs):
    plt.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}", ha="center")
plt.ylim(0, 1)
plt.show()
"""))

both(md("""
## Bridge to Day 2

The Vision Transformer worked by splitting the image into patches and letting them
attend to each other:

```
   IMAGE                                     SENTENCE
   [patch][patch][patch]  --embed-->         "the"  "cat"  "sat"  --embed-->
        vectors                                    vectors
          |                                          |
       attention  (which patches matter?)         attention  (which words matter?)
          |                                          |
       prediction: DR grade                       prediction: next word
```

That second column is a **Large Language Model**. Same machinery, different input.
Tomorrow we use one to read radiology reports. See you then.
"""))

both(md("""
## Stretch -- find a disagreement

If you finished early: find an eye image where the ResNet and the ViT predicted
*different* grades. Show the image and both predictions. What's unusual about it?
Which model would you trust, and why?
"""))

# --------------------------------------------------------------------------- #
def build():
    sol = new_nb()
    lab = new_nb()
    sol.cells = [p[0] for p in PAIRS]
    lab.cells = [p[1] for p in PAIRS]
    save(sol, ROOT / "notebooks/day1_ladder/day1_solution.ipynb")
    save(lab, ROOT / "notebooks/day1_ladder/day1.ipynb")
    print("wrote day1.ipynb (with TODO blanks) and day1_solution.ipynb")


if __name__ == "__main__":
    build()
