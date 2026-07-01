"""Build the Day 1 ladder notebooks from a single source.

Emits two notebooks from one definition:
  notebooks/day1_ladder/day1.ipynb           (with # TODO blanks -- the lab)
  notebooks/day1_ladder/day1_solution.ipynb  (filled in -- instructor reference)

The lab and solution never drift because they come from the same cell list:
code cells with blanks are declared once via nbutil.code_with_todos(solution, blanks).

Density target: MIT 6.S191 style -- numbered sub-sections, an explanation before
every code cell, and a figure (a live nbfig plot, or an embedded concept diagram)
wherever it helps. Live plots use notebooks/_shared/nbfig.py (Colab-safe); concept
diagrams under img/ are pre-rendered by img_gen.py with the build-figure skill.

Run:  python scripts/build_day1.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, code_with_todos, md, new_nb, save  # noqa: E402

PAIRS = []  # list of (solution_cell, lab_cell)


def both(cell):
    PAIRS.append((cell, cell))


def todo(solution_src, blanks):
    PAIRS.append(code_with_todos(solution_src, blanks))


# =========================================================================== #
# Intro
# =========================================================================== #
both(md("""
# Day 1 -- From pixels to vision transformers

Welcome. Over the next three hours you will build **five image classifiers** on the same
medical images and watch each one beat the last:

> **logistic regression -> MLP -> CNN -> ResNet -> Vision Transformer**

Each rung up the ladder adds one idea, and you will *see* the accuracy climb because of it.
That climb is the whole story of how computer vision got good.

### The dataset and the task
We use **APTOS-2019**: color photographs of the back of the eye (the *retina*, or *fundus*).
Diabetes damages the tiny blood vessels there, a disease called **diabetic retinopathy (DR)**.
Our task is the one actually deployed in clinics: **referable DR** -- does this eye show
enough disease that the person should see a specialist? **Yes or no.** A binary screening call.

### What you'll be able to do by the end
- Explain what an image *is* to a model, and why pixels alone are hard.
- Say what each model adds: non-linearity, spatial structure, pretraining, attention.
- Read a **learning curve** and a **confusion matrix**, and know why accuracy can mislead.
- Understand *transfer learning* -- the single most useful trick in applied vision.

### How the lab works
Fill in the `# TODO` lines. Stuck? Ask Claude, then make sure you understand the answer
before moving on. The point isn't to finish fastest, it's to understand *why each rung
beats the one below it.*
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
import os, sys
import torch
import numpy as np
import common
# nbfig lives in notebooks/_shared; add it relative to common.py so this cell
# works even if the setup cell above wasn't run, and from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(common.__file__)), "..", "_shared"))
import nbfig            # Colab-safe branded plotting (matches the slide figures)
nbfig.use()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)
results = {}   # we'll collect each model's validation accuracy here
"""))

# =========================================================================== #
# Section 0 -- The data
# =========================================================================== #
both(md("""
## 0. The data: pixels, eyes, and a yes/no question

Before any modeling, spend real time looking at the data. Good practitioners always do.

### 0.1 The clinical task
About **1 in 3** people with diabetes develop some retinopathy, and it is a leading cause
of blindness in working-age adults, yet it is **preventable** if caught early. The catch:
catching it needs a trained eye specialist, and there are far too few. That mismatch -- huge
routine need, scarce expertise -- is exactly where AI screening helps. Let's look at the eyes.
"""))

both(code("""
train_loader, val_loader = common.get_loaders(size=224, batch_size=32)
images, labels = next(iter(train_loader))
print("one batch:", images.shape, "  <- (batch, channels, height, width)")

# Look at eight real fundus images with their referable / not-referable labels.
fig, axes = nbfig.fig(2, 4, figsize=(11, 5.6))
for ax, img, lab in zip(axes.ravel(), images, labels):
    ax.imshow(common._denorm(img).permute(1, 2, 0).numpy())
    ax.set_title(common.GRADE_NAMES[int(lab)], fontsize=11,
                 color=(nbfig.DEEPPINK if int(lab) else nbfig.INK))
    ax.axis("off")
nbfig.show(fig, "Eight eyes: can you tell which need a doctor?")
"""))

both(md("""
### 0.2 An image is just numbers
To a model, this photo is not a picture -- it's a grid of numbers. A *color* image is
**three** grids stacked: how much red, green, and blue at each pixel. Everything today
operates on those numbers.
"""))

both(code("""
common.show_rgb_split(images[0])      # the same eye as red / green / blue grids
common.show_pixel_histogram(images[0])  # the distribution of pixel brightnesses
"""))

both(md("""
### 0.3 Normalization: put every channel on the same scale
Networks train best when their inputs are **centered at zero with a similar spread**.
**Standardization** does exactly that: for each color channel, subtract that channel's mean and
divide by its standard deviation. The result has mean ~0 and std ~1. Watch the per-channel means
collapse onto the zero line below.

Heads-up: the middle panel (the standardized image) looks washed-out and bluish, **not** like a
real photo -- that's expected. A retina is almost all red/orange with barely any blue, so
stretching every channel to the same spread blows the faint blue channel way up. There's no
"natural" way to display a standardized tensor; the model doesn't need one, it just reads the
numbers.
"""))

both(code("""
# Standardize each channel using THIS data's own mean and std -- the textbook recipe.
raw_batch = torch.stack([common._denorm(im) for im in images])  # N,3,H,W back in 0..1
ch_mean = raw_batch.mean(dim=(0, 2, 3))        # per-channel mean
ch_std = raw_batch.std(dim=(0, 2, 3))          # per-channel std
standardized = (raw_batch - ch_mean[:, None, None]) / ch_std[:, None, None]

raw0, std0 = raw_batch[0], standardized[0]
disp = (std0 - std0.min()) / (std0.max() - std0.min())  # rescale just to view

fig, (a1, a2, a3) = nbfig.fig(1, 3, figsize=(11, 3.6))
a1.imshow(raw0.permute(1, 2, 0).numpy()); a1.set_title("raw (0..1)", fontsize=11); a1.axis("off")
a2.imshow(disp.permute(1, 2, 0).numpy()); a2.set_title("standardized (rescaled to view)", fontsize=11); a2.axis("off")
x = np.arange(3)
a3.bar(x - 0.2, raw_batch.mean((0, 2, 3)).numpy(), 0.4, color=nbfig.TURQUOISE, label="raw mean")
a3.bar(x + 0.2, standardized.mean((0, 2, 3)).numpy(), 0.4, color=nbfig.DEEPPINK, label="standardized mean")
a3.axhline(0, color=nbfig.MUTED, lw=0.8); a3.set_xticks(x); a3.set_xticklabels(["R", "G", "B"])
a3.set_title("channel means: ~0 after", fontsize=11); a3.legend()
nbfig.show(fig, "Standardization: subtract the mean, divide by the std -> centered at zero")
print("standardized channel means:", [round(v, 3) for v in standardized.mean((0, 2, 3)).tolist()])
"""))

both(md("""
> **One practical footnote.** Our actual training pipeline standardizes with *fixed* **ImageNet**
> constants (mean `0.485/0.456/0.406`, std `0.229/0.224/0.225`) rather than each batch's own
> stats. Two reasons: the transform must be identical at train and test time (per-batch stats
> would drift), and the pretrained ResNet/ViT we load later were trained with exactly those
> numbers. It's the same idea -- a fixed, shared scale -- it just leaves our darker-than-average
> retinas sitting a little below zero instead of exactly on it.
"""))

both(md("""
### 0.4 Know your labels (this matters more than you think)
Before trusting any accuracy number, check how many examples of each class you have. If
most eyes are "not referable," a lazy model can score high by always guessing the majority.
Hold that thought -- we'll catch a model doing exactly that at the end.
"""))

both(code("""
import numpy as np
# count classes across the validation set
all_labels = np.concatenate([y.numpy() for _, y in val_loader])
counts = [int((all_labels == c).sum()) for c in range(common.NUM_CLASSES)]

fig, ax = nbfig.fig(figsize=(5.5, 3.2))
bars = ax.bar(common.GRADE_NAMES, counts, color=[nbfig.TURQUOISE, nbfig.DEEPPINK])
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom",
            fontweight="bold", family="DejaVu Sans Mono")
ax.set_ylabel("validation images")
nbfig.show(fig, "Class balance")
print("majority-class baseline accuracy:", f"{max(counts) / sum(counts):.3f}")
"""))

both(md("""
### 0.5 Augmentation: turn one eye into many
We have only a few thousand images. **Augmentation** makes small random changes -- flip,
rotate -- every time an image is used, so the model sees more variety and can't just
memorize specific photos. Here is the *same eye*, fifteen times, run through the real
training pipeline. Each one is a slightly different training example, for free.
"""))

both(code("""
import torchvision.transforms as T
pil = T.ToPILImage()(common._denorm(images[0]))
aug = common._transform(224, train=True)   # the actual training-time augmentation

fig, axes = nbfig.fig(3, 5, figsize=(11, 7))
for ax in axes.ravel():
    shown = common._denorm(aug(pil))        # apply augmentation, then de-normalize to view
    ax.imshow(shown.permute(1, 2, 0).numpy()); ax.axis("off")
nbfig.show(fig, "The same eye, 15 random augmentations")
"""))

both(md("""
### 0.6 Augmentations we deliberately *avoid* (and why)
Augmentation is not a free-for-all. In everyday-photo tasks (cats, cars) people throw on strong
**color jitter, shearing, perspective warps, vertical flips** -- a cat is still a cat upside-down
and tinted blue. **Medicine is different: some "harmless" augmentations destroy the signal.** The
exact things a clinician reads off a retina -- the *color* of a hemorrhage, the *shape* of the
vessels, the *position* of a lesion relative to the macula -- are the things these transforms
mangle. Below we crank each one up so you can *see* the damage, then we explain why it's off the table.
"""))

both(code("""
# Crank each "risky" augmentation way up so the damage is obvious.
risky = {
    "original":        pil,
    "color jitter":    T.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.45)(pil),
    "heavy shear":     T.functional.affine(pil, angle=0, translate=(0, 0), scale=1.0, shear=[35, 20]),
    "perspective warp": T.RandomPerspective(distortion_scale=0.6, p=1.0)(pil),
    "vertical flip":   T.functional.vflip(pil),
    "huge zoom-crop":  T.RandomResizedCrop(224, scale=(0.15, 0.15))(pil),
}
fig, axes = nbfig.fig(2, 3, figsize=(11, 7.2))
for ax, (name, im) in zip(axes.ravel(), risky.items()):
    ax.imshow(im); ax.set_title(name, fontsize=12,
              color=(nbfig.INK if name == "original" else nbfig.DEEPPINK)); ax.axis("off")
nbfig.show(fig, "Too far: augmentations that corrupt the diagnosis")
"""))

both(md("""
**Why each one is dangerous here:**
- **Color jitter / hue shift** -- diabetic retinopathy is graded partly on the *color* of
  lesions (bright yellow exudates, deep-red hemorrhages). Recolor the image and you can turn a
  textbook finding into something that looks like a different disease.
- **Shear / perspective warp** -- these bend straight anatomy. Vessel calibre and the shape of
  microaneurysms are diagnostic; warping invents geometry the eye never had.
- **Vertical flip** -- a retina has a real top and bottom. DR is graded by *where* lesions sit
  relative to the macula and optic disc; flipping vertically teaches the model anatomy that
  doesn't exist in any real patient. (A horizontal flip is the borderline-OK exception -- a left
  eye is roughly a mirror of a right eye, so we *do* allow that one.)
- **Huge zoom-crop** -- crop too aggressively and the actual lesion can fall outside the frame,
  so the image no longer supports its own label.

The rule of thumb: **augment only in ways a real patient or camera could plausibly produce.**
Small rotations and a horizontal flip pass that test; the transforms above do not -- which is
exactly why our training pipeline (Section 0.5) sticks to the gentle ones.
"""))

# =========================================================================== #
# Step 1 -- logistic regression
# =========================================================================== #
both(md("""
## 1. Logistic regression -- the simplest classifier

We start at the bottom rung. Flatten each image into one long row of numbers and fit a
**linear** model: it draws a single straight boundary between "referable" and "not." We
use small 64x64 images so it runs in seconds.

**Predict first:** a coin flip gets ~50%, and always-guessing-the-majority gets the
number you just printed. How much better can a straight line on raw pixels do? Write a guess.
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

preds = clf.predict(Xva)
acc = (preds == yva).mean()
results["logreg"] = acc
print(f"logistic regression val accuracy: {acc:.3f}")
""",
    [
        ("clf = LogisticRegression", "make a LogisticRegression with max_iter=500 and solver='saga'"),
        ("clf.fit(Xtr, ytr)", "fit the classifier on the training features Xtr, ytr"),
    ],
)

both(md("""
### 1.1 Look at *how* it's right and wrong
Accuracy is one number; a **confusion matrix** shows the mistakes. Rows are the truth,
columns are the guess. The diagonal is correct; off-diagonal is where it slips.
"""))

both(code("""
nbfig.confusion(yva, preds, common.GRADE_NAMES, text="Logistic regression").show()
"""))

both(md("""
Logistic regression treats every pixel as independent and can only draw straight-line
boundaries. It has no idea that neighboring pixels form vessels, spots, or shapes. Hold
that thought -- it's the ceiling every later model breaks through.
"""))

# =========================================================================== #
# Step 2 -- MLP
# =========================================================================== #
both(md("""
## 2. Multi-layer perceptron -- stacking non-linear layers

Same flattened pixels, but now we stack layers with non-linearities between them, so the
model can **bend** its decision boundary instead of using one straight cut. It still has
no notion of space, though -- shuffle the pixels and it wouldn't notice.
"""))

todo(
    """
mlp = common.make_mlp(in_features=3 * 64 * 64)
history = common.train_model(mlp, tr64, va64, epochs=15, lr=1e-3, device=device)
results["mlp"] = history[-1][1]
print(f"mlp val accuracy: {history[-1][1]:.3f}")
""",
    [
        ("mlp = common.make_mlp", "build an MLP with common.make_mlp; in_features is 3*64*64 (the flattened size)"),
        ("history = common.train_model", "train it: common.train_model(mlp, tr64, va64, epochs=15, lr=1e-3, device=device)"),
    ],
)

both(md("""
### 2.1 The learning curve
Unlike logreg, a neural net learns *gradually*, one epoch (one pass through the data) at a
time. The curve below shows validation accuracy climbing while the training loss falls --
that's learning happening. A flat curve means it's stuck; a falling-then-rising-loss curve
would mean trouble.
"""))

both(code("""
nbfig.learning_curve(history, text="MLP: accuracy up, loss down").show()
"""))

both(md("""
*Did it beat logreg by much?* Usually only a little. Extra layers help the model bend, but
it still can't see that pixels next to each other belong together. That missing idea --
**spatial structure** -- is the next rung, and it's a big one.
"""))

# =========================================================================== #
# Step 3 -- CNN
# =========================================================================== #
both(md("""
## 3. Convolutional neural network -- seeing in 2D

A **CNN** slides small filters across the image, so it understands *spatial* structure:
edges, blobs, textures, and where they are. Each filter is a little pattern-detector that
fires when it sees its pattern anywhere in the image. Now we switch to the full 224x224 images.

**Predict:** how much will accuracy jump versus the MLP?
"""))

todo(
    """
tr224, va224 = common.get_loaders(size=224, batch_size=32)

cnn = common.make_small_cnn()
history = common.train_model(cnn, tr224, va224, epochs=15, lr=1e-3, device=device)
results["cnn"] = history[-1][1]
print(f"cnn val accuracy: {history[-1][1]:.3f}")
""",
    [
        ("cnn = common.make_small_cnn()", "build the CNN with common.make_small_cnn()"),
        ("history = common.train_model(cnn", "train it for 15 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

both(code("""
nbfig.learning_curve(history, text="CNN: a steeper climb").show()
"""))

both(md("""
### 3.1 What did it learn to see?
We can peek at the filters in the very first convolutional layer. Nobody told the network
what to look for -- it *learned* these little edge and color detectors on its own, just
from trying to predict the label.
"""))

both(code("""
common.show_first_layer_filters(cnn)
"""))

both(md("""
### 3.2 Its mistakes
"""))

both(code("""
res = common.evaluate_classifier(lambda b: cnn(b).argmax(1), va224, device)
nbfig.confusion(res["y"], res["pred"], common.GRADE_NAMES, text="CNN confusion matrix").show()
"""))

both(md("""
That jump over the MLP is convolutions earning their keep: the model can finally use the
*arrangement* of pixels, not just their values. But we trained it from scratch on a few
thousand images. The next rung borrows a head start from millions.
"""))

both(md("""
### 3.3 Your turn: the knobs that change everything

A model isn't a fixed thing, it's a machine with **dials**. The ones you'll touch most:

- **Learning rate** -- the *step size* the model takes downhill while learning (remember the ball
  rolling down a hill). Too small and it crawls; too big and it overshoots and bounces around.
- **Epochs** -- how many times the model sees the whole training set. More isn't always better
  (too many causes *overfitting* -- memorizing instead of learning).
- **Regularization** -- deliberate handicaps that stop the model from memorizing. Two kinds here:
  **dropout** (randomly ignore some neurons each step) and **weight decay** (an L2 penalty that
  keeps the weights small). Turn them up when the model overfits.
- **Activation function** -- the little nonlinearity after each layer that lets the network bend.
  `relu` is the modern default; `sigmoid`/`tanh` are the classics (and can stall a deep net).

First, let's *see* the learning rate matter: the cell below trains the same small CNN at three
learning rates and overlays the curves. **Change the numbers and re-run** -- this is your playground.
"""))

both(code("""
# ---- the control panel: change these and re-run ----
LEARNING_RATES = [3e-4, 1e-3, 3e-3]   # try adding 1e-2 (too big) or 1e-5 (too small)
EPOCHS = 8
# ----------------------------------------------------

runs = {}
for lr in LEARNING_RATES:
    print(f"training a fresh CNN at lr={lr} ...")
    model = common.make_small_cnn().to(device)
    runs[lr] = common.train_model(model, tr224, va224, epochs=EPOCHS, lr=lr, device=device, verbose=False)

# overlay: validation accuracy (left) and training loss (right) for each learning rate
fig, (a1, a2) = nbfig.fig(1, 2, figsize=(11, 4.2))
colors = nbfig.palette(len(LEARNING_RATES))
for c, (lr, hist) in zip(colors, runs.items()):
    ep = [h[0] for h in hist]
    a1.plot(ep, [h[1] for h in hist], "-o", color=c, label=f"lr={lr}")
    a2.plot(ep, [h[2] for h in hist], "-o", color=c, label=f"lr={lr}")
a1.set_title("validation accuracy"); a1.set_xlabel("epoch"); a1.set_ylabel("accuracy"); a1.legend()
a2.set_title("training loss"); a2.set_xlabel("epoch"); a2.set_ylabel("loss"); a2.legend()
nbfig.show(fig, "Same model, three learning rates")
"""))

both(md("""
Look at the shapes. The **middle** learning rate usually climbs fastest and settles highest. The
**smallest** one learns too slowly to get there in time. The **largest** one is jumpy, its loss
zig-zags because each step overshoots. That's the whole intuition behind tuning: find the step
size that's big enough to make progress but small enough to be stable. Try `1e-2` to watch it
break, then `1e-5` to watch it crawl.
"""))

both(md("""
### 3.4 The full dashboard: every dial in one place
Now the real playground. The control panel below exposes **all** the knobs at once, learning rate,
epochs, dropout, weight decay, and the activation function. Change any of them, re-run, and read
the curve. Some things to try:

- Set `DROPOUT = 0.0` and `WEIGHT_DECAY = 0.0` -- watch training loss fall fast but validation
  accuracy stall or wobble (that gap is overfitting).
- Push `DROPOUT = 0.5` or `WEIGHT_DECAY = 1e-3` -- training gets *harder* (loss falls slower) but
  the model often *generalizes* better.
- Swap `ACTIVATION = "sigmoid"` -- feel how a classic nonlinearity can slow a deeper network down.
"""))

both(code("""
# ================= THE CONTROL PANEL: change anything, re-run =================
LEARNING_RATE = 1e-3      # step size downhill
EPOCHS        = 10        # passes over the data
DROPOUT       = 0.3       # regularization: 0.0 = none, 0.5 = heavy
WEIGHT_DECAY  = 0.0       # L2 regularization: try 1e-4, 1e-3
ACTIVATION    = "relu"    # "relu", "leaky_relu", "gelu", "elu", "tanh", "sigmoid"
# =============================================================================

model = common.make_small_cnn(dropout=DROPOUT, activation=ACTIVATION).to(device)
history = common.train_model(model, tr224, va224, epochs=EPOCHS, lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY, device=device, verbose=False)

print(f"lr={LEARNING_RATE}  dropout={DROPOUT}  weight_decay={WEIGHT_DECAY}  activation={ACTIVATION}")
print(f"final validation accuracy: {history[-1][1]:.3f}")
nbfig.learning_curve(history, text="Your custom CNN").show()
"""))

both(md("""
Keep a little log as you go, `"dropout 0.3 -> 0.5: val_acc 0.71 -> 0.74"`, exactly like you'll do
in the Day 3 capstone. That habit, change one dial, measure, write it down, is the entire craft of
tuning a model. There's no secret best setting; there's only *measured* better.
"""))

# =========================================================================== #
# Step 4 -- ResNet (transfer learning)
# =========================================================================== #
both(md("""
## 4. ResNet50 -- transfer learning

Instead of learning vision from scratch, we take a 50-layer network already trained on a
million everyday photos (ImageNet) and **reuse its vision.** We freeze it and train only a
small new final layer for our yes/no question. This is the single biggest practical trick
in modern computer vision.

![Transfer learning: reuse the backbone, train a tiny head](img/transfer_learning.png)
"""))

todo(
    """
resnet = common.make_resnet50(pretrained=True)
history = common.train_model(resnet, tr224, va224, epochs=3, lr=1e-3, device=device)
results["resnet"] = history[-1][1]
print(f"resnet val accuracy: {history[-1][1]:.3f}")
""",
    [
        ("resnet = common.make_resnet50", "build a pretrained ResNet50 with common.make_resnet50(pretrained=True)"),
        ("history = common.train_model(resnet", "finetune for 3 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

both(code("""
nbfig.learning_curve(history, text="ResNet: high accuracy in just 3 epochs").show()
"""))

both(md("""
### 4.1 Where is it looking?
A real worry in medical AI: is the model looking at the *disease*, or at some artifact (a
bright edge, the camera label)? **Grad-CAM** highlights the pixels that most drove the
prediction. We want the heat on the retina, not the border.
"""))

both(code("""
import numpy as np
from PIL import Image

imgs, _ = next(iter(va224))
cam, predicted = common.gradcam(resnet, imgs[0], device=device)
cam_up = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((224, 224))) / 255.0

fig, (a1, a2) = nbfig.fig(1, 2, figsize=(8.5, 4.4))
base = common._denorm(imgs[0]).permute(1, 2, 0).numpy()
a1.imshow(base); a1.set_title("the eye", fontsize=11); a1.axis("off")
a2.imshow(base); a2.imshow(cam_up, cmap="inferno", alpha=0.5)
a2.set_title(f"where ResNet looked (pred: {common.GRADE_NAMES[predicted]})", fontsize=11)
a2.axis("off")
nbfig.show(fig, "Grad-CAM: the model's attention")
"""))

both(md("""
*Why did pretraining help so much with so little training?* The network already knew how to
see edges, textures, and shapes. We only had to teach it which combinations mean "referable."
That's the power of standing on a million photos' worth of prior learning.
"""))

# =========================================================================== #
# Step 5 -- ViT
# =========================================================================== #
both(md("""
## 5. Vision Transformer -- patches that pay attention

The newest rung. A **ViT** chops the image into patches, turns each patch into a vector, and
lets the patches *pay attention* to each other -- deciding which patches matter for the call.
It's the exact machinery behind the language models you've used, just patches instead of
words. (Much more on that tomorrow.)

![A ViT: image to patches to attention to prediction](img/vit_patches.png)
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

both(code("""
nbfig.learning_curve(history, text="Vision Transformer").show()
"""))

both(md("""
On a few thousand images the ViT and the ResNet usually land close together -- transformers
are hungrier for data, so their real advantage shows at larger scale. The point isn't that
ViT always wins; it's that *attention* is a second, equally powerful way to see, and it's the
bridge to tomorrow.
"""))

# =========================================================================== #
# Leaderboard
# =========================================================================== #
both(md("""
## 6. The leaderboard you just built

Five models, same data, same yes/no question. Watch the climb -- and remember which idea
bought each step up.
"""))

both(code("""
names = list(results.keys())
accs = [results[n] for n in names]

fig, ax = nbfig.fig(figsize=(7.5, 4))
bars = ax.bar(names, accs, color=nbfig.palette(len(names)))
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}", ha="center",
            fontweight="bold", family="DejaVu Sans Mono")
ax.set_ylabel("validation accuracy"); ax.set_ylim(0, 1)
nbfig.show(fig, "Same data, five models")
"""))

both(md("""
Each rung added exactly one idea: **non-linearity** (MLP), **spatial structure** (CNN),
**pretraining** (ResNet), **attention** (ViT). That is, in miniature, the last fifteen years
of computer vision.
"""))

# =========================================================================== #
# Where it fails
# =========================================================================== #
both(md("""
## 7. Accuracy can lie -- and why that's dangerous here

Remember the class balance from section 0.3. In screening, the classes are lopsided and the
**costs are not symmetric**: telling a sick person they're fine (a *false negative*) is far
worse than a false alarm. A model can post a high accuracy while quietly missing the cases
that matter most. Always read the confusion matrix, not just the headline number.
"""))

both(code("""
best = common.evaluate_classifier(lambda b: resnet(b).argmax(1), va224, device)
nbfig.confusion(best["y"], best["pred"], common.GRADE_NAMES,
                text="Best model: count the missed 'referable' eyes").show()
print("per-class recall:", {k: round(v, 2) for k, v in
      common.evaluate_classifier(lambda b: resnet(b).argmax(1), va224, device)["per_class"].items()})
"""))

both(md("""
The bottom-left cell -- truly *referable* eyes the model called *not referable* -- is the one
a clinician loses sleep over. This is why medical AI is judged on sensitivity and specificity,
not raw accuracy. We dig into exactly that vocabulary in the slides.
"""))

# =========================================================================== #
# Bridge + stretch
# =========================================================================== #
both(md("""
## Bridge to Day 2

The Vision Transformer worked by splitting the image into patches and letting them attend to
each other:

```
   IMAGE                                     SENTENCE
   [patch][patch][patch]  --embed-->         "the"  "cat"  "sat"  --embed-->
        vectors                                    vectors
          |                                          |
       attention  (which patches matter?)         attention  (which words matter?)
          |                                          |
       prediction: referable?                     prediction: next word
```

That second column is a **Large Language Model**. Same machinery, different input. Tomorrow we
use one to read radiology reports and combine it with images. See you then.
"""))

both(md("""
## Stretch -- find a disagreement

If you finished early:

1. Find a validation eye where the **ResNet and the ViT predict differently**. Show the image
   and both predictions. What's unusual about it -- blurry, dark, an artifact?
2. Run Grad-CAM on a model's **wrong** prediction. Was it looking at the wrong thing?
3. Which model would you actually deploy in a clinic, and why? (Hint: it's not just accuracy.)
"""))

# =========================================================================== #
def build():
    sol = new_nb()
    lab = new_nb()
    sol.cells = [p[0] for p in PAIRS]
    lab.cells = [p[1] for p in PAIRS]
    save(sol, ROOT / "notebooks/day1_ladder/day1_solution.ipynb")
    save(lab, ROOT / "notebooks/day1_ladder/day1.ipynb")
    print(f"wrote day1.ipynb + day1_solution.ipynb ({len(PAIRS)} cells)")


if __name__ == "__main__":
    build()
