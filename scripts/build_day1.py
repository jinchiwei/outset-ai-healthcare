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
import os, sys                                    # os = talk to the computer; sys = talk to Python itself
if not os.path.exists("common.py"):              # if the course file isn't here, we must be on a fresh Colab
    os.system("git clone -q https://github.com/jinchiwei/outset-ai-healthcare.git")  # download the course repo
    os.chdir("outset-ai-healthcare/notebooks/day1_ladder")  # step into the folder we just downloaded
sys.path.insert(0, ".")                          # let Python find files in this folder (like common.py)
sys.path.insert(0, "../_shared")                 # also look one folder up in _shared (for nbfig, colab_setup)
import colab_setup                               # a helper that installs missing packages
colab_setup.ensure()                             # actually install anything that's missing
colab_setup.gpu_check()                          # print whether we have a fast GPU to train on
"""))

both(code("""
import os, sys           # os/sys again: file paths and Python settings
import torch             # PyTorch: the deep-learning library that runs the models
import numpy as np       # NumPy: fast math on arrays of numbers (nicknamed np)
import common            # the course helper module with all the get_loaders/make_* functions
# nbfig lives in notebooks/_shared; add it relative to common.py so this cell
# works even if the setup cell above wasn't run, and from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(common.__file__)), "..", "_shared"))
import nbfig            # Colab-safe branded plotting (matches the slide figures)
nbfig.use()             # switch matplotlib over to the course's plot styling

# pick the fastest hardware we have: cuda (NVIDIA GPU) > mps (Apple GPU) > cpu (slow fallback)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)   # show which one we ended up with
# keep the leaderboard across accidental re-runs of this cell (don't wipe what you trained)
results = globals().get("results", {})   # each model's validation accuracy lands here
"""))

# =========================================================================== #
# Toolbox -- the functions you can call (reference for the TODOs)
# =========================================================================== #
both(md("""
## Your toolbox: the functions you can call

Every `# TODO` below is filled by calling one of these functions -- you never write a model from
scratch. `common` is already imported for you. **Keep this section handy while you work.**

**How the pieces connect:** `get_loaders()` gives you **loaders** -> a `make_*()` gives you a
**model** -> `train_model(model, ...loaders...)` trains it -> `evaluate_classifier` / `gradcam`
inspect the trained model.

**What the data looks like** (so shapes never surprise you):
- A **loader** hands out **batches**; one batch is a pair `(images, labels)`.
- `images` has shape `(N, 3, H, W)`; `labels` has shape `(N,)` -- one 0 or 1 per image.
- A **model** turns images into scores: `model(images)` -> shape `(N, 2)` (a score for "no" and "yes").
"""))

both(md("""
### Get the data
| function | what you pass in | what you get back |
|---|---|---|
| `common.get_loaders(size=64)` | `size` = `64` (for the MLP) or `224` (CNN / ResNet / ViT) | `train_loader, val_loader` |
| `common.flatten_for_classical(loader)` | a loader | `X, y` -- flat rows of numbers for the logistic-regression step |

### Build a model (pick one per rung)
| function | what you pass in | what you get back |
|---|---|---|
| `common.make_mlp(in_features=3*64*64)` | `in_features` = pixels per image | a model |
| `common.make_small_cnn()` | nothing (or `dropout=0.3`, `activation="relu"`) | a model |
| `common.make_resnet50(pretrained=True)` | `pretrained=True` to reuse learned weights | a model |
| `common.make_vit_base(pretrained=True)` | `pretrained=True` | a model |

### Train it and score it
| function | what you pass in | what you get back |
|---|---|---|
| `common.train_model(model, train_loader, val_loader, epochs=3, lr=1e-3, device=device)` | a model, the two loaders, `epochs` (int), `lr` (small float like `1e-3`) | the trained model; prints accuracy each epoch |
| `common.evaluate_classifier(lambda b: model(b), val_loader, device=device)` | a small function that runs the model on a batch, plus a loader | accuracy and other metrics |

### Look inside the model
| function | what you pass in | what you get back |
|---|---|---|
| `common.gradcam(model, image, device=device)` | a trained model + one image `(3, H, W)` | a heatmap of *where* it looked |
| `common.show_first_layer_filters(model)` | a model | the patterns the first layer learned |
| `common.show_rgb_split(image)` | one image | the red / green / blue channels |
| `common.show_pixel_histogram(image)` | one image | a chart of pixel brightness |
"""))

both(md(r"""
### A full worked recipe

Every "your turn" is a slice of this. Task: *train a ResNet and see where it looks.*

```python
# 1. data  -- get_loaders returns the two loaders
train_loader, val_loader = common.get_loaders(size=224)

# 2. model -- a make_* function returns a model
model = common.make_resnet50(pretrained=True)

# 3. train -- pass the model and both loaders into train_model
common.train_model(model, train_loader, val_loader, epochs=3, lr=1e-3, device=device)

# 4. inspect -- grab ONE image from a batch, then ask where the model looked
images, labels = next(iter(val_loader))     # images is (N, 3, H, W); take images[0]
common.gradcam(model, images[0], device=device)
```

Read it as a chain: `get_loaders` -> loaders -> `make_resnet50` -> model -> `train_model` -> `gradcam`.
Once you see the chain, filling in a TODO is just choosing the right link.
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
train_loader, val_loader = common.get_loaders(size=224, batch_size=32)  # data for training and for checking, 32 eyes per batch
images, labels = next(iter(train_loader))   # pull the first batch: a stack of eye photos and their 0/1 labels
print("one batch:", images.shape, "  <- (batch, channels, height, width)")  # e.g. (32, 3, 224, 224)

# Look at eight real fundus images with their referable / not-referable labels.
fig, axes = nbfig.fig(2, 4, figsize=(11, 5.6))   # make a 2-row, 4-column grid of empty plots
for ax, img, lab in zip(axes.ravel(), images, labels):   # walk through each plot slot with one image + its label
    ax.imshow(common._denorm(img).permute(1, 2, 0).numpy())  # undo normalization, reorder to H,W,color, and draw it
    ax.set_title(common.GRADE_NAMES[int(lab)], fontsize=11,   # title = the human-readable class name
                 color=(nbfig.DEEPPINK if int(lab) else nbfig.INK))  # pink title if referable (label 1), dark if not
    ax.axis("off")     # hide the x/y number axes; it's a photo, not a chart
nbfig.show(fig, "Eight eyes: can you tell which need a doctor?")   # render the finished grid with a caption
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
# subtract each channel's mean, then divide by its std -> every channel now centered at ~0
standardized = (raw_batch - ch_mean[:, None, None]) / ch_std[:, None, None]

raw0, std0 = raw_batch[0], standardized[0]     # first eye, before and after standardizing
disp = (std0 - std0.min()) / (std0.max() - std0.min())  # rescale just to view

fig, (a1, a2, a3) = nbfig.fig(1, 3, figsize=(11, 3.6))   # three side-by-side panels
a1.imshow(raw0.permute(1, 2, 0).numpy()); a1.set_title("raw (0..1)", fontsize=11); a1.axis("off")   # left: original eye
a2.imshow(disp.permute(1, 2, 0).numpy()); a2.set_title("standardized (rescaled to view)", fontsize=11); a2.axis("off")  # middle: after standardizing
x = np.arange(3)     # positions 0,1,2 for the three color channels on the bar chart
a3.bar(x - 0.2, raw_batch.mean((0, 2, 3)).numpy(), 0.4, color=nbfig.TURQUOISE, label="raw mean")     # bars: mean of each channel BEFORE
a3.bar(x + 0.2, standardized.mean((0, 2, 3)).numpy(), 0.4, color=nbfig.DEEPPINK, label="standardized mean")  # bars: mean of each channel AFTER (~0)
a3.axhline(0, color=nbfig.MUTED, lw=0.8); a3.set_xticks(x); a3.set_xticklabels(["R", "G", "B"])   # draw the zero line and label channels R/G/B
a3.set_title("channel means: ~0 after", fontsize=11); a3.legend()   # title + a key for the two bar colors
nbfig.show(fig, "Standardization: subtract the mean, divide by the std -> centered at zero")   # render all three panels
print("standardized channel means:", [round(v, 3) for v in standardized.mean((0, 2, 3)).tolist()])  # confirm the means really are ~0
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
import numpy as np       # NumPy for the counting math below
# count classes across the validation set
all_labels = np.concatenate([y.numpy() for _, y in val_loader])   # gather every label from every batch into one long array
counts = [int((all_labels == c).sum()) for c in range(common.NUM_CLASSES)]   # how many eyes fall in each class (0 and 1)

fig, ax = nbfig.fig(figsize=(5.5, 3.2))   # one small bar chart
bars = ax.bar(common.GRADE_NAMES, counts, color=[nbfig.TURQUOISE, nbfig.DEEPPINK])   # one bar per class, sized by its count
for b, c in zip(bars, counts):   # walk each bar with its number
    ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom",   # print the count on top of the bar
            fontweight="bold", family="DejaVu Sans Mono")
ax.set_ylabel("validation images")   # y-axis label
nbfig.show(fig, "Class balance")   # render the chart
# if a lazy model always guessed the biggest class, this is the accuracy it would get for free
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
import torchvision.transforms as T   # torchvision's image-transform toolbox, nicknamed T
pil = T.ToPILImage()(common._denorm(images[0]))   # turn the first eye tensor back into a normal image object
aug = common._transform(224, train=True)   # the actual training-time augmentation

fig, axes = nbfig.fig(3, 5, figsize=(11, 7))   # a 3x5 grid = 15 slots
for ax in axes.ravel():   # fill each of the 15 slots
    shown = common._denorm(aug(pil))        # apply augmentation, then de-normalize to view
    ax.imshow(shown.permute(1, 2, 0).numpy()); ax.axis("off")   # draw this random version, hide the axes
nbfig.show(fig, "The same eye, 15 random augmentations")   # render the grid
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
risky = {   # a dictionary mapping each label to one deliberately-overdone version of the eye
    "original":        pil,   # the untouched image, for comparison
    "color jitter":    T.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.45)(pil),   # shift only the hue, hard
    "heavy shear":     T.functional.affine(pil, angle=0, translate=(0, 0), scale=1.0, shear=[35, 20]),   # slant the image
    "perspective warp": T.RandomPerspective(distortion_scale=0.6, p=1.0)(pil),   # tilt it like viewing from an angle
    "vertical flip":   T.functional.vflip(pil),   # flip top-to-bottom
    "huge zoom-crop":  T.RandomResizedCrop(224, scale=(0.15, 0.15))(pil),   # zoom way in, keeping only 15% of the area
}
fig, axes = nbfig.fig(2, 3, figsize=(11, 7.2))   # 2x3 grid = 6 slots, one per entry above
for ax, (name, im) in zip(axes.ravel(), risky.items()):   # pair each slot with a (label, image)
    ax.imshow(im); ax.set_title(name, fontsize=12,   # draw the image and label it
              color=(nbfig.INK if name == "original" else nbfig.DEEPPINK)); ax.axis("off")   # pink title for the damaged ones
nbfig.show(fig, "Too far: augmentations that corrupt the diagnosis")   # render the grid
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

both(md("""
### 0.7 Play with it: your own augmentation dials
Now grab the controls. Drag the sliders below and watch a real eye transform in real time, no
training, instant feedback. Try to answer for yourself: *which* of these still looks like a valid
patient photo, and which ones cross the line we just talked about? (The sliders need Colab or
Jupyter widgets; if you don't see them, run the previous cells first.)
"""))

both(code("""
import torchvision.transforms.functional as TF   # the one-shot versions of the transforms (nicknamed TF)
import matplotlib.pyplot as plt                   # the plotting library, nicknamed plt

# self-heal: works even if you jumped straight here after a kernel restart
try:
    common, device, tr224     # check these already exist from earlier cells
except NameError:             # if not (fresh kernel), rebuild them
    import sys; sys.path.insert(0, ".")
    import common
    _nb, device, tr224, _va = common.playground_setup()   # quick setup that returns the pieces we need

# one clean sample eye to experiment on
_imgs, _ = next(iter(tr224))                 # grab one batch (we ignore its labels with _)
BASE = TF.to_pil_image(common._denorm(_imgs[0]))   # first eye as a plain image we can transform

def show_augment(rotate=0, brightness=1.0, contrast=1.0, blur=0, zoom=1.0, hflip=False):   # runs once per slider change
    img = TF.adjust_contrast(TF.adjust_brightness(BASE, brightness), contrast)   # apply brightness then contrast
    if zoom > 1.0:                           # only zoom if the dial is above 1
        w, h = img.size                      # current width and height
        img = TF.resize(TF.center_crop(img, [int(h / zoom), int(w / zoom)]), [h, w])   # crop the center, then stretch back to full size
    img = TF.rotate(img, rotate)             # rotate by the chosen number of degrees
    if hflip:                                # if the checkbox is ticked
        img = TF.hflip(img)                  # mirror left-to-right
    if blur > 0:                             # if the blur dial is above 0
        img = TF.gaussian_blur(img, kernel_size=2 * blur + 1)   # soften it (kernel size must be odd)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))   # one square plot
    ax.imshow(img); ax.axis("off")           # draw the transformed eye, no axes
    ax.set_title(f"rotate {rotate}deg | bright {brightness:.1f} | contrast {contrast:.1f} | "   # caption showing every dial's value
                 f"blur {blur} | zoom {zoom:.1f} | hflip {hflip}", fontsize=8)
    plt.show()                               # display it

try:
    from ipywidgets import interact, FloatSlider, IntSlider, Checkbox   # the interactive slider widgets
    # coarse steps + continuous_update=False => it only redraws when you RELEASE the
    # slider, not on every pixel, so it stays snappy even on a slow Colab runtime.
    _S = dict(continuous_update=False)       # shared slider option: redraw on release, not while dragging
    interact(show_augment,                   # wire the function above to a set of sliders
             rotate=IntSlider(value=0, min=-45, max=45, step=15, **_S),        # rotation dial
             brightness=FloatSlider(value=1.0, min=0.4, max=2.0, step=0.2, **_S),   # brightness dial
             contrast=FloatSlider(value=1.0, min=0.4, max=2.0, step=0.2, **_S),     # contrast dial
             blur=IntSlider(value=0, min=0, max=8, step=2, **_S),              # blur dial
             zoom=FloatSlider(value=1.0, min=1.0, max=2.5, step=0.25, **_S),   # zoom dial
             hflip=Checkbox(value=False))    # mirror on/off checkbox
except ImportError:                          # ipywidgets not installed (some plain setups)
    print("No live sliders here (ipywidgets missing). For the interactive version, run")
    print("   !pip install ipywidgets   then restart the kernel and re-run this cell.")
    print("Showing a few fixed examples instead:")
    for kw in [dict(), dict(rotate=25), dict(brightness=1.6), dict(blur=5), dict(zoom=1.8, hflip=True)]:   # a few preset combos
        show_augment(**kw)                   # draw each preset
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
from sklearn.linear_model import LogisticRegression   # a classic straight-line classifier from scikit-learn

tr64, va64 = common.get_loaders(size=64, batch_size=64)   # small 64x64 loaders so this runs in seconds
Xtr, ytr = common.flatten_for_classical(tr64)   # training set as flat rows of numbers (X) + labels (y)
Xva, yva = common.flatten_for_classical(va64)   # same for the validation set
print("flattened features per image:", Xtr.shape[1])   # how many numbers describe one eye (3*64*64)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)

preds = clf.predict(Xva)         # ask the trained model for a 0/1 guess on each validation eye
acc = (preds == yva).mean()      # fraction of guesses that matched the truth = accuracy
results["logreg"] = acc          # save this score to the leaderboard
print(f"logistic regression val accuracy: {acc:.3f}")   # show the accuracy
""",
    [
        ("clf = LogisticRegression", "make a LogisticRegression with max_iter=1000"),
        ("clf.fit(Xtr, ytr)", "fit the classifier on the training features Xtr, ytr"),
    ],
)

both(md("""
### 1.1 Look at *how* it's right and wrong
Accuracy is one number; a **confusion matrix** shows the mistakes. Rows are the truth,
columns are the guess. The diagonal is correct; off-diagonal is where it slips.
"""))

both(code("""
nbfig.confusion(yva, preds, common.GRADE_NAMES, text="Logistic regression").show()   # truth vs guess grid: diagonal = correct
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
results["mlp"] = history[-1][1]   # history[-1] is the last epoch; [1] is its accuracy -> save to leaderboard
print(f"mlp val accuracy: {history[-1][1]:.3f}")   # print that final-epoch accuracy
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
nbfig.learning_curve(history, text="MLP: accuracy up, loss down").show()   # plot accuracy and loss over the epochs
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
tr224, va224 = common.get_loaders(size=224, batch_size=32)   # full-size 224x224 loaders for the CNN

cnn = common.make_small_cnn()
history = common.train_model(cnn, tr224, va224, epochs=15, lr=1e-3, device=device)
results["cnn"] = history[-1][1]   # last epoch's accuracy -> leaderboard
print(f"cnn val accuracy: {history[-1][1]:.3f}")   # print the final accuracy
""",
    [
        ("cnn = common.make_small_cnn()", "build the CNN with common.make_small_cnn()"),
        ("history = common.train_model(cnn", "train it for 15 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

both(code("""
nbfig.learning_curve(history, text="CNN: a steeper climb").show()   # accuracy/loss curves for the CNN
"""))

both(md("""
### 3.1 What did it learn to see?
We can peek at the filters in the very first convolutional layer. Nobody told the network
what to look for -- it *learned* these little edge and color detectors on its own, just
from trying to predict the label.
"""))

both(code("""
common.show_first_layer_filters(cnn)   # display the little edge/color detectors the first layer learned
"""))

both(md("""
The first layer is only the beginning. As an image flows **deeper** into the network, each layer
builds on the one before it: early layers react to fine edges, middle layers combine those into
textures, and the deepest layers respond to big, abstract shapes. Here are real feature maps at three
depths of your trained CNN, run on a single eye:
"""))

both(code("""
imgs, _ = next(iter(tr224))   # grab one batch of eyes (labels ignored with _)
common.show_feature_maps(cnn, imgs[0], device=device)   # early -> middle -> late, not just layer 1
"""))

both(md("""
### 3.2 Its mistakes
"""))

both(code("""
# run the CNN on the validation set; argmax(1) turns its two scores into a single 0/1 pick
res = common.evaluate_classifier(lambda b: cnn(b).argmax(1), va224, device)
nbfig.confusion(res["y"], res["pred"], common.GRADE_NAMES, text="CNN confusion matrix").show()   # truth vs guess grid
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
# lets this cell run on its own -- e.g. if you restart the kernel and jump here
try:
    common, nbfig, device     # do these already exist from earlier?
except NameError:             # if not, rebuild them
    import sys; sys.path.insert(0, ".")
    import common
    nbfig, device, _, _ = common.playground_setup()

# Small 64px images so the whole sweep finishes in seconds, not minutes -- we only
# need the *shape* of the curves, not top accuracy. (Re-run freely during class.)
trP, vaP = common.get_loaders(size=64, batch_size=64)

# ---- the control panel: change these and re-run ----
LEARNING_RATES = [3e-4, 1e-3, 3e-3]   # try adding 1e-2 (too big) or 1e-5 (too small)
EPOCHS = 6
# ----------------------------------------------------

runs = {}                     # will hold each learning rate's training history
for lr in LEARNING_RATES:     # train one fresh model per learning rate
    print(f"training a fresh CNN at lr={lr} ...")   # progress message
    model = common.make_small_cnn().to(device)      # brand-new CNN, moved onto the GPU/CPU
    runs[lr] = common.train_model(model, trP, vaP, epochs=EPOCHS, lr=lr, device=device, verbose=False)   # train and store its history

# overlay: validation accuracy (left) and training loss (right) for each learning rate
fig, (a1, a2) = nbfig.fig(1, 2, figsize=(11, 4.2))   # two panels side by side
colors = nbfig.palette(len(LEARNING_RATES))   # one distinct color per learning rate
for c, (lr, hist) in zip(colors, runs.items()):   # draw each run in its own color
    ep = [h[0] for h in hist]     # the epoch numbers (x-axis)
    a1.plot(ep, [h[1] for h in hist], "-o", color=c, label=f"lr={lr}")   # left panel: accuracy per epoch
    a2.plot(ep, [h[2] for h in hist], "-o", color=c, label=f"lr={lr}")   # right panel: loss per epoch
a1.set_title("validation accuracy"); a1.set_xlabel("epoch"); a1.set_ylabel("accuracy"); a1.legend()   # label left panel
a2.set_title("training loss"); a2.set_xlabel("epoch"); a2.set_ylabel("loss"); a2.legend()   # label right panel
nbfig.show(fig, "Same model, three learning rates")   # render both panels
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
# lets this cell run on its own -- e.g. if you restart the kernel and jump here
try:
    common, nbfig, device     # already defined from earlier cells?
except NameError:             # if not, rebuild them
    import sys; sys.path.insert(0, ".")
    import common
    nbfig, device, _, _ = common.playground_setup()

# small 64px images keep this fast enough to re-run live (seconds, not minutes)
trP, vaP = common.get_loaders(size=64, batch_size=64)

# ================= THE CONTROL PANEL: change anything, re-run =================
LEARNING_RATE = 1e-3      # step size downhill
EPOCHS        = 8         # passes over the data
DROPOUT       = 0.3       # regularization: 0.0 = none, 0.5 = heavy
WEIGHT_DECAY  = 0.0       # L2 regularization: try 1e-4, 1e-3
ACTIVATION    = "relu"    # "relu", "leaky_relu", "gelu", "elu", "tanh", "sigmoid"
# =============================================================================

model = common.make_small_cnn(dropout=DROPOUT, activation=ACTIVATION).to(device)   # build a CNN with your chosen dials
history = common.train_model(model, trP, vaP, epochs=EPOCHS, lr=LEARNING_RATE,   # train it with the panel's settings
                             weight_decay=WEIGHT_DECAY, device=device, verbose=False)

print(f"lr={LEARNING_RATE}  dropout={DROPOUT}  weight_decay={WEIGHT_DECAY}  activation={ACTIVATION}")   # echo the settings you used
print(f"final validation accuracy: {history[-1][1]:.3f}")   # the last epoch's accuracy
nbfig.learning_curve(history, text="Your custom CNN").show()   # plot how this run learned
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
results["resnet"] = history[-1][1]   # last epoch's accuracy -> leaderboard
print(f"resnet val accuracy: {history[-1][1]:.3f}")   # print it (notice how high, after only 3 epochs)
""",
    [
        ("resnet = common.make_resnet50", "build a pretrained ResNet50 with common.make_resnet50(pretrained=True)"),
        ("history = common.train_model(resnet", "finetune for 3 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

both(code("""
nbfig.learning_curve(history, text="ResNet: high accuracy in just 3 epochs").show()   # its accuracy/loss curves
"""))

both(md("""
### 4.1 Where is it looking?
A real worry in medical AI: is the model looking at the *disease*, or at some artifact (a
bright edge, the camera label)? **Grad-CAM** highlights the pixels that most drove the
prediction. We want the heat on the retina, not the border.
"""))

both(code("""
import numpy as np          # array math
from PIL import Image        # image resizing

imgs, _ = next(iter(va224))   # one batch of validation eyes (labels ignored)
cam, predicted = common.gradcam(resnet, imgs[0], device=device)   # heatmap of where it looked + its 0/1 guess
cam_up = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((224, 224))) / 255.0   # scale the small heatmap up to the image size

fig, (a1, a2) = nbfig.fig(1, 2, figsize=(8.5, 4.4))   # two panels: eye, and eye+heatmap
base = common._denorm(imgs[0]).permute(1, 2, 0).numpy()   # the viewable version of the eye
a1.imshow(base); a1.set_title("the eye", fontsize=11); a1.axis("off")   # left: the plain eye
a2.imshow(base); a2.imshow(cam_up, cmap="inferno", alpha=0.5)   # right: the eye with the heatmap laid semi-transparently on top
a2.set_title(f"where ResNet looked (pred: {common.GRADE_NAMES[predicted]})", fontsize=11)   # title shows the model's prediction
a2.axis("off")
nbfig.show(fig, "Grad-CAM: the model's attention")   # render both panels
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
results["vit"] = history[-1][1]   # last epoch's accuracy -> leaderboard
print(f"vit val accuracy: {history[-1][1]:.3f}")   # print the final accuracy
""",
    [
        ("vit = common.make_vit_base", "build a pretrained ViT with common.make_vit_base(pretrained=True)"),
        ("history = common.train_model(vit", "train the head for 5 epochs at lr=1e-3 on the 224px loaders"),
    ],
)

both(code("""
nbfig.learning_curve(history, text="Vision Transformer").show()   # the ViT's accuracy/loss curves
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
names = list(results.keys())        # the model names we saved, e.g. ["logreg", "mlp", ...]
accs = [results[n] for n in names]  # each model's accuracy, in the same order

fig, ax = nbfig.fig(figsize=(7.5, 4))   # one bar chart
bars = ax.bar(names, accs, color=nbfig.palette(len(names)))   # one colored bar per model
for b, a in zip(bars, accs):   # walk each bar with its accuracy
    ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}", ha="center",   # print the score above the bar
            fontweight="bold", family="DejaVu Sans Mono")
ax.set_ylabel("validation accuracy"); ax.set_ylim(0, 1)   # label y-axis, fix its range to 0..1
nbfig.show(fig, "Same data, five models")   # render the leaderboard
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
best = common.evaluate_classifier(lambda b: resnet(b).argmax(1), va224, device)   # run the ResNet on the validation set
nbfig.confusion(best["y"], best["pred"], common.GRADE_NAMES,   # draw its truth-vs-guess grid
                text="Best model: count the missed 'referable' eyes").show()
# per-class recall = for each class, what fraction did it actually catch (the sick ones matter most)
print("per-class recall:", {k: round(v, 2) for k, v in
      common.evaluate_classifier(lambda b: resnet(b).argmax(1), va224, device)["per_class"].items()})
"""))

both(md("""
The bottom-left cell -- truly *referable* eyes the model called *not referable* -- is the one
a clinician loses sleep over. This is why medical AI is judged on sensitivity and specificity,
not raw accuracy. We dig into exactly that vocabulary in the slides.
"""))

both(md("""
### 7.1 Play with it: slide the decision threshold
The model doesn't output "yes/no", it outputs a *probability*. **You** pick the cutoff for
action. Drag the slider and watch the trade-off happen live: lower the threshold to catch more
disease (sensitivity up) but trigger more false alarms (specificity down). There is no setting
that wins both, that's the whole lesson of the ROC curve, in your hands.
"""))

both(code("""
import numpy as np              # array math
import matplotlib.pyplot as plt  # plotting

# self-heal: make sure nbfig exists even if you jumped here after a restart
try:
    nbfig                        # is the plotting helper already loaded?
except NameError:
    import sys; sys.path.insert(0, "."); import common
    nbfig, device, tr224, va224 = common.playground_setup()

# use your trained model's real probabilities if they're in memory, else clear example scores
try:
    import torch
    resnet, device, va224        # do we still have a trained ResNet and the val data?
    resnet.eval(); _P, _Y = [], []   # switch model to eval mode; empty lists for probabilities and truths
    with torch.no_grad():        # no gradient tracking needed, we're only predicting
        for xb, yb in va224:     # each batch of eyes (xb) and their labels (yb)
            _P.append(torch.softmax(resnet(xb.to(device)), 1)[:, 1].cpu().numpy())   # probability of "referable" for each eye
            _Y.append(yb.numpy())   # the true labels
    scores, truth = np.concatenate(_P), np.concatenate(_Y)   # stitch batches into two long arrays
    source = "your ResNet's real predictions on the validation eyes"
except NameError:                # no trained model in memory -> fall back to made-up numbers
    _rng = np.random.RandomState(0)   # fixed random seed so the example is reproducible
    truth = np.r_[np.zeros(180), np.ones(120)].astype(int)   # 180 healthy + 120 referable fake labels
    scores = np.r_[np.clip(_rng.beta(2, 5, 180), 0, 1), np.clip(_rng.beta(5, 2, 120), 0, 1)]   # low scores for healthy, high for referable
    source = "example scores (run the notebook top-to-bottom to use your real model)"

def threshold_demo(threshold=0.5):   # redraws every time you move the slider
    pred = (scores >= threshold).astype(int)   # call it "referable" when the score clears the cutoff
    tp = int(((pred == 1) & (truth == 1)).sum()); fn = int(((pred == 0) & (truth == 1)).sum())   # true positives, missed cases
    tn = int(((pred == 0) & (truth == 0)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())   # true negatives, false alarms
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)   # sensitivity = caught sick; specificity = correctly cleared healthy
    fig, ax = plt.subplots(figsize=(8, 3.6))   # one plot
    ax.hist(scores[truth == 0], bins=20, alpha=0.6, color=nbfig.TURQUOISE, label="not referable")   # score spread for healthy eyes
    ax.hist(scores[truth == 1], bins=20, alpha=0.6, color=nbfig.DEEPPINK, label="referable")   # score spread for referable eyes
    ax.axvline(threshold, color=nbfig.INK, lw=2.5, ls="--")   # dashed line marking the current cutoff
    ax.set_xlabel("model's predicted probability of 'referable'"); ax.legend(loc="upper center")   # label x-axis + key
    ax.set_title(f"threshold {threshold:.2f}   ->   sensitivity {sens:.0%} (missed {fn}),  "   # live readout of the trade-off
                 f"specificity {spec:.0%} (false alarms {fp})", fontsize=9)
    plt.show()                   # display it

print("scoring:", source)        # tell the reader whether these are real or example scores
try:
    from ipywidgets import interact, FloatSlider   # the slider widget
    # coarse + continuous_update=False keeps it snappy: recompute on release, not every pixel
    interact(threshold_demo,     # attach the function to a single threshold slider
             threshold=FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, continuous_update=False))
except ImportError:              # no widgets available
    print("No live slider here (ipywidgets missing). Run  !pip install ipywidgets ,")
    print("restart the kernel, and re-run for the interactive version. Fixed cutoffs for now:")
    for t in (0.75, 0.5, 0.25):  # show three preset cutoffs instead
        threshold_demo(t)        # draw each one
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

both(md("""
## Level up with Claude

You have Claude Pro -- put it to work. Open **claude.ai** in another tab, pick a challenge, and
paste the starter prompt (edit it to fit). **Golden rule: understand every line it gives you.**
If you can't explain it, ask Claude to explain it more simply before you run it.

**Beat the leaderboard.**
> *"My best model on these fundus eye scans gets about ___% accuracy. Suggest ONE change to try,
> tell me exactly where in the notebook to make it, and what number to watch. I'll measure if it
> helped, then come back."*

**Break it on purpose (see overfitting).**
> *"Help me make the small CNN overfit deliberately -- train on very little data with no dropout
> or weight decay -- so I can watch the training accuracy climb while validation accuracy stalls.
> Explain what the gap means."*

**Understand the engine.**
> *"Walk me through what `common.train_model` does, line by line, like I'm new to this. What is
> the optimizer actually doing each step?"*

**Study its mistakes.**
> *"Write code to show me 6 validation eyes the ResNet got wrong, with the image and its
> confidence. What might these have in common?"*

**Tune it scientifically.** (use the section 3.4 control panel)
> *"Give me 3 experiments to run in the control panel to push accuracy up, one changed dial each,
> and tell me what to look for in the curves."*

**Swap the brain.**
> *"Show me how to swap the ResNet50 for a different pretrained model from `timm`, and compare
> their validation accuracy fairly."*

**Think like a doctor.**
> *"If this screener missed 1 in 10 people who actually have referable disease, is that okay? Help
> me reason about where to set the decision threshold and what the trade-off costs."*

Whatever you try: change **one** thing, measure, and write down what happened. That habit is the
entire job, and it's exactly what tomorrow's capstone rewards.
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
