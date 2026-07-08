"""Generate the G3 brain-CT SOLUTION notebook (the worked answer key).

Comprehensive, HS-accessible, and EVERY line of code is commented. Runs top to bottom on the
local brain-CT dataset (cc.get_loaders("brainct")): trains a stroke-vs-normal detector, evaluates
it honestly (confusion + sensitivity/specificity), asks whether it reads the brain or cheats off
the skull (Grad-CAM), stress-tests it with noise, and lands the real lesson -- because the dataset
ships NO age/sex/race/scanner, a fairness audit across those groups is simply impossible.
Edit this script + rebuild; never hand-edit the .ipynb.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, md, new_nb, save  # noqa: E402

HERE = Path(__file__).resolve().parent


def build():
    nb = new_nb()
    nb.cells = [
        md("""
# G3 SOLUTION — Spotting stroke on a brain CT (and the fairness you can't measure)

**This is the worked answer key.** It runs top to bottom and shows the whole project: the real
medicine, the idea, the code (every line commented), the result, and — the heart of this project —
one honest limit that is not a footnote but *the finding*.

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear — highlight the line and ask Claude "explain this."
"""),
        md("""
## 1. Background: what does head-CT triage AI actually do?

When someone arrives at the ER after a head injury or a suspected stroke, they get a **CT scan** of
the brain. A radiologist has to read a long queue of scans, and a bleed (hemorrhage) is an emergency
— minutes matter. So one of the most real, already-deployed uses of medical AI is **triage**: a
network scans the whole stack and pushes the scans that *probably* show bleeding to the top of the
list, so the dangerous ones get read first instead of in arrival order.

This is not science fiction. Chilamkurthy et al. trained networks on ~313,000 head CTs and detected
hemorrhage, fractures, and midline shift with AUCs above 0.90 [1]. The RSNA 2019 challenge had 60+
neuroradiologists label 25,000+ exams to build a public benchmark [2]. Deployed tools reach ~93%
accuracy and ~87% sensitivity — but do noticeably *worse* on some bleed subtypes [3]. So these tools
work, and they are not equally good on every case.

**Our toy version:** does a small network, trained in a few minutes, learn to tell a **stroke** brain
CT from a **normal** one? And — the real question — *who does it fail on?*
"""),
        md("""
## 2. The catch we will keep coming back to: a dataset with no patient labels

To know whether a tool is **fair**, you compare its accuracy across groups — young vs. old, women vs.
men, different races, different scanners. That comparison is only possible if the dataset **records**
who each scan came from. The "Datasheets for Datasets" idea says every dataset *should* ship that
documentation [4], and reviews show radiology datasets often simply don't [5]. Real audits found AI
that quietly missed disease more often in female, Black, and low-income patients — a bias you can
only *see* because those datasets recorded the groups [6][7].

Our brain-CT dataset ships **only the image and a normal/stroke label** — no age, sex, race, or
scanner. Hold onto that. We will build a working detector, and then hit a wall we cannot climb.
"""),
        code("""
# Tell Python where the course helper lives, then load the brain-CT data with it.
import sys                                          # sys lets us add a folder to Python's import path
sys.path.insert(0, "../../notebooks/day3_capstone") # that folder holds capstone_common.py
import capstone_common as cc                        # import it under the short name "cc"
import torch                                        # the deep-learning library we train with

device = "cuda" if torch.cuda.is_available() else "cpu"   # use a GPU if there is one, else the CPU
torch.manual_seed(0)                                # fix the random seed so results are repeatable

names = cc.class_names("brainct")                   # the human-readable class names for this dataset
print("classes:", names)                            # -> ['normal', 'stroke']

# get_loaders splits the images 70/15/15 into train / validation / test and batches them for us.
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders("brainct", size=64)
print("classes:", n_classes, " task:", task)        # 2 classes, a binary-classification task
xb, yb = next(iter(train_loader))                   # grab one batch just to see the shapes
print("one batch of images:", xb.shape, " labels:", yb.shape)  # (64, 3, 64, 64) images, (64, 1) labels
"""),
        md("""
Each image is a 64×64 brain slice, copied into 3 channels so a color-image network can read it. Notice
what the loader gives back: **images and labels, nothing else.** There is no column for age, sex, race,
or scanner — because the dataset never recorded them. Keep that in view; it is the whole point later.
"""),
        md("""
## 3. The idea: don't start from scratch — reuse a pretrained network

This is the same trick that won Day 1. Instead of teaching a network to see from zero on our few
hundred scans, we take a **ResNet18** already trained on a million everyday photos, freeze everything
it learned about edges and textures, and train only a small new **head** to map those features to
"normal vs. stroke." Borrowed knowledge beats from-scratch when data is small.
"""),
        code("""
# Build the model: a pretrained ResNet18 with its body frozen and a fresh 2-class head to train.
model = cc.make_model(n_classes, backbone="resnet18", pretrained=True)  # ImageNet weights, frozen body + new head

# Train just that head for a few passes over the data. Each epoch prints the validation accuracy.
model = cc.train(model, train_loader, val_loader, epochs=5, lr=1e-3, device=device)  # short = fine for a toy
"""),
        md("""
## 4. Result #1 — grade it honestly, and pick the metric that matters

Now the honest evaluation, on the **test** set the model never trained on. But accuracy alone is not
the right lens for a screening tool. Two numbers matter more:

- **Sensitivity** = of the real strokes, how many did we catch? A *missed* stroke is the dangerous
  error, so for screening this is the number to protect.
- **Specificity** = of the real normals, how many did we correctly clear? Low specificity means lots
  of false alarms — annoying and costly, but not deadly.

For triage, we would rather over-flag than miss a bleed, so we watch **sensitivity** first.
"""),
        code("""
# Run the trained model on the held-out test set and pull out the true labels and its guesses.
res = cc.evaluate(model, test_loader, device=device)  # returns accuracy + arrays of truth and predictions
y, p = res["y"], res["pred"]                          # y = the real answers, p = what the model guessed
stroke = names.index("stroke")                        # the label number that means "stroke"

acc = float((y == p).mean())                          # accuracy = fraction of all scans it got right
sens = float((p[y == stroke] == stroke).mean())       # sensitivity = of real strokes, fraction caught
spec = float((p[y != stroke] != stroke).mean())       # specificity = of real normals, fraction cleared
print(f"accuracy    = {acc:.2f}")                     # overall correctness (~0.71 typical)
print(f"sensitivity = {sens:.2f}   (of real strokes, how many we CAUGHT)")   # ~0.84 typical
print(f"specificity = {spec:.2f}   (of real normals, how many we CLEARED)")  # ~0.60 typical
"""),
        md("""
A confusion matrix shows the same thing as a picture: the diagonal is what it got right, and every
off-diagonal box is a specific kind of mistake. Look at *which* mistake is bigger.
"""),
        code("""
# Draw the confusion matrix on the test set: rows = the truth, columns = what the model guessed.
cc.audit_confusion(model, test_loader, device=device, class_names=names)  # diagonal = correct; off = errors
"""),
        md("""
Read it like a regulator. Sensitivity around **0.84** means the detector **catches most real strokes**
— good, that is the error we most wanted to avoid. But specificity around **0.60** means it **false-
alarms on a lot of normal scans**. In a real ER that is a tradeoff you would tune on purpose: for a
safety-net triage tool, crying wolf sometimes beats missing a bleed. The point is that *one accuracy
number would have hidden this* — you have to look at the two error types separately.
"""),
        md("""
## 5. Result #2 — is it reading the brain, or cheating off the skull?

A model can be right for the wrong reason. **Grad-CAM** paints a heatmap of *where* the network looked
to make each call. If the heat sits on the brain tissue, good. If it drifts onto the skull edge or the
black background, the model may have found a **shortcut** — a trick that works on this dataset but would
break on a new scanner. Shortcut-learning is exactly how "fair on one hospital" AI fails elsewhere [7].
"""),
        code("""
# Overlay Grad-CAM on a few test scans: red = the regions that most drove the model's prediction.
cc.audit_transparency(model, test_loader, device=device, class_names=names, n=4)  # top: scan, bottom: heatmap
"""),
        md("""
Honest read: some of the heat lands on brain tissue, but some bleeds along the **skull edge and the
image border** — a warning sign. Our toy model is partly reading anatomy and partly leaning on
edge/shape cues. That is a finding worth saying out loud, not hiding: *the model may be partly cheating,
and we would want a lot more data and validation before trusting it.*
"""),
        md("""
## 6. Result #3 — does it survive a messier scan?

Training images are clean. A real ER scanner is noisier — motion, low dose, different machines. We add
increasing random noise to the test images and watch accuracy. If it falls off a cliff, the model would
need monitoring (and retraining) after it ships.
"""),
        code("""
# Add more and more noise to the test images and plot how accuracy holds up (or doesn't).
cc.audit_monitoring(model, test_loader, device=device, class_names=names)  # x: noise added, y: accuracy
"""),
        md("""
Accuracy sags quickly as noise grows — this toy model is **brittle**. A deployable tool would need
augmentation, far more data, and live monitoring. Good to know *before* anyone relies on it.
"""),
        md("""
## 7. The gap that IS the finding — the fairness we cannot measure

Here is where this project is different. We *can* audit fairness **by class** — normal vs. stroke —
because the dataset records the label:
"""),
        code("""
# We CAN check accuracy per class, because the dataset records the class label (normal vs stroke).
cc.audit_fairness(model, test_loader, device=device, class_names=names)  # per-class accuracy bars
"""),
        md("""
But the audit that actually matters for fairness — *does it work as well for older patients as
younger? women as men? one scanner as another?* — we **cannot run at all.** Not because it is hard,
but because the dataset never recorded age, sex, race, or scanner. There is nothing to group by. The
figure below (from our experiment) is deliberately mostly blank: it is a picture of the metadata that
does not exist.
"""),
        code("""
# Display our 'what the dataset does NOT record' figure -- the blank where the demographics should be.
import matplotlib.pyplot as plt                       # the plotting library
import matplotlib.image as mpimg                       # helper to read a PNG into an array
img = mpimg.imread("figures/missing_metadata.png")     # load the figure the experiment saved
plt.figure(figsize=(7, 4.2))                           # set the display size
plt.imshow(img); plt.axis("off"); plt.show()           # show it with no axes
print("You cannot audit a bias across groups the data never records.")  # the one-sentence lesson
"""),
        md("""
This is the real lesson, tied straight to the literature. Datasheets-for-Datasets [4] and radiology-AI
reviews [5] say a dataset *should* document who is in it. Real audits [6][7] show hidden bias appears
exactly along age / sex / race / scanner lines — *and can only be seen if those labels exist*. Our
detector might be much worse for some group of patients, and **we would have no way to know.** That
silence is not a small caveat; for a tool that would touch patients, it is disqualifying on its own.
"""),
        md("""
## 8. Honest limits

- **It can** show a small network learns to flag stroke on brain CT, and it can be evaluated honestly
  (sensitivity vs. specificity, confusion, Grad-CAM, noise) — the same audits real tools get [1][3].
- **It cannot** match a deployed, regulated triage system trained on hundreds of thousands of scans [1][2].
- **It cannot** be trusted to read anatomy rather than shortcuts — the Grad-CAM heat drifts to the edges [7].
- **It cannot** be checked for fairness across age, sex, race, or scanner, because the dataset records
  none of them [4][5][6]. That gap is the headline, not the footnote.

**Bottom line:** a model that runs is easy; a model you can *audit* is the real bar — and this dataset
makes the most important audit impossible. Knowing that, and being able to say it, is the finding.

## References
[1] Chilamkurthy et al. 2018, *The Lancet* · [2] Flanders et al. 2020, *Radiology: AI* (RSNA challenge) ·
[3] Seyam et al. 2022, *Radiology: AI* · [4] Gebru et al. 2021, *Communications of the ACM* (Datasheets) ·
[5] Tripathi et al. 2023, *J Am Coll Radiol* · [6] Seyyed-Kalantari et al. 2021, *Nature Medicine* ·
[7] Yang et al. 2024, *Nature Medicine*.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
