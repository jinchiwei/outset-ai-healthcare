"""Generate the G2 skin-cancer SCREENING solution notebook (the worked answer key).

HS-accessible, and EVERY line of code is commented. It runs top to bottom: it loads HAM10000
(in its tidy DermaMNIST form, which downloads in seconds) and LOOKS at it to expose the class
imbalance that makes plain accuracy a trap; it trains a quick transfer-learning smoke test to
prove the pipeline runs; then it loads the canonical results.json + the saved audit figures to
tell the honest, WORKING screening story: CAFormer beats ResNet18, AUC 0.885, and -- the fix that
matters for screening -- tuning the decision threshold lifts melanoma recall from 74% to 90%.

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
# G2 SOLUTION -- Screening skin spots for melanoma

**This is the worked answer key.** It runs top to bottom and shows the whole project the way the
five-part rubric asks for it: the **background** (why this matters), the **data** (why HAM10000),
the **model** (which backbone, and exactly how it is built), the **results** (how we measured, the
working number, a fix, fairness, and where the model looks), and an honest **conclusion**.

Our job is **screening**: look at a photo of a skin spot and decide whether it could be **melanoma**,
the deadliest skin cancer. The priority is **sensitivity** -- it is far worse to *miss* a real
melanoma than to raise a false alarm. Keep that idea in your head; it drives every choice below.

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        # ---------------------------------------------------------------- 1. BACKGROUND
        md("""
## 1. Background: what are we building, and why?

Real dermatology-AI systems look at a photo of a skin spot and estimate how likely it is to be skin
cancer, especially **melanoma**. Landmark studies showed neural networks can match or even beat
dermatologists: Esteva et al. trained one network on ~129,000 photos and it matched 21 board-certified
dermatologists [1]; Haenssle et al. found a network detected melanoma as well as or better than most
of 58 dermatologists [3]; and a large international study found the best algorithms out-classified
human experts on average [4].

**The key idea for a screening tool:** plain accuracy is a trap. What matters is **sensitivity** --
the share of true melanomas the model catches -- because a missed cancer can be fatal while a false
alarm just means a follow-up visit [6]. So a screening tool deliberately favors sensitivity. That
single idea is what separates a real screen from a model that just chases a high accuracy score.
"""),
        # ---------------------------------------------------------------- 2. DATA
        md("""
## 2. The data: HAM10000, and look at it first

We use **HAM10000**, a public set of ~10,000 real **dermatoscopy** images across 7 lesion types,
including melanoma [2]. It is the right dataset for three reasons: the images are *real* clinical
dermatoscopy (not cartoons), it is *big enough* to train on, and every image carries **age and sex**,
so we can later check the model is **fair** across groups.

The golden rule of the whole course: **load the data and actually look at it before you model.** The
tidy, ready-to-download form of HAM10000 is called **DermaMNIST**, so we use that here to see the data
in seconds. The first thing to check is how *balanced* the classes are, because that decides whether
"accuracy" even means anything.
"""),
        code("""
# Load the course helper that knows how to download DermaMNIST (the tidy form of HAM10000) into loaders.
import sys                                              # sys lets us tell Python where to find helper files
sys.path.insert(0, "../../notebooks/day3_capstone")    # add the capstone helpers folder to the search path
import capstone_common as cc                            # import them under the short name "cc"
import numpy as np                                      # numpy for fast array math

# get_loaders downloads DermaMNIST once (tiny) and returns three data streams + how many classes there are.
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders("dermamnist", size=64, batch_size=128)
names = cc.class_names("dermamnist")                    # the human-readable name of each of the 7 lesion types
print("classes:", n_classes, " task:", task)           # 7 lesion types, multi-class image classification
for i, nm in enumerate(names):                          # walk through each class index...
    print(f"  {i}: {nm}")                               # ...and print its full medical name
"""),
        code("""
# Count how many TEST images are melanoma vs everything else -- the melanoma-vs-rest split we actually screen for.
mel = [i for i, nm in enumerate(names) if nm == "melanoma"][0]   # find WHICH class index is melanoma
counts = np.zeros(n_classes, dtype=int)                 # one running tally per class, all starting at 0
for _, yb in test_loader:                               # walk the test set one batch at a time (ignore the images)
    for c in cc._targets(yb).numpy():                   # _targets flattens the label shape; loop over each label
        counts[c] += 1                                  # add one to that class's tally
total = counts.sum()                                    # total number of test images
mel_frac = counts[mel] / total                          # melanoma's share of the test set
print(f"melanoma images : {counts[mel]:5d}   ({mel_frac:4.1%} of the test set)")   # the rare, deadly class
print(f"everything else : {total - counts[mel]:5d}   ({1 - mel_frac:4.1%})")        # the common, mostly-harmless rest
print(f"\\nMelanoma -- the one we must not miss -- is only about 1 in 9 spots.")     # the imbalance in a sentence
"""),
        md("""
About **1 in 9** spots is melanoma; the rest are mostly harmless. That imbalance is the whole story,
because it means a lazy model can score high on accuracy while being useless. Let's prove that next.
"""),
        # ---------------------------------------------------------------- accuracy trap
        md("""
### Why accuracy is a trap (the lazy baseline)

Before training anything, imagine the laziest possible "model": one that ignores the image and always
guesses "not melanoma." On imbalanced data it scores high on accuracy -- and catches **zero**
melanomas. This is exactly why screening tools are judged on sensitivity, not accuracy [6].
"""),
        code("""
# Score a "model" that always says "not melanoma," no matter what the image shows.
lazy_accuracy = 1 - mel_frac                            # it is "right" every time the spot really ISN'T melanoma
lazy_melanoma_recall = 0.0                              # it never says "melanoma," so it catches 0% of melanomas
print(f"Lazy 'always say NOT melanoma' model:")         # describe the lazy model
print(f"  overall accuracy   = {lazy_accuracy:.1%}   <- looks impressive!")           # high accuracy...
print(f"  melanoma recall    = {lazy_melanoma_recall:.0%}   <- but catches ZERO cancers")  # ...and useless
print("\\nLesson: on imbalanced data, high accuracy can hide a model that misses every case that matters.")
"""),
        md("""
**Melanoma recall** (also called *sensitivity*) is the share of true melanomas the model actually
catches. The lazy model's accuracy is ~89% and its recall is 0 -- a perfect illustration of why, for
screening, a missed cancer (a false negative) matters far more than a false alarm [6]. Every result
below therefore watches recall, not just accuracy.
"""),
        # ---------------------------------------------------------------- 3. MODEL
        md("""
## 3. The model: transfer learning, and which backbone

A blank neural network knows nothing about edges, colors, or textures, and ~10,000 images is not
enough to teach it from scratch. So we **borrow a brain**: start from a network already trained on
millions of everyday photos, then reuse those features for skin [5]. Below we run that pipeline for
one quick epoch just to prove it trains end to end and gives a real number.
"""),
        code("""
# Build a transfer-learning model and train just its new head for ONE epoch -- a quick pipeline smoke test.
import torch                                            # torch is the deep-learning engine
device = "cuda" if torch.cuda.is_available() else "cpu" # use a GPU if one is available, else the CPU
demo = cc.make_model(n_classes, backbone="resnet18",   # a ResNet18 backbone...
                     pretrained=True,                  # ...starting from weights learned on everyday photos (transfer!)
                     unfreeze_backbone=False)           # freeze the backbone; train only the fresh head
demo = cc.train(demo, train_loader, val_loader,        # train it, checking accuracy on the validation set
                epochs=1, lr=1e-3, device=device)      # just 1 epoch -- enough to show the pipeline works
print("\\nThe pipeline runs end to end and produces a real number, because it reused a pretrained brain.")
print("The FULL experiment (run_experiment.py) trains a stronger backbone at full 224px resolution -- next.")
"""),
        md("""
That runs and lands in a believable range within seconds. The full experiment goes further: it trains
on the **full-resolution 224-pixel color** images (where the melanoma color and texture actually live)
and frames the task as **melanoma vs. rest**. It also asks a sharper question -- *which backbone?* --
and saves the exact numbers and figures we analyze from here on.
"""),
        code("""
# Load the canonical results the full experiment saved, so every number here matches the slides exactly.
import json                                             # json reads the saved results file
R = json.load(open("results.json"))                     # load the numbers the full 224px experiment saved
print("Backbone bake-off (higher AUC wins):")           # WHY we picked the model we picked
for name, auc in R["backbone_auc"].items():             # walk the two backbones we compared...
    print(f"  {name:<10s} AUC = {auc:.3f}")             # ...and print each one's AUC on the held-out test set
print(f"\\nWinner: CAFormer (AUC {R['auc']:.3f}) -- a stronger backbone, so it becomes our screen.")
"""),
        md("""
**CAFormer beats ResNet18 (AUC 0.88 vs 0.83)**, so we keep CAFormer. This is the "which tool?" decision
made honestly -- by comparing on held-out data, not by guessing.

![Why CAFormer: a stronger backbone wins](figures/backbone_choice.png)
"""),
        md("""
### Model & data processing (the exact recipe)

So the whole thing is reproducible, here is the full pipeline in one place:

- **Model:** `caformer_s18`, pretrained, **backbone frozen**, a fresh 2-class head trained on top
  (transfer learning) [5].
- **Images:** every dermatoscopy image decoded to color and **resized to 224x224**, then
  **normalized** to a fixed mean/scale so the network sees inputs in the range it was trained on.
- **Task:** **melanoma vs. rest** (the 7 lesion types collapsed to one yes/no screening question).
- **Imbalance:** the loss is **class-weighted** to up-weight the rare melanomas, so the model can't
  win by ignoring them.
- **Split:** separate **train / validation / test** sets; every number we report is on the held-out
  **test** set the model never trained on.
"""),
        # ---------------------------------------------------------------- 4. RESULTS
        md("""
## 4. Results: how we measured, and what we found

**How we measure a screen: ROC and AUC.** A screen outputs a *score*, not a hard yes/no. The **ROC
curve** sweeps every possible cut-off and plots melanomas caught (sensitivity) against false alarms.
**AUC** is the area under that curve: 0.5 is a coin flip, 1.0 is perfect. We use AUC, not accuracy,
because accuracy is fooled by the imbalance we just saw.
"""),
        code("""
# Print the headline screening numbers the full experiment saved.
print(f"test images        : {R['n_test']}")                       # how many held-out spots we graded on
print(f"melanoma fraction  : {R['melanoma_frac']:.1%}")            # the same ~1-in-9 imbalance, on the test set
print(f"AUC (CAFormer)     : {R['auc']:.3f}   <- a working screen (well above 0.8)")   # the headline number
"""),
        md("""
**AUC 0.885** -- a genuinely working screen. The ROC curve below hugs the top-left corner: at a good
setting it catches most melanomas while keeping false alarms modest.

![A working melanoma screen, AUC 0.88](figures/roc.png)
"""),
        # ---------------------------------------------------------------- the fix
        md("""
### The fix that matters: tune the threshold to miss fewer cancers

A model's default cut-off (score >= 0.5) is tuned for *accuracy*, not for *catching cancers*. For a
screen we **lower the threshold**: flag a spot as "check this" at a lower score, so we catch more
melanomas. The price is more false alarms -- and for screening, that is the right trade [6].
"""),
        code("""
# Compare the default cut-off against the screening cut-off the experiment tuned, using the saved numbers.
print(f"DEFAULT cut-off (score >= 0.5):")                                  # the out-of-the-box setting
print(f"  melanomas caught (recall)      = {R['recall_default']:.0%}")     # sensitivity at the default cut-off
print(f"  healthy cleared (specificity)  = {R['specificity_default']:.0%}")# specificity at the default cut-off
print(f"\\nSCREENING cut-off (score >= {R['tuned_threshold']:.2f}):")       # the lowered, sensitivity-first setting
print(f"  melanomas caught (recall)      = {R['recall_tuned']:.0%}   <- up from {R['recall_default']:.0%}")
print(f"  healthy cleared (specificity)  = {R['specificity_tuned']:.0%}   <- the cost we accept")
print(f"\\nRecall {R['recall_default']:.0%} -> {R['recall_tuned']:.0%}: we now miss far fewer melanomas.")
"""),
        md("""
Recall climbs from **74% to 90%** -- we go from missing 1 in 4 melanomas to missing 1 in 10. Specificity
drops from 85% to 70%, meaning more healthy spots get a second look. That is exactly the trade a
screening tool should make: a few extra check-ups to avoid a missed cancer.

![Tuning the screen to miss fewer cancers](figures/recall_tuning.png)
"""),
        code("""
# Rebuild the confusion matrix at the screening cut-off from the saved counts, to see the mistakes directly.
import pandas as pd                                      # pandas reads the saved table of counts
M = pd.read_csv("figures/raw/confusion.csv").values     # a 2x2 grid: rows = truth, columns = what the screen said
tn, fp = M[0, 0], M[0, 1]                                # truly-not-melanoma: cleared (tn) vs false-alarmed (fp)
fn, tp = M[1, 0], M[1, 1]                                # truly-melanoma: missed (fn) vs caught (tp)
print(f"melanomas CAUGHT  : {tp}")                       # true positives -- the wins
print(f"melanomas MISSED  : {fn}    <- each one is a real danger")   # false negatives -- the ones that matter most
print(f"false alarms      : {fp}    <- each one is just an extra check-up")  # false positives -- the acceptable cost
print(f"recall = {tp} / ({tp} + {fn}) = {tp / (tp + fn):.0%}")       # confirm the 90% recall from the raw counts
"""),
        md("""
At the screening setting the model **catches 130 of 144 melanomas** and misses only 14, at the price of
340 false alarms out of ~1,140 healthy spots. Those are the numbers a clinician would weigh.

![Confusion matrix at the screening setting](figures/confusion.png)
"""),
        # ---------------------------------------------------------------- fairness
        md("""
### Fairness: does the screen work as well for everyone?

A screen that works only for some patients is not fair to the rest. Because HAM10000 records **sex**,
we can measure the AUC **separately for men and women** and check the gap.
"""),
        code("""
# Print the AUC computed separately for each sex, and the gap between them.
fair = R["auc_by_sex"]                                   # {'male': AUC, 'female': AUC} from the full run
for g, auc in fair.items():                              # walk each group...
    print(f"  {g:<7s} AUC = {auc:.3f}")                  # ...and print how well the screen ranks its spots
gap = abs(fair["male"] - fair["female"])                 # the size of the fairness gap
print(f"\\ngap = {gap:.3f}  ->  the screen is a bit weaker for women; a real tool would investigate and close this.")
"""),
        md("""
The screen works for both groups (both AUCs clear 0.8), but it is **stronger for men (0.91) than women
(0.85)** -- a real gap worth naming out loud. In a deployed tool you would dig into why (skin tone,
lesion location, sampling) and work to close it.

![Fairness check: works for women and men](figures/fairness_by_sex.png)
"""),
        # ---------------------------------------------------------------- Grad-CAM
        md("""
### Feature importance for images: Grad-CAM

For a table of numbers we would ask SHAP "which columns mattered?" For an image, the equivalent is
**Grad-CAM**: it highlights *where* on the picture the model looked to make its call. We want it
looking at the lesion, not at a ruler mark, a hair, or the corner of the frame.

![Where the screen looks on real melanomas](figures/gradcam.png)

The bright areas sit **on the pigmented lesion** -- reassuring evidence the screen is reading the spot
itself, the same cues a dermatologist uses, and not some artifact of the photo.
"""),
        # ---------------------------------------------------------------- 5. CONCLUSION
        md("""
## 5. Conclusion: honest limits

- **It works** -- a real melanoma screen on real dermatoscopy, **AUC 0.885**, tuned to catch **90%** of
  melanomas, and it looks at the lesion for the right reasons.
- **But** the melanoma-vs-women gap (0.85 vs 0.91) is real, and HAM10000 skews toward lighter skin, so
  performance on darker skin is unproven here [7].
- **A real tool** would train on more diverse patients, be validated across many hospitals, and keep a
  dermatologist in the loop before it is trusted near a patient [7].

**Bottom line:** the winning habit is the screening mindset -- never miss the dangerous case, tune the
threshold on purpose, and check fairness -- not chasing a shiny accuracy number.

## References
[1] Esteva et al. 2017, *Nature* -- dermatologist-level skin-cancer classification.
[2] Tschandl, Rosendahl & Kittler 2018, *Sci Data* -- the HAM10000 dataset.
[3] Haenssle et al. 2018, *Ann Oncol* -- CNN vs 58 dermatologists on melanoma.
[4] Tschandl et al. 2019, *Lancet Oncol* -- machines vs human readers, international study.
[5] Kim et al. 2022, *BMC Med Imaging* -- transfer learning for medical images.
[6] Trevethan 2017, *Front Public Health* -- sensitivity, specificity, and their pitfalls.
[7] Wei et al. 2024, *Front Med* -- AI and skin cancer: how well it works and the gaps.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
