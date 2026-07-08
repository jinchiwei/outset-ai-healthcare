"""Generate the G2 skin-cancer SCREENING solution notebook (the worked answer key).

HS-accessible, EVERY line of code commented. It runs top to bottom: it loads DermaMNIST live and
shows the class imbalance that makes plain accuracy a trap, trains a quick transfer-learning baseline
to prove the pipeline runs, then loads the canonical results.json + the four saved audit figures to
tell the honest screening story (the number that matters is melanoma recall, not overall accuracy).

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

**This is the worked answer key.** It runs top to bottom and shows the whole project: the medicine,
the one big idea, the code (every line commented), the result, and an honest look at the limits.

Our job is **screening**: look at a photo of a skin spot and decide whether it could be melanoma --
the deadliest skin cancer. The priority is **sensitivity**: it is far worse to *miss* a real melanoma
than to raise a false alarm. Keep that idea in your head; it drives every choice below.

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        md("""
## 1. Background: what are we building?

Real dermatology-AI systems look at a photo of a skin spot and estimate how likely it is to be skin
cancer, especially **melanoma**. Landmark studies showed that neural networks can match or even beat
dermatologists at this: Esteva et al. trained one network on ~129,000 photos and it matched 21
board-certified dermatologists [1]; Haenssle et al. found a network detected melanoma as well as or
better than most of 58 dermatologists [3]; and a large international study found the best algorithms
out-classified human experts on average [4].

Those models are trained on public image sets. Ours uses **DermaMNIST**, a tidy 7-class version of the
**HAM10000** dataset of ~10,000 dermatoscopy images [2]. And they lean on **transfer learning**:
reuse a network already trained on millions of everyday photos, then fine-tune it on the skin images,
so it works even with limited medical data [5].

**The key idea for a screening tool:** plain accuracy is a trap. What matters is **sensitivity** --
the share of true melanomas the model catches -- because a missed cancer can be fatal while a false
alarm just means a follow-up visit [6]. Our toy 64-pixel model can show this trade-off and the basic
pipeline, but it is far too small to trust on a real patient [7]. Let's see all of that in code.
"""),
        md("""
## 2. Load the data -- and look at it first

The golden rule of the whole course: **load the data and actually look at it before you model.** For a
screening problem, the first thing to check is how *balanced* the classes are, because that decides
whether "accuracy" even means anything.
"""),
        code("""
# Load the course helper that knows how to download DermaMNIST and build train/val/test loaders.
import sys                                              # sys lets us tell Python where to find helper files
sys.path.insert(0, "../../notebooks/day3_capstone")    # add the capstone helpers folder to the search path
import capstone_common as cc                            # import them under the short name "cc"
import numpy as np                                      # numpy for fast array math

# get_loaders downloads DermaMNIST once (tiny) and returns three data streams + how many classes there are.
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders("dermamnist", size=64, batch_size=128)
names = cc.class_names("dermamnist")                    # the human-readable name of each of the 7 classes
print("classes:", n_classes, " task:", task)           # 7 classes, multi-class image classification
for i, nm in enumerate(names):                          # walk through each class index...
    print(f"  {i}: {nm}")                               # ...and print its full medical name
"""),
        code("""
# Count how many test images belong to each class, so we can see how (im)balanced the data is.
counts = np.zeros(n_classes, dtype=int)                 # one running tally per class, all starting at 0
for _, yb in test_loader:                               # walk through the test set one batch at a time (ignore images)
    for c in cc._targets(yb).numpy():                   # _targets flattens the label shape; loop over each label
        counts[c] += 1                                  # add one to that class's tally
total = counts.sum()                                    # total number of test images
mel = names.index("melanoma")                           # find WHICH class index is melanoma (the one we care about)
for i, nm in enumerate(names):                          # print the breakdown, class by class...
    print(f"  {nm:<38s} {counts[i]:5d}   ({counts[i]/total:5.1%})")   # count and share of the test set
print(f"\\nBiggest class is '{names[counts.argmax()]}' at {counts.max()/total:.0%} of all images.")  # the imbalance
print(f"Melanoma (the one we must not miss) is only {counts[mel]/total:.1%} of the test set.")        # the rare, deadly class
"""),
        md("""
Look at that: one class, **melanocytic nevi** (harmless moles), is about two-thirds of the data, while
**melanoma** is a small slice. This imbalance is the whole story. It means a lazy model can score high
on accuracy while being useless -- let's prove that next.
"""),
        md("""
## 3. Why accuracy is a trap (the lazy baseline)

Before training anything, imagine the laziest possible "model": one that ignores the image and always
guesses the single most common class. On imbalanced data it scores shockingly high on accuracy -- and
catches **zero** melanomas. This is exactly why screening tools are judged on sensitivity, not accuracy.
"""),
        code("""
# Score a "model" that always predicts the biggest class, no matter what the image shows.
majority = counts.argmax()                              # the index of the most common class (harmless moles)
lazy_accuracy = counts[majority] / total               # it is "right" whenever the true class IS that class
lazy_melanoma_recall = 0.0                              # it never says "melanoma", so it catches 0% of melanomas
print(f"Lazy 'always guess {names[majority]}' model:")  # describe the lazy model
print(f"  overall accuracy   = {lazy_accuracy:.1%}   <- looks impressive!")   # high accuracy...
print(f"  melanoma recall    = {lazy_melanoma_recall:.0%}   <- but catches ZERO cancers")  # ...and totally useless
print("\\nLesson: on imbalanced data, high accuracy can hide a model that misses every case that matters.")
"""),
        md("""
**Melanoma recall** (also called *sensitivity*) is the share of true melanomas the model actually
catches. The lazy model's accuracy is high but its recall is 0 -- a perfect illustration of why, for
screening, a missed cancer (a false negative) matters far more than a false alarm [6]. Our whole
evaluation below will therefore watch melanoma recall, not just accuracy.
"""),
        md("""
## 4. The one idea: transfer learning

A blank neural network knows nothing about edges, colors, or textures, and we only have ~10,000 skin
images -- not enough to teach it from scratch. So we **borrow a brain**: start from a ResNet18 already
trained on millions of everyday photos, freeze what it learned, and train only a small new "head" to
map its features onto our 7 skin classes [5]. Let's run that pipeline for a couple of quick epochs to
prove it trains end to end and produces a real number.
"""),
        code("""
# Build a transfer-learning model and train just its new head for a couple of epochs (a quick demo run).
import torch                                            # torch is the deep-learning engine
device = "cuda" if torch.cuda.is_available() else "cpu" # use a GPU if one is available, else the CPU
model = cc.make_model(n_classes, backbone="resnet18",  # a ResNet18 backbone...
                      pretrained=True,                 # ...starting from weights learned on everyday photos (transfer!)
                      unfreeze_backbone=False)          # freeze the backbone; train only the fresh 7-class head
model = cc.train(model, train_loader, val_loader,      # train it, checking accuracy on the validation set each epoch
                 epochs=2, lr=1e-3, device=device)     # just 2 epochs -- enough to show the pipeline works
print("\\nThe pipeline runs end to end and produces a real validation number.")   # confirm success
print("Your exact number will wobble run to run; the full, carefully-trained results load next.")
"""),
        md("""
That runs, learns, and lands in a believable range within seconds -- because it reused a pretrained
brain instead of starting blank. The full experiment (`run_experiment.py`) trains three ways for more
epochs and saves the exact numbers and figures we analyze below, so our story stays consistent no
matter how this quick demo happens to wobble.
"""),
        md("""
## 5. Result #1 -- a better starting point beats a fancier model

The full experiment trained the network three ways: **from scratch** (a blank brain), **pretrained**
(borrowed eyes, transfer learning), and **pretrained + augmentation** (extra tricks). Here are the real
saved accuracies.
"""),
        code("""
# Load the canonical results the full experiment saved, so every number here matches the slides.
import json                                             # json reads the saved results file
results = json.load(open("results.json"))               # load the numbers the full experiment saved
print("from scratch (blank brain)    :", f"{results['scratch_acc']:.3f}")     # trained from zero
print("pretrained (borrowed eyes)    :", f"{results['baseline_acc']:.3f}")    # transfer learning
print("pretrained + augmentation     :", f"{results['improved_acc']:.3f}")    # transfer + extra data tricks
"""),
        md("""
The three numbers are almost the same. That is the honest, slightly humbling lesson: on this tiny
dataset, *which* model or trick you pick barely moves the needle -- the ceiling is set by the data, not
the cleverness. The picture below shows it at a glance.

![A better starting point beats a fancier model](figures/data_beats_model.png)
"""),
        md("""
## 6. Result #2 -- the confusion matrix tells the real story

Accuracy is one number; a **confusion matrix** shows *which* mistakes the model makes. Rows are the
truth, columns are the model's guess; the diagonal is correct, everything off it is an error. Watch the
melanoma row.

![Confusion matrix](figures/confusion.png)
"""),
        code("""
# Rebuild the confusion matrix from the saved raw numbers and compute the ONE screening metric that matters.
import pandas as pd                                      # pandas reads the saved table of counts
M = pd.read_csv("figures/raw/confusion.csv").values      # a 7x7 grid: row = true class, column = predicted class
mel = names.index("melanoma")                            # the melanoma row/column index again
true_melanomas = M[mel].sum()                            # total real melanomas in the test set (the whole melanoma row)
caught = M[mel, mel]                                     # melanomas the model correctly called melanoma (the diagonal cell)
melanoma_recall = caught / true_melanomas                # sensitivity = share of real melanomas we caught
print(f"real melanomas in the test set : {true_melanomas}")          # how many there were
print(f"correctly caught by the model  : {caught}")                  # how many we found
print(f"MELANOMA RECALL (sensitivity)  : {melanoma_recall:.1%}")     # the headline screening number
print(f"missed melanomas               : {true_melanomas - caught}   <- each one is a real danger")
"""),
        md("""
There it is, in the open: the model caught only a tiny fraction of real melanomas and sent almost all
of them into the "harmless mole" column. That is the single most important sentence in this whole
project -- **a 68% accurate model can still miss almost every melanoma** -- and it is exactly the
false-negative danger the background warned about [6].
"""),
        md("""
## 7. Result #3 -- fairness: screening lives or dies on the rare classes

Overall accuracy averages over classes and hides the weak ones. A per-class view (melanoma highlighted
in pink) shows the model is basically only good at the one common class and quietly fails the rare,
dangerous ones -- a fairness problem you would never accept in a real screening tool.

![Per-class accuracy, melanoma highlighted](figures/per_class_fairness.png)
"""),
        code("""
# Print the per-class accuracy the experiment saved, sorted worst-first, to name the classes the model fails.
per_class = results["per_class_acc"]                     # a dict of {class name -> accuracy} from the full run
for nm, acc in sorted(per_class.items(), key=lambda kv: kv[1]):   # sort from worst class to best
    flag = "   <- MELANOMA" if "melanoma" == nm[:8] and "melanocyti" not in nm else ""  # mark the melanoma row
    print(f"  {nm:<12s} {acc:6.1%}{flag}")               # class name, its accuracy, and the melanoma flag
print("\\nOne class scores near-perfect; the rest -- including melanoma -- are near the floor.")
"""),
        md("""
## 8. Result #4 -- look at the failures

The most interesting slide in any talk is the model's mistakes. Here are cases it got wrong: notice how
many true melanomas it waved through as harmless moles. A dermatologist would want a second look at
every one of these.

![The cases it got wrong](figures/failures.png)
"""),
        md("""
## 9. Honest limits

- **It can** show the real screening trade-off -- that accuracy is a trap and sensitivity is the metric
  that matters -- on the same kind of data the professionals use [2].
- **It cannot** be trusted on a patient: 64x64 grayscale thumbnails throw away the color and fine
  texture that dermatologists (and real systems [1][3]) rely on, and our melanoma recall is far too low.
- **A real tool** would deliberately tune its decision threshold toward high sensitivity, accept more
  false alarms, train on full-resolution color images, and be validated on many hospitals before anyone
  trusts it near a patient [7].

**Bottom line:** this project is how you *learn* the screening mindset -- never miss the dangerous case,
and never let a shiny accuracy number fool you. Use a validated tool and a real dermatologist to
actually screen a patient.

## References
[1] Esteva et al. 2017, *Nature* -- dermatologist-level skin-cancer classification ·
[2] Tschandl, Rosendahl & Kittler 2018, *Sci Data* -- the HAM10000 dataset (source of DermaMNIST) ·
[3] Haenssle et al. 2018, *Ann Oncol* -- CNN vs 58 dermatologists on melanoma ·
[4] Tschandl et al. 2019, *Lancet Oncol* -- machines vs human readers, international study ·
[5] Kim et al. 2022, *BMC Med Imaging* -- transfer learning for medical images ·
[6] Trevethan 2017, *Front Public Health* -- sensitivity, specificity, and their pitfalls ·
[7] Wei et al. 2024, *Front Med* -- AI and skin cancer: how well it works and the gaps.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
