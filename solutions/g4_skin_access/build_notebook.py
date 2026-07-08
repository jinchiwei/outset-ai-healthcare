"""Generate the G4 skin-cancer AI + ACCESS/EQUITY SOLUTION notebook (the worked answer key).

Same DermaMNIST model as the screening group, but the whole point is the EQUITY lens: a model that
is "accurate on average" can still fail a whole condition -- and the dataset hides who it was trained
to see. Runs top to bottom (trains a small transfer model on GPU/CPU), reproduces the real results
(overall ~0.68 accuracy but a ~0.99 per-class equity gap), and EVERY code line is commented.

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
# G4 SOLUTION -- Who does skin-cancer AI help, and who is missing from the data?

**This is the worked answer key for the ACCESS / EQUITY project.** It runs top to bottom and shows
the whole investigation: the promise, the catch, the code (every line commented), the result, and an
honest look at who gets left out.

We use the **exact same DermaMNIST skin-lesion model** as the screening group. The difference is the
*question we ask of it*. The screening group asks "how good is it?" We ask a harder question:
**good for whom?** A model can look great on average and still quietly fail an entire group of people.

You do **not** need to memorize the Python. You *do* need to be able to say, in plain English, what
each cell is for. Anything unclear -- highlight the line and ask Claude "explain this."
"""),
        md("""
## 1. Background: a promise, and a catch

**The promise.** There are roughly 3 billion people in the world with little or no access to a
dermatologist [7]. Skin-cancer AI is often sold as the fix: point a phone at a mole, get an answer,
no specialist required. If it works, that is a genuinely big deal for access to care.

**The catch.** A model only learns what it is shown -- and the images these models learn from are
badly skewed. A systematic review of 21 public skin datasets (106,950 images) found that almost all
came from Europe, North America, and Oceania, with **almost no images labeled as darker skin** [5].
On the Fitzpatrick skin-type scale (I = lightest -> VI = darkest) [1], the training data piles up on
the left and nearly vanishes on the right:

![Most training images are light skin](figures/intro_dataset_skew.png)

Groh et al. built the Fitzpatrick 17k dataset and showed the consequence directly: **a model is most
accurate on the skin types it saw most during training** [2]. Daneshjou et al. then tested
state-of-the-art models on a diverse, biopsy-confirmed set and found accuracy **drops sharply on dark
skin** [3]. Same model, very different results depending on who you are -- and the single "average"
number a report shows sits in between and hides it:

![One model, very different accuracy by group](figures/intro_accuracy_by_group.png)
"""),
        md("""
## 2. The one idea: "accurate on average" can still be unfair

Here is the trap, and it is the whole lesson of this project. Obermeyer et al. studied a health
algorithm used on **millions** of patients that looked fair on paper but quietly underserved Black
patients [4]. Why? It was trained to predict **health-care cost** as a stand-in for **illness**.
Because less money was historically spent on equally-sick Black patients, the algorithm learned to
rate them as *less* sick. It scored well on its target and was unfair anyway -- because the target
was the wrong thing:

![A model can score great on the wrong target](figures/intro_proxy_gone_wrong.png)

The equity lens we bring to our own model is exactly this suspicion: **do not trust one average
number.** Break the score apart and ask who it works for and who it fails. Let's do that.
"""),
        code("""
# Tell Python where the course's shared helpers live, then import the ones we need.
import sys                                                   # sys lets us add folders to the import path
sys.path.insert(0, "../../notebooks/day3_capstone")          # <- the folder with the capstone helpers
import capstone_common as cc                                 # loaders, model factory, train/evaluate -- all here
import torch                                                 # the deep-learning library that runs the model
import numpy as np                                           # for array math
import matplotlib.pyplot as plt                              # for our own charts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"      # use the GPU if there is one, else the CPU
torch.manual_seed(0)                                         # fix the randomness so results are reproducible
FLAG = "dermamnist"                                          # the MedMNIST skin-lesion dataset (7 conditions)

names = cc.class_names(FLAG)                                 # the full medical name of each of the 7 conditions
short = [n.split()[0].rstrip(",")[:10] for n in names]       # a short nickname for each, for chart labels
print("device:", DEVICE)                                     # confirm GPU vs CPU
for i, n in enumerate(names):                                # list every condition the model must tell apart
    print(f"  class {i}: {short[i]:<11} = {n}")              # short nickname = full name
"""),
        md("""
The seven conditions are not equally common. In real skin clinics -- and in this dataset -- one class,
**melanocytic nevi** (ordinary moles), massively outnumbers the rest, while dangerous-but-rarer
conditions like **actinic keratoses** and **dermatofibroma** show up only a little. Let's *look* at
that imbalance before we train anything, because it is going to drive everything that follows.
"""),
        code("""
# Load DermaMNIST as train / validation / test sets (downloads once, then reads from cache).
train_loader, val_loader, test_loader, n_classes, task = cc.get_loaders(  # get three batched data streams...
    FLAG, size=64, augment=True)                             # ...at 64x64 pixels, with light augmentation on train

# Count how many TRAINING images the model gets to see of each condition.
train_labels = np.concatenate(                              # glue every batch's labels into one long array
    [cc._targets(yb).numpy() for _, yb in train_loader])    # (_targets flattens MedMNIST's (B,1) label shape)
counts = [int((train_labels == c).sum()) for c in range(n_classes)]  # tally images per class

plt.figure(figsize=(8, 3.2))                                # set the chart size
plt.bar(short, counts, color="#40E0D0", edgecolor="#14141C")  # one turquoise bar per condition
plt.ylabel("training images"); plt.title("The model sees some conditions FAR more than others")  # labels
plt.xticks(rotation=45, ha="right")                        # tilt the condition names so they fit
plt.tight_layout(); plt.show()                             # draw it
print("Most common:", short[int(np.argmax(counts))], "=", max(counts), "images")   # the giant class
print("Rarest:     ", short[int(np.argmin(counts))], "=", min(counts), "images")   # the tiny class
"""),
        md("""
One bar towers over the rest. Keep that picture in your head: **whatever the model sees most, it will
get best at.** That is not a bug we introduced -- it is the same skew Groh and Daneshjou describe for
skin tone [2][3], just along the axis DermaMNIST actually labels (the condition).

## 3. Train the exact same model the screening group uses

Nothing fancy and nothing different here -- we deliberately use the standard transfer-learning recipe
(a ResNet18 pretrained on ordinary photos, with a fresh classifier head) so that any gap we find is
about **the data and the question**, not a weak model.
"""),
        code("""
# Build the standard transfer-learning model: an ImageNet-pretrained ResNet18 with a new 7-way head.
model = cc.make_model(n_classes, backbone="resnet18", pretrained=True)   # frozen backbone + fresh head
model = cc.train(model, train_loader, val_loader,          # train it, printing validation accuracy each epoch
                 epochs=3, lr=1e-3, device=DEVICE)         # 3 quick passes is enough to make the point
"""),
        md("""
## 4. Result #1 -- the headline number looks fine

Now we grade the model on the held-out test set (images it never trained on). We get a single overall
accuracy. On its own, this is the number that would go on a slide and make everyone nod.
"""),
        code("""
# Run the trained model on the test set and read off the overall accuracy.
res = cc.evaluate(model, test_loader, device=DEVICE)       # returns overall accuracy + the raw answers
y = res["y"]                                                # y  = the TRUE condition for each test image
pred = res["pred"]                                          # pred = the model's GUESS for each test image
overall = res["accuracy"]                                   # the single headline accuracy number
print(f"Overall test accuracy: {overall:.1%}")             # looks respectable... but for WHOM?
"""),
        md("""
Around **68%**. Not amazing, not embarrassing -- the kind of number you would quietly accept and move
on. **This is exactly the moment the equity lens says: stop and break it apart.** An average is a blend;
a blend can hide that one group is carried by another. Who is carrying this 68%?
"""),
        code("""
# Compute accuracy SEPARATELY for each of the 7 conditions -- the equity audit.
per_class = {}                                             # will map condition -> its own accuracy
for c in range(n_classes):                                 # go through each condition one at a time
    mask = (y == c)                                        # pick out just the test images that really are condition c
    if mask.sum() > 0:                                     # (guard against a condition with zero test images)
        per_class[short[c]] = float((pred[mask] == c).mean())   # of those, what fraction did the model get right?

order = np.argsort(list(per_class.values()))              # sort conditions from worst-caught to best-caught
labels = np.array(list(per_class.keys()))[order]          # sorted condition names
accs = np.array(list(per_class.values()))[order]          # sorted per-condition accuracies

plt.figure(figsize=(8, 3.4))                              # set the chart size
colors = ["#FF1493" if a < overall else "#40E0D0" for a in accs]  # pink = below the average, turquoise = above
plt.bar(labels, accs, color=colors, edgecolor="#14141C")  # one bar per condition
plt.axhline(overall, ls="--", color="#14141C", label=f"overall average {overall:.0%}")  # the average line
plt.ylabel("accuracy for THIS condition"); plt.ylim(0, 1)  # y-axis 0..1
plt.title("'Accurate overall' can still fail a whole condition"); plt.legend()  # title + legend
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()  # tilt labels, draw it

gap = max(per_class.values()) - min(per_class.values())   # the equity gap: best class minus worst class
print("Best-caught condition: ", max(per_class, key=per_class.get), f"{max(per_class.values()):.0%}")
print("Worst-caught condition:", min(per_class, key=per_class.get), f"{min(per_class.values()):.0%}")
print(f"EQUITY GAP (best - worst): {gap:.0%}")            # the headline of this whole project
"""),
        md("""
## 5. Result #2 -- the average was hiding a near-total failure

Look at what the average hid. One class -- **melanocytic nevi**, the ordinary-mole class with thousands
of training images -- is caught almost perfectly, near **99%**. It is so common that getting it right
is enough to prop the overall number up to 68%. But several rarer conditions are caught **almost never**
-- some sit flat at **0%**. The gap between the best-served and worst-served condition is about
**99 percentage points**.

That is the entire lesson in one chart. A model can be "68% accurate" and, for a patient with one of
the rare conditions, be **worse than useless** -- confidently wrong nearly every time. The average was
a comforting blend that hid a group being failed completely. This is the same shape as the Obermeyer
story [4]: a fair-looking number sitting on top of an unfair reality.
"""),
        md("""
## 6. The blind spot we CANNOT measure -- and must be honest about

Here is where we have to be careful and honest, which is itself part of doing this well.

Everything above is measured along the axis DermaMNIST actually labels: the **condition**. But the gap
the literature warns about most loudly is along **skin tone** [2][3][5]. And DermaMNIST **does not ship
skin-tone labels** -- it records the lesion type and nothing about the patient's skin tone, age, sex, or
country:

![What the dataset hides](figures/skin_tone_gap.png)

So we literally cannot check, in this dataset, whether our model works as well on dark skin as on light
skin. That does not mean the risk is gone -- it means **it is invisible to us here.** Given that public
dermatology datasets skew heavily toward light skin [5][6] and that models are most accurate on the
skin types they saw most [2][3], the honest statement is: **we can demonstrate the RISK of a blind
spot, not measure the real skin-tone gap.** Naming what you cannot see is more honest than pretending
a good average means the model is fair for everyone.
"""),
        md("""
## 7. Honest limits -- and who gets left out

- **What this project CAN show:** that one overall accuracy number can hide a whole condition being
  failed, on the same DermaMNIST model the screening group uses. The ~99-point per-class gap is real
  and reproducible.
- **What it CANNOT show:** the actual skin-tone disparity, because DermaMNIST carries no skin-tone
  labels [5][6]. We show the risk, not the measured gap.
- **Who gets left out:** the people the pitch was supposed to help most. Skin-cancer AI is aimed at
  the ~3 billion people with little dermatology access [7], who are disproportionately in regions and
  skin types barely present in the training data [5]. A tool that is worst exactly where care is
  scarcest can **widen** the gap it was sold to close [6].

**Bottom line:** before trusting a medical AI, ask *good for whom?* Break the average apart, look at
who it fails, and say out loud what the data does not let you check. A model you can question honestly
is worth more than a high number you cannot.

## References
[1] Fitzpatrick 1988, *Arch Dermatol* · [2] Groh et al. 2021, *CVPR Workshops (Fitzpatrick 17k)* ·
[3] Daneshjou et al. 2022, *Sci Adv (Diverse Dermatology Images)* · [4] Obermeyer et al. 2019, *Science* ·
[5] Wen et al. 2022, *Lancet Digital Health* · [6] Adamson & Smith 2018, *JAMA Dermatology* ·
[7] Buster, Stevens & Elmets 2012, *Dermatologic Clinics*.
"""),
    ]
    save(nb, HERE / "solution.ipynb")
    print("wrote solution.ipynb")


if __name__ == "__main__":
    build()
