"""Build the Day 2 multimodal notebooks (lab + solution) from one source.

Emits:
  notebooks/day2_multimodal/day2.ipynb           (# TODO blanks)
  notebooks/day2_multimodal/day2_solution.ipynb  (filled)

MIT 6.S191 density: numbered sub-sections, an explanation before every code cell, and a
figure wherever it helps. Live plots use notebooks/_shared/nbfig.py (Colab-safe);
concept diagrams under img/ are pre-rendered by img_gen.py with the build-figure skill.

Run:  python scripts/build_day2.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nbutil import code, code_with_todos, md, new_nb, save  # noqa: E402

PAIRS = []


def both(cell):
    PAIRS.append((cell, cell))


def todo(solution_src, blanks):
    PAIRS.append(code_with_todos(solution_src, blanks))


# =========================================================================== #
# Intro
# =========================================================================== #
both(md("""
# Day 2 -- Language models and the whole patient

Yesterday we pointed one big neural network at eye photos. Today we change two things at
once, and they're the two ideas behind most modern medical AI:

1. **Language models.** Every scan comes with a written radiology report. We'll see what an
   LLM actually does, and use one to turn that free text into numbers.
2. **Multimodal fusion.** Real clinical decisions use the scan *and* the notes *and* the
   patient history. We'll combine three different signals about one patient into a single
   prediction.

The dataset is **Open-i chest X-rays** with real radiologist reports. The task: does this
patient have **cardiomegaly** (an enlarged heart)? Each patient gives us three signals, and
each casts a *vote*:

- **image vote** -- a trained image model's probability for the X-ray (transfer learning,
  like Day 1): one number, `img_pred`. Combining per-modality predictions like this is
  called **late fusion**, or **stacking**.
- **text features** -- yes/no findings an LLM pulled out of the report.
- **demographics** -- age, sex, smoking history.

### By the end you'll be able to
- Explain what an LLM does (predict the next token) and where it goes wrong (hallucination).
- Turn three different data types into one table and let a foundation model (**TabPFN**) decide.
- Spot **target leakage** -- the trap that makes a useless model look brilliant.

Fill in the `# TODO`s. Ask Claude when stuck, then make sure you understand the answer.
"""))

both(code("""
# Setup: on Colab, grab the course files. Locally this is a no-op.
import os, sys
if not os.path.exists("common.py"):
    os.system("git clone -q https://github.com/jinchiwei/outset-ai-healthcare.git")
    os.chdir("outset-ai-healthcare/notebooks/day2_multimodal")
sys.path.insert(0, ".")
sys.path.insert(0, "../_shared")
import colab_setup
colab_setup.ensure(*colab_setup.DAY2)
"""))

both(code("""
import os, sys
import numpy as np
import pandas as pd
import common
# nbfig lives in notebooks/_shared; add it relative to common.py so this cell
# works even if the setup cell above wasn't run, and from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(common.__file__)), "..", "_shared"))
import nbfig          # Colab-safe branded plotting (matches the slide figures)
nbfig.use()
"""))

# =========================================================================== #
# Section 0 -- Meet the data
# =========================================================================== #
both(md("""
## 0. Meet the data: X-rays, reports, and a yes/no question

Before any modeling, look at the data, exactly like Day 1. Our task: does this chest X-ray
show **cardiomegaly** (an enlarged heart)? A binary call a radiologist makes every day.

### 0.1 Three real chest X-rays
These are real Open-i chest films. The heart is the bright shape in the lower-middle. In an
enlarged heart it takes up more than half the chest width. See if you can spot which is which
before you read the label.
"""))

both(code("""
import json
from PIL import Image

reports = json.loads(open("../../datasets/openi_sample_reports.json").read())
sample_ids = ["3840", "1164", "3187"]   # the three we shipped images for

fig, axes = nbfig.fig(1, 3, figsize=(11, 4.2))
for ax, cid in zip(axes, sample_ids):
    ax.imshow(Image.open(f"sample_images/{cid}.png").convert("L"), cmap="gray")
    lab = "CARDIOMEGALY" if reports[cid]["label"] else "normal heart"
    ax.set_title(lab, fontsize=12, color=(nbfig.DEEPPINK if reports[cid]["label"] else nbfig.INK),
                 family="DejaVu Sans Mono")
    ax.axis("off")
nbfig.show(fig, "Three real chest X-rays")
"""))

both(md("""
### 0.2 Every scan comes with a report
Here is what makes chest imaging different from yesterday's eye photos: each scan is paired
with a **radiologist's written report** -- a `FINDINGS` section and an `IMPRESSION`. That free
text is a second channel of data, and historically it's where most of the clinical signal was
recorded. Read one real report below.
"""))

both(code("""
cid = "3840"
rec = reports[cid]
print(f"CASE {cid}   (label: {'cardiomegaly' if rec['label'] else 'normal'})")
print("=" * 64)
print("FINDINGS:   ", rec["findings"])
print("IMPRESSION: ", rec["impression"])
print("MeSH tags:  ", rec["mesh_majors"])
"""))

both(md("""
### 0.3 Where does the label come from?
We didn't hand-label these. Each Open-i report carries **MeSH** tags -- standardized medical
keywords a librarian assigned. Our label is simply: *does `Cardiomegaly` appear in the tags?*
Notice that the impression text and the MeSH tag agree -- hold onto that, it becomes the
sharpest lesson of the day (Section 6).
"""))

both(code("""
df = pd.read_csv("../../datasets/openi_features.csv")
counts = df.label.value_counts().sort_index()

fig, ax = nbfig.fig(figsize=(5.2, 3.2))
bars = ax.bar(["normal", "cardiomegaly"], counts.values, color=[nbfig.TURQUOISE, nbfig.DEEPPINK])
for b, c in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom",
            fontweight="bold", family="DejaVu Sans Mono")
ax.set_ylabel("patients")
nbfig.show(fig, f"A balanced dataset: {len(df)} patients")
print("balanced on purpose, so accuracy is a meaningful number (50% = coin flip).")
"""))

# =========================================================================== #
# Section 1 -- What is a language model
# =========================================================================== #
both(md("""
## 1. What is a language model?

### 1.1 The bridge from yesterday
Yesterday's Vision Transformer split an image into patches and used **attention** to decide
which patches mattered. A language model does the exact same thing with **words instead of
patches**. So none of today is magic -- it's the same machinery you already built, pointed
at text. Let's start with a real radiology report.
"""))

both(code("""
report = ("FINDINGS: The heart is enlarged. There is a small left pleural effusion. "
          "No pneumothorax. IMPRESSION: Cardiomegaly with small effusion.")

# A model can't read letters; text is chopped into *tokens* (word-pieces), then numbers.
tokens = report.replace(".", " .").replace(":", " :").split()
print(f"{len(tokens)} rough tokens")

# Visualize the report as a strip of tokens -- the sequence the model actually sees.
from matplotlib.patches import FancyBboxPatch
show = tokens[:16]
fig, ax = nbfig.fig(figsize=(11, 1.7))
ax.axis("off"); ax.set_xlim(0, len(show)); ax.set_ylim(0, 1); ax.grid(False)
for i, t in enumerate(show):
    c = nbfig.palette(len(show))[i]
    ax.add_patch(FancyBboxPatch((i + 0.05, 0.25), 0.9, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=c, edgecolor="none"))
    ax.text(i + 0.5, 0.5, t, ha="center", va="center", fontsize=9,
            color=nbfig.txt_on(c), family="DejaVu Sans Mono")
nbfig.show(fig, "Text becomes a sequence of tokens")
"""))

both(md("""
### 1.2 Attention, next-token, and the catch
Once text is tokens, the **same attention** from the ViT decides which tokens carry signal
("heart", "enlarged") and which are filler ("the", "is"). And under all the hype, an LLM does
exactly one thing: **predict the next token**, over and over. That simple objective at huge
scale is enough to summarize, answer questions, and pull structured facts out of messy notes.

**The catch (important in medicine):** "predict something plausible" is not "tell the truth."
When an LLM doesn't know, it doesn't stop -- it **hallucinates** a fluent, confident, wrong
answer. So today's rule: *use the LLM, but verify what it says.*
"""))

both(md("""
### 1.3 The LLM already read every report for us
Calling an LLM on hundreds of reports costs money and time, so the instructor ran it once
(Anthropic API) and **saved** the structured findings -- which findings are present, plus a
severity word. You'll load those; no API key needed. That's the second signal handled.

The prompt was essentially: *"Read this radiology report. Return JSON: is cardiomegaly
present? effusion? opacity? atelectasis? pneumothorax? and a severity word."* Below, see a
real report next to the structured findings the LLM pulled out of it.
"""))

both(code("""
# A real report (free text) -> the LLM's structured findings (numbers we can model).
cid = "3840"
print("THE REPORT THE LLM READ:")
print(" ", reports[cid]["findings"])
print(" ", reports[cid]["impression"])
print("\\nWHAT THE LLM RETURNED (cached):")
for k, v in common.load_cached_llm_features(cid).items():
    print(f"  {k:28s} {v}")
"""))

both(md("""
### 1.4 How often is each finding present?
Quick sanity check on the text channel: across all patients, how often did the LLM mark each
finding present? If a feature is almost always 0 (or always 1) it carries little signal.
Cardiomegaly itself sits near 50% -- because we balanced the dataset on exactly that label.
"""))

both(code("""
llm_cols = [c for c in df.columns if c.startswith("llm_") and c.endswith("_present")]
rates = df[llm_cols].mean().sort_values()

fig, ax = nbfig.fig(figsize=(7.5, 3.4))
ax.barh([c.replace("llm_", "").replace("_present", "") for c in rates.index], rates.values,
        color=nbfig.palette(len(rates)))
ax.set_xlabel("fraction of patients where the LLM marked it present"); ax.set_xlim(0, 1)
nbfig.show(fig, "How common is each finding?")
"""))

# =========================================================================== #
# Section 2 -- The three signals, and stacking
# =========================================================================== #
both(md("""
## 2. Three signals, one table

The whole plan in one picture: turn each signal into numbers, lay them side by side as one
row per patient, and hand the table to one model.

![Late fusion: each signal becomes a column, then TabPFN decides](img/multimodal_stack.png)

### 2.1 The image's vote (late fusion)
How do we get the X-ray into the table? The classic way is **radiomics**: hand-craft numbers
like brightness and texture (shown below for contrast). But there's a cleaner move: train an
actual image model -- transfer learning, exactly like Day 1 -- and feed the table just *its
prediction*, one probability `img_pred`. That's the image's vote.

**One honesty catch:** if the image model scores a patient it trained on, that score is too
optimistic (it has seen the answer). So every `img_pred` was pre-computed **out-of-fold** --
each patient scored by a model trained only on the *others*. (See
`scripts/cache_openi_image_preds.py`.) Same fairness instinct that runs through the whole day.
"""))

both(code("""
from pathlib import Path
import json

# The classic handcrafted way (radiomics) -- shown once for contrast:
sample = sorted(Path("sample_images").glob("*.png"))[0]
feats = common.extract_image_features(sample)
print("radiomics: ~12 handcrafted numbers per image (the old way)")
for k, v in list(feats.items())[:4]:
    print(f"  {k:22s} {v:.4f}")
print("  ...")

# What we actually use: the trained image model's single out-of-fold vote.
img_preds = json.loads(Path("../../datasets/openi_image_preds.json").read_text())
print(f"\\nimg_pred: one probability per case (we use THIS). {len(img_preds)} cached.")
print("  examples:", {k: img_preds[k] for k in list(img_preds)[:3]})
"""))

# =========================================================================== #
# Section 3 -- The table
# =========================================================================== #
both(md("""
## 3. Everything becomes one table

The instructor pre-built the table: one row per patient, all three signals plus the label.
`img_pred` is the image vote, `llm_` are the text features, then the demographics.
"""))

both(code("""
df = pd.read_csv("../../datasets/openi_features.csv")
print("table shape:", df.shape)
print("\\ncolumn groups:")
print("  image:", [c for c in df.columns if c == "img_pred"])
print("  text :", [c for c in df.columns if c.startswith("llm_")])
print("  demo :", [c for c in df.columns if c in ("age", "sex_male", "smoker")])
df.head(3)
"""))

both(md("""
### 3.1 Is the image vote any good on its own?
Before modeling, look at the image vote split by the truth. If `img_pred` is a real signal,
patients *with* cardiomegaly should score higher than those without -- the two histograms
should pull apart. (They overlap, because the image alone isn't the whole story. That's why
we add the other signals.)
"""))

both(code("""
fig, ax = nbfig.fig(figsize=(7, 3.4))
ax.hist(df.loc[df.label == 0, "img_pred"], bins=20, alpha=0.7, color=nbfig.TURQUOISE, label="no cardiomegaly")
ax.hist(df.loc[df.label == 1, "img_pred"], bins=20, alpha=0.7, color=nbfig.DEEPPINK, label="cardiomegaly")
ax.set_xlabel("img_pred (image model's probability)"); ax.set_ylabel("patients"); ax.legend()
nbfig.show(fig, "The image vote separates the classes -- partly")
"""))

# =========================================================================== #
# Section 4 -- Build + TabPFN
# =========================================================================== #
both(md("""
## 4. Build the feature matrix, then let TabPFN decide

**Predict first:** which group will matter most for cardiomegaly -- the image, the report
text, or demographics? Now assemble the inputs: `X` is every feature column; `y` is the label.
"""))

todo(
    """
feature_cols = [c for c in df.columns if c not in ("case_id", "label")]
X = df[feature_cols].fillna(0).values
y = df["label"].values
print("X:", X.shape, " positives:", int(y.sum()), "/", len(y))
""",
    [
        ("feature_cols = [c for c in df.columns", "select every column except 'case_id' and 'label'"),
        ("X = df[feature_cols]", "make X from those columns (use .fillna(0).values)"),
    ],
)

both(md("""
**TabPFN** is a foundation model pretrained on millions of synthetic tables. It doesn't train
in the usual sense -- you `fit` (it just studies your examples) and `predict`, both in
seconds. Same pretraining-and-reuse idea as ImageNet yesterday, now for tables.
"""))

todo(
    """
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

clf = TabPFNClassifier()
clf.fit(Xtr, ytr)
acc_all = (clf.predict(Xte) == yte).mean()
print(f"multimodal (image + text + demographics) accuracy: {acc_all:.3f}")
""",
    [
        ("clf = TabPFNClassifier()", "make a TabPFNClassifier"),
        ("clf.fit(Xtr, ytr)", "fit it on the training split Xtr, ytr"),
    ],
)

# =========================================================================== #
# Section 5 -- Ablation
# =========================================================================== #
both(md("""
## 5. How much did each modality help?

**Predict:** drop the report (text) features and keep only the image vote + demographics --
how much does accuracy fall? Fill in the ablation, then we'll plot the two side by side.
"""))

todo(
    """
no_text_cols = [c for c in feature_cols if not c.startswith("llm_")]
Xnt = df[no_text_cols].fillna(0).values
Xnt_tr, Xnt_te, _, _ = train_test_split(Xnt, y, test_size=0.25, random_state=0, stratify=y)

clf2 = TabPFNClassifier()
clf2.fit(Xnt_tr, ytr)
acc_no_text = (clf2.predict(Xnt_te) == yte).mean()
print(f"image + demographics only: {acc_no_text:.3f}")
print(f"with text:                 {acc_all:.3f}")
print(f"the text features were worth {acc_all - acc_no_text:+.3f}")
""",
    [
        ("no_text_cols = [c for c in feature_cols", "keep feature columns that do NOT start with 'llm_'"),
        ("clf2.fit(Xnt_tr, ytr)", "fit a fresh TabPFN on the no-text training data"),
    ],
)

both(code("""
fig, ax = nbfig.fig(figsize=(6, 3.6))
bars = ax.bar(["image + demo", "+ text"], [acc_no_text, acc_all],
              color=[nbfig.TURQUOISE, nbfig.DEEPPINK], width=0.55)
for b, a in zip(bars, [acc_no_text, acc_all]):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.0%}", ha="center",
            fontweight="bold", family="DejaVu Sans Mono")
ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1)
nbfig.show(fig, "Adding the text features looks amazing")
"""))

both(md("""
### 5.1 The confusion matrix of the full model
"""))

both(code("""
nbfig.confusion(yte, clf.predict(Xte), ["no cardiomegaly", "cardiomegaly"],
                text="Multimodal model").show()
"""))

# =========================================================================== #
# Section 6 -- Leakage
# =========================================================================== #
both(md("""
## 6. Wait -- that's too good. Target leakage.

That text boost should make you *suspicious*, not excited. Here's the catch, and it's the
single most important lesson of the day.

![The report names the diagnosis we are predicting](img/leakage.png)

The report we read the text from literally **states the diagnosis** we're trying to predict
("IMPRESSION: Cardiomegaly"). So the LLM's `cardiomegaly: yes` matches the label almost
perfectly. The model isn't detecting disease -- it's copying the answer off the report.
"""))

both(code("""
llm_cols = [c for c in df.columns if c.startswith("llm_")]
corrs = {c.replace("llm_", ""): np.corrcoef(df[c], df["label"])[0, 1] for c in llm_cols}

fig, ax = nbfig.fig(figsize=(7.5, 3.4))
names = list(corrs); vals = [corrs[n] for n in names]
colors = [nbfig.DEEPPINK if abs(v) > 0.7 else nbfig.TURQUOISE for v in vals]
ax.barh(names, vals, color=colors)
ax.set_xlabel("correlation with the true label"); ax.set_xlim(-0.1, 1)
ax.axvline(0, color=nbfig.MUTED, lw=0.8)
nbfig.show(fig, "One text feature basically IS the answer")
print("cardiomegaly_present vs label:", round(corrs.get("cardiomegaly_present", float('nan')), 2))
"""))

both(md("""
### 6.1 Prove it: remove the one leaked feature
If `llm_cardiomegaly_present` really is the answer in disguise, deleting *just that one column*
(keeping every other text feature) should make the near-perfect score collapse. Let's check.
"""))

both(code("""
leaked = "llm_cardiomegaly_present"
fair_cols = [c for c in feature_cols if c != leaked]
Xf = df[fair_cols].fillna(0).values
Xf_tr, Xf_te, _, _ = train_test_split(Xf, y, test_size=0.25, random_state=0, stratify=y)

clf_fair = TabPFNClassifier()
clf_fair.fit(Xf_tr, ytr)
acc_fair = (clf_fair.predict(Xf_te) == yte).mean()
print(f"with the leaked feature:    {acc_all:.3f}")
print(f"after deleting just it:     {acc_fair:.3f}")
print(f"one column was worth {acc_all - acc_fair:+.3f} -- it WAS the answer.")
"""))

both(md("""
### 6.2 What a fair test looks like
This trap is **target leakage**, and catching it is real expertise.

- **No peeking at the label.** A feature that encodes the answer is cheating. The report's
  impression basically *is* the answer.
- **Use inputs that come before the label.** Fair signals exist before the diagnosis is
  known: the raw pixels (our `img_pred`), the demographics.
- **Report it honestly.** Show the leaked and de-leaked numbers side by side. The honest
  result -- the image vote + demographics, ~72% -- is lower, and real.
"""))

# =========================================================================== #
# Recap + stretch
# =========================================================================== #
both(md("""
## 7. Two paradigms, both yours now

In two days you've built the two dominant approaches in medical AI:

- **Day 1 -- end-to-end deep learning:** feed raw data to one big network that learns its own
  features.
- **Day 2 -- combine signals + a foundation model:** turn everything into a table and let a
  pretrained model handle it.

Knowing which to reach for -- and being able to *catch your own leakage* -- is half the job.
A closing caution: clinical **text** carries even more identifying detail than pixels; handle
it like the patient record it is.
"""))

both(md("""
## Stretch

The image vote (`img_pred`) is the one honest signal -- it comes from pixels, not a report
that names the answer. Train TabPFN on **just `img_pred`**, then on **just the demographics**,
and compare. Which single modality is strongest on its own? You'll see the leaked text wins on
paper, but the image vote is the number you could actually defend to a clinician.
"""))

both(md("""
## Tomorrow: capstone

You now have the whole toolkit: end-to-end deep learning, transfer learning, multimodal
feature stacks, and a foundation model. Tomorrow you pick a problem and build it start to
finish, with Claude as your engineer. See you there.
"""))


def build():
    sol, lab = new_nb(), new_nb()
    sol.cells = [p[0] for p in PAIRS]
    lab.cells = [p[1] for p in PAIRS]
    save(sol, ROOT / "notebooks/day2_multimodal/day2_solution.ipynb")
    save(lab, ROOT / "notebooks/day2_multimodal/day2.ipynb")
    print(f"wrote day2.ipynb + day2_solution.ipynb ({len(PAIRS)} cells)")


if __name__ == "__main__":
    build()
