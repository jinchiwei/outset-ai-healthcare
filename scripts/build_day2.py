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

Fill in the `# TODO`s, and watch for **Design decision** boxes along the way. At each one, real
engineers face a genuine choice with no single right answer. Pick one, jot down *why*, then
**confirm with Claude: "which makes the most sense here, and why?"** and see whether your
reasoning survives. That habit, making a call and pressure-testing it, is the whole point of today.
"""))

both(code("""
# Setup: on Colab, grab the course files. Locally this is a no-op.
import os, sys, subprocess
if not os.path.exists("common.py"):
    os.system("git clone -q https://github.com/jinchiwei/outset-ai-healthcare.git")
    os.chdir("outset-ai-healthcare/notebooks/day2_multimodal")
else:
    os.system("git pull -q 2>/dev/null")   # already cloned -> refresh the course code
sys.path.insert(0, ".")
sys.path.insert(0, "../_shared")

# TabPFN weights: PIN 2.2.1. Newer (8.x) releases -- what Colab grabs by default --
# gate the weight download behind a license token that can't be accepted on Colab
# (TabPFNLicenseError). Force the working version explicitly, before anything imports it.
# Belt-and-suspenders: also set a throwaway course token, so that even if a newer
# TabPFN ends up loaded, its license check passes. (Disposable 30-day key; harmless
# for 2.2.1, which ignores it. Instructor: let it expire / revoke after the course.)
os.environ.setdefault("TABPFN_TOKEN", "tabpfn_sk_wHI64x4RQU89RmBI8R-JUGSy18LeLHHNuSm0VgVCRbw")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==2.2.1"], check=False)
if "tabpfn" in sys.modules and not getattr(sys.modules["tabpfn"], "__version__", "2.2").startswith("2.2"):
    print("\\n*** ACTION NEEDED: an old TabPFN is still loaded in memory.")
    print("    Go to  Runtime -> Restart session , then  Runtime -> Run all .  ***\\n")

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
### 0.2b Explore: watch one patient become a row of numbers
This is the single most important thing to *understand* today, so don't rush it. Use the dropdown
to flip through real patients. For each one you're seeing the **whole pipeline**: the messy
free-text report the LLM read, the tidy yes/no findings it pulled out, the demographics, the
image model's vote, and finally the true label. **Everything to the right of the report is just
numbers a model can use.**

As you click through a few, watch one thing: does the LLM's `llm_cardiomegaly_present` line up
with the true label? Keep that observation, it's the trap you'll spring in Section 6.
"""))

both(code("""
def show_patient(case=list(reports)[0]):
    rec = reports[case]
    print("=" * 68)
    print("THE REPORT (free text the LLM read):")
    print("  FINDINGS:  ", (rec["findings"] or "(none)")[:280])
    print("  IMPRESSION:", (rec["impression"] or "(none)")[:200])
    print("-" * 68)
    print("LLM-EXTRACTED TEXT FEATURES (numbers):")
    for k, v in common.load_cached_llm_features(case).items():
        print(f"    {k:28s} {v}")
    print("DEMOGRAPHICS:", common.load_demographics(case))
    print("IMAGE VOTE  : img_pred =", round(common.load_cached_image_pred(case)["img_pred"], 3))
    print("-" * 68)
    print("TRUE LABEL  :", "CARDIOMEGALY" if rec["label"] else "normal (no cardiomegaly)")

try:
    from ipywidgets import interact, Dropdown
    interact(show_patient, case=Dropdown(options=list(reports), description="patient"))
except ImportError:
    print("(no dropdown -- ipywidgets missing; showing the first patient)\\n")
    show_patient()
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

### 2.1 The image's vote -- your first design decision
How do we get the X-ray into the table? Three real options:

- dump all the **raw pixels** as thousands of columns,
- hand-craft a dozen **radiomics** numbers (brightness, texture),
- train an **image model** and feed the table just *its prediction*, one probability `img_pred`.

> **Design decision.** Which makes the most sense for a small dataset (~700 patients), and why?
> Pick one and jot down your reason, then **confirm with Claude: "for ~700 patients, what's the
> smartest way to feed an X-ray into a tabular model, and why?"** (We'll use `img_pred` -- called
> **late fusion** or **stacking** -- and the code below shows why the alternatives struggle.)

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
### 3.0b Read the table like a detective
Don't model yet, just make sure you can *read* it. One row = one patient; one column = one number
about them. Print a few rows and answer these for yourself (out loud with your partner):

1. Which columns came from the **image**? the **report text**? the **demographics**?
2. Look at each patient's `llm_cardiomegaly_present` next to their `label`. Do they match? Every
   time? (Suspicious, isn't it. Remember this for Section 6.)
"""))

both(code("""
show_cols = (["label", "img_pred"]
             + [c for c in df.columns if c.startswith("llm_")][:3]
             + ["age", "sex_male", "smoker"])
print(df[show_cols].head(6).to_string())
print("\\n^ 'label' is what we PREDICT. everything else is what we predict FROM.")
print("  does llm_cardiomegaly_present always equal label? scroll up and check every row.")
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
> **Design decision.** You have a table of ~700 patients and ~10 columns. How should you model
> it? (a) train a neural network from scratch, (b) a simple logistic regression, (c) a *pretrained*
> tabular foundation model. Which fits this amount of data best, and why? Decide, then **confirm
> with Claude: "for a small tabular medical dataset, which model makes the most sense and why?"**

**TabPFN** is a foundation model pretrained on millions of synthetic tables. It doesn't train
in the usual sense -- you `fit` (it just studies your examples) and `predict`, both in
seconds. Same pretraining-and-reuse idea as ImageNet yesterday, now for tables. (A network from
scratch would overfit ~700 rows; TabPFN's pretraining is exactly what a tiny dataset needs.)
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

> **Design decision.** Accuracy is one number, but it hides *which* mistakes the model makes.
> For a cardiomegaly screener, which error is worse: missing a truly enlarged heart (a false
> negative), or a false alarm on a healthy one? Which metric should you optimize, accuracy or
> sensitivity? Decide, then **confirm with Claude: "for a screening tool, what should I optimize
> and why?"** Then read the matrix below with that answer in mind.
"""))

both(code("""
nbfig.confusion(yte, clf.predict(Xte), ["no cardiomegaly", "cardiomegaly"],
                text="Multimodal model").show()
"""))

both(md("""
### 5.2 Explore: mix and match the signals yourself
Now stop reading and start poking. Tick which signals to feed the model, hit **Run Interact**, and
read the accuracy. Try to answer with experiments, not guesses:

- Which **single** signal is strongest on its own? (image only vs text only vs demographics only)
- Does adding demographics to the image actually help?
- The text-only score should raise an eyebrow. Hold that thought for the next section.
"""))

both(code("""
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

def try_signals(use_image=True, use_text=True, use_demographics=True):
    cols = []
    if use_image:        cols += ["img_pred"]
    if use_text:         cols += [c for c in df.columns if c.startswith("llm_")]
    if use_demographics: cols += ["age", "sex_male", "smoker"]
    if not cols:
        print("Pick at least one signal!"); return
    Xc = df[cols].fillna(0).values
    yc = df["label"].values
    Xa, Xb, ya, yb = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
    used = ', '.join(c for c, u in [('image', use_image), ('text', use_text), ('demographics', use_demographics)] if u)
    try:
        m = TabPFNClassifier(); m.fit(Xa, ya)
        acc = (m.predict(Xb) == yb).mean()
        print(f"signals used: {used}")
        print(f"  -> {len(cols)} feature columns,  test accuracy = {acc:.3f}")
    except Exception:
        # TabPFN refuses a set with too little variation (e.g. our synthetic demographics alone).
        print(f"signals used: {used}")
        print("  -> TabPFN couldn't train on just this -- too few / too flat features.")
        print("     That's itself a finding: demographics alone barely move the needle. Add another signal.")

try:
    from ipywidgets import interact_manual, Checkbox
    interact_manual(try_signals, use_image=Checkbox(True, description="image"),
                    use_text=Checkbox(True, description="text"),
                    use_demographics=Checkbox(True, description="demographics"))
except ImportError:
    print("(no checkboxes -- ipywidgets missing; running a few combos)\\n")
    for combo in [(True, False, False), (False, True, False), (False, False, True), (True, False, True)]:
        try_signals(*combo)
"""))

both(md("""
### 5.3 Sanity check: is it learning real signal, or memorizing?
Here's a trick real ML engineers use. If a model learns **real** patterns, then **scrambling the
answers** should destroy it, accuracy should crash to a coin flip (~50%). If it *didn't* crash,
the model was just memorizing noise. **Predict:** what accuracy will the shuffled-label model get?
"""))

both(code("""
import numpy as np
y_shuffled = np.random.RandomState(1).permutation(y)   # same features, scrambled answers
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X, y_shuffled, test_size=0.25, random_state=0, stratify=y_shuffled)
clf_shuf = TabPFNClassifier(); clf_shuf.fit(Xs_tr, ys_tr)
acc_shuf = (clf_shuf.predict(Xs_te) == ys_te).mean()

print(f"accuracy with REAL labels    : {acc_all:.3f}")
print(f"accuracy with SHUFFLED labels: {acc_shuf:.3f}   (~0.50 = pure chance)")
print("\\nThe gap between them is the real signal. If shuffled were also high, we'd be fooling ourselves.")
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
### 6.2 The rule you just discovered
That trap is **target leakage**: a feature that secretly carries the answer. The test for a
*fair* feature is one question:

> **Would this information exist *before* the diagnosis was made, and does it avoid just
> restating it?**

- `img_pred` -- comes from the **raw pixels**. Exists before anyone writes a report. **Fair.**
- age, sex, smoker -- known at intake. **Fair.**
- the `llm_` findings -- all read off a report a radiologist wrote *knowing the diagnosis*. They
  don't just correlate with the answer, they *are* a paraphrase of it. **Not fair.**

Catching this is real expertise. But finding the cheat isn't the finish line -- **building an
honest model is.**
"""))

# =========================================================================== #
# Section 7 -- Design the honest model (student agency)
# =========================================================================== #
both(md("""
## 7. Your turn: design a model you can defend

You diagnosed the problem. Now **you** decide the fix. The one real design choice is: *which
features are fair to keep?* There's no single right answer, only choices you can defend, so this
is exactly the kind of question to think through **with Claude as a design partner** (not to fill
in a blank for you).

**Ask Claude something like:**
> *"I'm predicting cardiomegaly. My features are: img_pred (from the X-ray pixels), age, sex,
> smoker, and several yes/no findings an LLM pulled from the radiology report. Which of these are
> fair to use, and which risk leaking the answer? Walk me through your reasoning."*

Read its reasoning, decide if you agree, then set your feature list below and build your model.
"""))

both(code("""
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

# YOUR DESIGN DECISION -- edit this list to whatever you can defend.
# Starting point: signals that exist independent of the diagnostic report.
FAIR_COLS = ["img_pred", "age", "sex_male", "smoker"]

Xfair = df[FAIR_COLS].fillna(0).values
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xfair, df["label"].values,
                                              test_size=0.25, random_state=0, stratify=df["label"].values)
model = TabPFNClassifier(); model.fit(Xf_tr, yf_tr)
acc_honest = (model.predict(Xf_te) == yf_te).mean()

print("features you chose:", FAIR_COLS)
print(f"honest model accuracy: {acc_honest:.3f}")
print(f"(the leaky version got {acc_all:.3f} -- but you couldn't defend a single one of its text features)")
nbfig.confusion(yf_te, model.predict(Xf_te), ["no cardiomegaly", "cardiomegaly"],
                text="Your honest, defensible model").show()
"""))

both(md("""
## 8. You designed this

Look at what you actually did today: you combined three kinds of data, you got a suspiciously
perfect score, **you caught your own model cheating**, and then you made a design call and built
a model whose every feature you can explain to a doctor. That last model is lower on paper and
**worth far more** -- it's a real detector of disease, not a paraphrase of the answer.

That whole loop -- build, distrust the too-good result, find the leak, redesign -- *is* the job.
Most people never learn to do the third and fourth steps.

**Push it further with Claude, as your design partner:**
- *"How could I make this honest model more trustworthy? What signal would actually help,
  and how would I check it isn't leaking too?"*
- *"If a hospital wanted to deploy this, what would you test before trusting it?"*
- *"Is my 72% good enough for a screening tool? How should I think about the trade-off?"*

The goal isn't a higher number. It's a model, and a set of choices, you can stand behind.
"""))

both(md("""
## Two paradigms, both yours now, and onward

In two days you built the two dominant approaches in medical AI: **end-to-end deep learning**
(Day 1) and **combining signals with a foundation model** (Day 2). You also learned the habit that
separates a careful engineer from a fooled one, distrust a too-good result and hunt the leak.

One caution to carry forward: clinical **text** holds even more identifying detail than pixels;
treat it like the patient record it is.

**Tomorrow, the capstone:** you pick a problem and build it start to finish, making the design
calls yourself, with Claude as your engineer. You're ready. See you there.
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
