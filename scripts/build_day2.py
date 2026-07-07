"""Build the Day 2 multimodal notebooks (lab + solution) from one source.

Emits:
  notebooks/day2_multimodal/day2.ipynb           (runnable; design-decision + explore driven)
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
- Ask the deployment question -- *which signals will I actually have when I need a prediction?* --
  and design a model that stays useful when one goes missing.

The code is all here and runs top to bottom, **your job is to understand it and make the calls**,
not to type boilerplate. Run each cell, read what it prints, and watch for two things:

- **Design decision** boxes -- a real choice with no single right answer. Pick one, jot down
  *why*, then **confirm with Claude: "which makes the most sense here, and why?"**
- **Explore** cells -- poke the sliders and dropdowns and see what changes.

Making a call and pressure-testing it with Claude is the whole point of today.
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

both(md("""
### 3.2 The image branch: *where* the model learned matters most
The single biggest lever of the whole day is not the tabular model, it's the image model. The
image vote can come from two very different places:

- **`img_pred_imagenet`** -- a ResNet18 pretrained on **ImageNet** (cats, cars, everyday photos),
  then pointed at chest X-rays. Generic transfer learning, like Day 1.
- **`img_pred`** -- a DenseNet pretrained on **~500,000 real chest X-rays** (it has seen
  cardiomegaly tens of thousands of times). Domain pretraining.

**Design decision.** Which do you expect to win, and by how much? Guess, then **confirm with
Claude: "for a medical imaging task with only ~700 examples, does it matter more what the model
was pretrained on, or the exact architecture? why?"** Then run the cell.
"""))

both(code("""
from sklearn.metrics import roc_auc_score
for col, label in [("img_pred_imagenet", "ImageNet-pretrained (generic)"),
                   ("img_pred",          "chest-X-ray-pretrained (domain)")]:
    p = df[col].values
    print(f"{label:34s}: accuracy={((p >= 0.5).astype(int) == df.label.values).mean():.3f}   "
          f"AUC={roc_auc_score(df.label.values, p):.3f}")
print("\\nSame task, same pixels. The model that learned on chest X-rays wins by a mile.")
print("This is why cardiomegaly felt 'impossible' at ~0.72 -- that was a cat-photo model guessing.")
print("A model that studied half a million chest X-rays makes it look easy. In medical AI,")
print("WHERE a model was pretrained beats almost everything else, including a fancier tabular model.")
"""))

# =========================================================================== #
# Section 4 -- Build + TabPFN
# =========================================================================== #
both(md("""
## 4. Build the feature matrix, then let TabPFN decide

**Predict first:** which group will matter most for cardiomegaly -- the image, the report
text, or demographics? Now assemble the inputs: `X` is every feature column; `y` is the label.
"""))

both(code("""
# assemble the inputs: X = every feature column, y = the label
feature_cols = [c for c in df.columns if c not in ("case_id", "label", "img_pred_imagenet")]
X = df[feature_cols].fillna(0).values
y = df["label"].values
print("X:", X.shape, " positives:", int(y.sum()), "/", len(y))
print("feature columns:", feature_cols)
"""))

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

both(code("""
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

# hold out 25% to grade on -- the model never sees these during fit
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

clf = TabPFNClassifier()   # the foundation model you chose above
clf.fit(Xtr, ytr)          # "fit" = it just studies your examples (seconds, no real training)
acc_all = (clf.predict(Xte) == yte).mean()
print(f"multimodal (image + text + demographics) accuracy: {acc_all:.3f}")
"""))

both(md("""
### 4.1 Now YOU are the model designer
That was our baseline. Your turn: **build your own model from choices.** Pick which signals to
feed it, pick the model, pick how much data to hold out for the test, and hit **Run Interact**.
Run a few combinations and watch closely, one thing will jump out:

- Swap the **model** (logreg -> Random Forest -> CatBoost -> TabPFN) but keep the same signals.
  How much does the accuracy actually move?
- Now instead keep the model and **add or remove a signal** (toggle "text"). How much does *that*
  move it?
- Which mattered more: changing the model, or changing the data?
"""))

both(code("""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier

# the two branches of the late-fusion pipeline, each with real choices:
IMAGE_BRANCH = {"chest-X-ray model (domain)": "img_pred",        # DenseNet, ~500k CXRs
                "ImageNet model (baseline)": "img_pred_imagenet"} # ResNet18, everyday photos
MODELS = {   # the tabular branch
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "CatBoost":            lambda: CatBoostClassifier(verbose=0, random_state=0),
    "TabPFN (foundation)": lambda: TabPFNClassifier(),
}

def build_my_model(signals=("image", "text", "demographics"),
                   image_model="chest-X-ray model (domain)",
                   tabular_model="TabPFN (foundation)", test_percent=25):
    cols = []
    if "image" in signals:        cols += [IMAGE_BRANCH[image_model]]   # your chosen image branch
    if "text" in signals:         cols += [c for c in df.columns if c.startswith("llm_")]
    if "demographics" in signals: cols += ["age", "sex_male", "smoker"]
    if not cols:
        print("Pick at least one signal!"); return
    Xc, yc = df[cols].fillna(0).values, df["label"].values
    Xa, Xb, ya, yb = train_test_split(Xc, yc, test_size=test_percent / 100, random_state=0, stratify=yc)
    try:
        m = MODELS[tabular_model](); m.fit(Xa, ya)
        pred = np.asarray(m.predict(Xb)).ravel()   # CatBoost returns 2D; flatten to be safe
        acc = (pred == yb).mean()
        print(f"signals:       {', '.join(signals)}")
        if "image" in signals:
            print(f"image branch:  {image_model}")
        print(f"tabular model: {tabular_model}")
        print(f"test {test_percent}% ({len(yb)} patients)   ->   ACCURACY = {acc:.3f}")
        nbfig.confusion(yb, pred, ["no cardiomegaly", "cardiomegaly"], text=f"{tabular_model}").show()
    except Exception:
        print("That combination wouldn't train (too few / too flat features). Add a signal and retry.")

try:
    from ipywidgets import interact_manual, SelectMultiple, RadioButtons, IntSlider
    interact_manual(build_my_model,
        signals=SelectMultiple(options=["image", "text", "demographics"],
                               value=("image", "text", "demographics"), description="signals"),
        image_model=RadioButtons(options=list(IMAGE_BRANCH), value="chest-X-ray model (domain)", description="image"),
        tabular_model=RadioButtons(options=list(MODELS), value="TabPFN (foundation)", description="tabular"),
        test_percent=IntSlider(value=25, min=10, max=40, step=5, description="test %"))
except ImportError:
    print("(no widgets -- ipywidgets missing; showing example builds)\\n")
    build_my_model(("image", "demographics"), "chest-X-ray model (domain)", "TabPFN (foundation)", 25)
    build_my_model(("image", "demographics"), "ImageNet model (baseline)", "TabPFN (foundation)", 25)
"""))

both(md("""
### The big takeaway of Day 2
Did you notice which knobs actually moved the number?

- Swapping the **tabular model** (logreg -> Random Forest -> CatBoost -> TabPFN): **barely a blip.**
- Swapping the **image branch** (ImageNet -> chest-X-ray-pretrained): **a huge jump.**
- Adding or removing a **signal** (text): **a huge jump.**

So the fancy tabular algorithm was never the point. What matters is **the data and where your
models learned**: a domain-pretrained image model, and which signals you feed in.

That's the whole lesson, and it's the opposite of Day 1:

> **Day 1 was about the model** (the ladder climbed as the architecture got smarter).
> **Day 2 is about the data.** Once you've picked a reasonable tabular model, the way to win is
> **better signals and better pretraining** (what your image model already learned from real chest
> X-rays), **not a fancier tabular algorithm.**

Real ML engineers learn this the expensive way: they spend weeks tuning models when they should
have spent them finding better data. You just saw it in 30 seconds. So the interesting question
isn't "which model?" -- it's **"which signals, and will I actually have them when it counts?"**
That question is the rest of today.
"""))

# =========================================================================== #
# Section 5 -- Ablation
# =========================================================================== #
both(md("""
## 5. How much did each modality help?

**Predict first:** drop the report (text) features and keep only the image vote + demographics --
how much do you think accuracy falls? Write down a guess, then run the cell and see.
"""))

both(code("""
# drop the report (llm_) columns, keep image + demographics
no_text_cols = [c for c in feature_cols if not c.startswith("llm_")]
Xnt = df[no_text_cols].fillna(0).values
Xnt_tr, Xnt_te, _, _ = train_test_split(Xnt, y, test_size=0.25, random_state=0, stratify=y)

clf2 = TabPFNClassifier()
clf2.fit(Xnt_tr, ytr)
acc_no_text = (clf2.predict(Xnt_te) == yte).mean()
print(f"image + demographics only: {acc_no_text:.3f}")
print(f"with text:                 {acc_all:.3f}")
print(f"the text features were worth {acc_all - acc_no_text:+.3f}")
"""))

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
# Section 6 -- Will you actually have these features?
# =========================================================================== #
both(md("""
## 6. Wait -- will you *have* the report when it counts?

Adding the report text took you from ~0.82 to ~0.98. That's the power of fusion, real and worth it. But
before you celebrate, ask the question a deployment engineer always asks: **when will each of
these signals actually be available?**

![Where each finding comes from](img/leakage.png)

These text findings are genuine, and they're powerful for a subtle reason: a radiologist wrote
that report *after* examining the patient, so a finding like "cardiomegaly: yes" is almost a
**restatement of the diagnosis**. That's exactly why it's such a strong feature -- and exactly
why you might not have it when you need it:

- At **screening time**, the X-ray exists but the report may not be written yet.
- In a busy or under-resourced clinic, a full report may **never** arrive.
- You're building a tool to *catch* disease -- leaning on a feature that already *names* it means
  the model learned to copy the report, not to read the scan.

Let's see just how much the model is leaning on that one finding.
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
nbfig.show(fig, "The 'cardiomegaly' finding basically restates the answer")
print("cardiomegaly_present vs label:", round(corrs.get("cardiomegaly_present", float('nan')), 2))
"""))

both(md("""
### 6.1 The stress test: what if the report vanishes?
The honest question isn't "is this cheating" -- these are real findings. It's **"how does my
model hold up if the report isn't there?"** Let's simulate exactly that: drop the report-derived
text features and keep only what you'd *always* have at imaging time -- the image and the
patient's demographics.
"""))

both(code("""
always_available = [c for c in feature_cols if not c.startswith("llm_")]   # image vote + demographics
Xa2 = df[always_available].fillna(0).values
Xa_tr, Xa_te, _, _ = train_test_split(Xa2, y, test_size=0.25, random_state=0, stratify=y)

clf_robust = TabPFNClassifier()
clf_robust.fit(Xa_tr, ytr)
acc_robust = (clf_robust.predict(Xa_te) == yte).mean()
print(f"with the full report (text present): {acc_all:.3f}")
print(f"report GONE, image + demographics:   {acc_robust:.3f}")
print(f"\\nso if the report disappears, this system still works at {acc_robust:.0%} -- it degrades, it doesn't die.")
"""))

both(md("""
### 6.2 The question that actually matters
Notice the reframe. The findings aren't "bad" -- they're real and useful **when you have them.**
The design question is about **availability**: which signals can you count on at prediction time?

> **When will this signal exist, relative to the moment you need a prediction?**

- `img_pred` -- from the **pixels**, available the instant the X-ray is taken. **Always there.**
- age, sex, smoker -- known at **intake**. **Always there.**
- the `llm_` report findings -- written *later*, by someone who already saw the answer. **Maybe there, maybe not.**

A model that only works when the report is present is fragile. A model that stays useful without
it is one you can actually deploy. So the finish line isn't "remove the strong feature" -- it's
**design a system that uses everything when available and doesn't fall apart when it isn't.**
"""))

# =========================================================================== #
# Section 7 -- Design a deployable model (student agency)
# =========================================================================== #
both(md("""
## 7. Your turn: design a model you'd actually deploy

You've seen the whole picture: fusion makes the model strong, but its strength leans on a report
you may not always have. Now **you** make the design call: *which features should your deployable
model rely on?* There's no single right answer, only trade-offs, so this is exactly the kind of
question to think through **with Claude as a design partner** (not to fill in a blank for you).

**Ask Claude something like:**
> *"I'm predicting cardiomegaly at screening time. My features are: img_pred (from the X-ray
> pixels), age, sex, smoker, and yes/no findings an LLM pulled from the radiology report -- but the
> report may not be written yet. Which features should my deployed model depend on, and why?"*

Read its reasoning, decide if you agree, then **tick the features you'd bet on** below and hit Run
Interact. (Tip: hold Ctrl/Cmd to multi-select. Try keeping the report findings, then drop them,
and think about which version you could actually ship.)
"""))

both(code("""
ALL_FEATURES = ["img_pred"] + [c for c in df.columns if c.startswith("llm_")] + ["age", "sex_male", "smoker"]

def build_honest_model(keep_features=("img_pred", "age", "sex_male", "smoker")):
    if not keep_features:
        print("Select at least one feature."); return
    cols = list(keep_features)
    Xh = df[cols].fillna(0).values
    Xa, Xb, ya, yb = train_test_split(Xh, df["label"].values, test_size=0.25, random_state=0, stratify=df["label"].values)
    m = TabPFNClassifier(); m.fit(Xa, ya)
    pred = np.asarray(m.predict(Xb)).ravel()
    acc = (pred == yb).mean()
    print("features you kept:", cols)
    print(f"accuracy: {acc:.3f}   (the everything-in model got {acc_all:.3f})")
    if any(c.startswith("llm_") for c in cols):
        print("\\n  you're depending on the report findings -- great WHEN the report exists,")
        print("  but this model breaks the moment it doesn't. powerful, but fragile.")
    else:
        print("\\n  image + demographics only: lower, but it runs on data you ALWAYS have at")
        print("  imaging time. this is the version you could actually deploy for screening.")
    nbfig.confusion(yb, pred, ["no cardiomegaly", "cardiomegaly"], text="Your model").show()

try:
    from ipywidgets import interact_manual, SelectMultiple
    interact_manual(build_honest_model,
        keep_features=SelectMultiple(options=ALL_FEATURES,
                                     value=("img_pred", "age", "sex_male", "smoker"),
                                     rows=len(ALL_FEATURES), description="keep"))
except ImportError:
    print("(no widgets -- ipywidgets missing; building the deployable default (image + demographics))\\n")
    build_honest_model(("img_pred", "age", "sex_male", "smoker"))
"""))

both(md("""
## 8. You designed this

Look at what you actually did today: you combined three kinds of data and saw **fusion work** --
the report findings took the model from ~82% to ~98%. Then you asked the question that separates a
demo from a deployable system: *will I have that report when I need a prediction?* And you designed
for the answer, a model that uses everything when it's there and **still works at ~82% when it
isn't.**

Two lessons you'll carry a long way:
- **Better signals beat a fancier model.** Swapping logreg -> CatBoost -> TabPFN barely moved the
  number; adding a signal moved it a lot.
- **Design for the data you'll actually have,** not the data that makes the demo look best.

**Push it further with Claude, as your design partner:**
- *"How could I make my image + demographics model better? What signal would help, and would I
  actually have it at screening time?"*
- *"If a hospital wanted to deploy this, what would you test before trusting it?"*
- *"Is ~82% good enough for a cardiomegaly screening tool? How should I think about the trade-off?"*

The goal isn't the highest number. It's a system, and a set of choices, you can stand behind.
"""))

both(md("""
## Two paradigms, both yours now, and onward

In two days you built the two dominant approaches in medical AI: **end-to-end deep learning**
(Day 1, where the *model* mattered) and **combining signals** (Day 2, where the *data* mattered
more). You also learned the habit that separates a careful engineer from a fooled one: ask whether
you'll actually have each signal at the moment you need to predict.

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
