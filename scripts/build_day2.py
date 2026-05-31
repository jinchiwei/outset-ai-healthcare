"""Build the Day 2 multimodal notebooks (lab + solution) from one source.

Emits:
  notebooks/day2_multimodal/day2.ipynb           (# TODO blanks)
  notebooks/day2_multimodal/day2_solution.ipynb  (filled)

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


# --------------------------------------------------------------------------- #
both(md("""
# Day 2 -- LLMs and multimodal medical AI

Yesterday: end-to-end deep learning on eye images. Today a different problem and a
different paradigm.

We move to **chest X-rays** (Open-i, with real radiologist reports). Instead of one
big neural net, we combine three kinds of signal, where each one casts a *vote*:

- **image vote**: a trained image model's probability for the X-ray (transfer
  learning, like Day 1) -- a single number, not a raw embedding. This is **late
  fusion**, or **stacking**. (Pre-computed out-of-fold; you load it.)
- **text features**: findings an LLM pulled out of the radiology report
- **demographics**: age, sex, smoking history

We stack all three into one table and hand it to **TabPFN**, a foundation model for
tabular data. The big idea: *everything becomes a row of numbers, then a model handles
the table.*
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

# ---- What is an LLM ------------------------------------------------------ #
both(md("""
## Recap the bridge: ViT patches -> LLM tokens

Yesterday's Vision Transformer split an image into patches and used attention. An LLM
does the same with text: it splits text into **tokens** (word pieces) and uses
attention to decide which tokens matter. Let's look at a real radiology report.
"""))

both(code("""
report = ("FINDINGS: The heart is enlarged. There is a small left pleural effusion. "
          "No pneumothorax. IMPRESSION: Cardiomegaly with small effusion.")
# A real LLM uses subword tokens; here's the rough idea -- text becomes pieces:
tokens = report.replace(".", " .").split()
print(f"{len(tokens)} rough tokens:")
print(tokens[:20])
"""))

both(md("""
## The LLM already read every report for us

Calling an LLM live for hundreds of reports costs money and time, so the instructor
ran it once with the Anthropic API and saved the structured findings. You'll load
those. Here's what the model pulled out of each report: which findings are present,
and a severity word.
"""))

# ---- Image vote: late fusion / stacking ---------------------------------- #
both(md("""
## The image's vote: a trained model, not raw features

How do we get the X-ray into the table? The classic way is **radiomics**: hand-craft
numbers like brightness and texture (the cell below shows it). But there's a cleaner
move. Train an actual image model -- transfer learning, exactly like Day 1 -- and feed
the table just *its prediction*: one probability, `img_pred`. That's the image's vote.
Combining each modality's prediction this way is called **late fusion** or **stacking**.

One honesty catch: if the image model scores a patient it trained on, that score is too
optimistic (it has seen the answer). So the instructor pre-computed every `img_pred`
**out-of-fold** -- each patient scored by a model trained only on the *others*. You just
load the result. (See `scripts/cache_openi_image_preds.py`.)
"""))

both(code("""
import common
from pathlib import Path

# The classic handcrafted way (radiomics) -- shown once for contrast:
sample = sorted(Path("sample_images").glob("*.png"))[0]
feats = common.extract_image_features(sample)
print("radiomics: ~12 handcrafted numbers per image (the old way)")
for k, v in list(feats.items())[:4]:
    print(f"  {k:22s} {v:.4f}")
print("  ...")

# What we actually use: the trained image model's single out-of-fold vote.
import json
img_preds = json.loads(Path("../../datasets/openi_image_preds.json").read_text())
print(f"\\nimg_pred: one probability per case (we use THIS). {len(img_preds)} cached.")
print("  examples:", {k: img_preds[k] for k in list(img_preds)[:3]})
"""))

# ---- The table ----------------------------------------------------------- #
both(md("""
## Everything becomes one table

The instructor pre-built a table: one row per patient, with all three kinds of signal
plus the label (does this patient have cardiomegaly?). Notice the columns -- `img_pred`
is the image model's vote, `llm_` are the text features, then the demographics.
"""))

both(code("""
import pandas as pd
df = pd.read_csv("../../datasets/openi_features.csv")
print("table shape:", df.shape)
print("\\ncolumn groups:")
print("  image:", [c for c in df.columns if c == "img_pred"])
print("  text :", [c for c in df.columns if c.startswith("llm_")])
print("  demo :", [c for c in df.columns if c in ("age", "sex_male", "smoker")])
df.head(3)
"""))

# ---- Build X / y (TODO) -------------------------------------------------- #
both(md("""
## Build your feature matrix

**Predict first:** which feature group do you think will matter most for detecting
cardiomegaly (an enlarged heart) -- the image, the report text, or demographics?

Now assemble the inputs. `X` is every feature column; `y` is the label.
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

# ---- TabPFN (TODO) ------------------------------------------------------- #
both(md("""
## TabPFN: a foundation model for tables

TabPFN is pretrained on millions of synthetic tables. It does not train in the usual
sense -- you `fit` (it just memorizes the examples) and `predict` in seconds.
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

# ---- Ablation (TODO) ----------------------------------------------------- #
both(md("""
## How much did each modality help?

**Predict:** if we throw away the report (text) features and keep only the image vote +
demographics, how much will accuracy drop?

Fill in the ablation: build an image+demographics-only matrix (drop the `llm_` columns),
refit, compare. The image vote alone is an honest signal -- how far does the text take us
*beyond* it?
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

both(md("""
## Honest discussion: is this a fair test?

That text-feature boost is suspiciously large. Look at why:
"""))

both(code("""
import numpy as np
if "llm_cardiomegaly_present" in df.columns:
    corr = np.corrcoef(df["llm_cardiomegaly_present"], df["label"])[0, 1]
    print(f"correlation between the LLM's 'cardiomegaly_present' and the true label: {corr:.2f}")
    print("\\nThe report often *names* the diagnosis we're trying to predict.")
    print("So the text features can leak the answer. In real clinical AI this is a")
    print("constant danger: what counts as a fair input vs. a peek at the label?")
"""))

both(md("""
## Stretch

The image vote (`img_pred`) is the one honest signal here -- it comes from pixels, not
from a report that names the answer. Train TabPFN on **just `img_pred`**, then on **just
the demographics**, and compare. Which single modality is strongest on its own? Now you
have the full picture: the leaked text wins on paper, but the image vote is the number you
could actually defend.
"""))

both(md("""
## Tomorrow: capstone

You've now seen two paradigms: end-to-end deep learning (Day 1) and feature-based
modeling with a foundation model (Day 2). Tomorrow you pick a problem and build, with
Claude as your pair programmer.
"""))


def build():
    sol, lab = new_nb(), new_nb()
    sol.cells = [p[0] for p in PAIRS]
    lab.cells = [p[1] for p in PAIRS]
    save(sol, ROOT / "notebooks/day2_multimodal/day2_solution.ipynb")
    save(lab, ROOT / "notebooks/day2_multimodal/day2.ipynb")
    print("wrote day2.ipynb + day2_solution.ipynb")


if __name__ == "__main__":
    build()
