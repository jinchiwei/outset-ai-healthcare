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
big neural net, we combine three kinds of signal:

- **image features**: handcrafted numbers from the X-ray (texture, brightness) --
  radiomics-style, the way radiologists have quantified images for years
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

# ---- Image features demo ------------------------------------------------- #
both(md("""
## Image features: radiomics, the quick version

Radiologists have long summarized images with handcrafted numbers: how bright, how
much contrast, how much texture. Let's compute some on a sample chest X-ray. (The
production tool for this is PyRadiomics; we use scikit-image so it runs anywhere.)
"""))

both(code("""
import common
from pathlib import Path

sample = sorted(Path("sample_images").glob("*.png"))[0]
feats = common.extract_image_features(sample)
print("image features for one chest X-ray:")
for k, v in feats.items():
    print(f"  {k:22s} {v:.4f}")
"""))

# ---- The table ----------------------------------------------------------- #
both(md("""
## Everything becomes one table

The instructor pre-built a table: one row per patient, with all three kinds of feature
plus the label (does this patient have cardiomegaly?). Notice the column name prefixes
-- `intensity_`/`glcm_` are image, `llm_` are text, then the demographics.
"""))

both(code("""
import pandas as pd
df = pd.read_csv("../../datasets/openi_features.csv")
print("table shape:", df.shape)
print("\\ncolumn groups:")
print("  image:", [c for c in df.columns if c.startswith(("intensity_", "glcm_"))])
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

**Predict:** if we throw away the report (text) features and keep only image +
demographics, how much will accuracy drop?

Fill in the ablation: build an image+demographics-only matrix, refit, compare.
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

Try predicting something the report does *not* directly state, like the patient's age
group, from the **image features alone**. Is it harder? Why might that be a more honest
test of what the X-ray actually contains?
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
