# Outset AI in Healthcare — Build Plan

> **For agentic workers:** Use the standard subagent-driven-development pattern to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Adapted from a software-TDD plan template for an educational product: "test" = "artifact runs end-to-end and produces expected output."

**Goal:** Build all course materials (slides, tiered notebooks, capstone kits, quiz) for the 3-session AI in Healthcare course defined in `syllabus.md`, ready for delivery Mon Jul 6 – Wed Jul 8, 2026.

**Architecture:** Each day ships a `python-pptx`-generated slide deck (built from a script — diff-friendly, theme-consistent) and a set of T1/T2/T3 + solution notebooks. Notebooks share a per-day `common.py` for data loading, eval, and plotting. A smoke-test script runs every notebook headless to verify end-to-end execution. Capstone is structured as project-option starter kits.

**Tech Stack:** Python 3.12 (conda env `outset`), python-pptx, jupyter/nbformat, torch + torchvision, medmnist, transformers, anthropic, matplotlib.

---

## File structure

```
outset-ai-healthcare/
  syllabus.md                          # done
  README.md                            # done (will refresh at end)
  requirements.txt                     # done
  .gitignore                           # done
  prep/
    hours.md                           # done, keep updating
    build-plan.md                      # this file

  assessment/
    tier_quiz.md                       # printable + Colab-pasteable
    tier_quiz.ipynb                    # optional notebook variant

  scripts/
    smoke_notebooks.py                 # run every .ipynb headless, fail on cell errors
    build_all_slides.py                # invokes day1/day2/day3 builders
    nbutil.py                          # helpers: make_cell, save_nb, etc.

  slides/
    theme.py                           # Jin brand: Geist, palette, layout helpers
    day1.py                            # builds slides/build/day1.pptx
    day2.py
    day3.py
    build/                             # gitignored output
      day1.pptx
      day2.pptx
      day3.pptx

  notebooks/
    _shared/
      colab_setup.py                   # one-cell pip install + sanity checks
    day1_imaging/
      common.py                        # PneumoniaMNIST loader, eval, plots
      solution.ipynb                   # gold reference (Jin runs in lecture)
      t1_scaffold.ipynb                # most cells written, fill blanks
      t2_partial.ipynb                 # students write training/eval loops
      t3_spec.ipynb                    # spec only, students implement
    day2_llm/
      common.py                        # Open-i loader, text utilities
      solution.ipynb
      t1_scaffold.ipynb
      t2_partial.ipynb
      t3_spec.ipynb
    day3_capstone/
      project_options.md
      rubric.md
      starter_kits/
        skin_lesion/                   # HAM10000
          starter_t1.ipynb
          starter_t2.ipynb
          spec_t3.md
        ecg_arrhythmia/                # PTB-XL
        pneumonia_hard/                # RSNA full
        medmnist_choose/               # any MedMNIST
        multimodal/                    # Open-i image+report

  datasets/
    README.md                          # done
    download_pneumoniamnist.py
    download_openi.py
```

**Decomposition rationale:** slides as Python (not .pptx in git) makes diffs reviewable and themes consistent. Per-day `common.py` keeps the three tier notebooks DRY without `nbgrader` overhead. Starter kits as folders so each capstone option ships with its own minimal context.

---

## Phase 0 — Foundations

These are reusable across all three days. Build first.

### Task 1: Brand theme module

**Files:**
- Create: `slides/theme.py`
- Test: `slides/test_theme.py`

- [ ] **Step 1: Write theme module**

```python
# slides/theme.py
"""Brand theme for Jinchi's Outset AI in Healthcare deck.

Colors and fonts mirror Jin's personal branding (turquoise / deeppink / amber /
blueviolet, Geist + Geist Mono).
"""
from dataclasses import dataclass
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


# Brand palette — see memory/user_branding.md
TURQUOISE = RGBColor(0x40, 0xE0, 0xD0)
DEEPPINK  = RGBColor(0xFF, 0x14, 0x93)
AMBER     = RGBColor(0xF0, 0xC8, 0x40)  # NOT #FFD700
BLUEVIOLET = RGBColor(0x8A, 0x2B, 0xE2)
INK       = RGBColor(0x10, 0x10, 0x10)
PAPER     = RGBColor(0xFA, 0xFA, 0xFA)
MUTED     = RGBColor(0x66, 0x66, 0x66)

# Fonts
SANS = "Geist"
MONO = "Geist Mono"

# Standard 16:9 slide is 13.333 x 7.5 in
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

@dataclass
class Slide:
    title: str
    body: str = ""
    accent: RGBColor = TURQUOISE


def title_slide(prs, *, title, subtitle="", accent=TURQUOISE):
    """Add a title slide with Jin's brand layout."""
    from pptx.util import Inches, Pt
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    # Accent bar
    from pptx.shapes.autoshape import Shape
    from pptx.enum.shapes import MSO_SHAPE
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.5), Inches(0.4), Inches(6.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()

    # Title
    tx = slide.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(11.5), Inches(2.0))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = SANS
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = INK

    if subtitle:
        sx = slide.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(11.5), Inches(1.0))
        sp = sx.text_frame.paragraphs[0]
        sp.text = subtitle
        sp.font.name = SANS
        sp.font.size = Pt(24)
        sp.font.color.rgb = MUTED

    return slide


def content_slide(prs, *, title, bullets, accent=TURQUOISE):
    """Add a content slide with title and bullets."""
    from pptx.util import Inches, Pt
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    # Title
    tx = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12), Inches(0.8))
    p = tx.text_frame.paragraphs[0]
    p.text = title
    p.font.name = SANS
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = INK

    # Underline accent
    from pptx.enum.shapes import MSO_SHAPE
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.25), Inches(1.6), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()

    # Bullets
    bx = slide.shapes.add_textbox(Inches(0.6), Inches(1.6), Inches(12), Inches(5.5))
    tf = bx.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.text = b
        para.font.name = SANS
        para.font.size = Pt(22)
        para.font.color.rgb = INK
        para.level = 0
    return slide


def section_divider(prs, *, label, accent=DEEPPINK):
    """Big section break. Used between major topics in a deck."""
    from pptx.util import Inches, Pt
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = INK
    bg.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(1), Inches(3), Inches(11), Inches(1.5))
    p = tx.text_frame.paragraphs[0]
    p.text = label
    p.font.name = SANS
    p.font.size = Pt(60)
    p.font.bold = True
    p.font.color.rgb = accent
    return slide
```

- [ ] **Step 2: Write smoke test**

```python
# slides/test_theme.py
from pptx import Presentation
from slides.theme import title_slide, content_slide, section_divider, TURQUOISE


def test_theme_builds():
    prs = Presentation()
    title_slide(prs, title="Test", subtitle="Smoke")
    content_slide(prs, title="Bullets", bullets=["a", "b", "c"])
    section_divider(prs, label="Break")
    assert len(prs.slides) == 3
```

- [ ] **Step 3: Run test**

```bash
cd /Users/jinchiwei/arcadia/educator/outset-ai-healthcare
python -m pytest slides/test_theme.py -v
```
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add slides/theme.py slides/test_theme.py
git commit -m "Add brand theme module for course slides"
```

---

### Task 2: Notebook smoke-test infrastructure

**Files:**
- Create: `scripts/smoke_notebooks.py`
- Create: `scripts/nbutil.py`

- [ ] **Step 1: Write nbutil helpers**

```python
# scripts/nbutil.py
"""Helpers for programmatic notebook construction."""
import nbformat as nbf
from pathlib import Path


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())

def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())

def save(nb: nbf.NotebookNode, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))

def new_nb() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3 (outset)"},
        "language_info": {"name": "python"}
    })
```

- [ ] **Step 2: Write smoke runner**

```python
# scripts/smoke_notebooks.py
"""Execute every notebook in notebooks/ headless. Fail on cell errors.

Skips T1/T2/T3 student notebooks (they're intentionally incomplete);
runs only solution.ipynb and starter_t*.ipynb completed variants.
"""
import sys
from pathlib import Path
from nbclient import NotebookClient
import nbformat


SKIP_PATTERNS = ("t1_scaffold", "t2_partial", "t3_spec", "spec_t3")
ROOT = Path(__file__).resolve().parents[1]


def should_run(p: Path) -> bool:
    name = p.name
    if name.endswith(".ipynb_checkpoints"):
        return False
    return not any(s in name for s in SKIP_PATTERNS)


def run_one(path: Path) -> tuple[bool, str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    try:
        client.execute()
    except Exception as e:
        return False, f"{path.relative_to(ROOT)}: {type(e).__name__}: {e}"
    return True, f"{path.relative_to(ROOT)}: ok"


def main() -> int:
    nbs = [p for p in ROOT.glob("notebooks/**/*.ipynb") if should_run(p)]
    if not nbs:
        print("no notebooks to smoke test")
        return 0
    fails = 0
    for nb_path in nbs:
        ok, msg = run_one(nb_path)
        print(("OK  " if ok else "FAIL"), msg)
        fails += 0 if ok else 1
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify it runs (with no notebooks)**

```bash
python scripts/smoke_notebooks.py
```
Expected: "no notebooks to smoke test", exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "Add notebook smoke-test runner and nbutil helpers"
```

---

### Task 3: Colab setup helper for student notebooks

**Files:**
- Create: `notebooks/_shared/colab_setup.py`

- [ ] **Step 1: Write the helper**

```python
# notebooks/_shared/colab_setup.py
"""Single-cell setup for Colab. Idempotent — students can rerun safely."""
import subprocess, sys, importlib

REQUIRED = [
    "torch", "torchvision", "medmnist", "matplotlib", "scikit-learn",
    "pandas", "numpy", "pillow", "tqdm",
]

def ensure(*pkgs):
    missing = []
    for p in (pkgs or REQUIRED):
        mod = p.replace("-", "_")
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(p)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    print(f"setup ok: {len(pkgs or REQUIRED)} packages ready")
```

- [ ] **Step 2: Smoke test it locally**

```bash
python -c "import sys; sys.path.insert(0, 'notebooks/_shared'); import colab_setup; colab_setup.ensure()"
```
Expected: "setup ok: 9 packages ready" (or similar).

- [ ] **Step 3: Commit**

```bash
git add notebooks/_shared/colab_setup.py
git commit -m "Add idempotent Colab setup helper"
```

---

## Phase 1 — Day 1 (medical imaging)

D1 sets the pattern; T2 and T3 notebooks are derived from the solution.

### Task 4: Day 1 common module (data, eval, plots)

**Files:**
- Create: `notebooks/day1_imaging/common.py`
- Create: `notebooks/day1_imaging/test_common.py`

- [ ] **Step 1: Write common module**

```python
# notebooks/day1_imaging/common.py
"""Shared helpers for Day 1: PneumoniaMNIST classification.

Importable from all three tier notebooks and the solution notebook.
Keeps boilerplate DRY without making notebooks magic.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T


def get_loaders(batch_size: int = 64):
    """Return train/val/test DataLoaders for PneumoniaMNIST (28x28, binary)."""
    from medmnist import PneumoniaMNIST
    tfm = T.Compose([T.ToTensor()])
    train = PneumoniaMNIST(split="train", download=True, transform=tfm)
    val   = PneumoniaMNIST(split="val", download=True, transform=tfm)
    test  = PneumoniaMNIST(split="test", download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


def evaluate(model, loader, device="cpu"):
    """Return dict with accuracy, sensitivity, specificity, predictions, labels."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logit = model(xb).squeeze(-1)
            pred = (torch.sigmoid(logit) > 0.5).long().cpu().numpy()
            ys.append(yb.numpy().reshape(-1))
            ps.append(pred.reshape(-1))
    y = np.concatenate(ys); p = np.concatenate(ps)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return {
        "accuracy": (tp + tn) / max(1, len(y)),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "y": y, "pred": p,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def show_misclassified(images, y, pred, n: int = 8):
    """Plot up to n misclassified examples with true/pred labels."""
    import matplotlib.pyplot as plt
    idx = np.where(y != pred)[0][:n]
    if len(idx) == 0:
        print("no misclassifications (suspicious — check splits)")
        return
    fig, axes = plt.subplots(1, len(idx), figsize=(2 * len(idx), 2.5))
    if len(idx) == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"y={int(y[i])}, p̂={int(pred[i])}")
        ax.axis("off")
    plt.tight_layout()
    return fig
```

- [ ] **Step 2: Write smoke test**

```python
# notebooks/day1_imaging/test_common.py
"""Smoke test for Day 1 common module. Skipped if medmnist data not downloaded."""
import pytest
import torch
import torch.nn as nn

from notebooks.day1_imaging import common


@pytest.mark.slow
def test_loaders_smoke():
    train, val, test = common.get_loaders(batch_size=8)
    xb, yb = next(iter(train))
    assert xb.shape[1:] == (1, 28, 28)
    assert yb.shape[0] == 8


@pytest.mark.slow
def test_evaluate_with_random_model():
    _, _, test = common.get_loaders(batch_size=32)
    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 1))
    res = common.evaluate(model, test)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert set(["accuracy", "sensitivity", "specificity"]).issubset(res)
```

- [ ] **Step 3: Smoke run (will download data first time)**

```bash
python -m pytest notebooks/day1_imaging/test_common.py -v -m slow
```
Expected: 2 passed (after ~30s download).

- [ ] **Step 4: Commit**

```bash
git add notebooks/day1_imaging/common.py notebooks/day1_imaging/test_common.py
git commit -m "Add Day 1 common module: loaders, eval, misclassification viewer"
```

---

### Task 5: Day 1 solution notebook (gold reference)

**Files:**
- Create: `notebooks/day1_imaging/solution.ipynb`

This is the notebook Jin demos live. T2 partial and T1 scaffold are derived by deleting cells.

- [ ] **Step 1: Build the notebook programmatically**

```python
# (write this as scripts/build_day1_solution.py — keeps notebook content in
#  a reviewable .py file, regenerates the .ipynb on demand)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nbutil import new_nb, md, code, save

OUT = Path(__file__).resolve().parents[1] / "notebooks/day1_imaging/solution.ipynb"

nb = new_nb()
nb.cells = [
    md("""# Day 1 — Pneumonia detection on chest X-ray (Solution)

This is the instructor reference. Students get scaffolded versions.

We're going to:
1. Load PneumoniaMNIST (small, fast, real medical images)
2. Train a tiny CNN to classify Normal vs. Pneumonia
3. Evaluate it — and look at what it gets wrong
4. Talk about what would have to be true to actually deploy this
"""),
    code("""# Colab one-time setup
import sys, os
sys.path.insert(0, "../_shared")
import colab_setup; colab_setup.ensure()
"""),
    code("""import torch, torch.nn as nn
import sys; sys.path.insert(0, "..")
from day1_imaging import common

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
"""),
    md("## 1. Load the data"),
    code("""train_loader, val_loader, test_loader = common.get_loaders(batch_size=64)
xb, yb = next(iter(train_loader))
print("batch:", xb.shape, "labels:", yb.shape, "classes:", yb.unique().tolist())
"""),
    code("""# Show a few examples — what does pneumonia look like to a model?
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(xb[i].squeeze(), cmap="gray")
    ax.set_title("Pneumonia" if yb[i].item()==1 else "Normal", fontsize=10)
    ax.axis("off")
plt.tight_layout()
"""),
    md("## 2. Build a tiny CNN"),
    code("""class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

model = TinyCNN().to(device)
sum(p.numel() for p in model.parameters()), "parameters"
"""),
    md("## 3. Train"),
    code("""opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(3):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.float().to(device).squeeze(-1)
        opt.zero_grad()
        out = model(xb).squeeze(-1)
        loss = loss_fn(out, yb)
        loss.backward(); opt.step()
        running += loss.item() * xb.size(0)
    val = common.evaluate(model, val_loader, device=device)
    print(f"epoch {epoch+1}  loss={running/len(train_loader.dataset):.3f}  val_acc={val['accuracy']:.3f}  sens={val['sensitivity']:.3f}  spec={val['specificity']:.3f}")
"""),
    md("""## 4. Evaluate on held-out test set

Accuracy alone is a trap in healthcare. We also care about:
- **Sensitivity** (recall on pneumonia): of the kids who DO have pneumonia, how many did we catch?
- **Specificity**: of the kids who DON'T, how many did we correctly say no?

Missing pneumonia is much worse than a false alarm.
"""),
    code("""res = common.evaluate(model, test_loader, device=device)
for k in ("accuracy", "sensitivity", "specificity"):
    print(f"{k:12s} {res[k]:.3f}")
print("confusion:", res["confusion"])
"""),
    md("## 5. Look at what it got wrong"),
    code("""# Pull all test images so we can show misclassified ones
xs, ys = [], []
for xb, yb in test_loader:
    xs.append(xb); ys.append(yb)
images = torch.cat(xs)
y_all = torch.cat(ys).numpy().reshape(-1)
common.show_misclassified(images, y_all, res["pred"], n=8)
"""),
    md("""## Discussion

- Where did the model fail? What do the images that fooled it have in common?
- If sensitivity is 0.92 and specificity is 0.88 — would you deploy this in a hospital?
- What would you need to test before deploying?
- The training set was clean. What might be different about real ER images?
"""),
]
save(nb, OUT)
print("wrote", OUT)
```

- [ ] **Step 2: Run the builder**

```bash
python scripts/build_day1_solution.py
```
Expected: "wrote notebooks/day1_imaging/solution.ipynb".

- [ ] **Step 3: Smoke test the notebook**

```bash
python scripts/smoke_notebooks.py
```
Expected: "OK  notebooks/day1_imaging/solution.ipynb: ok".

- [ ] **Step 4: Commit**

```bash
git add scripts/build_day1_solution.py notebooks/day1_imaging/solution.ipynb
git commit -m "Add Day 1 solution notebook (instructor reference)"
```

---

### Task 6: Day 1 T2 partial notebook (some Python, write the loops)

**Files:**
- Create: `scripts/build_day1_t2.py`
- Create: `notebooks/day1_imaging/t2_partial.ipynb`

T2 deletes the training loop and evaluation reasoning, keeps data + model.

- [ ] **Step 1: Write the T2 builder**

Builder reads the same source structure but blanks specific cells. Reuse `nbutil`. Critical cells to blank:
- Training loop body → `# YOUR CODE HERE: write the inner training loop. Use opt, loss_fn, model.`
- Evaluation discussion is kept; the call to `common.evaluate` is kept (helper, not pedagogy).
- Misclassified-viewer call kept; analysis questions kept.

Code outline (full file pattern same as Task 5; only diffs shown — engineer should copy-paste Task 5's builder and apply these changes):

```python
# In the training cell, replace the inner for-loop body with:
TRAIN_LOOP_BLANK = '''opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(3):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.float().to(device).squeeze(-1)
        # YOUR CODE: zero grads, forward pass, compute loss, backward, opt.step()
        # Track running loss as `running`.
        pass
    val = common.evaluate(model, val_loader, device=device)
    print(f"epoch {epoch+1}  loss={running/len(train_loader.dataset):.3f}  val_acc={val['accuracy']:.3f}")
'''
```

Add a markdown cell *before* training:
```
## 3. Train

You write the training loop. Hints:
- `opt.zero_grad()` clears old gradients
- `out = model(xb).squeeze(-1)` is your prediction
- `loss = loss_fn(out, yb)` compares prediction to truth
- `loss.backward()` then `opt.step()` updates weights
- Add `loss.item() * xb.size(0)` to `running` to track average loss
```

- [ ] **Step 2: Build it**

```bash
python scripts/build_day1_t2.py
```

- [ ] **Step 3: Eyeball the notebook in Jupyter**

```bash
jupyter notebook notebooks/day1_imaging/t2_partial.ipynb
```
Verify training cell is blanked, surrounding cells intact. (No smoke test — T2 is intentionally non-runnable.)

- [ ] **Step 4: Commit**

```bash
git add scripts/build_day1_t2.py notebooks/day1_imaging/t2_partial.ipynb
git commit -m "Add Day 1 T2 partial notebook (students write training loop)"
```

---

### Task 7: Day 1 T1 scaffold notebook (fill-in-the-blank)

**Files:**
- Create: `scripts/build_day1_t1.py`
- Create: `notebooks/day1_imaging/t1_scaffold.ipynb`

T1 is heavily scaffolded. Most code is pre-written. Students fill in 4-5 specific blanks (e.g., `BATCH_SIZE = ___`, `model = TinyCNN()`, `epochs = ___`). Reasoning prompts in markdown after every section.

- [ ] **Step 1: Write builder**

Pattern: copy Task 5's builder, then for each "student blank," replace a single line with `# FILL IN: <hint>` and surround it with a markdown cell:

```
**Stop and predict:**
Before running the next cell, what do you think will happen? Write your guess in the cell below as a comment.
```

Specific blanks:
1. `BATCH_SIZE = ___  # try 32 or 64`
2. `model = ___()  # which class did we define above?`
3. `epochs = ___  # 3 is a reasonable starting place`
4. `threshold = ___  # what value separates "pneumonia" from "normal" in sigmoid output?`
5. After eval: a free-response markdown cell — "Pick one misclassified image. Why might the model have gotten it wrong?"

- [ ] **Step 2: Build it**

```bash
python scripts/build_day1_t1.py
```

- [ ] **Step 3: Eyeball + commit**

```bash
jupyter notebook notebooks/day1_imaging/t1_scaffold.ipynb
git add scripts/build_day1_t1.py notebooks/day1_imaging/t1_scaffold.ipynb
git commit -m "Add Day 1 T1 scaffold notebook (fill-in-the-blank)"
```

---

### Task 8: Day 1 T3 spec notebook (Claude Code capable)

**Files:**
- Create: `notebooks/day1_imaging/t3_spec.ipynb`

T3 is mostly markdown. The notebook is a *spec*, not code. Students write everything.

- [ ] **Step 1: Write the spec notebook directly (no builder needed; it's mostly markdown)**

```python
# scripts/build_day1_t3.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nbutil import new_nb, md, code, save

OUT = Path(__file__).resolve().parents[1] / "notebooks/day1_imaging/t3_spec.ipynb"

nb = new_nb()
nb.cells = [
    md("""# Day 1 T3 — Pneumonia chest X-ray spec

You're at the "I can use Claude Code" tier. You get a spec, not code.

## Goal
Train a binary classifier on PneumoniaMNIST that achieves at least 85% test accuracy
AND at least 90% sensitivity (recall on positive class).

## Constraints
- Use PyTorch.
- Don't use a pretrained model — train from scratch (it's a 28×28 dataset, you don't need ImageNet).
- Train for at most 5 minutes on Colab CPU.
- Report: test accuracy, sensitivity, specificity, confusion matrix.
- Show 8 misclassified images with predictions.

## Stretch goals (pick one)
- Compare against the same task on RSNA Pneumonia (full-resolution chest X-ray). What changes?
- Add data augmentation. Does it help? Show before/after.
- Re-train with class weights to favor sensitivity. What's the trade-off?

## Allowed tools
- The `common.py` in this folder has helpers — read it before reimplementing.
- Claude Code is encouraged. Treat it like a pair programmer, not an autocomplete.
- The instructor's `solution.ipynb` is off-limits until after presentations.

## What "done" looks like
- Your notebook runs end-to-end on a fresh kernel.
- A markdown cell at the end answers: "If your model said this patient has pneumonia, would you trust it? Why or why not?"
"""),
    code("""# Your imports here. Helpful starter:
import torch, torch.nn as nn
import sys; sys.path.insert(0, "..")
from day1_imaging import common
"""),
    md("## Build below."),
    code(""),
]
save(nb, OUT)
print("wrote", OUT)
```

- [ ] **Step 2: Build + commit**

```bash
python scripts/build_day1_t3.py
git add scripts/build_day1_t3.py notebooks/day1_imaging/t3_spec.ipynb
git commit -m "Add Day 1 T3 spec notebook"
```

---

### Task 9: Day 1 slide deck

**Files:**
- Create: `slides/day1.py`
- Output: `slides/build/day1.pptx`

Slide outline (~22 slides, ~30 min lecture + transitions). Use `theme.title_slide`, `content_slide`, `section_divider`.

- [ ] **Step 1: Write builder**

```python
# slides/day1.py
"""Day 1 deck: medical imaging + pneumonia chest X-ray."""
from pathlib import Path
from pptx import Presentation
from slides.theme import (
    title_slide, content_slide, section_divider,
    TURQUOISE, DEEPPINK, AMBER, BLUEVIOLET,
)


def build():
    prs = Presentation()
    prs.slide_width  = int(13.333 * 914400)  # 16:9
    prs.slide_height = int(7.5 * 914400)

    title_slide(
        prs,
        title="AI in Healthcare",
        subtitle="Day 1: Medical imaging — Outset, July 6, 2026",
        accent=TURQUOISE,
    )
    content_slide(prs, title="Today",
        bullets=[
            "Welcome + 5-min self-placement quiz",
            "Lecture: what AI in medicine actually is",
            "Lab: train a real chest X-ray model",
            "Share-back",
        ])
    content_slide(prs, title="Three tiers, same project",
        bullets=[
            "T1 — fill-in-the-blank, focus on reasoning",
            "T2 — write the training loop yourself",
            "T3 — spec only, build from scratch (Claude Code OK)",
            "Pick now; you can switch tomorrow.",
        ])

    section_divider(prs, label="What is medical AI?", accent=DEEPPINK)
    content_slide(prs, title="Three real systems",
        bullets=[
            "Diabetic retinopathy screening (Google, India + Thailand clinics)",
            "Sepsis early warning (Epic — controversial)",
            "Mammography triage (multiple FDA-cleared products)",
        ])
    content_slide(prs, title="What they have in common",
        bullets=[
            "Narrow task — not 'diagnose anything'",
            "Big labeled dataset — usually retrospective",
            "Evaluated on held-out hospitals, not just held-out patients",
            "Sensitivity matters more than accuracy",
        ])
    content_slide(prs, title="What goes wrong",
        bullets=[
            "Distribution shift — your hospital is not the training hospital",
            "Shortcut learning — model uses the chest tube, not the pneumonia",
            "Calibration — '90% confident' actually means what?",
            "Operator effect — does the radiologist still pay attention?",
        ])

    section_divider(prs, label="How a CNN sees an X-ray", accent=AMBER)
    # ... ~5 more slides on convolution intuition, pooling, hierarchy
    # (keep simple — these are 10th graders)

    section_divider(prs, label="Lab: build it", accent=BLUEVIOLET)
    content_slide(prs, title="The plan",
        bullets=[
            "Open your tier's notebook in Colab",
            "Run the setup cell first",
            "Stop at every 'discussion' marker — talk to your neighbor",
            "I'll walk through the solution at 4:35",
        ])
    content_slide(prs, title="If you get stuck",
        bullets=[
            "T1/T2: ask your neighbor first, then me",
            "T3: ask Claude Code first, then me",
            "Always: print shapes (`print(x.shape)`)",
        ])

    section_divider(prs, label="Share-back", accent=DEEPPINK)
    content_slide(prs, title="One thing each tier",
        bullets=[
            "T1: one image the model got wrong — why?",
            "T2: what was your final sensitivity?",
            "T3: what design choice did you make?",
        ])
    content_slide(prs, title="Tomorrow",
        bullets=[
            "LLMs and clinical text",
            "Combine the X-ray with its radiology report",
            "Same dataset, more signal",
        ])

    out = Path(__file__).resolve().parents[1] / "slides/build/day1.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


if __name__ == "__main__":
    p = build()
    print("wrote", p)
```

- [ ] **Step 2: Run + verify**

```bash
python -m slides.day1
```
Expected: "wrote slides/build/day1.pptx".

Open the file in PowerPoint or Keynote and eyeball: brand colors present, no broken layout, ~14-18 slides.

- [ ] **Step 3: Commit (deck file is gitignored — only the builder is committed)**

```bash
git add slides/day1.py
git commit -m "Add Day 1 slide deck builder"
```

---

## Phase 2 — Day 2 (LLMs and multimodal)

D2 follows the same pattern as D1: solution → T2 → T1 → T3 → slides. Anchor dataset is **Open-i / Indiana University Chest X-ray** (paired image + report, no PhysioNet credentialing).

### Task 10: Day 2 dataset prep — Open-i loader

**Files:**
- Create: `datasets/download_openi.py`
- Create: `notebooks/day2_llm/common.py`

- [ ] **Step 1: Download script**

Open-i has a public XML+image archive. Cache to `datasets/raw/openi/`. Keep this offline-friendly: if files exist, skip download.

```python
# datasets/download_openi.py
"""Download a curated subset of Open-i Indiana University Chest X-ray reports.

This script downloads ~3,800 frontal CXR images + their associated radiology
reports (XML). Cache to datasets/raw/openi/.
"""
import sys, urllib.request, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets/raw/openi"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"


def fetch(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract(archive: Path, into: Path):
    if (into / ".extracted").exists():
        return
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(into)
    (into / ".extracted").touch()


def main():
    img_tar = ROOT / "NLMCXR_png.tgz"
    rep_tar = ROOT / "NLMCXR_reports.tgz"
    fetch(IMAGES_URL, img_tar)
    fetch(REPORTS_URL, rep_tar)
    extract(img_tar, ROOT / "images")
    extract(rep_tar, ROOT / "reports")
    print("ok")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: common.py with loader**

```python
# notebooks/day2_llm/common.py
"""Open-i CXR helpers: parse reports, pair with images, build small splits."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import re

OPENI_ROOT = Path(__file__).resolve().parents[2] / "datasets/raw/openi"

def parse_report(xml_path: Path) -> dict:
    """Extract findings + impression + assigned MeSH labels from an Open-i report."""
    root = ET.parse(xml_path).getroot()
    abstract = root.find(".//Abstract")
    text = {}
    if abstract is not None:
        for at in abstract.findall("AbstractText"):
            label = at.get("Label", "").lower()
            text[label] = (at.text or "").strip()
    images = [img.get("id") for img in root.findall(".//parentImage")]
    mesh = []
    mesh_node = root.find(".//MeSH")
    if mesh_node is not None:
        for m in mesh_node.findall(".//major"):
            if m.text:
                mesh.append(m.text.strip())
    return {
        "findings": text.get("findings", ""),
        "impression": text.get("impression", ""),
        "images": images,
        "labels": mesh,
    }


def has_finding(record: dict, term: str) -> int:
    """Binary label: does this report mention `term` (case-insensitive)?"""
    blob = " ".join([record["findings"], record["impression"], *record["labels"]]).lower()
    return int(bool(re.search(re.escape(term.lower()), blob)))


def list_pairs(label_term: str = "cardiomegaly", n: int | None = None):
    """Yield (image_path, report_dict, label) triples. Keeps it small for Colab."""
    reports_dir = OPENI_ROOT / "reports/ecgen-radiology"
    images_dir = OPENI_ROOT / "images/NLMCXR_png"
    out = []
    for xml in sorted(reports_dir.glob("*.xml")):
        rec = parse_report(xml)
        if not rec["images"]:
            continue
        img = images_dir / f"{rec['images'][0]}.png"
        if not img.exists():
            continue
        out.append((img, rec, has_finding(rec, label_term)))
        if n and len(out) >= n:
            break
    return out
```

- [ ] **Step 3: Run + verify**

```bash
python datasets/download_openi.py    # ~3-5 min, ~1.5GB
python -c "from notebooks.day2_llm import common; print(len(common.list_pairs(n=20)))"
```
Expected: 20.

- [ ] **Step 4: Commit (data files gitignored)**

```bash
git add datasets/download_openi.py notebooks/day2_llm/common.py
git commit -m "Add Open-i CXR downloader and report parser for Day 2"
```

---

### Task 11: Day 2 solution notebook

**Files:**
- Create: `scripts/build_day2_solution.py`
- Create: `notebooks/day2_llm/solution.ipynb`

Outline:
1. What is an LLM? Tokenize a real radiology sentence, show the tokens.
2. Use the Anthropic API to extract structured findings from a free-text report. (Keep API call cells optional with a fallback to cached responses to avoid live-cost in class.)
3. Build a multimodal predictor: image features (from D1's CNN, or a fresh small one on 224×224) + text features (TF-IDF over impression text) + age/sex demographics → predict cardiomegaly.
4. Compare image-only vs. text-only vs. multimodal.
5. Discussion: hallucination risk, why the LLM is great at structuring text but not at being the final word.

- [ ] **Step 1: Write builder** — same pattern as Task 5, with full Python content for each cell. (Engineer: copy Task 5's builder structure; replace with Day 2 cell list per outline above. Use `common.list_pairs("cardiomegaly", n=600)` for the dataset.)

- [ ] **Step 2: Build + smoke run**

```bash
python scripts/build_day2_solution.py
python scripts/smoke_notebooks.py
```
Expected: solution.ipynb passes smoke test.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_day2_solution.py notebooks/day2_llm/solution.ipynb
git commit -m "Add Day 2 solution notebook (LLM + multimodal)"
```

---

### Task 12: Day 2 T2, T1, T3 notebooks

**Files:**
- Create: `scripts/build_day2_t2.py`, `_t1.py`, `_t3.py`
- Create: `notebooks/day2_llm/t2_partial.ipynb`, `t1_scaffold.ipynb`, `t3_spec.ipynb`

Same blanking strategy as Day 1.

**T2 blanks**: the multimodal feature concatenation cell + the train loop. Students figure out how to combine image embeddings with text vectors.

**T1 blanks**: 5 fill-in-the-blanks (e.g., `LABEL_TERM = ___`, the threshold for prediction, demographics column names). Reasoning prompts after every section.

**T3 spec**: "Build a multimodal predictor for one finding from Open-i. Beat the image-only baseline by ≥3 points sensitivity. Use Claude Code if you want."

- [ ] **Step 1: Write builders (3 files, parallel to D1 pattern)**
- [ ] **Step 2: Build all three**
- [ ] **Step 3: Eyeball each in Jupyter; commit**

```bash
git add scripts/build_day2_t*.py notebooks/day2_llm/t*.ipynb
git commit -m "Add Day 2 T1/T2/T3 notebooks"
```

---

### Task 13: Day 2 slide deck

**Files:**
- Create: `slides/day2.py`

Slide outline (~18 slides):
- Recap: what we did D1
- What is an LLM? (tokens → context → completion)
- Live demo slide: Anthropic API on a real report
- Why text matters in healthcare (notes, history, demographics)
- Hallucination + clinical safety
- Multimodal: how to combine image + text
- Lab plan
- Share-back

- [ ] **Step 1: Write builder, mirroring Task 9 pattern**
- [ ] **Step 2: Build + eyeball**
- [ ] **Step 3: Commit**

```bash
git add slides/day2.py
git commit -m "Add Day 2 slide deck builder"
```

---

## Phase 3 — Day 3 (capstone)

### Task 14: Capstone project options + rubric

**Files:**
- Create: `notebooks/day3_capstone/project_options.md`
- Create: `notebooks/day3_capstone/rubric.md`

- [ ] **Step 1: project_options.md**

Document each of the five options with: dataset, source URL, baseline goal, stretch goal, T1/T2/T3 expectations, and one "watch out for" caveat. Use the syllabus.md capstone block as the source.

- [ ] **Step 2: rubric.md**

Codify the 5-point rubric from syllabus.md. Add example presentations at each score level (1, 3, 5).

- [ ] **Step 3: Commit**

```bash
git add notebooks/day3_capstone/project_options.md notebooks/day3_capstone/rubric.md
git commit -m "Add capstone project options and presentation rubric"
```

---

### Task 15: Capstone starter kits (5 projects × T1/T2 + T3 spec)

**Files** — one folder per project:
- `notebooks/day3_capstone/starter_kits/skin_lesion/{starter_t1,starter_t2}.ipynb`, `spec_t3.md`
- ...same pattern for `ecg_arrhythmia`, `pneumonia_hard`, `medmnist_choose`, `multimodal`

Each starter_t1.ipynb: end-to-end runnable baseline notebook with explicit "improve me" markers.
Each starter_t2.ipynb: data loading + eval scaffolding only — students build the model.
Each spec_t3.md: a short markdown spec like Day 1 T3.

- [ ] **Step 1: Write a single builder per project. Five builders total.**

`scripts/build_capstone_skin.py`, `..._ecg.py`, `..._pneumonia_hard.py`, `..._medmnist.py`, `..._multimodal.py`.

- [ ] **Step 2: Build all five. Smoke-test starter_t1 notebooks.**

```bash
for s in skin ecg pneumonia_hard medmnist multimodal; do python scripts/build_capstone_${s}.py; done
python scripts/smoke_notebooks.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build_capstone_*.py notebooks/day3_capstone/starter_kits/
git commit -m "Add capstone starter kits for 5 project options"
```

---

### Task 16: Day 3 slide deck

**Files:**
- Create: `slides/day3.py`

Slide outline (~10 slides — D3 is mostly work time):
- Welcome + recap of D1+D2
- Capstone format: pairs, options, rubric
- The 5 options (one per slide w/ headline + dataset + key challenge)
- "How to use Claude Code well" — 3 tips
- Presentation format reminder
- Closing: where to go next

- [ ] **Step 1: Write builder + build + eyeball + commit**

```bash
git add slides/day3.py
git commit -m "Add Day 3 slide deck builder"
```

---

## Phase 4 — Quiz, polish, ship

### Task 17: Self-assessment quiz

**Files:**
- Create: `assessment/tier_quiz.md`
- Create: `assessment/tier_quiz.ipynb`

5 questions, ~5 min:
1. Have you written Python before? (none / a little / a lot)
2. Do you know what a `for` loop does without looking it up?
3. Read this 6-line snippet — what does it print? (a tiny snippet that loops and accumulates)
4. Have you used Claude (Claude Code or Claude.ai) to help you write code?
5. Pick a tier honestly. (T1 / T2 / T3, with a one-sentence description of each)

Score map: 0-3 → T1, 4-6 → T2, 7+ and used Claude Code → T3. Students may override one tier up or down.

- [ ] **Step 1: Write tier_quiz.md with full questions + scoring**
- [ ] **Step 2: Generate tier_quiz.ipynb (mostly markdown, one Python cell for the code-reading question)**
- [ ] **Step 3: Commit**

```bash
git add assessment/
git commit -m "Add tier self-assessment quiz"
```

---

### Task 18: Final smoke run + README refresh

- [ ] **Step 1: Full smoke**

```bash
python scripts/smoke_notebooks.py
python -m slides.day1 && python -m slides.day2 && python -m slides.day3
ls slides/build/
```
All notebooks: OK. All three .pptx files present.

- [ ] **Step 2: Refresh README.md** with what was actually built — list of decks, list of notebooks per day, link to syllabus.md, build commands.

- [ ] **Step 3: Update prep/hours.md** with cumulative time.

- [ ] **Step 4: Final commit**

```bash
git add README.md prep/hours.md
git commit -m "Polish: refresh README with delivered materials, update hour log"
```

---

## Self-review checklist

Run this before handing off to build:

**Spec coverage:**
- [ ] D1 imaging anchor (pneumonia chest X-ray): tasks 4-9 ✓
- [ ] D2 LLM + multimodal (Open-i): tasks 10-13 ✓
- [ ] D3 capstone (5 options + rubric): tasks 14-16 ✓
- [ ] Tiered notebook system (T1/T2/T3 + solution): every day ✓
- [ ] Self-assessment quiz: task 17 ✓
- [ ] Slides via python-pptx: tasks 9, 13, 16 ✓
- [ ] Brand theme (Geist, palette): task 1 ✓
- [ ] Smoke testing: task 2 ✓
- [ ] Datasets requiring no PhysioNet credentialing: PneumoniaMNIST + Open-i ✓

**Placeholders:**
- Tasks 11, 12, 13, 16 reference "same pattern as Task 5/9" — engineer must copy Task 5/9 structure and only swap content. This is acceptable here because the *pattern is the deliverable*. Each builder ends up with full code in its own file.

**Type / API consistency:**
- `common.evaluate(model, loader, device=...)` is the eval signature. Day 2 reuses it. ✓
- `nbutil.{md, code, save, new_nb}` is the notebook construction API. Used everywhere. ✓
- `theme.{title_slide, content_slide, section_divider}` used by all three slide builders. ✓
