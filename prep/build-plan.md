# Outset AI in Healthcare — Build Plan (v2)

> **For agentic workers:** Use the standard subagent-driven-development pattern to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Adapted from a software-TDD plan template for an educational product: "test" = "artifact runs end-to-end and produces expected output."

**Goal:** Build all course materials (slides, tracked notebooks, capstone kits) for the 3-session AI in Healthcare course defined in `syllabus.md`, ready for delivery Mon Jul 6 – Wed Jul 8, 2026.

**Architecture:** Each day ships a `python-pptx`-generated slide deck and a set of Track A/B/C + solution notebooks. D1 is the 6-model ladder (logreg → ViT) on APTOS-2019 fundus DR. D2 is PyRadiomics + Anthropic API + demographics → TabPFN. D3 is open-ended capstone with starter kits. Notebooks share a per-day `common.py` for data loading, eval, and plotting. A smoke-test script runs every executable notebook headless.

**Tech Stack:** Python 3.12 (conda env `outset`), python-pptx, jupyter/nbformat, torch + torchvision, timm (for ViT), scikit-learn, xgboost, pyradiomics, transformers, tabpfn, anthropic, matplotlib.

---

## Naming convention update

Three notebook variants, framed as **tracks** (not tiers — no hierarchy):

| Track | Filename suffix | Profile |
|-------|----------------|---------|
| A — Predict & Compare | `_track_a.ipynb` | No code experience. Predict accuracy before each cell, run it, find one weird saliency case. |
| B — Implement | `_track_b.ipynb` | Some Python. Fill implementation gaps (forward passes, training loops, saliency function). |
| C — Build | `_track_c.ipynb` | Claude Code capable. Spec-only. |

Solution notebook stays `_solution.ipynb`.

---

## File structure

```
outset-ai-healthcare/
  syllabus.md                          # done (v2)
  README.md                            # done
  requirements.txt                     # update with timm/pyradiomics/tabpfn/xgboost
  .gitignore                           # done
  prep/
    hours.md                           # done
    build-plan.md                      # this file (v2)

  scripts/
    smoke_notebooks.py                 # run every executable .ipynb headless
    build_all_slides.py                # invokes day1/day2/day3 builders
    nbutil.py                          # helpers
    pretrain_warmup.py                 # standalone: pre-download HF weights to populate Colab cache

  slides/
    theme.py                           # brand: Geist, palette, layout helpers
    day1.py                            # builds slides/build/day1.pptx
    day2.py
    day3.py
    build/                             # gitignored output

  notebooks/
    _shared/
      colab_setup.py                   # one-cell pip install + GPU check
    day1_ladder/
      common.py                        # APTOS loader, model factories, eval, viz
      solution.ipynb                   # full 6-model ladder, instructor reference
      track_a.ipynb                    # predict-then-run + saliency analysis
      track_b.ipynb                    # implementation gaps to fill
      track_c.ipynb                    # spec only
    day2_multimodal/
      common.py                        # PyRadiomics extractor, Anthropic helper, TabPFN wrapper
      solution.ipynb
      track_a.ipynb                    # human-vs-model labeling + multimodal compare
      track_b.ipynb
      track_c.ipynb
    day3_capstone/
      project_options.md
      rubric.md
      starter_kits/
        pneumonia_rsna/
          starter_a.ipynb
          starter_b.ipynb
          spec_c.md
        skin_ham10000/
        ecg_ptbxl/
        medmnist_choose/
        multimodal_extension/

  datasets/
    README.md                          # done; update for APTOS
    download_aptos.py
    download_openi.py                  # for D3 multimodal extension
    grading_rubric.json                # synthetic-but-realistic rubric for D2 text features
    synthetic_demographics.csv         # synthetic demographics for D2
```

---

## Phase 0 — Foundations (unchanged from v1)

Reusable across all three days. Build first.

### Task 1: Brand theme module

**Files:** `slides/theme.py`, `slides/test_theme.py`

Same as v1 — Geist + turquoise/deeppink/amber/blueviolet palette + python-pptx layout helpers (`title_slide`, `content_slide`, `section_divider`). See v1 plan or just port directly.

- [ ] Build theme.py with palette, helpers
- [ ] Smoke test
- [ ] Commit `Add brand theme module for course slides`

### Task 2: Notebook smoke-test infrastructure

**Files:** `scripts/smoke_notebooks.py`, `scripts/nbutil.py`

Same as v1. Smoke runner skips track_a/track_b/track_c/spec_c notebooks (intentionally non-runnable).

- [ ] Build nbutil + smoke runner
- [ ] Verify it runs (no notebooks present yet → exit 0)
- [ ] Commit `Add notebook smoke-test runner and nbutil helpers`

### Task 3: Colab setup helper

**Files:** `notebooks/_shared/colab_setup.py`

Idempotent pip install. Update package list to include `timm`, `pyradiomics`, `tabpfn`, `xgboost`. Add a GPU-check function that prints whether CUDA is available; if not, warn the student to use Runtime → Change runtime type.

```python
# notebooks/_shared/colab_setup.py
import subprocess, sys, importlib

REQUIRED = [
    "torch", "torchvision", "timm",
    "scikit-learn", "xgboost", "matplotlib", "seaborn", "pillow",
    "tqdm", "pandas", "numpy",
    "transformers", "anthropic", "tabpfn",
    "pyradiomics", "SimpleITK",
]

def ensure(*pkgs):
    missing = []
    for p in (pkgs or REQUIRED):
        mod = p.replace("-", "_").lower()
        # Special-case modules whose package and import names differ
        import_name = {"scikit-learn": "sklearn", "pillow": "PIL"}.get(p, mod)
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(p)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    print(f"setup ok: {len(pkgs or REQUIRED)} packages ready")

def gpu_check():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU OK: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU. Go to Runtime → Change runtime type → T4 GPU.")
    except ImportError:
        print("torch not installed yet — run ensure() first")
```

- [ ] Write helper
- [ ] Smoke test locally
- [ ] Commit `Add Colab setup with GPU check`

### Task 3.5: Pretrained-weights warmup script (NEW)

**Files:** `scripts/pretrain_warmup.py`, `notebooks/_shared/pretrain_warmup.ipynb`

Send students a notebook the day before D1 that downloads ResNet50 (~100MB) and ViT-Base (~330MB) into the Colab cache. Avoids 30-60s download stalls during the live lab.

```python
# notebooks/_shared/pretrain_warmup.ipynb (one cell)
import torchvision.models as tvm
from transformers import ViTModel, ViTImageProcessor

print("downloading ResNet50...")
_ = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
print("downloading ViT-Base (patch16-224, ImageNet)...")
_ = ViTModel.from_pretrained("google/vit-base-patch16-224")
_ = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
print("done. weights cached.")
```

- [ ] Write the notebook
- [ ] Verify timings
- [ ] Commit `Add pretrained weight warmup notebook for students`

---

## Phase 1 — Day 1: The Ladder (REWRITTEN)

D1 anchor is APTOS-2019 diabetic retinopathy. Same dataset across all 6 models. Live training on Colab T4.

### Task 4: APTOS dataset loader + download script

**Files:** `datasets/download_aptos.py`, `notebooks/day1_ladder/common.py`

APTOS-2019 is on Kaggle. Students will need a Kaggle API token. Document the friction.

- [ ] **Step 1: Download script with Kaggle API**

```python
# datasets/download_aptos.py
"""Download APTOS-2019 from Kaggle. Requires KAGGLE_USERNAME and KAGGLE_KEY env vars
(or a ~/.kaggle/kaggle.json file). Free Kaggle account required.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets/raw/aptos"

def main():
    if (ROOT / "train.csv").exists():
        print("aptos already downloaded")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    print("downloading APTOS 2019 from Kaggle (~9 GB compressed)...")
    subprocess.check_call([
        "kaggle", "competitions", "download",
        "-c", "aptos2019-blindness-detection",
        "-p", str(ROOT),
    ])
    # unzip
    for z in ROOT.glob("*.zip"):
        subprocess.check_call(["unzip", "-q", "-o", str(z), "-d", str(ROOT)])
    print("done.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: common.py — loader, transforms, splits**

```python
# notebooks/day1_ladder/common.py
"""Day 1 helpers: APTOS DR data loading, model factories, eval, visualization.

Shared across solution + track_a + track_b + track_c.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

APTOS_ROOT = Path(__file__).resolve().parents[2] / "datasets/raw/aptos"

NUM_CLASSES = 5
IMG_SIZE = 224

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class APTOSDataset(Dataset):
    """APTOS 2019 diabetic retinopathy dataset.

    Returns (image_tensor, label_int). label is 0-4 severity grade.
    """
    def __init__(self, csv_path: Path, img_dir: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / f"{row['id_code']}.png").convert("RGB")
        img = self.transform(img)
        return img, int(row["diagnosis"])


def default_transform(train: bool = False):
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])


def get_loaders(batch_size: int = 32, val_frac: float = 0.15, seed: int = 0):
    """Split APTOS train.csv into train/val. Returns (train_loader, val_loader)."""
    csv = APTOS_ROOT / "train.csv"
    img_dir = APTOS_ROOT / "train_images"
    df = pd.read_csv(csv)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    df.iloc[train_idx].to_csv(APTOS_ROOT / "_train_split.csv", index=False)
    df.iloc[val_idx].to_csv(APTOS_ROOT / "_val_split.csv", index=False)

    train_ds = APTOSDataset(APTOS_ROOT / "_train_split.csv", img_dir, default_transform(train=True))
    val_ds   = APTOSDataset(APTOS_ROOT / "_val_split.csv", img_dir, default_transform(train=False))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=batch_size, num_workers=2),
    )


def flatten_loader(loader, max_n: Optional[int] = None):
    """Flatten images to (N, H*W*C) for non-CNN models.
    Returns (X, y) numpy arrays. Subsample to max_n if given.
    """
    Xs, ys = [], []
    for xb, yb in loader:
        Xs.append(xb.numpy().reshape(xb.size(0), -1))
        ys.append(yb.numpy())
        if max_n and sum(x.shape[0] for x in Xs) >= max_n:
            break
    X = np.concatenate(Xs)[:max_n]
    y = np.concatenate(ys)[:max_n]
    return X, y


def evaluate_classifier(predict_fn, loader, device="cpu"):
    """Generic eval for any model with a predict_fn(batch_of_images) -> labels."""
    ys, ps = [], []
    for xb, yb in loader:
        if hasattr(xb, "to"):
            xb_dev = xb.to(device)
        else:
            xb_dev = xb
        pred = predict_fn(xb_dev)
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        ys.append(yb.numpy() if hasattr(yb, "numpy") else yb)
        ps.append(pred)
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {
        "accuracy": float((y == p).mean()),
        "y": y,
        "pred": p,
    }
```

Add CNN, ResNet, ViT factory functions to `common.py` (kept short here; engineer expands):

```python
def make_small_cnn(num_classes=NUM_CLASSES):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(128, num_classes),
    )

def make_resnet50(num_classes=NUM_CLASSES, pretrained=True):
    import torchvision.models as tvm
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = tvm.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_vit_base(num_classes=NUM_CLASSES, pretrained=True):
    import timm
    return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
```

Plus `train_one_model(model, train_loader, val_loader, epochs, lr, device)` and `show_filters(model)`, `gradcam(model, image, target_class)` helpers.

- [ ] Write all factories + helpers
- [ ] Write `test_common.py` smoke tests (subset, batch=4, 1 epoch each)
- [ ] Run smoke
- [ ] Commit `Add Day 1 common module: APTOS loader, model factories, eval, viz`

### Task 5: Day 1 solution notebook (the ladder)

**Files:** `scripts/build_day1_solution.py`, `notebooks/day1_ladder/solution.ipynb`

The notebook is the instructor reference. Walks through all 6 models in order on APTOS DR.

Notebook structure (cells):

1. Markdown: course intro, "today we build six models on the same data"
2. Code: `import colab_setup; colab_setup.ensure(); colab_setup.gpu_check()`
3. Code: imports + load data + show batch
4. Markdown: "Step 0 — Look at the data"
5. Code: RGB channel split visualization
6. Code: pixel histogram per channel
7. Code: augmentation effects (rotation, flip, normalization side-by-side)
8. Markdown: "Step 1 — Logistic regression"
9. Code: flatten train + val, fit `sklearn.linear_model.LogisticRegression(max_iter=200)`, report accuracy
10. Markdown: "Step 2 — Gradient boosting"
11. Code: same flat features, `xgboost.XGBClassifier`, report
12. Markdown: "Step 3 — MLP"
13. Code: small MLP on flat features, train 5 epochs, report
14. Markdown: "Step 4 — CNN from scratch"
15. Code: `make_small_cnn()`, train 5 epochs on raw images, report. Show first-layer filter visualizations.
16. Markdown: "Step 5 — ResNet50 (transfer learning)"
17. Code: `make_resnet50(pretrained=True)`, finetune 3 epochs, report
18. Markdown: "Step 6 — Vision Transformer"
19. Code: `make_vit_base(pretrained=True)`, finetune 3 epochs, report
20. Markdown: "Step 7 — What did it learn?"
21. Code: Grad-CAM on the ResNet over 6 sample images
22. Markdown: leaderboard table + bridge to D2

- [ ] Build notebook programmatically (mirror v1 pattern)
- [ ] Run on a Colab-equivalent local GPU or mark deferred to Jinchi's local validation
- [ ] Commit `Add Day 1 solution notebook (the ladder)`

### Task 6: Day 1 Track B notebook (implement)

**Files:** `scripts/build_day1_track_b.py`, `notebooks/day1_ladder/track_b.ipynb`

Track B keeps the structure of solution.ipynb but blanks specific implementation cells. Helper functions in `common.py` are still available.

Cells to blank with `# YOUR CODE` markers:
- Step 3 (MLP): blank the forward pass; provide skeleton class
- Step 4 (CNN): blank the training loop body (zero_grad / forward / loss / backward / step)
- Step 7 (Grad-CAM): blank the saliency function; provide signature and one-line description

Add markdown cells before each blank with hints (1-2 sentences, not the full answer).

- [ ] Write builder
- [ ] Build, eyeball
- [ ] Commit `Add Day 1 Track B notebook (implement gaps)`

### Task 7: Day 1 Track A notebook (predict & compare)

**Files:** `scripts/build_day1_track_a.py`, `notebooks/day1_ladder/track_a.ipynb`

Track A runs everything end-to-end. Their work is *prediction* before each cell + *analysis* after each cell.

Structure:
- Same data + augmentation cells as solution.
- Before each model cell, insert a markdown cell: **"PREDICT: what accuracy do you think this model will reach on validation? Write your guess in the cell below as a comment, then run."**
- Add a Python cell with `# my_guess_step_N = ___  # write your guess here`.
- After each model cell, insert markdown: **"How close was your guess? What surprised you?"** + a markdown cell for the student's note.
- At Step 7 (Grad-CAM): they look at 6 saliency maps and write down which one looks most suspicious and why.
- At end: a "what would you tell the doctor about this model" reflection cell.

No code blanks. Track A is about reasoning, not implementation.

- [ ] Write builder
- [ ] Build, eyeball
- [ ] Commit `Add Day 1 Track A notebook (predict & compare)`

### Task 8: Day 1 Track C notebook (build)

**Files:** `scripts/build_day1_track_c.py`, `notebooks/day1_ladder/track_c.ipynb`

Track C is mostly markdown spec. Two code cells: imports + a final empty cell.

```markdown
# Day 1 Track C — The Ladder, your build

You're at the "I can use Claude Code" track. Here's the spec.

## Goal
Build six classifiers on APTOS-2019 DR, in this order:
1. Logistic regression on flattened pixels
2. Gradient boosting on flattened pixels
3. Small MLP on flattened pixels
4. Small CNN trained from scratch on raw images
5. ResNet50 with ImageNet pretrained weights, finetuned
6. ViT-Base with ImageNet pretrained weights, finetuned

For each model, report validation accuracy. Plot a leaderboard at the end.

## Constraints
- Use Colab T4 GPU.
- The whole notebook must run end-to-end in under 60 minutes.
- Use `notebooks/day1_ladder/common.py` helpers if it saves time.
- No pre-cached weights — train everything live.

## Stretch
- Visualize first-layer CNN filters.
- Run Grad-CAM on the ResNet, find one suspicious-looking saliency case.
- Try unfreezing the ResNet and full-finetuning vs. linear-head only. Compare.

## Allowed
- Claude Code is encouraged. Treat it like a pair programmer, not autocomplete.
- The instructor's `solution.ipynb` is off-limits until after share-back.

## Done means
- Notebook runs end-to-end on a fresh Colab kernel.
- A markdown cell at the end answers: "Pick the architectural step where you saw the biggest accuracy jump. Why did it help?"
```

- [ ] Write builder + spec
- [ ] Commit `Add Day 1 Track C spec notebook`

### Task 9: Day 1 slide deck

**Files:** `slides/day1.py`

Slide outline (~25 slides, ~45 min lecture):

1. Title — AI in Healthcare, Day 1
2. Today's plan — three blocks
3. Tracks A/B/C explained, pick now
4. **Section: Why does this matter?**
5. Real deployed medical AI: 3 examples
6. Real deployed: things that go wrong
7. Today's anchor — diabetic retinopathy, the first AI screening deployed at scale
8. **Section: What is an image?**
9. Pixels → grid of numbers
10. RGB → three stacked grids
11. Voxels (briefly, for 3D)
12. Code-on-slide: `image.shape == (224, 224, 3)`
13. Augmentation effects visualized
14. **Section: What is "learning"?**
15. A classifier is a function: numbers → label
16. The ladder we'll build, all six on one slide
17. Logistic regression — the dumbest model that still works
18. Boosting — trees, but better
19. MLP — neural net basics
20. CNN — encoding spatial structure
21. ResNet — deep + transfer learning
22. ViT — attention on patches
23. The one-slide bridge: ViT eats images, LLM eats words
24. **Section: Lab plan**
25. Time-box per step + how to use Claude Code well
26. **Section: Share-back**
27. Closing: the leaderboard you just built, and what changed at each step

- [ ] Write `slides/day1.py` with all 25-27 slides
- [ ] Build + eyeball
- [ ] Commit `Add Day 1 slide deck builder`

---

## Phase 2 — Day 2: Multimodal stack (REWRITTEN)

PyRadiomics + Anthropic API + demographics → TabPFN.

### Task 10: D2 common module

**Files:** `notebooks/day2_multimodal/common.py`, `datasets/grading_rubric.json`, `datasets/synthetic_demographics.csv`

- [ ] **Step 1: PyRadiomics feature extractor**

```python
# notebooks/day2_multimodal/common.py
"""Day 2 helpers: extract image features (PyRadiomics), text features (Anthropic),
demographics. Concat into one tabular row per patient. Hand to TabPFN.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import json

APTOS_ROOT = Path(__file__).resolve().parents[2] / "datasets/raw/aptos"
RUBRIC_PATH = Path(__file__).resolve().parents[2] / "datasets/grading_rubric.json"
DEMO_PATH = Path(__file__).resolve().parents[2] / "datasets/synthetic_demographics.csv"


def extract_radiomics_features(image_path: Path) -> dict:
    """Extract PyRadiomics features from a 2D image.

    Returns a flat dict of feature_name -> float. Roughly 100 features.
    """
    import SimpleITK as sitk
    from radiomics import featureextractor
    from PIL import Image as PILImage

    pil = PILImage.open(image_path).convert("L")  # grayscale; PyRadiomics is happier with single-channel
    arr = np.array(pil, dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(arr)
    mask_arr = np.ones_like(arr, dtype=np.uint8)  # whole-image mask
    sitk_mask = sitk.GetImageFromArray(mask_arr)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("shape2D")

    out = extractor.execute(sitk_img, sitk_mask)
    return {k: float(v) for k, v in out.items() if not k.startswith("diagnostics_")}


def llm_extract_findings(report_text: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Use Anthropic API to extract structured findings from a free-text report.

    Returns dict like:
      {"hemorrhages_present": bool, "exudates_present": bool, "severity_word": str, ...}
    """
    import anthropic, os
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    schema_hint = (
        '{"hemorrhages_present": bool, "exudates_present": bool, '
        '"microaneurysms_present": bool, "macular_edema_present": bool, '
        '"severity_word": "none|mild|moderate|severe|proliferative"}'
    )
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        system=(
            "You are extracting structured findings from a fundus exam report. "
            f"Return ONLY valid JSON matching this schema: {schema_hint}"
        ),
        messages=[{"role": "user", "content": report_text}],
    )
    return json.loads(msg.content[0].text)


def build_feature_row(image_path: Path, report_text: str, demographics: dict) -> pd.Series:
    """Combine image radiomics + text features + demographics into one row."""
    img_feats = extract_radiomics_features(image_path)
    text_feats = llm_extract_findings(report_text)
    # convert booleans to ints, severity_word to ordinal
    severity_map = {"none": 0, "mild": 1, "moderate": 2, "severe": 3, "proliferative": 4}
    text_feats["severity_ord"] = severity_map.get(text_feats.pop("severity_word", "none"), 0)
    text_feats = {k: int(v) if isinstance(v, bool) else v for k, v in text_feats.items()}

    row = {**img_feats, **{f"txt_{k}": v for k, v in text_feats.items()}, **demographics}
    return pd.Series(row)


def load_synthetic_reports_and_demographics():
    """Returns DataFrame with columns id_code, report_text, age, sex, diabetes_years, diagnosis."""
    return pd.read_csv(DEMO_PATH)
```

- [ ] **Step 2: synthetic grading rubric + demographics**

Generate `grading_rubric.json` (template language for fundus reports), then a script that walks APTOS train.csv and generates a plausible report per image based on its 0-4 grade. Demographics randomized but correlated with diagnosis (older + longer diabetes duration = higher grade probability).

- [ ] **Step 3: TabPFN integration test**

```python
# scripts/smoke_d2.py
import pandas as pd
from tabpfn import TabPFNClassifier
from notebooks.day2_multimodal.common import build_feature_row, APTOS_ROOT, load_synthetic_reports_and_demographics

df = load_synthetic_reports_and_demographics()
X_rows = []
ys = []
for _, row in df.head(100).iterrows():
    img_path = APTOS_ROOT / "train_images" / f"{row['id_code']}.png"
    feat_row = build_feature_row(img_path, row["report_text"], {
        "age": row["age"], "sex_male": int(row["sex"] == "M"),
        "diabetes_years": row["diabetes_years"],
    })
    X_rows.append(feat_row)
    ys.append(row["diagnosis"])
X = pd.DataFrame(X_rows).fillna(0).values
y = np.array(ys)
clf = TabPFNClassifier()
clf.fit(X[:80], y[:80])
print("test accuracy:", (clf.predict(X[80:]) == y[80:]).mean())
```

- [ ] **Step 4: smoke run, eyeball**
- [ ] **Step 5: commit** `Add Day 2 multimodal common module + synthetic data`

### Task 11: D2 solution notebook

Walks through:
1. What's an LLM (warmup)
2. Anthropic API live demo on a sample report
3. Pick 200 APTOS images, extract PyRadiomics features (cache to disk for speed)
4. For the same 200, extract LLM features from the synthetic reports
5. Stack with demographics → tabular
6. Hand to TabPFN, report accuracy
7. Compare to D1's image-only baseline (load result from D1 leaderboard)
8. Discussion: when does multimodal help?

- [ ] Build solution notebook
- [ ] Smoke test
- [ ] Commit `Add Day 2 solution notebook (multimodal stack)`

### Task 12: D2 Track A/B/C notebooks

- **Track A (predict & compare + human-vs-model labeler):**
  Notebook runs end-to-end. Their work: label 20 fundus images themselves using the rubric (rubric.md provided). Then look at the D1 model's predictions on the same 20. Compute "your accuracy vs. the D1 model's accuracy on the same set." Find the most surprising disagreement. Write it up.
- **Track B (implement):**
  Blanked: `build_feature_row` integration, the demographic encoding, the TabPFN call. Helpers visible.
- **Track C (build):**
  Spec only. "Build a multimodal predictor that beats D1's image-only baseline by ≥3 points. Use any combination of PyRadiomics, LLM, demographics, TabPFN."

- [ ] Build all three
- [ ] Eyeball
- [ ] Commit `Add Day 2 Track A/B/C notebooks`

### Task 13: D2 slide deck

**Files:** `slides/day2.py`

Outline (~22 slides):

1. Recap: what we built D1
2. **Section: What is an LLM?**
3. Tokens, context, completion
4. The transformer block (one-slide visualization)
5. Why text matters in healthcare (notes, history, demographics)
6. Hallucination + clinical safety
7. **Section: The Anthropic API live demo**
8. Anthropic API on a real report → JSON
9. Tool use briefly
10. **Section: Multimodal medical AI**
11. Modality 1: image features (PyRadiomics)
12. Modality 2: text features (LLM-extracted)
13. Modality 3: demographics (tabular)
14. The big idea: everything becomes a tabular row
15. **Section: TabPFN**
16. Foundation model for tabular data
17. No traditional training, just inference
18. **Section: Lab plan**
19. Track-specific work
20. Cost discipline (don't burn the API budget)
21. **Section: Share-back**
22. Capstone preview

- [ ] Write deck builder
- [ ] Build + eyeball
- [ ] Commit `Add Day 2 slide deck builder`

---

## Phase 3 — Day 3: capstone

### Task 14: project_options.md + rubric.md

Document each of the 5 options:
1. **Pneumonia chest X-ray** (RSNA dataset)
2. **Skin lesion** (HAM10000)
3. **ECG arrhythmia** (PTB-XL)
4. **MedMNIST cross-modality** (any sub-dataset)
5. **Multimodal extension** (Open-i CXR, extend the D2 stack)

For each: dataset URL + size, baseline goal, stretch goal, Track A/B/C expectations, one "watch out for" caveat. Pull from syllabus.md.

Rubric.md codifies the 5-point rubric with example presentations at score 1, 3, 5.

- [ ] Write both
- [ ] Commit `Add capstone project options and rubric`

### Task 15: capstone starter kits

Five folders: `pneumonia_rsna/`, `skin_ham10000/`, `ecg_ptbxl/`, `medmnist_choose/`, `multimodal_extension/`.

Per kit:
- `starter_a.ipynb` — fully runnable baseline (a working but improvable model)
- `starter_b.ipynb` — data loading + eval scaffolding, model is the gap
- `spec_c.md` — short spec for Track C builders

Build effort: ~5 hr per kit, ~25 hr total. Worth it given Outset's quality bar.

- [ ] Build 5 kits (one builder script per kit)
- [ ] Smoke test all `starter_a.ipynb`
- [ ] Commit `Add 5 capstone starter kits`

### Task 16: D3 slide deck

Short — ~12 slides, mostly work time.

1. Recap D1+D2
2. Capstone format: pairs, options, rubric
3-7. The 5 options (one per slide)
8. Using Claude Code well — 3 tips
9. Time-boxing the sprint
10. Presentation format (3 min, 5-point rubric)
11. What now? Where to keep learning.
12. Closing.

- [ ] Build + eyeball + commit `Add Day 3 slide deck builder`

---

## Phase 4 — Polish

### Task 17: Final smoke + README refresh

- [ ] Run full smoke: every executable notebook end-to-end
- [ ] Build all 3 decks
- [ ] Refresh README with delivered materials
- [ ] Update prep/hours.md
- [ ] Commit `Polish: full smoke + README refresh`

### Task 18: Pre-class warmup notebook (NEW)

Single notebook to send students 1-2 days before D1. Contains:
- Colab account check
- Pip install warmup
- HF cache warmup (downloads ResNet + ViT)
- Optional: a tiny "look at one fundus image" cell as a teaser

- [ ] Write notebook
- [ ] Commit `Add pre-class warmup notebook`

---

## Self-review checklist

**Spec coverage:**
- [x] D1: 6-model ladder on APTOS DR (Tasks 4-9)
- [x] D2: PyRadiomics + LLM + TabPFN (Tasks 10-13)
- [x] D3: 5 capstone kits (Tasks 14-16)
- [x] Track A (predict & compare): D1 Task 7, D2 Task 12
- [x] Track B (implement): D1 Task 6, D2 Task 12
- [x] Track C (spec): D1 Task 8, D2 Task 12, D3 Task 14
- [x] Slides via python-pptx: Tasks 9, 13, 16
- [x] Brand theme: Task 1
- [x] Smoke testing: Task 2
- [x] Datasets without PhysioNet credentialing: APTOS (Kaggle), Open-i (free), HAM10000 (Kaggle), MedMNIST (pip)
- [x] Pre-class warmup: Tasks 3.5, 18
- [x] No pre-course tier quiz (per instructor decision)

**Type / API consistency:**
- `common.evaluate_classifier(predict_fn, loader)` is the eval signature for D1.
- `common.build_feature_row(image_path, report_text, demographics)` is the D2 row builder.
- `nbutil.{md, code, save, new_nb}` is the notebook construction API everywhere.
- `theme.{title_slide, content_slide, section_divider}` used by all three slide builders.

**Prep budget:** This v2 is more ambitious than v1. Rough estimate: 35-50 hr total. Instructor said "don't worry about cap, want quality" — proceed.
