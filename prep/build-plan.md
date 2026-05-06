# Outset AI in Healthcare — Build Plan (v3)

> **For agentic workers:** Use the standard subagent-driven-development pattern to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. "Test" = "artifact runs end-to-end and produces expected output."

**Goal:** Build all course materials (slides, notebooks, capstone kits) for the 3-session AI in Healthcare course defined in `syllabus.md`, ready for delivery Mon Jul 6 – Wed Jul 8, 2026.

**Architecture:** Each day ships one slide deck (`.pptx` via python-pptx) and **one main notebook**. No tier system. D1 is a 5-model ladder (logreg → ViT) on APTOS-2019 fundus DR. D2 is PyRadiomics + cached LLM features + demographics → TabPFN, on Open-i chest X-ray. D3 ships 3 capstone starter kits. A smoke-test script runs every executable notebook headless.

**v3 deltas from v2:**
- Track A/B/C system removed. One notebook per day, everyone builds.
- D1 dropped XGBoost (5 models now: logreg, MLP, CNN, ResNet, ViT).
- D1 flat-feature steps use subsampled images (64×64) for speed.
- D2 anchor swapped from APTOS to Open-i chest X-ray (real images + real reports, no synthetic-report leakage).
- D2 LLM access via pre-cached extractions (instructor pre-runs Anthropic API, commits JSON).
- Capstone kits cut from 5 to 3 (pneumonia, skin, MedMNIST).
- Removed: tier-quiz materials, Anthropic safety demo (modern models handle most adversarial cases reasonably).

**Tech Stack:** Python 3.12 (conda env `outset`), python-pptx, jupyter/nbformat, torch + torchvision, timm, scikit-learn, pyradiomics + SimpleITK, transformers, tabpfn, anthropic, matplotlib.

---

## File structure

```
outset-ai-healthcare/
  syllabus.md                          # done (v3)
  README.md                            # done; refresh at end
  requirements.txt                     # done (v2 deps still apply)
  .gitignore                           # done
  prep/
    hours.md                           # tracking
    build-plan.md                      # this file (v3)

  scripts/
    smoke_notebooks.py                 # run every executable .ipynb headless
    nbutil.py                          # notebook construction helpers
    build_all_slides.py                # invokes day1/day2/day3 builders
    cache_openi_llm.py                 # one-shot: instructor runs to populate cache

  slides/
    theme.py                           # brand: Geist, palette, layout helpers
    day1.py                            # builds slides/build/day1.pptx
    day2.py
    day3.py
    build/                             # gitignored output

  notebooks/
    _shared/
      colab_setup.py                   # pip install + GPU check
      pretrain_warmup.ipynb            # pre-class: warm HF cache for ResNet+ViT
    day1_ladder/
      common.py                        # APTOS loader, model factories, eval, viz
      day1.ipynb                       # the lab notebook (everyone runs this)
    day2_multimodal/
      common.py                        # Open-i loader, PyRadiomics, cached-LLM helper, TabPFN wrapper
      day2.ipynb
    day3_capstone/
      project_options.md
      rubric.md
      starter_kits/
        pneumonia_rsna/
          starter.ipynb                # working baseline + improve-me markers
          README.md
        skin_ham10000/
          starter.ipynb
          README.md
        medmnist_choose/
          starter.ipynb
          README.md

  datasets/
    README.md                          # done; update for v3 datasets
    download_aptos.py
    download_openi.py
    openi_llm_extractions.json         # pre-cached LLM output, committed
    synthetic_demographics.csv         # for Open-i (Open-i lacks demographics)
```

---

## Phase 0 — Foundations

Reusable across all three days. Build first.

### Task 1: Brand theme module

**Files:** `slides/theme.py`, `slides/test_theme.py`

Same as v1/v2 — Geist + turquoise/deeppink/amber/blueviolet palette + python-pptx layout helpers (`title_slide`, `content_slide`, `section_divider`).

- [ ] Build theme.py with palette, helpers
- [ ] Smoke test
- [ ] Commit `Add brand theme module for course slides`

### Task 2: Notebook smoke-test infrastructure

**Files:** `scripts/smoke_notebooks.py`, `scripts/nbutil.py`

With Track A/B/C gone, the smoke runner runs *every* `.ipynb` in `notebooks/**` (no skip patterns needed for student variants). Capstone starter notebooks should also run end-to-end with a working baseline.

- [ ] Build nbutil + smoke runner
- [ ] Verify it runs (no notebooks present yet → exit 0)
- [ ] Commit `Add notebook smoke-test runner and nbutil helpers`

### Task 3: Colab setup helper

**Files:** `notebooks/_shared/colab_setup.py`

```python
# notebooks/_shared/colab_setup.py
import subprocess, sys, importlib

REQUIRED = [
    "torch", "torchvision", "timm",
    "scikit-learn", "matplotlib", "seaborn", "pillow",
    "tqdm", "pandas", "numpy",
    "transformers", "tabpfn",
    "pyradiomics", "SimpleITK",
]

# Map pip name -> import name when they differ
IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "pyradiomics": "radiomics",
}

def ensure(*pkgs):
    missing = []
    for p in (pkgs or REQUIRED):
        import_name = IMPORT_NAMES.get(p, p.replace("-", "_").lower())
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

(Notice: dropped `xgboost` and `anthropic` from REQUIRED. Anthropic is only used by the *instructor's* one-shot caching script, not by student notebooks. Students use cached JSON.)

- [ ] Write helper
- [ ] Smoke test locally
- [ ] Commit `Add Colab setup with GPU check`

### Task 4: Pre-class warmup notebook

**Files:** `notebooks/_shared/pretrain_warmup.ipynb`

Send students this notebook 1-2 days before D1. Downloads ResNet50 and ViT-Base into the HF/torchvision cache so the live lab doesn't stall on weight downloads.

```python
# pretrain_warmup.ipynb (one cell)
import torchvision.models as tvm
from transformers import ViTModel, ViTImageProcessor

print("downloading ResNet50...")
_ = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
print("downloading ViT-Base (patch16-224, ImageNet)...")
_ = ViTModel.from_pretrained("google/vit-base-patch16-224")
_ = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
print("done. weights cached. see you Monday.")
```

- [ ] Write the notebook
- [ ] Verify weights pull on a fresh Colab session
- [ ] Commit `Add pre-class warmup notebook for students`

---

## Phase 1 — Day 1: The Ladder

D1 anchor is APTOS-2019 diabetic retinopathy. Same dataset across all 5 models. Live training on Colab T4. **One notebook for everyone.**

### Task 5: APTOS dataset loader + download script

**Files:** `datasets/download_aptos.py`, `notebooks/day1_ladder/common.py`

- [ ] **Step 1: Download script with Kaggle API**

```python
# datasets/download_aptos.py
"""Download APTOS-2019 from Kaggle. Requires KAGGLE_USERNAME and KAGGLE_KEY env vars
(or a ~/.kaggle/kaggle.json file). Free Kaggle account required.
"""
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
    for z in ROOT.glob("*.zip"):
        subprocess.check_call(["unzip", "-q", "-o", str(z), "-d", str(ROOT)])
    print("done.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: common.py — loader, transforms, splits, model factories**

Two loaders in `common.py`:
- `get_loaders_full(batch_size, val_frac)` → 224×224 RGB tensors, used for CNN/ResNet/ViT
- `get_loaders_subsampled(batch_size, val_frac, size=64)` → 64×64 RGB tensors, used for logreg/MLP

Plus: `flatten_for_classical(loader)` returns (X, y) numpy arrays for sklearn. Transforms include normalization and (train-only) augmentation.

Model factories:
- `make_mlp(in_features, num_classes=5, hidden=(256, 128))`
- `make_small_cnn(num_classes=5)`
- `make_resnet50(num_classes=5, pretrained=True)`
- `make_vit_base(num_classes=5, pretrained=True)` (uses timm)

Eval helper: `evaluate_classifier(predict_fn, loader, device)` → dict with accuracy, confusion matrix, per-class accuracy, predictions, labels.

Viz helpers:
- `show_rgb_split(image)` — shows R/G/B channels separately
- `show_pixel_histogram(image)` — three histograms for the three channels
- `show_augmentations(image)` — original + rotated + flipped + brightness-shifted side-by-side
- `show_first_layer_filters(model)` — visualize learned conv filters
- `gradcam(model, image, target_class)` — saliency map for a CNN/ResNet
- `attention_rollout(vit_model, image)` — attention-based saliency for ViT

- [ ] Write common.py with all helpers
- [ ] Write `test_common.py` smoke tests (subset, batch=4, 1 epoch each model)
- [ ] Run smoke
- [ ] Commit `Add Day 1 common module: APTOS loader, model factories, eval, viz`

### Task 6: Day 1 lab notebook (the ladder)

**Files:** `scripts/build_day1.py`, `notebooks/day1_ladder/day1.ipynb`

This is THE notebook everyone runs. Walks through all 5 models in order on APTOS DR, with viz between steps and stretch goals at the end.

Notebook outline:

1. Markdown: course intro, "today we build five models on the same data and watch each one improve"
2. Code: `import colab_setup; colab_setup.ensure(); colab_setup.gpu_check()`
3. Code: imports + load data + show batch
4. Markdown: **Step 0 — Look at the data**
5. Code: RGB channel split, pixel histogram, augmentation effects
6. Markdown: **Step 1 — Logistic regression**
7. Code: subsampled (64×64) flatten + sklearn `LogisticRegression(max_iter=500, solver='saga')`. Print accuracy + confusion matrix.
8. Markdown: **Step 2 — MLP**
9. Code: same subsampled features → MLP, train 5 epochs, report.
10. Markdown: **Step 3 — Small CNN from scratch**
11. Code: full 224×224 images → `make_small_cnn()`, train 5 epochs. Show first-layer filter visualizations.
12. Markdown: **Step 4 — ResNet50 (transfer learning)**
13. Code: `make_resnet50(pretrained=True)`, finetune 3 epochs, report. Run Grad-CAM on 4 sample images.
14. Markdown: **Step 5 — Vision Transformer**
15. Code: `make_vit_base(pretrained=True)`, finetune 3 epochs, report. Run attention-rollout viz on 4 images.
16. Markdown: **Leaderboard**
17. Code: bar chart of all 5 model accuracies side-by-side.
18. Markdown: **Bridge to Day 2** — image patch → patch embedding → attention vs. word → token embedding → attention. (One markdown cell with both ASCII diagrams.)
19. Markdown: **Stretch — find a disagreement**
20. Code: find images where ResNet and ViT disagree confidently. Show 4 examples with both predictions and saliency maps.

- [ ] Build notebook programmatically (mirror v1 nbutil pattern)
- [ ] Run end-to-end on a Colab-equivalent local GPU (or mark deferred to Jinchi's local validation week before course)
- [ ] Commit `Add Day 1 ladder notebook`

### Task 7: Day 1 slide deck

**Files:** `slides/day1.py`

Slide outline (~22 slides, ~45 min lecture):

1. Title — AI in Healthcare, Day 1
2. Today's plan — three blocks (lecture, lab, share-back)
3. How to ask for help — instructor circulates, no shame in asking
4. **Section: Why does this matter?**
5. Real deployed medical AI: 3 examples (DR screening, mammo triage, sepsis warning)
6. Real deployed: things that go wrong (shortcut learning, distribution shift)
7. Today's anchor — diabetic retinopathy, the first AI screening deployed at scale
8. **Section: What is an image?**
9. Pixels → grid of numbers
10. RGB → three stacked grids
11. Voxels (briefly, for 3D)
12. Code-on-slide: `image.shape == (224, 224, 3)`
13. Augmentation effects visualized
14. **Section: What is "learning"?**
15. A classifier is a function: numbers → label
16. The ladder we'll build, all five on one slide
17. Logistic regression — the dumbest model that still works
18. MLP — neural net basics
19. CNN — encoding spatial structure
20. ResNet — deep + transfer learning
21. ViT — attention on patches
22. **Section: Lab plan**
23. Time-box per step + how to ask for help
24. **Section: Share-back & bridge**
25. The leaderboard you just built
26. **The bridge**: image patches vs. words. Side-by-side. Same architecture, different modality. *This slide gets 5 minutes, not one sentence.*
27. See you tomorrow

- [ ] Write `slides/day1.py` with all slides
- [ ] Build + eyeball
- [ ] Commit `Add Day 1 slide deck builder`

---

## Phase 2 — Day 2: Multimodal stack on chest X-ray

D2 anchor pivots to **Open-i Indiana University CXR**. Real chest X-ray images + real radiologist reports + MeSH labels. No PhysioNet credentialing. PyRadiomics is on-domain on grayscale CXR.

### Task 8: Open-i loader + cached LLM extractions + demographics

**Files:** `datasets/download_openi.py`, `notebooks/day2_multimodal/common.py`, `scripts/cache_openi_llm.py`, `datasets/openi_llm_extractions.json`, `datasets/synthetic_demographics.csv`

- [ ] **Step 1: Download script for Open-i**

Open-i has a public XML+image archive (~1.5GB images, ~50MB reports). Download once, cache locally.

```python
# datasets/download_openi.py
"""Download Open-i Indiana University CXR. No PhysioNet, no Kaggle, no auth."""
import urllib.request, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets/raw/openi"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"

def fetch(url, dest):
    if dest.exists(): return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)

def extract(archive, into):
    if (into / ".extracted").exists(): return
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(into)
    (into / ".extracted").touch()

def main():
    fetch(IMAGES_URL,  ROOT / "NLMCXR_png.tgz")
    fetch(REPORTS_URL, ROOT / "NLMCXR_reports.tgz")
    extract(ROOT / "NLMCXR_png.tgz",     ROOT / "images")
    extract(ROOT / "NLMCXR_reports.tgz", ROOT / "reports")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: common.py — image loader, report parser, PyRadiomics helper, TabPFN wrapper**

```python
# notebooks/day2_multimodal/common.py
"""Day 2 helpers: Open-i CXR loader, PyRadiomics extractor, cached LLM features,
demographics, TabPFN integration.
"""
from __future__ import annotations
from pathlib import Path
import json, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

OPENI_ROOT = Path(__file__).resolve().parents[2] / "datasets/raw/openi"
LLM_CACHE_PATH = Path(__file__).resolve().parents[2] / "datasets/openi_llm_extractions.json"
DEMO_PATH = Path(__file__).resolve().parents[2] / "datasets/synthetic_demographics.csv"


def parse_report(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    abstract = root.find(".//Abstract")
    text = {}
    if abstract is not None:
        for at in abstract.findall("AbstractText"):
            label = at.get("Label", "").lower()
            text[label] = (at.text or "").strip()
    images = [img.get("id") for img in root.findall(".//parentImage")]
    mesh_majors = []
    mesh_node = root.find(".//MeSH")
    if mesh_node is not None:
        for m in mesh_node.findall(".//major"):
            if m.text: mesh_majors.append(m.text.strip())
    return {
        "findings": text.get("findings", ""),
        "impression": text.get("impression", ""),
        "images": images,
        "mesh_majors": mesh_majors,
    }


def list_cases(target_label: str = "Cardiomegaly", n: int | None = None):
    """Yield (case_id, image_path, report_dict, label) triples.
    label = 1 if target_label appears in MeSH majors (positive case).
    """
    reports_dir = OPENI_ROOT / "reports/ecgen-radiology"
    images_dir = OPENI_ROOT / "images/NLMCXR_png"
    cases = []
    for xml in sorted(reports_dir.glob("*.xml")):
        rec = parse_report(xml)
        if not rec["images"]: continue
        img = images_dir / f"{rec['images'][0]}.png"
        if not img.exists(): continue
        label = int(any(target_label.lower() in m.lower() for m in rec["mesh_majors"]))
        cases.append((xml.stem, img, rec, label))
        if n and len(cases) >= n: break
    return cases


def extract_radiomics(image_path: Path) -> dict:
    """PyRadiomics features on a 2D grayscale CXR. Whole-image mask.
    Drops constant features. Returns flat dict.
    """
    import SimpleITK as sitk
    from radiomics import featureextractor
    from PIL import Image as PILImage

    pil = PILImage.open(image_path).convert("L")
    arr = np.array(pil, dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(arr)
    sitk_mask = sitk.GetImageFromArray(np.ones_like(arr, dtype=np.uint8))

    ex = featureextractor.RadiomicsFeatureExtractor()
    ex.disableAllFeatures()
    ex.enableFeatureClassByName("firstorder")
    ex.enableFeatureClassByName("glcm")
    ex.enableFeatureClassByName("glrlm")

    out = ex.execute(sitk_img, sitk_mask)
    return {k: float(v) for k, v in out.items()
            if not k.startswith("diagnostics_") and not isinstance(v, (tuple, list))}


def load_cached_llm_features(case_id: str) -> dict:
    """Load pre-cached LLM extraction for a case. Returns flat dict of features."""
    cache = json.loads(LLM_CACHE_PATH.read_text())
    rec = cache.get(case_id, {})
    # Convert booleans to ints, severity words to ordinals
    severity_map = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    feats = {}
    for k, v in rec.items():
        if isinstance(v, bool): feats[f"llm_{k}"] = int(v)
        elif isinstance(v, str): feats[f"llm_{k}_ord"] = severity_map.get(v.lower(), 0)
        elif isinstance(v, (int, float)): feats[f"llm_{k}"] = v
    return feats


def load_demographics(case_id: str) -> dict:
    df = pd.read_csv(DEMO_PATH).set_index("case_id")
    if case_id in df.index:
        row = df.loc[case_id]
        return {
            "age": int(row["age"]),
            "sex_male": int(row["sex"] == "M"),
            "smoker": int(row["smoker"]),
        }
    return {"age": 50, "sex_male": 0, "smoker": 0}


def build_feature_row(case_id: str, image_path: Path) -> pd.Series:
    img = extract_radiomics(image_path)
    txt = load_cached_llm_features(case_id)
    demo = load_demographics(case_id)
    return pd.Series({**img, **txt, **demo})
```

- [ ] **Step 3: Synthetic demographics generator**

Open-i doesn't include patient demographics. Generate plausible synthetic demographics (age, sex, smoking history) per case_id with mild correlation to common findings (e.g., older + smoker → higher cardiomegaly prevalence). Commit as `datasets/synthetic_demographics.csv`. Document the generation process in a comment header so it's transparent it's synthetic.

- [ ] **Step 4: Instructor caching script (`scripts/cache_openi_llm.py`)**

One-shot script the instructor runs *once* with their own API key. Iterates over Open-i reports, extracts structured findings using Anthropic API, writes to `datasets/openi_llm_extractions.json`.

```python
# scripts/cache_openi_llm.py
"""Instructor one-shot: pre-cache Anthropic LLM extractions on Open-i reports.
Run with ANTHROPIC_API_KEY set. Costs ~$1-2 for ~200 reports with Haiku 4.5.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebooks.day2_multimodal.common import list_cases, LLM_CACHE_PATH

import anthropic
client = anthropic.Anthropic()

SCHEMA = (
    '{"cardiomegaly_present": bool, "effusion_present": bool, '
    '"opacity_present": bool, "atelectasis_present": bool, '
    '"pneumothorax_present": bool, "severity_word": "none|mild|moderate|severe"}'
)

def extract(report_text: str) -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=(
            "You are extracting structured findings from a chest X-ray radiology report. "
            f"Return ONLY valid JSON matching: {SCHEMA}"
        ),
        messages=[{"role": "user", "content": report_text}],
    )
    return json.loads(msg.content[0].text)

def main(target_label="Cardiomegaly", n=200):
    cache = {}
    for case_id, img_path, rec, label in list_cases(target_label, n=n):
        report_text = f"FINDINGS: {rec['findings']}\nIMPRESSION: {rec['impression']}"
        try:
            cache[case_id] = extract(report_text)
        except Exception as e:
            print(f"failed {case_id}: {e}")
    LLM_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"cached {len(cache)} extractions to {LLM_CACHE_PATH}")

if __name__ == "__main__":
    main()
```

**Note:** instructor runs this *once* before D2, commits the resulting JSON to the repo. Students never call the API.

- [ ] **Step 5: Smoke test the multimodal pipeline**

```python
# scripts/smoke_d2.py
import numpy as np, pandas as pd
from tabpfn import TabPFNClassifier
from notebooks.day2_multimodal.common import list_cases, build_feature_row

cases = list_cases("Cardiomegaly", n=80)
rows, ys = [], []
for case_id, img_path, _, label in cases:
    rows.append(build_feature_row(case_id, img_path))
    ys.append(label)
X = pd.DataFrame(rows).fillna(0).values
y = np.array(ys)
clf = TabPFNClassifier()
clf.fit(X[:60], y[:60])
print("smoke acc:", (clf.predict(X[60:]) == y[60:]).mean())
```

- [ ] Write all common helpers + caching script + demographics
- [ ] Run smoke (instructor pre-runs Anthropic caching first)
- [ ] Commit `Add Day 2 multimodal common module + Open-i loader + cached LLM extractions`

### Task 9: Day 2 lab notebook

**Files:** `scripts/build_day2.py`, `notebooks/day2_multimodal/day2.ipynb`

Notebook outline:

1. Markdown: D2 intro — "yesterday: end-to-end deep learning. Today: handcrafted features + foundation model on tabular."
2. Code: `colab_setup.ensure(); gpu_check()`
3. Markdown: **What's an LLM?** Show tokenization in code, walk through one prompt+completion in cells.
4. Markdown: **Live demo (the only API call of the day)**
5. Code: load instructor's API key from Colab secret, run *one* extraction on a sample Open-i report. (Hard-capped at 1 call. If the secret isn't available, fall back to showing the cached output for that report.)
6. Markdown: **Now we use cached extractions for everyone**
7. Code: load `openi_llm_extractions.json`, show one entry.
8. Markdown: **Image features (PyRadiomics)**
9. Code: extract radiomics features from one CXR image, show the feature vector.
10. Markdown: **Demographics**
11. Code: load synthetic demographics for the same case.
12. Markdown: **Combine: everything becomes a tabular row**
13. Code: build feature rows for ~150 cases, show DataFrame head.
14. Markdown: **TabPFN**
15. Code: train/test split, fit TabPFN on train, predict on test, report accuracy.
16. Markdown: **Compare to image-only baseline**
17. Code: a small pretrained ResNet head trained on the same train/test split as a unimodal baseline. Compare numbers.
18. Markdown: **Honest discussion: target leakage**
19. Code: a simple ablation — drop the LLM features, just radiomics + demographics. How does accuracy change? Discuss what the gap means.
20. Markdown: **Stretch — predict something the report doesn't directly mention** (e.g., synthetic age bin from imaging features alone). Show that's harder.
21. Markdown: capstone preview — what you'll pick tomorrow.

- [ ] Build notebook
- [ ] Smoke test
- [ ] Commit `Add Day 2 multimodal notebook`

### Task 10: Day 2 slide deck

**Files:** `slides/day2.py`

Outline (~20 slides):

1. Recap: what we built D1
2. **Section: What is an LLM?**
3. Tokens, context, completion
4. The transformer block (visualization)
5. The bridge from yesterday: ViT patches vs. LLM tokens, side-by-side
6. Why text matters in clinical care (notes, history, demographics)
7. Hallucination + clinical safety (one slide; surface but don't dwell)
8. Foundation models everywhere
9. **Section: Today's dataset shift**
10. Why we move from fundus (D1) to chest X-ray (D2)
11. Open-i — real images, real reports
12. **Section: The multimodal stack**
13. Modality 1: image features (PyRadiomics on grayscale CXR — radiomics is on-domain here)
14. Modality 2: text features (LLM-extracted from real reports)
15. Modality 3: demographics (synthetic for Open-i)
16. The big idea: everything becomes a tabular row
17. **Section: TabPFN**
18. Foundation model for tabular data
19. **Section: Lab plan**
20. Cost discipline: why we pre-cached the LLM outputs (one slide, brief)
21. Lab time-box
22. **Closing**: capstone preview

- [ ] Write deck builder
- [ ] Build + eyeball
- [ ] Commit `Add Day 2 slide deck builder`

---

## Phase 3 — Day 3: capstone

### Task 11: project_options.md + rubric.md

**Files:** `notebooks/day3_capstone/project_options.md`, `notebooks/day3_capstone/rubric.md`

Document the 3 options:
1. **Pneumonia chest X-ray** (RSNA dataset)
2. **Skin lesion** (HAM10000)
3. **MedMNIST cross-modality** (any sub-dataset)

For each: dataset URL + size, baseline goal, stretch goal, one "watch out for" caveat. Note that pairs can propose off-menu options at 2:30 (ECG, multimodal extension, their own dataset) and the instructor will green-light if reasonable.

`rubric.md` codifies the 5-point rubric with example presentations at score 1, 3, 5.

- [ ] Write both
- [ ] Commit `Add capstone project options and rubric`

### Task 12: Capstone starter kits (3)

Three folders: `pneumonia_rsna/`, `skin_ham10000/`, `medmnist_choose/`.

Per kit:
- `starter.ipynb` — fully runnable baseline (a working but improvable model). Clear "improve me" markdown markers (e.g., "try data augmentation here," "can you add class weighting?").
- `README.md` — one-page: dataset description, what the baseline does, suggested improvements, gotchas.

Build effort: ~5 hr per kit, ~15 hr total. Less than v2 because we cut from 5 to 3.

- [ ] Build 3 kits (one builder script per kit)
- [ ] Smoke test all 3 starter notebooks
- [ ] Commit `Add 3 capstone starter kits`

### Task 13: Day 3 slide deck

**Files:** `slides/day3.py`

Short — ~10 slides, mostly work time.

1. Recap D1+D2
2. Capstone format: pairs, 3 options (or propose your own), rubric
3. Option 1: Pneumonia RSNA
4. Option 2: Skin HAM10000
5. Option 3: MedMNIST
6. Using Claude Code well — 3 tips
7. Time-boxing the sprint (2:50-4:25 build, 4:30 present)
8. Presentation format (3 min, 5-point rubric)
9. What now? Where to keep learning.
10. Closing.

- [ ] Build + eyeball + commit `Add Day 3 slide deck builder`

---

## Phase 4 — Polish

### Task 14: Final smoke + README refresh

- [ ] Run full smoke: every executable notebook end-to-end
- [ ] Build all 3 decks
- [ ] Refresh README with delivered materials
- [ ] Update prep/hours.md
- [ ] Commit `Polish: full smoke + README refresh`

### Task 15: D1 wall-clock validation on free-tier T4

**Critical pre-class check, not in materials.**

Instructor runs the D1 ladder notebook end-to-end on a free-tier Colab T4 (not Pro), times each step, and confirms the lab fits in 70 min. If it doesn't, knobs to turn:
- Reduce ResNet/ViT epochs from 3 → 2
- Subsample further (224 → 192 for ResNet/ViT)
- Drop the final viz step

Run this the *week before* July 6, not the night before.

- [ ] Schedule the wall-clock test (suggest: Thu Jul 2)
- [ ] Document actual timings in `prep/hours.md`
- [ ] If overrun, apply mitigations + re-test

---

## Self-review checklist

**Spec coverage:**
- [x] D1: 5-model ladder on APTOS DR (Tasks 5-7)
- [x] D2: PyRadiomics + cached LLM + TabPFN on Open-i CXR (Tasks 8-10)
- [x] D3: 3 capstone kits + slides (Tasks 11-13)
- [x] No tier system: one notebook per day for everyone
- [x] Slides via python-pptx: Tasks 7, 10, 13
- [x] Brand theme: Task 1
- [x] Smoke testing: Task 2
- [x] Pre-class warmup: Task 4
- [x] Real ViT→LLM bridge slide (5 min): Task 7 slide 26 + Task 9 cell 18 + Task 10 slide 5
- [x] D2 LLM access via pre-cached extractions, one live demo cell: Task 8 step 4 + Task 9 cell 5
- [x] D1 wall-clock pre-validation: Task 15
- [x] Capstone cuts: 5 → 3 (Task 12)

**Type / API consistency:**
- `common.evaluate_classifier(predict_fn, loader)` (D1) and `common.build_feature_row(case_id, image_path)` (D2) are the eval/feature-row signatures.
- `nbutil.{md, code, save, new_nb}` is the notebook construction API everywhere.
- `theme.{title_slide, content_slide, section_divider}` used by all three slide builders.

**Prep budget v3:** ~25-35 hr total (down from v2's 35-50 because no tier variants). Instructor said "don't worry about cap, want quality" — proceed.
