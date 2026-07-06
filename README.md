# outset-ai-healthcare

Three-session "AI in Healthcare" course for high school students, taught for Outset.

**Dates**: Mon Jul 6 – Wed Jul 8, 2026, 2:30–5:00pm PT
**Instructor**: Jinchi Wei
**Audience**: high schoolers, mixed coding background (zero through Claude-Code-fluent)

## Sessions

| Day | Focus | Anchor |
|-----|-------|--------|
| Mon Jul 6 | Intro to deep learning, lab applies it | A 5-model ladder (logreg → MLP → CNN → ResNet → ViT) on APTOS-2019 fundus DR |
| Tue Jul 7 | LLMs + multimodal medical AI | Image-model vote (late fusion) + cached LLM features + demographics → TabPFN, on Open-i chest X-ray |
| Wed Jul 8 | Capstone in pairs | 3 starter kits (pneumonia, skin, MedMNIST) or propose your own |

## Course structure

Each day is one lab notebook with `# TODO` blanks to fill in, plus a paired solution notebook released after the session (MIT 6.S191 pattern). No tier system. Every student has Claude Pro and uses it as an ambient pair programmer when stuck. D1/D2 are scaffolded; D3 capstone is build-from-a-goal with Claude as the full deployment tool.

## Repo layout

```
syllabus.md                        anchor doc — read this first
prep/build-plan.md                 task-by-task implementation plan
slides/                            python-pptx generators per session
notebooks/
  _shared/                         colab setup, pre-class warmup
  day1_ladder/                     APTOS DR + 5-model ladder (lab + solution)
  day2_multimodal/                 Open-i CXR: image-model vote + LLM + TabPFN (lab + solution)
  day3_capstone/                   3 starter kits + project options + rubric
datasets/                          download scripts, pre-cached LLM JSON, no raw data
scripts/                           smoke runner, deck builder, instructor caching script
```

## Running the notebooks

### Option A — Google Colab (recommended, zero install)
Open a day's notebook in Colab and run top to bottom. The first cell clones this repo and
installs anything missing. For speed, enable a GPU: **Runtime → Change runtime type → T4 GPU**.

### Option B — one-click local (Windows / macOS)
For running on a laptop without Colab. **Download this repo** (green "Code" button → Download
ZIP → unzip), then:

- **Windows**: double-click **`START_HERE_windows.bat`** (if SmartScreen warns: More info → Run anyway)
- **macOS**: double-click **`START_HERE_mac.command`** (if blocked: right-click → Open → Open)

It installs a private Python + all the libraries (via [`uv`](https://astral.sh/uv), no prior
Python needed) and opens JupyterLab in your browser. First run takes a few minutes; after that
it's instant. Uses `requirements-student.txt` (the lean runtime set). No GPU required — the
heavier cells are tuned to run in seconds on a CPU.

### Instructor / dev env
Conda env `outset` (Python 3.12.8). Full toolchain (slides, LLM caching, etc.):
```
pip install -r requirements.txt
```

## Build commands

```
# regenerate notebooks from source (lab + solution)
python scripts/build_day1.py
python scripts/build_day2.py
python scripts/build_capstone.py

# build slide decks (-> slides/build/*.pptx)
python -m slides.day1
python -m slides.day2
python -m slides.day3

# run every solution + starter notebook headless (skips blanked labs)
python scripts/smoke_notebooks.py
```

## Instructor one-time data prep

Students never download raw datasets. The instructor builds small artifacts once:

```
# Day 1: APTOS subset -> HuggingFace (students load dreamxjei/aptos-mini)
KAGGLE_USERNAME=... KAGGLE_KEY=... python datasets/download_aptos.py
HF_TOKEN=... python datasets/prep_aptos_subset.py --per-class 400 --push

# Day 2: Open-i feature table (committed to repo; students load datasets/openi_features.csv)
python datasets/download_openi.py
ANTHROPIC_API_KEY=... python scripts/cache_openi_llm.py --mode anthropic   # or --mode rules
python datasets/make_synthetic_demographics.py
python datasets/build_openi_features.py
```

## Validated results (local, Apple MPS)

- **D1 ladder**: logreg 0.53 -> mlp 0.48 -> cnn 0.55 -> resnet 0.58 -> vit 0.64 (transfer learning is the jump)
- **D2 multimodal**: image+text+demo 0.89 vs image+demo only 0.64; leakage surfaced (report names the diagnosis, corr 0.81)
- **D3 pneumonia kit**: baseline test 0.795, improvable
