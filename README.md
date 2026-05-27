# outset-ai-healthcare

Three-session "AI in Healthcare" course for high school students, taught for Outset.

**Dates**: Mon Jul 6 – Wed Jul 8, 2026, 2:30–5:00pm PT
**Instructor**: Jinchi Wei
**Audience**: high schoolers, mixed coding background (zero through Claude-Code-fluent)

## Sessions

| Day | Focus | Anchor |
|-----|-------|--------|
| Mon Jul 6 | Intro to deep learning, lab applies it | A 5-model ladder (logreg → MLP → CNN → ResNet → ViT) on APTOS-2019 fundus DR |
| Tue Jul 7 | LLMs + multimodal medical AI | PyRadiomics + cached LLM features + demographics → TabPFN, on Open-i chest X-ray |
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
  day2_multimodal/                 Open-i CXR + PyRadiomics + LLM + TabPFN (lab + solution)
  day3_capstone/                   3 starter kits + project options + rubric
datasets/                          download scripts, pre-cached LLM JSON, no raw data
scripts/                           smoke runner, deck builder, instructor caching script
```

## Local setup

Conda env `outset` (Python 3.12.8) on instructor machine. Pip-only installs.

```
pip install -r requirements.txt
```

Students use Google Colab; the local env exists for prototyping and the one-shot LLM caching script.

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
