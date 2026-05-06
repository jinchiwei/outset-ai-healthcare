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

Everyone runs the same notebook each day. No tier system. Students who arrive with zero coding get help one-on-one; students fluent with Claude Code blast through the main path and pick up stretch goals at the end.

## Repo layout

```
syllabus.md                        anchor doc — read this first
prep/build-plan.md                 task-by-task implementation plan
slides/                            python-pptx generators per session
notebooks/
  _shared/                         colab setup, pre-class warmup
  day1_ladder/                     APTOS DR + 5-model ladder
  day2_multimodal/                 Open-i CXR + PyRadiomics + LLM + TabPFN
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
python scripts/smoke_notebooks.py   # runs every executable .ipynb headless
python -m slides.day1               # builds slides/build/day1.pptx
python scripts/cache_openi_llm.py   # instructor only, one-shot, requires ANTHROPIC_API_KEY
```
