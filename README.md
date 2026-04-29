# outset-ai-healthcare

Three-session "AI in Healthcare" course for high school students, taught for Outset.

**Dates**: Mon Jul 6 – Wed Jul 8, 2026, 2:30–5:00pm PT
**Instructor**: Jinchi Wei
**Audience**: high schoolers, mixed coding background (zero through Claude-Code-fluent)

## Sessions

| Day | Focus | Anchor project |
|-----|-------|----------------|
| Mon Jul 6 | Intro + medical imaging | Pneumonia detection on chest X-ray |
| Tue Jul 7 | LLMs + multimodal | Image + radiology report (Open-i CXR) |
| Wed Jul 8 | Capstone in pairs | Pick from 3-5 project options w/ public data |

## Tiered notebooks

Each lab ships in three variants. Students self-place via a 5-min quiz on D1 and can move tiers between days.

- **T1 (no code)**: heavy scaffolding, fill-in-the-blank, reasoning prompts
- **T2 (some Python)**: partial scaffolding, students write loops/eval
- **T3 (Claude Code capable)**: spec + dataset path, design and implement

## Repo layout

```
slides/                 python-pptx generators per session
notebooks/
  day1_imaging/         pneumonia chest x-ray
  day2_llm/             LLMs + multimodal
  day3_capstone/        project options + starter kits
datasets/               download scripts and pointers (no data committed)
scripts/                helpers (e.g., notebook tier generation)
prep/                   prep notes, hour log
```

## Local setup

Conda env `outset` (Python 3.12.8) on instructor machine. Pip-only installs.

```
pip install -r requirements.txt
```

Students will use Google Colab; the local env exists for prototyping.
