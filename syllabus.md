# AI in Healthcare — 3-Session Course

**Outset summer 2026** · Mon Jul 6 – Wed Jul 8, 2026 · 2:30–5:00pm PT each day · taught by Jinchi Wei

## Course promise

By the end of three afternoons, every student will have **trained, evaluated, and reasoned about an AI model that diagnoses disease from real medical data**, and will leave with their own capstone project they can show to anyone.

Students arrive with anything from "never opened a Python file" to "I use Claude Code daily." All three should leave the course feeling like they did real work — not a watered-down version of it.

## Audience and tiering

High school students, mixed coding background. Each lab ships in three variants; students self-place via a 5-min quiz on D1 and may move tiers between days.

| Tier | Profile | What their notebook looks like |
|------|---------|-------------------------------|
| T1 | Never coded, or scared of it | Heavy scaffolding. Most cells pre-written. Fill-in-the-blank for key lines. Conceptual reasoning prompts ("the model missed this image. What's a hypothesis?") |
| T2 | Comfortable with Python basics, may not know ML | Partial scaffolding. Students write training/eval loops with helper functions provided. Open analysis questions. |
| T3 | Can use Claude Code; wants to build | A spec, a dataset path, and constraints. Students design and implement the whole project. Encouraged to use Claude Code as a pair programmer. |

The same end goal across tiers: *every student trains and evaluates a real model on real medical data.* Tiers differ in how much code is handed to them, not in ambition.

## Day 1 — Mon Jul 6: medical imaging
**Anchor project: pneumonia detection on chest X-ray**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 2:45 | Welcome + tier quiz | 5-min self-assessment, students pick T1/T2/T3 |
| 2:45 – 3:20 | Lecture | What is medical AI? Why imaging? How a CNN sees an x-ray. Failure modes. Real clinical deployment stories. |
| 3:20 – 3:30 | Break | |
| 3:30 – 4:45 | Lab: pneumonia classifier | T1/T2/T3 notebooks. PneumoniaMNIST primary, RSNA dataset for T3 stretch. Live coding from instructor; students follow at their tier. |
| 4:45 – 5:00 | Share-back + intro to D2 | Each tier shares one finding (a misclassification, a confusion, an idea). |

**Learning outcomes**:
- Articulate what AI in healthcare does and doesn't do well.
- Train a working CNN classifier on real medical images.
- Read a confusion matrix; explain why "accuracy" alone is dangerous in healthcare.
- Identify at least one failure mode visually.

**Datasets**: PneumoniaMNIST (primary), RSNA Pneumonia (stretch).

## Day 2 — Tue Jul 7: LLMs and multimodal
**Anchor project: combining a chest X-ray with its radiology report**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 3:00 | Lecture | What is an LLM, really? Tokenization → context → completion. Why text matters in healthcare (notes, reports, history). Hallucination and clinical safety. |
| 3:00 – 3:20 | Live demo | Use the Anthropic API to extract findings from a radiology report. Discuss tool use. |
| 3:20 – 3:30 | Break | |
| 3:30 – 4:45 | Lab: multimodal | Open-i CXR dataset. T1/T2/T3 notebooks: combine the D1 image model with text features from reports + demographic features. Predict a finding. |
| 4:45 – 5:00 | Share-back + capstone preview | Show capstone options for D3. |

**Learning outcomes**:
- Explain what an LLM is and what tokenization does.
- Use the Anthropic API (or local equivalent) to analyze clinical text.
- Combine image features and text features into a single predictor.
- Identify at least one risk specific to LLMs in clinical contexts.

**Datasets**: Open-i / Indiana University Chest X-ray (paired images + reports, no credentialing).

## Day 3 — Wed Jul 8: capstone
**Format: pairs, open-ended, presentations at 4:30**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 2:50 | Capstone kickoff | Recap of options, pair formation, pick a project. |
| 2:50 – 4:25 | Build sprint | Pairs work. Instructor circulates. Two short check-ins (3:15, 4:00). |
| 4:25 – 4:30 | Reset | |
| 4:30 – 4:55 | Presentations | 3 min/pair: what they built, one finding, one limitation. |
| 4:55 – 5:00 | Closing | What now? How to keep learning. |

**Capstone options (draft, pick 3-5)**:

1. **Skin lesion triage** — HAM10000. Build a classifier that flags melanoma vs. benign.
2. **ECG arrhythmia detection** — PTB-XL subset. Treat 12-lead ECG as time series; classify rhythm.
3. **Pneumonia, harder** — same task as D1, but on the full RSNA dataset with a held-out demographic split. Measure fairness across age groups.
4. **MedMNIST cross-modality** — pick any MedMNIST dataset (organ, tissue, derma, etc.) and ship a baseline + improvement.
5. **Multimodal triage** — use Open-i to predict a finding from image + report, beat the image-only baseline.

**Tier expectations**:
- T1 pairs use a starter kit notebook. Goal: working baseline + one improvement they can explain.
- T2 pairs use a spec sheet. Goal: from-scratch model + analysis.
- T3 pairs get a problem statement only. Encouraged to use Claude Code; goal: clean implementation + thoughtful evaluation choice.

**Presentation rubric** (5 points, 1 each):
1. Did the model train? (it ran end-to-end)
2. Did you evaluate it honestly? (held-out data, real metric)
3. Did you find one failure mode? (and can you show it)
4. Did you make one design decision and defend it?
5. Did your partner contribute? (both speak)

## Materials students leave with

- Three notebooks they ran end-to-end (D1, D2, capstone)
- A capstone they can put on GitHub
- A small "what's next" reading list (suggested papers, courses, datasets)

## Materials Outset gets

- Slides per session (`.pptx` via python-pptx)
- All notebooks (T1/T2/T3 + solutions) in this repo
- Syllabus, capstone options, rubric, this doc

## Risks and mitigations

- **Colab GPU starvation** — mitigated by anchoring on PneumoniaMNIST and small subsets. RSNA / HAM10000 are stretch.
- **Wide skill range** — mitigated by tiered notebooks and tier-mobility between days.
- **Datasets requiring auth** — avoided. PneumoniaMNIST and Open-i need no credentialing. PTB-XL needs a free PhysioNet account, doable.
- **LLM cost on D2** — capped via prompt limits in the notebook; instructor pre-runs expensive cells.
- **Time overruns on D3** — strict 4:25 stop; presentations are 3 min hard.
