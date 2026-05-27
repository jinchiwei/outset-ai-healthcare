# AI in Healthcare — 3-Session Course

**Outset summer 2026** · Mon Jul 6 – Wed Jul 8, 2026 · 2:30–5:00pm PT each day · taught by Jinchi Wei

## Course promise

Three afternoons. By Wednesday, every student will have built five different machine-learning models on real medical images, used a foundation model on real clinical text, and shipped a capstone project they can put on GitHub. Students who arrive having never written Python leave understanding what an AI model actually is and why it works.

## How the days fit together

| Day | Pitch | What you build |
|-----|-------|----------------|
| Mon Jul 6 | "What is an image, what is learning, and how did we get from logistic regression to vision transformers?" | A 5-model ladder on diabetic retinopathy: logreg → MLP → CNN → ResNet → ViT |
| Tue Jul 7 | "Transformers aren't just for images. Here's how they read clinical text, and how to combine three signals at once." | A multimodal predictor on chest X-ray: PyRadiomics features + LLM-extracted text features + demographics → TabPFN |
| Wed Jul 8 | "Pick a problem, build something, present it." | A capstone you can show to anyone, in pairs, with a 3-minute presentation |

The progression is intentional. D1 ends with a vision transformer. D2 opens with: *transformers aren't just for images, they're how LLMs work too.* The dataset shifts from fundus (where end-to-end deep learning crushes it) to chest X-ray (where the workflow is different — clinicians have notes, radiomics is on-domain, and a foundation model on tabular data can compete).

## Audience and pace

High school students, mixed coding background. Every student has a Claude Pro subscription.

**Each day is one lab notebook with `# TODO` blanks to fill in, plus a paired solution notebook** released after the session. This is the standard ML-lab format (MIT 6.S191, fast.ai). Students fill in the missing pieces: define the conv layer, write the training loop, complete the eval call. Each blank is small, specific, and calibrated to teach one concept. Nobody stares at an empty cell with no idea where to start.

No tier system, no labels, no pre-course placement. The format self-levels: a confident student fills blanks fast and moves to stretch goals; a true beginner leans on Claude and the instructor and still gets through, understanding each piece.

## How we use Claude

Every student has Claude Pro. We treat it as what it is in the real world: an auto-programming tool that takes you from an idea to a working result. Not the subject of the course, not a crutch that writes everything. The tool a working ML engineer reaches for when stuck.

- **D1 and D2** are scaffolded TODO labs. Claude is your pair programmer when a blank stumps you. Early on D1, the instructor spends ~3 minutes modeling *good* Claude use on one TODO: how to ask, how to sanity-check the answer, why you should understand what you paste. Then gets out of the way.
- **D3 capstone** is where Claude becomes the full deployment tool: here's a dataset and a goal, build the whole thing. The real workflow.

The rule we say out loud: *use Claude freely, but always be able to explain what your code does.* The D3 rubric checks for it.

## Day 1 — Mon Jul 6: from pixels to vision transformers
**Anchor: APTOS-2019 diabetic retinopathy (color fundus photos, 5-class severity). The first AI in medicine deployed at scale.**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 2:45 | Welcome + course shape | What we'll do over 3 days, how to ask for help, Colab setup check. |
| 2:45 – 3:30 | Lecture: what is an image, what is learning | Clinical motivation (Google's DR screening in India). Pixels → numbers → arrays → channels. What a classifier is. High-altitude tour: logreg → MLP → CNN → ResNet → ViT, with the *why* of each. |
| 3:30 – 3:40 | Break | |
| 3:40 – 4:50 | Lab: the ladder | Same dataset, five models, trained live on Colab T4 GPU. One TODO-blank notebook; fill in the missing pieces of each model. | 
| 4:50 – 5:00 | Share-back + bridge to D2 | A few volunteer findings. Then a 5-min walkthrough of the ViT-as-LLM bridge: image patch → patch embedding → attention vs. word → token embedding → attention. Same architecture, different modality. See you tomorrow. |

**The Ladder Notebook (lab structure):** one notebook, `# TODO` blanks at each step (define the conv layer, write the training loop body, complete the eval call), paired with a `_solution.ipynb` released after class.

| # | Model | What students see + fill in | Approx live time |
|---|-------|-------------------|------------------|
| 0 | Look at the data | Fundus images. RGB channel split. Pixel histograms. Augmentation effects (rotation, flip, brightness, normalization) shown side-by-side. | 5 min |
| 1 | Logistic regression | Subsampled images (64×64, ~12K features) for speed. Flatten → logreg. Baseline accuracy. "Simple linear models can't see structure." | 6 min |
| 2 | MLP | Same subsampled flat features, 2-3 hidden layers. Slight improvement, plateau. "Neural nets without the right inductive bias still struggle." | 8 min |
| 3 | Small CNN from scratch | Raw images at 224×224 now. Big jump. "Convolutions encode spatial structure." Visualize early-layer filters. | 12 min |
| 4 | ResNet50 finetune | ImageNet-pretrained backbone, finetuned on DR. Another big jump. "Standing on ImageNet's shoulders." | 15 min |
| 5 | ViT-Base finetune | Vision Transformer, same task. Comparable or better. Saliency / Grad-CAM on the ViT to close out: where is it looking? | 18 min |

All training happens live on Colab T4. No pre-cached weights (a pre-class warmup notebook downloads ResNet + ViT weights to populate the HF cache so the lab doesn't stall on a 330MB download). Students see loss curves, watch numbers improve.

**Stretch (for fast finishers):** find one image where the ViT was confidently wrong and the ResNet was confidently right (or vice versa). What's different about it?

**Learning outcomes**:
- Explain what an image is at the pixel/array level.
- Articulate why each architectural step exists (logreg → MLP → CNN → ResNet → ViT).
- Read a confusion matrix; explain why accuracy alone is misleading in healthcare.
- Find one failure mode visually using a saliency map.

**Datasets**: APTOS-2019 (Kaggle, free, no credentialing).

## Day 2 — Tue Jul 7: LLMs and multimodal medical AI
**Anchor: Open-i Indiana University chest X-ray, with real radiologist reports.**

The dataset shift from D1 is intentional. Fundus is where end-to-end deep learning is the right hammer. Chest X-ray is where the historical workflow looked very different — clinicians have written reports, radiologists used handcrafted measurements, and a modern multimodal stack has to combine them. Today we build that stack.

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 3:00 | Lecture | Transformers, recap. What an LLM is — tokens, context, completion. Why text matters in clinical care (notes, reports, history). Hallucination and clinical safety. Foundation models everywhere. |
| 3:00 – 3:20 | Live demo | Anthropic API extracting structured findings from a real Open-i report. Tool use. Cost discipline. *(See "LLM access" below — this is the only live API call of the day.)* |
| 3:20 – 3:30 | Break | |
| 3:30 – 4:50 | Lab: multimodal stack | TODO-blank notebook: fill in the feature concatenation, the TabPFN fit/predict, the baseline comparison. PyRadiomics features (CXR image) + cached LLM features (real reports) + demographics → TabPFN. | 
| 4:50 – 5:00 | Share-back + capstone preview | One volunteer shares their best multimodal vs. unimodal delta. Capstone options revealed. |

**The Multimodal Stack:**

The lab teaches one big idea: **everything becomes a tabular row, then a foundation model handles it.**

1. **Image features** — PyRadiomics extracts handcrafted features (intensity, shape, texture statistics) from the chest X-ray. Output: a fixed-length vector per patient. Radiomics on grayscale CXR is on-domain; this is what radiologists historically did with these images, formalized.
2. **Text features** — Anthropic API has parsed an associated report (real Open-i radiology report) into a structured JSON of findings (effusion present, opacity location, etc.). **Pre-cached by the instructor — students load JSON, no API key needed.**
3. **Demographics** — age, sex, comorbidities. Tabular.
4. **Concatenate** the three → one wide tabular row per patient.
5. **TabPFN** — a 2023+ foundation model for tabular data. No traditional training. Run inference, compare to a CXR-image-only baseline (e.g., a pretrained ResNet head on the same images).

**Honest discussion at the end:** the LLM-extracted features from a radiology report can leak the answer (the report often *names* the finding we're trying to predict). This is a real problem in clinical ML — what counts as data, what counts as a label, what's a clinician's note that postdates the imaging. We'll surface this and discuss what a fair evaluation would look like.

**Stretch:** try predicting a finding the report doesn't directly mention (e.g., predict patient age bin from imaging features alone, or predict a comorbidity).

**Learning outcomes**:
- Explain what an LLM does at the token level.
- Read a structured-extraction prompt and an LLM JSON output.
- Articulate the "everything becomes tabular" insight.
- Use a foundation model for tabular prediction (TabPFN) and explain why it's different from gradient boosting.
- Recognize a target-leakage failure mode in clinical ML.

**Datasets**: Open-i Indiana University CXR (free, no credentialing). Pre-cached LLM extractions committed to repo.

## LLM access (D2)

Anthropic API costs money. Students are on free Colab and don't have API keys. The course handles this in three layers:

1. **Pre-cached LLM outputs.** Instructor pre-runs the structured-extraction prompt over all D2 lab reports using their own API key. The JSON outputs are committed to the repo (`datasets/openi_llm_extractions.json`). Students load the JSON in the lab notebook. Zero student-side API cost. Deterministic.
2. **One live demo cell** during the lecture portion uses the instructor's key (loaded from a Colab secret, hard-capped at one call per session). Students see the actual API round-trip happen once, then use cached for the rest.
3. **Optional advanced section** for students who want to try LLM inference themselves: load a small open-source instruction-tuned model (e.g., Qwen 2.5 7B Instruct) on the Colab T4 GPU and run extraction locally. Free, no key needed, slower. This is stretch material.

## Day 3 — Wed Jul 8: capstone
**Format: pairs, open-ended, presentations at 4:30.**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 2:50 | Capstone kickoff | Recap of options, pair formation, pick a project. |
| 2:50 – 4:25 | Build sprint | Pairs work. Instructor circulates. Two check-ins (3:15, 4:00). |
| 4:25 – 4:30 | Reset | |
| 4:30 – 4:55 | Presentations | 3 min/pair: what you built, one finding, one limitation. |
| 4:55 – 5:00 | Closing | "What now?" — concrete next projects, papers to read, ways to keep learning. |

**Capstone options (3):**

1. **Pneumonia chest X-ray** — RSNA dataset, full resolution. Build a classifier; measure performance across patient subgroups.
2. **Skin lesion triage** — HAM10000. Melanoma vs. benign. Stretch: run on a phone-camera photo and see what happens.
3. **MedMNIST cross-modality** — pick any sub-dataset (organ, tissue, retina, derma, blood, path). Ship a baseline + one principled improvement.

Pairs that want to go off-menu (ECG, multimodal extension, their own dataset) can propose at 2:30; if the instructor green-lights it, they go.

Each option ships with a starter notebook that has data loading, a working baseline, and "improve me" markers. Pairs decide how deep to go.

**Presentation rubric (5 points, 1 each):**

1. **Built it**: ran end-to-end, you can show the model in action.
2. **Evaluated honestly**: held-out data, the right metric, sensible baseline.
3. **Found a failure mode**: at least one case where the model was wrong, and you can explain why.
4. **Defended a design decision**: each partner names one choice they personally made and defends it.
5. **Both contributed**: each partner can answer a question about the other's part of the work.

## Materials students leave with

- Three notebooks they filled in and ran end-to-end (D1 ladder, D2 multimodal, capstone), plus the solution notebooks.
- A capstone they can put on GitHub.
- A "what's next" page: 3 concrete next projects, links to Kaggle competitions, recommended papers, instructor's preferred contact for follow-up.

## Materials Outset gets

- Slides per session (`.pptx`, generated via python-pptx).
- All lab notebooks (TODO-blank) + solution notebooks + capstone starter kits in the course repo.
- Syllabus, capstone options, rubric.
- Optional: a short demo-reel video of student capstone presentations (with consent).

## Risks and mitigations

- **Colab GPU lottery** — free-tier Colab usually gives a T4; sometimes it doesn't. Mitigation: instructor uses Colab Pro for the day to share GPU runtimes if a student lands CPU-only. Pre-class warmup notebook downloads pretrained ResNet/ViT weights to populate the HF cache.
- **D1 lab time pressure** — five models in 70 min. Mitigation: instructor pre-validates total wall-clock on free-tier T4 (not Pro) the week before. Subsampled images for the flat-feature steps. Hard time-box per step.
- **D2 LLM cost** — solved by pre-caching extractions. Single live demo cell uses instructor key, hard-capped.
- **Datasets requiring auth** — APTOS and HAM10000 use Kaggle (free account). Open-i needs nothing. PTB-XL (capstone alternative if a pair proposes ECG) needs a free PhysioNet account.
- **D3 time overruns** — strict 4:25 stop. Presentations are 3 min hard.
