# AI in Healthcare — 3-Session Course

**Outset summer 2026** · Mon Jul 6 – Wed Jul 8, 2026 · 2:30–5:00pm PT each day · taught by Jinchi Wei

## Course promise

Three afternoons. By Wednesday, every student will have built six different machine-learning models on real medical images, used a foundation model on real clinical text, and shipped a capstone project they can put on GitHub. Each tier does *real* work, not a watered-down version. Students who arrive having never written Python leave understanding what an AI model actually is and why it works.

## How the days fit together

| Day | Pitch | What you build |
|-----|-------|----------------|
| Mon Jul 6 | "What is an image, what is learning, and how did we get from logistic regression to vision transformers?" | A 6-model ladder on diabetic retinopathy: logreg → boosting → MLP → CNN → ResNet → ViT |
| Tue Jul 7 | "Transformers aren't just for images. Here's how they read clinical text, and how to combine three signals at once." | A multimodal predictor: PyRadiomics features + LLM-extracted text features + demographics → TabPFN |
| Wed Jul 8 | "Pick a problem, build something, present it." | A capstone you can show to anyone, in pairs, with a 3-minute presentation |

The progression is intentional. D1 ends with a vision transformer. D2 opens with: *transformers aren't just for images, they're how LLMs work too.* D2 ends with a multimodal stack. D3 lets students pick which thread to pull on.

## Audience and tracks

High school students, mixed coding background. Each lab ships in three variants. Students self-place at the start of D1 and may switch tracks between days.

| Track | Profile | What their notebook looks like |
|-------|---------|-------------------------------|
| Track A — Predict & Compare | Never coded, or new to ML | All cells run end-to-end. Their work is to **predict the accuracy of each model before running it**, write down why, then check. At the saliency-map step, find one weird case and explain it. *Real ML skill: error analysis and intuition building.* |
| Track B — Implement | Comfortable with Python, may not know ML | Notebook has implementation gaps. Students write the MLP forward pass, the CNN training loop, the saliency function. Helper functions are provided. |
| Track C — Build | Can use Claude Code, wants to design | A spec, a dataset path, and a target metric. Students implement the whole ladder themselves. Encouraged to use Claude Code as a pair programmer. |

Same end goal across tracks: every student goes through the full pedagogical arc and leaves with the same conceptual understanding. Tracks differ in *what kind of work* the student does, not in ambition. Track A's "predict-then-run + error analysis" is real ML work, not consolation. Track A's share-back is the highlight slot, not the throwaway.

## Day 1 — Mon Jul 6: from pixels to vision transformers
**Anchor: APTOS-2019 diabetic retinopathy (color fundus photos, 5-class severity). The first AI in medicine deployed at scale.**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 2:45 | Welcome + track self-pick | Three tracks described in plain language; students pick one. They can switch tomorrow. |
| 2:45 – 3:30 | Lecture: what is an image, what is learning | Clinical motivation (Google's DR screening in India). Pixels → numbers → arrays → channels. What a classifier is. High-altitude tour: logreg → boosting → MLP → CNN → ResNet → ViT, with the *why* of each. |
| 3:30 – 3:40 | Break | |
| 3:40 – 4:50 | Lab: the ladder | Same dataset, six models, all trained live on Colab GPU. Each tier walks through the whole ladder. See section below. |
| 4:50 – 5:00 | Share-back + bridge to D2 | One finding per track. Bridge: "ViT eats images, LLM eats words, same architecture. See you tomorrow." |

**The Ladder Notebook (lab structure):**

| # | Model | What students see | Approx live time |
|---|-------|-------------------|------------------|
| 0 | Look at the data | Fundus images. RGB channel split. Pixel histograms. Augmentation effects (rotation, flip, brightness, normalization) shown side-by-side. | 5 min |
| 1 | Logistic regression | Flatten image → ~150K features → logreg. Baseline accuracy. "Simple models can't see structure." | 5 min |
| 2 | Gradient boosting | Same flat features, XGBoost or LightGBM. Modest improvement. "Trees model nonlinearity but still don't see space." | 5 min |
| 3 | MLP | Same flat features, 2-3 hidden layers. Slight improvement, plateau. "Neural nets without the right inductive bias still struggle." | 8 min |
| 4 | Small CNN from scratch | Raw image as input now. Big jump. "Convolutions encode spatial structure." Visualize early-layer filters. | 12 min |
| 5 | ResNet50 finetune | ImageNet-pretrained backbone, finetuned on DR. Another big jump. "Standing on ImageNet's shoulders." | 15 min |
| 6 | ViT-Base finetune | Vision Transformer, same task. Comparable or better. *Bridge to D2: this same architecture eats words too.* | 18 min |
| 7 | What did it learn? | Saliency / Grad-CAM on the ResNet. Where is the model looking? Any shortcut-learning red flags? | 10 min |

All training happens live on Colab T4. No pre-cached weights. Students see loss curves, hear GPU fans (metaphorically), watch numbers improve.

**Learning outcomes**:
- Explain what an image is at the pixel/array level.
- Articulate why each architectural step exists (logreg → boosting → MLP → CNN → ResNet → ViT).
- Read a confusion matrix; explain why accuracy alone is misleading in healthcare.
- Find one failure mode visually using a saliency map.

**Datasets**: APTOS-2019 (Kaggle, free, no credentialing).

## Day 2 — Tue Jul 7: LLMs and multimodal medical AI
**Anchor: a multimodal predictor combining image features, LLM-extracted text features, and demographics.**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 3:00 | Lecture | Transformers, recap. What an LLM is — tokens, context, completion. Why text matters in clinical care (notes, reports, history). Hallucination and clinical safety. Foundation models everywhere. |
| 3:00 – 3:20 | Live demo | Anthropic API extracting structured findings from a free-text report. Tool use. Cost discipline. |
| 3:20 – 3:30 | Break | |
| 3:30 – 4:50 | Lab: multimodal stack | PyRadiomics features (image) + Anthropic API features (text) + demographics → TabPFN. Compare to D1 image-only. | 
| 4:50 – 5:00 | Share-back + capstone preview | One pair shares their best multimodal vs. unimodal delta. Capstone options revealed. |

**The Multimodal Stack:**

The lab teaches one big idea: **everything becomes a tabular row, then a foundation model handles it.**

1. **Image features** — PyRadiomics extracts handcrafted features (intensity, shape, texture statistics). Output: a fixed-length vector per image.
2. **Text features** — Anthropic API parses an associated report (or structured grading rubric for fundus DR) into a structured JSON of findings. Convert to a fixed-length vector.
3. **Demographics** — age, prior diagnoses, structured clinical info. Tabular.
4. **Concatenate** the three → one wide tabular row per patient.
5. **TabPFN** — a 2023+ foundation model for tabular data. No traditional training. Run inference, compare to the D1 unimodal baseline.

Track-specific work:
- *Track A:* notebook runs end-to-end. Their job is **human-vs-model labeling on D1's outputs**. First, label 20 fundus images themselves using the rubric. Then look at the D1 model's predictions. Notebook auto-computes "your accuracy vs. model accuracy on the same set." Find the most surprising disagreement.
- *Track B:* implement the feature concatenation and TabPFN integration. Helpers provided for PyRadiomics + Anthropic API calls.
- *Track C:* build the whole multimodal pipeline. Beat the image-only D1 baseline by ≥3 points on the metric of their choice.

**Learning outcomes**:
- Explain what an LLM does at the token level.
- Use the Anthropic API to extract structured features from clinical text.
- Articulate the "everything becomes tabular" insight.
- Use a foundation model for tabular prediction (TabPFN) and explain why it's different from gradient boosting.
- Name one risk specific to LLMs in clinical contexts.

**Datasets**: APTOS-2019 (extending D1) plus structured grading rubric and synthetic demographic data. Open-i CXR available as a stretch dataset.

## Day 3 — Wed Jul 8: capstone
**Format: pairs, open-ended, presentations at 4:30.**

| Time | Block | What happens |
|------|-------|-------------|
| 2:30 – 2:50 | Capstone kickoff | Recap of options, pair formation, pick a project. |
| 2:50 – 4:25 | Build sprint | Pairs work. Instructor circulates. Two check-ins (3:15, 4:00). |
| 4:25 – 4:30 | Reset | |
| 4:30 – 4:55 | Presentations | 3 min/pair: what you built, one finding, one limitation. |
| 4:55 – 5:00 | Closing | "What now?" — concrete next projects, papers to read, ways to keep learning. |

**Capstone options (5):**

1. **Pneumonia chest X-ray** — RSNA dataset, full resolution. Build a classifier; measure performance across patient subgroups.
2. **Skin lesion triage** — HAM10000. Melanoma vs. benign. Stretch: run on a phone-camera photo and see what happens.
3. **ECG arrhythmia detection** — PTB-XL subset. 12-lead ECG as time series; classify rhythm.
4. **MedMNIST cross-modality** — pick any sub-dataset (organ, tissue, retina, derma, blood, path). Ship a baseline + one principled improvement.
5. **Multimodal extension** — extend the D2 stack to a new dataset (Open-i CXR or your own). Beat unimodal.

**Track expectations**:
- *Track A:* starter-kit notebook with a working baseline pre-trained. Goal: characterize one failure mode they didn't expect, plus one principled comparison.
- *Track B:* spec sheet + scaffolded data loading. Goal: build a model from scratch and analyze it honestly.
- *Track C:* problem statement only. Encouraged to use Claude Code. Goal: clean implementation, principled evaluation choice, defended design decisions.

**Presentation rubric (5 points, 1 each):**

1. **Built it**: ran end-to-end, you can show the model in action.
2. **Evaluated honestly**: held-out data, the right metric, sensible baseline.
3. **Found a failure mode**: at least one case where the model was wrong, and you can explain why.
4. **Defended a design decision**: each partner names one choice they personally made and defends it.
5. **Both contributed**: each partner can answer a question about the other's part of the work.

## Materials students leave with

- Three notebooks they ran end-to-end (D1 ladder, D2 multimodal, capstone).
- A capstone they can put on GitHub.
- A "what's next" page: 3 concrete next projects, links to Kaggle competitions, recommended papers, instructor's preferred contact for follow-up.

## Materials Outset gets

- Slides per session (`.pptx`, generated via python-pptx).
- All notebooks (Track A/B/C + solutions) in the course repo.
- Syllabus, capstone options, rubric.
- Optional: a short demo-reel video of student capstone presentations (with consent).

## Risks and mitigations

- **Colab GPU lottery** — free-tier Colab usually gives a T4; sometimes it doesn't. Mitigation: instructor uses Colab Pro for the day to share GPU runtimes if a student lands CPU-only. Pre-class warmup notebook downloads pretrained ResNet/ViT weights to populate the HF cache.
- **Wide skill range** — three tracks, with track mobility between days. Track A is reframed as *different* work, not less work — error analysis and prediction are real ML skills.
- **API costs (D2)** — instructor distributes a single shared Anthropic key with a hard usage cap; expensive cells can be pre-cached if needed.
- **Datasets requiring auth** — APTOS and HAM10000 use Kaggle (free account). PTB-XL needs a free PhysioNet account, doable. Open-i needs nothing.
- **D1 lab time pressure** — six models in 70 min. Mitigation: hard time-box each step. Instructor pre-validates total compute time on Colab T4 the night before.
- **D3 time overruns** — strict 4:25 stop. Presentations are 3 min hard.
