# Datasets

No raw data is committed to this repo (see `.gitignore`). This directory holds **pointers and download scripts**.

## D1 — Diabetic retinopathy (fundus photos)

- **APTOS-2019 Blindness Detection** (primary): ~3,500 train images, color RGB fundus, 5-class severity grading.
  - Source: https://www.kaggle.com/competitions/aptos2019-blindness-detection
  - Requires a free Kaggle account + API token (`~/.kaggle/kaggle.json`).
  - Download: `python datasets/download_aptos.py` (after Kaggle auth).

## D2 — Multimodal (image-model vote + cached LLM + TabPFN)

D2 anchor: **Open-i Indiana University CXR** — real chest X-ray images with real radiologist reports. No credentialing required.

- `download_openi.py` — fetches images and reports from Open-i.
- `openi_image_preds.json` — **pre-computed** image-model votes. Instructor runs `scripts/cache_openi_image_preds.py` once: a transfer-learning image model (frozen ResNet18 + logistic head) produces one out-of-fold probability `img_pred` per case (each scored by a model that never trained on it, so the image feature can't leak). Late fusion / stacking. Students load this JSON; they never touch the raw X-rays.
- `openi_llm_extractions.json` — **pre-cached** Anthropic API output. Instructor runs `scripts/cache_openi_llm.py` once before D2; students load this JSON in the lab notebook (zero student-side API cost).
- `openi_features.csv` — the assembled table students load: `case_id, label, img_pred, llm_*, age, sex_male, smoker`. Built by `build_openi_features.py`.
- `synthetic_demographics.csv` — Open-i lacks demographics, so we generate plausible synthetic ones (age, sex, smoking history) per case_id, mildly correlated with common findings. Header documents the generation process.

## D3 — Capstone options (3)

- **Pneumonia chest X-ray (RSNA)**: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge (Kaggle, free)
- **Skin lesion (HAM10000)**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 (Kaggle, free)
- **MedMNIST** (cross-modality variety): pip-installable, no auth.

Pairs may propose off-menu options at the start of D3 (e.g., ECG arrhythmia on PTB-XL, multimodal extension). Instructor green-lights if reasonable.
