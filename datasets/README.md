# Datasets

No raw data is committed to this repo (see `.gitignore`). This directory holds **pointers and download scripts**.

## D1 — Diabetic retinopathy (fundus photos)

- **APTOS-2019 Blindness Detection** (primary): ~3,500 train images, color RGB fundus, 5-class severity grading.
  - Source: https://www.kaggle.com/competitions/aptos2019-blindness-detection
  - Requires a free Kaggle account + API token (`~/.kaggle/kaggle.json`).
  - Download: `python datasets/download_aptos.py` (after Kaggle auth).

## D2 — Multimodal (PyRadiomics + LLM + TabPFN)

D2 extends APTOS-2019 with synthetic-but-plausible radiology-style reports and demographics. Files in this directory:

- `grading_rubric.json` — fundus DR grading rubric (template for synthetic report generation).
- `synthetic_demographics.csv` — generated demographics + reports per APTOS image, correlated with severity.
- `download_openi.py` — Open-i Indiana University Chest X-ray (paired images + reports), used as a **stretch dataset** for D2 Track C and the D3 multimodal extension. No PhysioNet credentialing.

## D3 — Capstone options

- **Pneumonia chest X-ray (RSNA)**: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge (Kaggle, free)
- **Skin lesion (HAM10000)**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 (Kaggle, free)
- **ECG arrhythmia (PTB-XL)**: https://physionet.org/content/ptb-xl/ (requires free PhysioNet account, no credentialing)
- **MedMNIST** (cross-modality variety): pip-installable, no auth.
- **Multimodal extension**: extend the D2 stack to Open-i CXR.
