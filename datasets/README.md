# Datasets

No raw data is committed to this repo (see `.gitignore`). This directory holds **pointers and download scripts**.

## D1 — Pneumonia chest X-ray

- **PneumoniaMNIST** (primary): tiny, fast, no-auth. Ships via the `medmnist` pip package.
  - Source: https://medmnist.com/
- **RSNA Pneumonia Detection** (extension for T3 / capstone): larger, bounding boxes for detection.
  - Source: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge
  - Requires Kaggle account (free).

## D2 — LLMs + multimodal

- **Open-i / Indiana University Chest X-ray**: paired images + radiology reports. **No PhysioNet credentialing required.**
  - Source: https://openi.nlm.nih.gov/
- **MedQA / PubMedQA**: medical question answering for LLM exercises.
  - Source: HuggingFace `datasets`.

## D3 — Capstone options (draft)

- **HAM10000** (skin lesion classification): https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **PTB-XL** (ECG arrhythmia): https://physionet.org/content/ptb-xl/ (requires free PhysioNet account, no credentialing)
- **MedMNIST** (cross-modality variety): pip-installable.
- **NIH ChestX-ray14**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Multimodal variant: image + report from Open-i.
