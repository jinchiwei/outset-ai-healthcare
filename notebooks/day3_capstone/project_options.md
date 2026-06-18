# Capstone project options

Work in pairs. Pick one of the three kits below, or propose your own at the start of the
session. Each kit has a `starter.ipynb` with a working baseline. Your job is to make it
better and understand how, then present for 3 minutes.

All three use MedMNIST: pip-installable, downloads in seconds, no account or token needed.
So you spend the sprint building, not fighting downloads.

## Option 1: Pneumonia detection (chest X-ray)
`starter_kits/pneumonia/`

Binary classification: does this chest X-ray show pneumonia? Connects straight to the
chest X-rays from Day 2. Good if you want a clear, focused yes/no problem.

**Level up:** swap in the full-resolution RSNA Pneumonia dataset from Kaggle (real
1024x1024 images, bounding boxes, real class imbalance).

## Option 2: Skin lesion triage
`starter_kits/skin/`

7-class classification of dermatoscopic skin lesion images, including melanoma. Good if
you want a multi-class problem with real stakes.

**Level up:** swap in the full HAM10000 dataset, then run your model on a phone photo of
a mole and watch how it behaves on data that looks nothing like training.

## Option 3: Choose your own
`starter_kits/choose/`

Pick any MedMNIST dataset and build on it. Defaults to RetinaMNIST (diabetic retinopathy,
like Day 1). Good if something specific caught your interest. Set `DATASET` in `config.py`
to one of these:

| `DATASET` | modality | task |
|-----------|----------|------|
| `pneumoniamnist` | chest X-ray | pneumonia? (binary) |
| `retinamnist`    | fundus photo | retinopathy grade (5) |
| `dermamnist`     | dermoscopy | skin lesion (7) |
| `breastmnist`    | breast ultrasound | malignant? (binary) |
| `bloodmnist`     | microscopy | blood-cell type (8) |
| `pathmnist`      | histology | colon tissue (9) |
| `octmnist`       | retinal OCT | retinal disease (4) |
| `organamnist`    | abdominal CT | organ (11) |

A dozen more exist (organ coronal/sagittal views, tissue, ...). See https://medmnist.com.

**Level up:** try a harder dataset, combine two, or bring your own images.

## Stuck? Grab one of these
Don't burn ten minutes deciding. Each idea below works with the `project_template/`: set
`DATASET`, run the baseline, then chase the one goal in bold. A narrow goal makes a better talk.

1. **A pneumonia screener you'd trust** (`pneumoniamnist`) -- tune the decision threshold so you
   miss **fewer than 5% of pneumonia cases**, and report what that costs in false alarms.
2. **Catch the melanoma** (`dermamnist`) -- maximize **recall on the melanoma class**, then look
   at which lesions it confuses melanoma with.
3. **Re-grade the retina** (`retinamnist`) -- redo Day 1's diabetic-retinopathy task and try to
   **beat your Day 1 accuracy**.
4. **Which blood cells confuse the model?** (`bloodmnist`) -- build the 8-class classifier, then
   **read the confusion matrix** and explain the mix-ups.
5. **Does pretraining actually help?** (any set) -- compare **pretrained vs from-scratch** and
   measure the gap honestly.
6. **Does augmentation help here?** (any set) -- add augmentation, measure the **real delta**,
   and log it for your presentation.

## Propose your own
Want to do ECG, a multimodal stack like Day 2, or something with your own data? Pitch it
to the instructor at 2:30. If it's feasible in the time, go for it.

## What makes a good capstone
Not the highest accuracy. The rubric rewards:
- a model that actually runs end to end,
- honest evaluation (held-out test data, the right metric),
- finding and explaining a failure mode,
- a design decision each partner can defend,
- both partners understanding the whole thing.

See `rubric.md`.
