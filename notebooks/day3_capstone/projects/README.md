# Day 3 capstone: the six project starters

One tailored notebook per group. Each opens in Colab, clones the repo, installs what it needs,
and runs top-to-bottom to a **working baseline**. Your job is to make it better and be able to
explain every change. Same rubric for everyone: build it, evaluate it honestly, find a failure
mode, defend one design decision, and make sure both partners can explain the work.

| Group | Project | Kind | Priority | Open in Colab |
|---|---|---|---|---|
| G1 | CRISPR guide efficiency | sequence (DNA) | interpretability | [open](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/notebooks/day3_capstone/projects/g1_crispr/starter.ipynb) |
| G2 | Skin cancer screening | image | sensitivity (don't miss cancer) | [open](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/notebooks/day3_capstone/projects/g2_skin_screening/starter.ipynb) |
| G3 | Head trauma / stroke on brain CT | image | fairness you can't even measure | [open](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/notebooks/day3_capstone/projects/g3_brain_ct/starter.ipynb) |
| G4 | Skin cancer AI + access/equity | image | fairness and access | [open](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/notebooks/day3_capstone/projects/g4_skin_access/starter.ipynb) |
| G5 | Heart disease from a checkup | tabular | fairness across sex | [open](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/notebooks/day3_capstone/projects/g5_heart/starter.ipynb) |
| G6 | Estrogen, cognition & the patient's account | tabular | confounding vs causation | [open](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/notebooks/day3_capstone/projects/g6_estrogen/starter.ipynb) |

## Three kinds of data, three helper files

- **Image** groups (G2, G3, G4) use `capstone_common.py`: MedMNIST or a local npz, transfer
  learning, and the "regulator's toolkit" of audits.
- **Tabular** groups (G5, G6) use `capstone_tabular.py`: pick features, pick a model, and audit
  for fairness and confounding.
- **Sequence** group (G1) uses `capstone_seq.py`: turn a DNA guide into numbers two different
  ways and watch which representation wins.

## Datasets

- G1 CRISPR: `datasets/crispr_guides.csv` (Doench et al., *Nature Biotechnology* 2016 -- the
  canonical guide-efficiency benchmark, 5310 guides).
- G2/G4 skin: DermaMNIST (downloads in seconds, no account).
- G3 brain CT: `datasets/brain_ct.npz`, built by `datasets/build_brain_ct.py` from
  `little-duck/brain_CT_transform` on the HuggingFace Hub (normal vs stroke, 64px, balanced).
- G5 heart: UCI Heart Disease (fetched live via `ucimlrepo`).
- G6 estrogen: NHANES 2013-14 (fetched live from the CDC).

Notebooks are generated from `scripts/build_projects.py` -- edit the script and rebuild, never
hand-edit the `.ipynb`.
