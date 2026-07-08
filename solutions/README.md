# Day 3 capstone — worked solutions ("answer keys")

A fully-worked solution for each of the six capstone projects: a **literature-grounded**, real
experiment run to completion, a **commented `solution.ipynb`** (the answer key), a **hand-authored
freeform branded deck**, and saved figures + raw data. Audience: HS sophomores who barely know
Python but all have Claude Code — so the teaching is about **judgment** (what to build, which tool,
why), not syntax.

| Group | Project | Notebook (Colab) | Deck |
|---|---|---|---|
| G1 | CRISPR guide efficiency | [solution](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/solutions/g1_crispr/solution.ipynb) | `g1_crispr/slides/g1_crispr.pptx` / `.pdf` |
| G2 | Skin-cancer screening | [solution](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/solutions/g2_skin_screening/solution.ipynb) | `g2_skin_screening/slides/g2_skin_screening.pptx` / `.pdf` |
| G3 | Brain-CT stroke + un-auditable fairness | [solution](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/solutions/g3_brain_ct/solution.ipynb) | `g3_brain_ct/slides/g3_brain_ct.pptx` / `.pdf` |
| G4 | Skin-cancer AI + access/equity | [solution](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/solutions/g4_skin_access/solution.ipynb) | `g4_skin_access/slides/g4_skin_access.pptx` / `.pdf` |
| G5 | Heart disease + sex disparity | [solution](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/solutions/g5_heart/solution.ipynb) | `g5_heart/slides/g5_heart.pptx` / `.pdf` |
| G6 | Estrogen → cognition (confounding) | [solution](https://colab.research.google.com/github/jinchiwei/outset-ai-healthcare/blob/main/solutions/g6_estrogen/solution.ipynb) | `g6_estrogen/slides/g6_estrogen.pptx` / `.pdf` |

## What's in each group folder

- `literature.md` — 7–8 real, verified references (DOIs/PMIDs) + a positioning paragraph + intro-figure specs.
- `run_experiment.py` — the experiment, run to completion → `figures/*.png`, `figures/raw/*.csv`, `results.json`.
- `intro_figures.py` — bespoke matplotlib figures **redrawing foundational reference papers** (attributed) for the deck's background section.
- `build_notebook.py` → `solution.ipynb` — the worked answer key, every code line commented.
- `slides/<group>.md` + `<group>.md.layout.json` — the deck markdown + hand-authored freeform layout sidecar (build-pptx expressive; every content slide `_provenance: agent`, theme `bone`).
- `slides/<group>.pptx` / `.pdf` — the rendered deck.

## Headline results

- **G1 CRISPR:** one-hot (order-aware) AUC 0.77 beats position-blind k-mer; per-position importance peaks at guide position 20 — the seed next to the PAM. The model rediscovered the biology.
- **G2 skin:** better pretraining beats a fancier model; the audit is per-class recall, because melanoma is the rare class a screen must not miss.
- **G3 brain CT:** stroke sensitivity 0.84; the dataset records no demographics → a fairness audit is impossible, and that gap is the finding.
- **G4 skin equity:** per-class accuracy gap ~0.99 — the rarest condition is essentially never caught; the dataset hides skin tone.
- **G5 heart:** four models tie (~0.79–0.81) — the data sets the ceiling; cholesterol alone is weak; a ~0.09 accuracy gap by sex on ~2:1-male data.
- **G6 estrogen:** crude effect −0.334 → adjusted −0.112 after controlling age/education/income (**67% was confounding** — healthy-user bias).

## Rebuild

Experiments: `python solutions/<group>/run_experiment.py` (CPU groups) or via `solutions/_common/run_gpu.sbatch` (SLURM, image groups). Figures: `python solutions/<group>/intro_figures.py`. Notebook: `python solutions/<group>/build_notebook.py`. Deck: hand-author the sidecar per `BUILD_SPEC.md`, then `python ~/.claude/skills/build-pptx/build.py --input slides/<group>.md --output slides/<group>.pptx --qa`. Env: `/data/rauschecker1/jkw/envs/outset`.
