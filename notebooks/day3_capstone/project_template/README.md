# Capstone project template (scripts, not a notebook)

A real health-AI project is a few small scripts that each do one job, not one giant notebook.
This is that structure. Clone it, run it, then improve it.

## The files

| file | what it does | do you edit it? |
|------|--------------|-----------------|
| `config.py`   | the knobs: dataset, image size, epochs, learning rate | **yes, first** |
| `data.py`     | loads train / val / test splits (MedMNIST)            | rarely |
| `model.py`    | defines the model (baseline = transfer learning)      | **yes, to improve** |
| `train.py`    | trains and saves `model.pt`                           | run it |
| `evaluate.py` | grades on the **held-out test set**, saves failures   | run it |

## Run it

```bash
pip install medmnist torch torchvision matplotlib
python train.py       # trains a baseline, saves model.pt, prints val accuracy
python evaluate.py    # honest test-set number + confusion matrix + failures.png
```

It works out of the box on `pneumoniamnist`. That first number is your **baseline**.

## The build loop (this is the whole skill)

1. **Make it run** — done; the baseline already trains end to end.
2. **Get a number** — whatever `train.py` prints. Write it down.
3. **Improve ONE thing** — edit `config.py` (more epochs? bigger images?) or `model.py`
   (unfreeze the backbone?). Re-run. Compare.
4. **Keep what helps, log it** — "more epochs: +4 points." That log is half your presentation.

Change one thing at a time. If you change five things and the number moves, you've learned
nothing about *why*.

## Pick your dataset

Set `DATASET` in `config.py`. Good starts: `pneumoniamnist` (binary, like Day 2),
`dermamnist` (skin, includes melanoma), `retinamnist` (Day 1's diabetic-retinopathy problem).

## What gets graded

Not the highest accuracy. It runs, you evaluated **honestly** (held-out test, the right
metric), you found a **failure mode**, and you can **explain every line**. A simple model you
understand beats a fancy one you can't.
