"""Project config -- the knobs you'll actually turn.

This is the FIRST file to edit. Change a value, re-run `python train.py`, and compare the
number to your last run. Change ONE thing at a time so you know what helped.
"""

# Which dataset? Any 2D MedMNIST key (downloads in seconds, no account needed):
#   pneumoniamnist  -- chest X-ray, pneumonia yes/no   (closest to Day 2)
#   dermamnist      -- skin lesions, 7 classes incl. melanoma
#   retinamnist     -- retina, diabetic-retinopathy grade (Day 1's problem!)
#   bloodmnist, organamnist, pathmnist, ...   (a dozen more)
DATASET = "pneumoniamnist"

IMG_SIZE = 64          # 28, 64, 128, or 224. Bigger = slower but can be sharper.
BATCH_SIZE = 64
EPOCHS = 3             # TODO: more passes usually helps at first -- try 8, then 15.
LR = 1e-3              # learning rate. TODO: if it won't improve, try 3e-4 or 3e-3.
PRETRAINED = True      # transfer learning (Day 1's trick). Try False to feel the difference.

CHECKPOINT = "model.pt"   # where train.py saves the model and evaluate.py reads it
