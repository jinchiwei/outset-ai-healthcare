"""Train the model and save a checkpoint.

    python train.py

This runs end to end as a BASELINE right now -- it will print a (probably mediocre) number.
That number is your starting point. Your whole job today is to make it go up, ONE change at a
time, measuring after each. A change you cannot measure is a change you cannot trust.

The build loop:
    1. make it run        (done -- this script works out of the box)
    2. get a number       (the val_acc it prints)
    3. improve one thing  (edit config.py or model.py), re-run, compare
    4. keep what helps, log it ("more epochs: +4 points"), repeat
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capstone_common as cc  # noqa: E402

import config  # noqa: E402
from data import get_data  # noqa: E402
from model import build_model  # noqa: E402

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    train_loader, val_loader, test_loader, n_classes, task = get_data()
    print(f"dataset={config.DATASET}  classes={n_classes}  task='{task}'  device={device}")

    model = build_model(n_classes)
    model = cc.train(model, train_loader, val_loader,
                     epochs=config.EPOCHS, lr=config.LR, device=device)

    torch.save(model.state_dict(), config.CHECKPOINT)
    print(f"\nsaved checkpoint -> {config.CHECKPOINT}")
    print("next:  python evaluate.py   (grades on the held-out TEST set -- the honest number)")


if __name__ == "__main__":
    main()
