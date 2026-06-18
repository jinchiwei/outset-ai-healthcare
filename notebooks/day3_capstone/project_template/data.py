"""Load the dataset as train / val / test loaders.

Provided for you -- you usually won't need to touch this. It wraps the shared MedMNIST
loader. The important idea: THREE splits. You train on `train`, tune on `val`, and you only
ever grade on `test` -- once, at the end. That last rule is what keeps your number honest.
"""
import os
import sys

# make the shared helpers (one directory up) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capstone_common as cc  # noqa: E402

import config  # noqa: E402


def get_data():
    """Return (train_loader, val_loader, test_loader, n_classes, task)."""
    return cc.get_loaders(config.DATASET, size=config.IMG_SIZE, batch_size=config.BATCH_SIZE)
