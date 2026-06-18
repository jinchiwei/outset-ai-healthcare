"""Define the model.

The baseline is the Day 1 trick: a ResNet18 pretrained on ImageNet, frozen, with a fresh
trainable head. It runs as-is. This is one of the most rewarding files to experiment in --
the TODOs below are the highest-leverage changes you can make.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capstone_common as cc  # noqa: E402

import config  # noqa: E402


def build_model(n_classes):
    """Build the classifier. Start from the baseline, then improve it."""
    model = cc.make_baseline(n_classes, pretrained=config.PRETRAINED)

    # ----------------------------------------------------------------------- #
    # TODO (improve): fine-tune the last block instead of freezing everything.
    # Unfreezing lets the backbone adapt to medical images; often a real boost.
    #   for p in model.layer4.parameters():
    #       p.requires_grad = True
    #
    # TODO (explore): swap the architecture entirely, e.g. resnet50 or a ViT.
    # ----------------------------------------------------------------------- #

    return model
