"""Smoke tests for Day 1 common module, on synthetic data (no APTOS download needed).

Verifies the 5 model factories forward + train + eval + Grad-CAM run end-to-end.
Uses pretrained=False for ResNet/ViT to avoid weight downloads in CI.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

DEV = "cpu"  # keep tests deterministic + portable


def test_flatten():
    tr, _ = common.synthetic_loaders(size=64, batch_size=8, n=24)
    X, y = common.flatten_for_classical(tr, max_n=24)
    assert X.shape == (24, 3 * 64 * 64)
    assert y.shape == (24,)


def test_mlp_trains():
    tr, va = common.synthetic_loaders(size=64, batch_size=8, n=24)
    mlp = common.make_mlp(in_features=3 * 64 * 64)
    hist = common.train_model(mlp, tr, va, epochs=1, device=DEV)
    assert 0.0 <= hist[-1][1] <= 1.0


def test_cnn_trains_and_gradcam():
    tr, va = common.synthetic_loaders(size=224, batch_size=4, n=12)
    cnn = common.make_small_cnn()
    common.train_model(cnn, tr, va, epochs=1, device=DEV)
    img = next(iter(va))[0][0]
    cam, cls = common.gradcam(cnn, img, device=DEV)
    assert cam.ndim == 2
    assert 0 <= cls < common.NUM_CLASSES


def test_resnet_vit_forward():
    x = torch.rand(2, 3, 224, 224)
    rn = common.make_resnet50(pretrained=False)
    assert rn(x).shape == (2, common.NUM_CLASSES)
    vit = common.make_vit_base(pretrained=False)
    assert vit(x).shape == (2, common.NUM_CLASSES)


def test_evaluate_shapes():
    _, va = common.synthetic_loaders(size=64, batch_size=8, n=16)
    mlp = common.make_mlp(in_features=3 * 64 * 64)
    res = common.evaluate_classifier(lambda b: mlp(b).argmax(1), va, device=DEV)
    assert res["confusion"].shape == (common.NUM_CLASSES, common.NUM_CLASSES)
    assert 0.0 <= res["accuracy"] <= 1.0


if __name__ == "__main__":
    test_flatten(); test_mlp_trains(); test_cnn_trains_and_gradcam()
    test_resnet_vit_forward(); test_evaluate_shapes()
    print("all common.py smoke tests passed")
