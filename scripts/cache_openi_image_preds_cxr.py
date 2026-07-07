"""Instructor-only, one-shot: cache the DOMAIN image branch's cardiomegaly vote.

The Day 2 image vote (`img_pred`) comes from a chest-X-ray-pretrained model, not
ImageNet transfer. That distinction is the point: a DenseNet pretrained on ~500,000
real chest X-rays (torchxrayvision, weights 'densenet121-res224-all', trained on
NIH + CheXpert + MIMIC + Open-i + Kaggle) already knows cardiomegaly cold. On the
balanced Open-i set it scores AUC ~0.88 / acc ~0.78 -- vs ~0.77 / 0.72 for an
ImageNet-transfer ResNet18. WHERE a model pretrained beats the architecture.

Pipeline:
  torchxrayvision DenseNet -> P(Cardiomegaly) per X-ray            [domain pretraining]
  out-of-fold logistic calibration of that single probability      [honest + calibrated]
  -> datasets/openi_image_preds_cxr.json  (also copied to openi_image_preds.json,
     the default cache the notebook loads)

Uses the SAME balanced case set (target + seed) as build_openi_features.py, so the
cached probability lines up row-for-row with the feature table's `img_pred` column.

Requires: pip install torchxrayvision scikit-image
Usage:    python scripts/cache_openi_image_preds_cxr.py --target Cardiomegaly
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/day2_multimodal"))
import common  # noqa: E402

OUT_CXR = ROOT / "datasets/openi_image_preds_cxr.json"
OUT_DEFAULT = ROOT / "datasets/openi_image_preds.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Cardiomegaly")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import torch
    import skimage.io
    import torchxrayvision as xrv
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    cases = common.list_cases(args.target, balanced=True, seed=0)
    ids = [c[0] for c in cases]
    paths = [str(c[1]) for c in cases]
    y = np.array([c[3] for c in cases])
    print(f"{len(cases)} balanced cases, {int(y.sum())} positive")

    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval().to(device)
    if args.target not in model.pathologies:
        raise SystemExit(f"{args.target} not among model pathologies: {model.pathologies}")
    idx = model.pathologies.index(args.target)

    crop, resize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
    raw = []
    with torch.no_grad():
        for p in paths:
            img = skimage.io.imread(p)
            if img.ndim == 3:
                img = img[..., :3].mean(2)                 # grayscale
            img = xrv.datasets.normalize(img, 255)         # -> [-1024, 1024]
            img = resize(crop(img[None, ...]))
            t = torch.from_numpy(img)[None, ...].float().to(device)
            raw.append(torch.sigmoid(model(t))[0, idx].item())
    raw = np.array(raw)

    # out-of-fold logistic calibration of the single probability (honest img_pred)
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    pred = cross_val_predict(LogisticRegression(max_iter=1000),
                             raw.reshape(-1, 1), y, cv=cv, method="predict_proba")[:, 1]
    print(f"domain img_pred: acc={((pred >= 0.5).astype(int) == y).mean():.3f}  "
          f"auc={roc_auc_score(y, pred):.3f}")

    OUT_CXR.write_text(json.dumps({cid: float(v) for cid, v in zip(ids, pred)}, indent=1))
    shutil.copyfile(OUT_CXR, OUT_DEFAULT)   # make the domain vote the notebook's default img_pred
    print(f"wrote {OUT_CXR.name} and {OUT_DEFAULT.name} ({len(ids)} cases)")
    print("Now regenerate the feature table so its img_pred column matches:")
    print("   python datasets/build_openi_features.py")


if __name__ == "__main__":
    main()
