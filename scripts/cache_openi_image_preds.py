"""Instructor-only, one-shot: cache an image model's honest prediction per case.

Late fusion / stacking. Instead of feeding the table a pile of handcrafted radiomics
numbers (or a raw 512-d image embedding, which swamps a tabular model like TabPFN), we
train an actual image classifier -- transfer learning, exactly like Day 1 -- and let
its single probability be the image's vote in the multimodal table.

The honesty catch: if the image model predicts a case it was trained on, that
prediction is optimistically overfit -- the image feature would quietly leak. So we use
OUT-OF-FOLD predictions: each case's probability comes from a model trained only on the
*other* cases. That is exactly what cross_val_predict does.

Pipeline:
  frozen ResNet18 (ImageNet)  ->  512-d embedding per chest X-ray   [the 'transfer' part]
  out-of-fold logistic head on those embeddings                    [the head + honesty]
  ->  P(target | image) for every case, cached to JSON keyed by case_id

Students load datasets/openi_image_preds.json; they never touch the raw 1.5 GB of
X-rays. Uses the SAME balanced case set (same target + seed) as build_openi_features.py,
so the cached probability lines up row-for-row with the feature table.

Usage:  python scripts/cache_openi_image_preds.py --target Cardiomegaly
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/day2_multimodal"))
import common  # noqa: E402

OUT = ROOT / "datasets/openi_image_preds.json"
SEED = 0


def embed_images(image_paths, device, batch_size=32):
    """Frozen ResNet18 (ImageNet) -> 512-d embedding per image."""
    import torch
    from torchvision import models, transforms
    from torchvision.models import ResNet18_Weights
    from PIL import Image

    weights = ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()          # 512-d penultimate features
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    # ImageNet preprocessing; grayscale X-ray -> 3 channels via the transform's expand
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="embed"):
            batch = image_paths[i:i + batch_size]
            x = torch.stack([tf(Image.open(p).convert("L")) for p in batch]).to(device)
            embs.append(net(x).cpu().numpy())
    return np.concatenate(embs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Cardiomegaly")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score

    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # identical case selection to build_openi_features.py
    cases = common.list_cases(args.target, balanced=True, seed=SEED)
    case_ids = [c[0] for c in cases]
    img_paths = [c[1] for c in cases]
    y = np.array([c[3] for c in cases])
    print(f"{len(cases)} cases ({int(y.sum())} positive / {len(y) - int(y.sum())} negative)")

    emb = embed_images(img_paths, device)
    print(f"embeddings: {emb.shape}")

    # out-of-fold probabilities: each case scored by a model that never saw it
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    proba = cross_val_predict(pipe, emb, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, (proba >= 0.5).astype(int))
    print(f"out-of-fold image model:  AUC {auc:.3f}   acc {acc:.3f}")

    cache = {cid: round(float(p), 6) for cid, p in zip(case_ids, proba)}
    OUT.write_text(json.dumps(cache, indent=0))
    print(f"wrote {len(cache)} image predictions -> {OUT}")


if __name__ == "__main__":
    main()
