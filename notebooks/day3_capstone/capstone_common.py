"""Shared helpers for the Day 3 capstone starter kits.

All three kits use MedMNIST (pip-installable, tiny, no Kaggle token) so students spend
the sprint improving a model, not fighting downloads. The baseline is transfer learning
with a frozen ResNet18 + new head -- the same trick that won Day 1.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T


def get_loaders(flag: str = "pneumoniamnist", size: int = 64, batch_size: int = 64):
    """Return (train, val, test) loaders + (n_classes, task) for a MedMNIST dataset.

    `flag` is any 2D MedMNIST key, e.g. 'pneumoniamnist', 'dermamnist', 'retinamnist',
    'bloodmnist', 'organamnist', 'pathmnist'.
    """
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    n_classes = len(info["label"])

    # MedMNIST images may be 1 or 3 channel; force 3 channels for ResNet, resize, normalize.
    tfm = T.Compose([
        T.Resize((size, size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    def loader(split, shuffle):
        ds = DataClass(split=split, transform=tfm, download=True, size=size)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (loader("train", True), loader("val", False), loader("test", False),
            n_classes, info["task"])


def make_baseline(n_classes: int, pretrained: bool = True):
    """Transfer-learning ResNet18: frozen backbone + a fresh trainable head."""
    import torchvision.models as tvm
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.resnet18(weights=weights)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def _targets(yb):
    # MedMNIST labels come as shape (B, 1); flatten to (B,)
    return yb.squeeze(-1).long() if yb.ndim > 1 else yb.long()


def train(model, train_loader, val_loader, epochs: int = 3, lr: float = 1e-3, device: str = "cpu"):
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), _targets(yb).to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        acc = evaluate(model, val_loader, device)["accuracy"]
        print(f"epoch {epoch+1}/{epochs}  val_acc={acc:.3f}")
    return model


def evaluate(model, loader, device: str = "cpu"):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            ys.append(_targets(yb).numpy())
            ps.append(pred)
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {"accuracy": float((y == p).mean()), "y": y, "pred": p}
