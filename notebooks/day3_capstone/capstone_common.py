"""Shared helpers for the Day 3 capstone starter kits.

All three kits use MedMNIST (pip-installable, tiny, no Kaggle token) so students spend
the sprint improving a model, not fighting downloads. The baseline is transfer learning
with a frozen ResNet18 + new head -- the same trick that won Day 1.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


# Locally-built datasets (not on MedMNIST). flag -> (npz filename, task description).
# brainct: Group 3's brain-CT set, built by datasets/build_brain_ct.py.
LOCAL_NPZ = {"brainct": ("brain_ct.npz", "normal vs stroke on a brain CT slice")}


def _npz_path(flag: str) -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / LOCAL_NPZ[flag][0]


def class_names(flag: str):
    """Human-readable class names for a dataset flag (MedMNIST key or a LOCAL_NPZ flag)."""
    if flag in LOCAL_NPZ:
        d = np.load(_npz_path(flag), allow_pickle=True)
        return [str(x) for x in d["class_names"]]
    from medmnist import INFO
    return list(INFO[flag]["label"].values())


def get_loaders(flag: str = "pneumoniamnist", size: int = 64, batch_size: int = 64,
                augment: bool = False):
    """Return (train, val, test) loaders + (n_classes, task) for an image dataset.

    `flag` is any 2D MedMNIST key ('pneumoniamnist', 'dermamnist', 'retinamnist',
    'bloodmnist', 'organamnist', 'pathmnist', ...) OR a locally-built set in LOCAL_NPZ
    (e.g. 'brainct'). `augment=True` adds light flips/rotations to the TRAIN split only.
    """
    if flag in LOCAL_NPZ:
        return _local_npz_loaders(flag, size, batch_size, augment)

    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    n_classes = len(info["label"])

    # MedMNIST images may be 1 or 3 channel; force 3 channels for ResNet, resize, normalize.
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(15)] if augment else []
    train_tfm = T.Compose([T.Resize((size, size)), *aug, T.Grayscale(3), T.ToTensor(),
                           T.Normalize([0.5] * 3, [0.5] * 3)])
    eval_tfm = T.Compose([T.Resize((size, size)), T.Grayscale(3), T.ToTensor(),
                          T.Normalize([0.5] * 3, [0.5] * 3)])

    def loader(split, shuffle, tfm):
        ds = DataClass(split=split, transform=tfm, download=True, size=size)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (loader("train", True, train_tfm), loader("val", False, eval_tfm),
            loader("test", False, eval_tfm), n_classes, info["task"])


def _local_npz_loaders(flag, size, batch_size, augment):
    """Loaders for a locally-built grayscale-uint8 npz (X:[N,H,W], y:[N]).

    Splits 70/15/15 stratified into train/val/test, replicates to 3 channels, resizes and
    normalizes exactly like the MedMNIST path so the model code is identical either way.
    """
    d = np.load(_npz_path(flag), allow_pickle=True)
    X, y = d["X"], d["y"].astype(np.int64)
    n_classes = int(y.max()) + 1

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
    splits = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    aug = [T.RandomHorizontalFlip(), T.RandomRotation(15)] if augment else []
    train_tfm = T.Compose([T.ToPILImage(), T.Resize((size, size)), *aug, T.Grayscale(3),
                           T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    eval_tfm = T.Compose([T.ToPILImage(), T.Resize((size, size)), T.Grayscale(3),
                          T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def loader(split, shuffle, tfm):
        sel = splits[split]
        imgs = torch.stack([tfm(X[i]) for i in sel])                 # (n,3,size,size)
        labels = torch.from_numpy(y[sel]).unsqueeze(1)               # (n,1) like MedMNIST
        return DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=shuffle)

    return (loader("train", True, train_tfm), loader("val", False, eval_tfm),
            loader("test", False, eval_tfm), n_classes, "binary-class")


# backbones offered in the interactive model builder (timm names)
BACKBONES = ["resnet18", "resnet50", "densenet121", "mobilenetv2_100", "efficientnet_b0"]


def make_baseline(n_classes: int, pretrained: bool = True):
    """Transfer-learning ResNet18: frozen backbone + a fresh trainable head."""
    import torchvision.models as tvm
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.resnet18(weights=weights)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def make_model(n_classes: int, backbone: str = "resnet18", pretrained: bool = True,
               unfreeze_backbone: bool = False):
    """Flexible model factory for the interactive builder. Any `backbone` in BACKBONES.

    pretrained: start from ImageNet weights (True) or from scratch (False).
    unfreeze_backbone: train the WHOLE network (True) or just a fresh head (False).
    """
    import timm
    m = timm.create_model(backbone, pretrained=pretrained, num_classes=n_classes)
    if not unfreeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.get_classifier().parameters():   # train only the new head
            p.requires_grad = True
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


# --------------------------------------------------------------------------- #
# The regulator's toolkit: apply Day 3's priorities to YOUR OWN model.
# Each function takes (model, test_loader, device, class_names) and draws one audit.
# --------------------------------------------------------------------------- #
def _names(class_names, n):
    return class_names if class_names else [str(i) for i in range(n)]


def audit_confusion(model, loader, device="cpu", class_names=None):
    """SAFETY / EVIDENCE -- honest evaluation: a confusion matrix on the TEST set.
    The diagonal is right; everything off it is a mistake, and WHICH mistakes matter."""
    import matplotlib.pyplot as plt
    res = evaluate(model, loader, device)
    y, p = res["y"], res["pred"]
    n = int(max(y.max(), p.max())) + 1
    names = _names(class_names, n)
    M = np.zeros((n, n), int)
    for t, q in zip(y, p):
        M[t, q] += 1
    fig, ax = plt.subplots(figsize=(1.4 + 0.55 * n, 1.4 + 0.55 * n))
    ax.imshow(M, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"TEST accuracy = {res['accuracy']:.3f}")
    plt.tight_layout(); plt.show()
    print("Look off the diagonal: which real class gets missed most? For screening, a missed")
    print("disease (a false negative) is usually worse than a false alarm.")


def audit_fairness(model, loader, device="cpu", class_names=None):
    """FAIRNESS -- per-class accuracy. MedMNIST has no demographics, so we audit by CLASS:
    a high overall score can still hide one group the model quietly fails."""
    import matplotlib.pyplot as plt
    res = evaluate(model, loader, device)
    y, p = res["y"], res["pred"]
    n = int(y.max()) + 1
    names = _names(class_names, n)
    accs = [float((p[y == c] == c).mean()) if (y == c).any() else 0.0 for c in range(n)]
    weak = max(accs) - 0.15
    colors = ["#FF1493" if a < weak else "#40E0D0" for a in accs]
    fig, ax = plt.subplots(figsize=(1.8 + 0.7 * n, 3))
    ax.bar(names, accs, color=colors)
    ax.axhline(res["accuracy"], ls="--", color="gray", label=f"overall {res['accuracy']:.2f}")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("per-class accuracy (fairness audit)")
    plt.xticks(rotation=45, ha="right", fontsize=7); plt.tight_layout(); plt.show()
    worst = names[int(np.argmin(accs))]
    print(f"Weakest class: '{worst}' at {min(accs):.2f} vs overall {res['accuracy']:.2f}.")
    print("Would you deploy a model that works this differently across groups?")


def audit_transparency(model, loader, device="cpu", class_names=None, n=4):
    """TRANSPARENCY -- Grad-CAM: overlay WHERE the model looked to make each call.
    If the heat is off the anatomy, the model may be right for the wrong reason."""
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    model = model.to(device).eval()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        print("This model has no conv layer to visualize (not a CNN)."); return
    last = convs[-1]
    xb, yb = next(iter(loader))
    xb = xb[:n].to(device)
    acts, grads = {}, {}
    h1 = last.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))
    h2 = last.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))
    try:
        x = xb.clone().requires_grad_(True)
        logits = model(x)
        cls = logits.argmax(1)
        model.zero_grad()
        logits.gather(1, cls[:, None]).sum().backward()
        w = grads["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["v"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=xb.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[:, 0].cpu().numpy()
    finally:
        h1.remove(); h2.remove()          # always remove hooks -> safe to re-run
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for i in range(n):
        img = xb[i, 0].detach().cpu().numpy()
        c = cam[i]; c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        lab = _names(class_names, int(cls.max()) + 1)[int(cls[i])]
        axes[0, i].imshow(img, cmap="gray"); axes[0, i].set_title(f"pred: {lab}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(img, cmap="gray"); axes[1, i].imshow(c, cmap="jet", alpha=0.5)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("image", fontsize=8); axes[1, 0].set_ylabel("where it looked", fontsize=8)
    plt.tight_layout(); plt.show()
    print("Top row: the X-ray. Bottom: red = what drove the prediction. Is it looking at the right place?")


def audit_monitoring(model, loader, device="cpu", class_names=None, max_batches=15):
    """MONITORING -- robustness / drift: does accuracy hold as image quality degrades?
    A deployed model meets messier images than it trained on; this simulates that."""
    import matplotlib.pyplot as plt
    model = model.to(device).eval()
    sigmas = [0.0, 0.1, 0.2, 0.35, 0.5]
    accs = []
    for s in sigmas:
        ys, ps = [], []
        with torch.no_grad():
            for b, (xb, yb) in enumerate(loader):
                if b >= max_batches:
                    break
                xn = (xb + s * torch.randn_like(xb)).to(device)
                ps.append(model(xn).argmax(1).cpu().numpy())
                ys.append(_targets(yb).numpy())
        y, p = np.concatenate(ys), np.concatenate(ps)
        accs.append(float((y == p).mean()))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(sigmas, accs, "o-", color="#8A2BE2", lw=2)
    ax.set_xlabel("noise added to the image"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("monitoring: accuracy vs image degradation")
    plt.tight_layout(); plt.show()
    print(f"Clean images: {accs[0]:.2f}  ->  noisy images: {accs[-1]:.2f}.")
    print("If this falls off a cliff, your model needs monitoring (and retraining) after it ships.")


def audit_failures(model, loader, device="cpu", class_names=None, k=8):
    """FAILURE ANALYSIS -- show the cases the model got WRONG. Rubric point 3, and usually
    the most interesting slide: is there a pattern a doctor would recognize?"""
    import matplotlib.pyplot as plt
    model = model.to(device).eval()
    imgs, ts, prs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            t = _targets(yb)
            for i in range(len(xb)):
                if int(pred[i]) != int(t[i]) and len(imgs) < k:
                    imgs.append(xb[i, 0].numpy()); ts.append(int(t[i])); prs.append(int(pred[i]))
            if len(imgs) >= k:
                break
    if not imgs:
        print("No mistakes found in the scanned batches -- your model is doing great here!"); return
    names = _names(class_names, max(max(ts), max(prs)) + 1)
    cols = min(k, len(imgs)); fig, axes = plt.subplots(1, cols, figsize=(2.1 * cols, 2.6))
    axes = np.atleast_1d(axes)
    for ax, im, t, pr in zip(axes, imgs, ts, prs):
        ax.imshow(im, cmap="gray")
        ax.set_title(f"true {names[t]}\npred {names[pr]}", fontsize=7, color="#FF1493")
        ax.axis("off")
    plt.tight_layout(); plt.show()
    print("What do the mistakes have in common -- blurry, unusual, a rare class? That's your finding.")


# name -> tool, for the interactive "regulator's toolkit" dropdown in the starter kits
REGULATOR_TOOLS = {
    "Safety/Evidence -- confusion matrix (honest test eval)": audit_confusion,
    "Fairness -- accuracy per class": audit_fairness,
    "Transparency -- Grad-CAM (where the model looked)": audit_transparency,
    "Monitoring -- accuracy as images degrade": audit_monitoring,
    "Failure analysis -- cases it got wrong": audit_failures,
}
