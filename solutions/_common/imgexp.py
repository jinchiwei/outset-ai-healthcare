"""Shared helper for the real-image solution experiments (G2 HAM10000, G4 PAD-UFES-20).

Loads a HuggingFace image dataset into arrays, trains a transfer-learning binary classifier
(frozen ResNet18 + head), and returns test-set probabilities so we can report AUC, tune a
threshold, and audit/mitigate by subgroup. Reuses capstone_common for the model + train loop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks/day3_capstone"))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
import capstone_common as cc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_NORM = T.Normalize([0.5] * 3, [0.5] * 3)


def hf_to_arrays(split, image_key, size, limit=None):
    """HF dataset split -> (X uint8 [N,size,size,3]). Decodes + resizes each image."""
    n = len(split) if limit is None else min(limit, len(split))
    X = np.zeros((n, size, size, 3), np.uint8)
    for i in range(n):
        im = split[i][image_key].convert("RGB").resize((size, size))
        X[i] = np.asarray(im)
    return X


def _loader(X, y, size, batch=64, shuffle=False, augment=False):
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(12)] if augment else []
    tfm = T.Compose([T.ToPILImage(), T.Resize((size, size)), *aug, T.ToTensor(), _NORM])
    imgs = torch.stack([tfm(X[i]) for i in range(len(X))])
    ys = torch.tensor(np.asarray(y), dtype=torch.long)
    return DataLoader(TensorDataset(imgs, ys), batch_size=batch, shuffle=shuffle)


def train_binary(Xtr, ytr, Xva, yva, size=224, epochs=6, class_weight=None, augment=True,
                 backbone="caformer_s18", unfreeze=False):
    """Transfer-learning backbone + 2-class head. unfreeze=True fine-tunes the whole net (lower lr)."""
    model = cc.make_model(2, backbone=backbone, pretrained=True, unfreeze_backbone=unfreeze)
    tr = _loader(Xtr, ytr, size, shuffle=True, augment=augment)
    va = _loader(Xva, yva, size)
    model = model.to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=(1e-4 if unfreeze else 1e-3))
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        print(f"  epoch {ep+1}/{epochs} done", flush=True)
    return model


def probs(model, X, y, size=224):
    """Return positive-class probabilities on (X,y)."""
    model = model.to(DEVICE).eval()
    ld = _loader(X, y, size)
    out = []
    with torch.no_grad():
        for xb, _ in ld:
            out.append(torch.softmax(model(xb.to(DEVICE)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def gradcam_fig(model, X, size, base, name, headline, labels=None, n=4):
    """Grad-CAM: overlay WHERE the model looks on a few images. The image analogue of SHAP."""
    import solfig as sf
    import torch.nn.functional as F
    model = model.to(DEVICE).eval()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    last = convs[-1]
    ld = _loader(X[:n], np.zeros(n), size)
    xb, _ = next(iter(ld)); xb = xb.to(DEVICE)
    acts, grads = {}, {}
    h1 = last.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))
    h2 = last.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))
    try:
        x = xb.clone().requires_grad_(True)
        logits = model(x); cls = logits.argmax(1)
        model.zero_grad(); logits.gather(1, cls[:, None]).sum().backward()
        w = grads["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["v"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=xb.shape[-2:], mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    fig, axes = sf.plt.subplots(2, n, figsize=(2.3 * n, 4.7))
    for i in range(n):
        img = X[i]
        c = cam[i]; c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        axes[0, i].imshow(img); axes[0, i].axis("off")
        if labels is not None:
            axes[0, i].set_title(labels[i], fontsize=8)
        axes[1, i].imshow(img); axes[1, i].imshow(c, cmap="magma", alpha=0.45); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("image", fontsize=8); axes[1, 0].set_ylabel("where it looked", fontsize=8)
    fig.suptitle(headline, fontsize=13, fontweight="bold", family="Geist Mono", color=sf.INK, y=1.02)
    sf.save(fig, base, name)
