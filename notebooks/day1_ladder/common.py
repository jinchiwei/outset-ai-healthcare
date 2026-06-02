"""Day 1 helpers: APTOS DR data loading, model factories, eval, visualization.

Shared across the Day 1 lab notebook and its solution. Students load a small,
pre-resized APTOS-2019 subset from HuggingFace (no Kaggle token, no 9 GB download).

Public API:
    get_loaders(size=224, batch_size=32)   -> (train_loader, val_loader)
    flatten_for_classical(loader, max_n)    -> (X, y) numpy for sklearn
    make_mlp / make_small_cnn / make_resnet50 / make_vit_base
    train_model(model, train, val, epochs, lr, device)
    evaluate_classifier(predict_fn, loader, device)
    show_rgb_split / show_pixel_histogram / show_augmentations
    show_first_layer_filters / gradcam
"""
from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Binary "referable DR" task: does this eye need to see a doctor?
# Grades 0-1 (none/mild) = not referable; grades 2-4 (moderate+) = referable.
# This is the task actually deployed in clinics, and far more learnable (and
# motivating) than 5-class severity grading.
NUM_CLASSES = 2
REFERABLE_THRESHOLD = 2
HF_DATASET = "dreamxjei/aptos-mini"  # instructor-hosted pre-resized subset (labels 0-4)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
GRADE_NAMES = ["not referable", "referable"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class _HFImageDataset(Dataset):
    """Wraps a HuggingFace split (columns: 'image' PIL, 'diagnosis' int) as a
    torch Dataset that applies a transform."""
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        label = int(row["diagnosis"] >= REFERABLE_THRESHOLD)  # binary: referable DR
        return self.transform(img), label


def _transform(size: int, train: bool):
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(15)] if train else []
    return T.Compose(
        [T.Resize((size, size)), *aug, T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]
    )


def get_loaders(size: int = 224, batch_size: int = 32):
    """Load the APTOS-mini subset from HuggingFace and return train/val loaders.

    `size` controls image resolution: use 64 for the flat-feature models
    (logreg, MLP) and 224 for the CNN/ResNet/ViT steps.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)
    train = _HFImageDataset(ds["train"], _transform(size, train=True))
    val = _HFImageDataset(ds["validation"], _transform(size, train=False))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
    )


def synthetic_loaders(size: int = 224, batch_size: int = 8, n: int = 32):
    """Random-noise loaders with the right shapes. For local smoke tests only,
    when the real HF dataset isn't downloaded."""
    class _Rand(Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return torch.rand(3, size, size, generator=g), int(torch.randint(0, NUM_CLASSES, (1,), generator=g))

    return (
        DataLoader(_Rand(), batch_size=batch_size, shuffle=True),
        DataLoader(_Rand(), batch_size=batch_size),
    )


def flatten_for_classical(loader, max_n: Optional[int] = None):
    """Flatten image batches into (N, H*W*C) for sklearn models."""
    Xs, ys = [], []
    seen = 0
    for xb, yb in loader:
        Xs.append(xb.reshape(xb.size(0), -1).numpy())
        ys.append(np.asarray(yb))
        seen += xb.size(0)
        if max_n and seen >= max_n:
            break
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    if max_n:
        X, y = X[:max_n], y[:max_n]
    return X, y


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def make_mlp(in_features: int, num_classes: int = NUM_CLASSES, hidden=(256, 128)):
    """MLP on flattened pixels. Accepts image-shaped batches (N, C, H, W) and
    flattens them, so it drops into the same training loop as the CNN.
    `in_features` must equal C*H*W of the input images.
    """
    layers, d = [nn.Flatten()], in_features
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, num_classes))
    return nn.Sequential(*layers)


def make_small_cnn(num_classes: int = NUM_CLASSES):
    # BatchNorm after each conv: from-scratch CNNs train much more reliably with it,
    # which matters on the small medical dataset used in class.
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
    )


def make_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True):
    import torchvision.models as tvm
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = tvm.resnet50(weights=weights)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # new head, trainable
    return m


def make_vit_base(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True):
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.get_classifier().parameters():  # train only the new head
            p.requires_grad = True
    return m


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_model(model, train_loader, val_loader, epochs: int = 3, lr: float = 1e-3,
                device: str = "cpu", verbose: bool = True):
    """Standard supervised loop.

    Returns list of (epoch, val_acc, train_loss) per epoch. The val_acc stays at
    index [1] so older callers (history[-1][1]) keep working; train_loss at [2]
    lets the notebook draw a learning curve.
    """
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss) * len(xb); n += len(xb)
        train_loss = running / max(n, 1)
        acc = evaluate_classifier(lambda b: model(b).argmax(1), val_loader, device)["accuracy"]
        history.append((epoch + 1, acc, train_loss))
        if verbose:
            print(f"  epoch {epoch + 1:2d}/{epochs}  train_loss {train_loss:.3f}  val_acc {acc:.3f}")
    return history


def evaluate_classifier(predict_fn: Callable, loader, device: str = "cpu"):
    """Eval any model. `predict_fn(batch_on_device) -> predicted label tensor/array.`"""
    ys, ps = [], []
    was_training = None
    for xb, yb in loader:
        xb = xb.to(device) if hasattr(xb, "to") else xb
        with torch.no_grad():
            pred = predict_fn(xb)
        pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)
        ys.append(np.asarray(yb))
        ps.append(pred.reshape(-1))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, q in zip(y, p):
        conf[int(t), int(q)] += 1
    per_class = {GRADE_NAMES[c]: (conf[c, c] / conf[c].sum() if conf[c].sum() else 0.0) for c in range(NUM_CLASSES)}
    return {"accuracy": float((y == p).mean()), "confusion": conf, "per_class": per_class, "y": y, "pred": p}


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _denorm(img_tensor):
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


def show_rgb_split(img_tensor):
    """Show an image and its R, G, B channels separately."""
    import matplotlib.pyplot as plt
    img = _denorm(img_tensor)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(img.permute(1, 2, 0)); axes[0].set_title("RGB")
    for i, (name, cmap) in enumerate([("Red", "Reds"), ("Green", "Greens"), ("Blue", "Blues")]):
        axes[i + 1].imshow(img[i], cmap=cmap); axes[i + 1].set_title(name)
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    return fig


def show_pixel_histogram(img_tensor):
    import matplotlib.pyplot as plt
    img = _denorm(img_tensor)
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, c in enumerate(["red", "green", "blue"]):
        ax.hist(img[i].flatten().numpy(), bins=40, color=c, alpha=0.5, label=c)
    ax.set_title("Pixel intensity per channel"); ax.legend()
    return fig


def show_augmentations(pil_image, size: int = 224):
    """Show the same image under several augmentations side-by-side."""
    import matplotlib.pyplot as plt
    base = T.Resize((size, size))(pil_image.convert("RGB"))
    variants = {
        "original": base,
        "h-flip": T.functional.hflip(base),
        "rotate 15": T.functional.rotate(base, 15),
        "brighter": T.functional.adjust_brightness(base, 1.6),
    }
    fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3))
    for ax, (name, im) in zip(axes, variants.items()):
        ax.imshow(im); ax.set_title(name); ax.axis("off")
    plt.tight_layout()
    return fig


def show_first_layer_filters(model):
    """Visualize the first conv layer's learned filters (for the from-scratch CNN)."""
    import matplotlib.pyplot as plt
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.detach().cpu()
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    n = min(16, w.size(0))
    fig, axes = plt.subplots(2, n // 2, figsize=(n, 3))
    for i, ax in enumerate(axes.flat):
        ax.imshow(w[i].permute(1, 2, 0)); ax.axis("off")
    fig.suptitle("First-layer filters")
    return fig


def gradcam(model, img_tensor, target_class: Optional[int] = None, device: str = "cpu"):
    """Grad-CAM heatmap for a CNN/ResNet. Returns (heatmap HxW numpy, predicted_class).

    Hooks the last Conv2d layer. Works for make_small_cnn and make_resnet50.
    """
    model = model.to(device).eval()
    last_conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    acts, grads = {}, {}

    def fwd_hook(_, __, out):
        acts["v"] = out.detach()

    def bwd_hook(_, gin, gout):
        grads["v"] = gout[0].detach()

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(bwd_hook)
    try:
        # requires_grad on the input forces autograd to build the full backward
        # graph through the conv layers even when the backbone is frozen, so the
        # gradient hook on the last conv actually fires.
        x = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        logits = model(x)
        cls = int(logits.argmax(1)) if target_class is None else target_class
        model.zero_grad()
        logits[0, cls].backward()
        a, g = acts["v"][0], grads["v"][0]
        weights = g.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * a).sum(0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), cls
    finally:
        h1.remove()
        h2.remove()
