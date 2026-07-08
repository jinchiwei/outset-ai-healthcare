"""Shared image-model training + figure helpers for the G2/G3/G4 solution experiments.

Reuses the course helpers (capstone_common) for the heavy lifting -- get_loaders, make_model,
train, evaluate -- and adds figure-savers that write brand-locked PNGs + raw CSVs via solfig,
so a solution deck can embed them directly. Everything here runs on CPU or GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # solfig
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks/day3_capstone"))
import solfig as sf
import capstone_common as cc
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_preds(model, loader):
    """Return (y_true, y_pred, prob_positive) over a loader."""
    model = model.to(DEVICE).eval()
    ys, ps, pr = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            prob = torch.softmax(logits, 1).cpu().numpy()
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(cc._targets(yb).numpy())
            pr.append(prob)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(pr)


def fig_confusion(y, pred, class_names, base, name, headline):
    n = len(class_names)
    M = np.zeros((n, n), int)
    for t, q in zip(y, pred):
        M[t, q] += 1
    fig, ax = sf.plt.subplots(figsize=(1.6 + 0.6 * n, 1.6 + 0.6 * n))
    ax.imshow(M, cmap="BuPu")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() / 2 else sf.INK)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("what the model guessed"); ax.set_ylabel("the truth")
    sf.title(ax, headline)
    sf.save(fig, base, name)
    sf.save_raw(sf.pd.DataFrame(M, index=class_names, columns=class_names), base, name)
    return M


def fig_perclass(y, pred, class_names, base, name, headline, highlight=None):
    n = len(class_names)
    accs = [float((pred[y == c] == c).mean()) if (y == c).any() else 0.0 for c in range(n)]
    overall = float((y == pred).mean())
    colors = [sf.DEEPPINK if (highlight is not None and c == highlight) else sf.TURQUOISE for c in range(n)]
    fig, ax = sf.plt.subplots(figsize=(1.8 + 0.7 * n, 3.4))
    ax.bar(class_names, accs, color=colors, edgecolor=sf.INK, linewidth=0.5)
    ax.axhline(overall, ls="--", color=sf.MUTED, lw=1, label=f"overall {overall:.2f}")
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy for this group"); ax.legend(fontsize=8)
    sf.title(ax, headline)
    sf.plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    sf.save(fig, base, name)
    sf.save_raw(sf.pd.DataFrame({"class": class_names, "accuracy": accs}), base, name)
    return dict(zip(class_names, accs))


def fig_failures(loader, model, class_names, base, name, headline, k=8):
    model = model.to(DEVICE).eval()
    imgs, ts, prs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).argmax(1).cpu(); t = cc._targets(yb)
            for i in range(len(xb)):
                if int(pred[i]) != int(t[i]) and len(imgs) < k:
                    imgs.append(xb[i, 0].numpy()); ts.append(int(t[i])); prs.append(int(pred[i]))
            if len(imgs) >= k:
                break
    cols = max(1, len(imgs))
    fig, axes = sf.plt.subplots(1, cols, figsize=(2.0 * cols, 2.6))
    axes = np.atleast_1d(axes)
    for ax, im, t, pr in zip(axes, imgs, ts, prs):
        ax.imshow(im, cmap="gray")
        ax.set_title(f"true: {class_names[t]}\nguess: {class_names[pr]}", fontsize=7, color=sf.DEEPPINK)
        ax.axis("off")
    fig.suptitle(headline, fontsize=13, fontweight="bold", family="Geist Mono", color=sf.INK, y=1.06)
    sf.save(fig, base, name)


def fig_gradcam(loader, model, class_names, base, name, headline, n=4):
    import torch.nn.functional as F
    model = model.to(DEVICE).eval()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    last = convs[-1]
    xb, yb = next(iter(loader)); xb = xb[:n].to(DEVICE)
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
    fig, axes = sf.plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for i in range(n):
        img = xb[i, 0].detach().cpu().numpy()
        c = cam[i]; c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        axes[0, i].imshow(img, cmap="gray"); axes[0, i].set_title(class_names[int(cls[i])], fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(img, cmap="gray"); axes[1, i].imshow(c, cmap="magma", alpha=0.5); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("scan", fontsize=8); axes[1, 0].set_ylabel("where it looked", fontsize=8)
    fig.suptitle(headline, fontsize=13, fontweight="bold", family="Geist Mono", color=sf.INK, y=1.02)
    sf.save(fig, base, name)


def fig_noise(loader, model, base, name, headline):
    model = model.to(DEVICE).eval()
    sigmas = [0.0, 0.1, 0.2, 0.35, 0.5]; accs = []
    for s in sigmas:
        ys, ps = [], []
        with torch.no_grad():
            for b, (xb, yb) in enumerate(loader):
                if b >= 15:
                    break
                xn = (xb + s * torch.randn_like(xb)).to(DEVICE)
                ps.append(model(xn).argmax(1).cpu().numpy()); ys.append(cc._targets(yb).numpy())
        y, p = np.concatenate(ys), np.concatenate(ps); accs.append(float((y == p).mean()))
    fig, ax = sf.plt.subplots(figsize=(5.2, 3.4))
    ax.plot(sigmas, accs, "o-", color=sf.BLUEVIOLET, lw=2)
    ax.set_xlabel("noise added to the scan"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    sf.title(ax, headline)
    sf.save(fig, base, name)
    sf.save_raw(sf.pd.DataFrame({"noise": sigmas, "accuracy": accs}), base, name)
    return accs
