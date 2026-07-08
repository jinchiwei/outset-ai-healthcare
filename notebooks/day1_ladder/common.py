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
from __future__ import annotations   # lets us write type hints without them running at import time
import os                            # to read/set environment variables (settings the OS holds)
import warnings                      # to hide warning messages we know are harmless
from typing import Callable, Optional   # labels for "a function argument" and "might be None"

# Keep the teaching notebook output clean: silence HuggingFace download chatter
# and the harmless "IProgress not found" tqdm warning so students don't mistake
# routine log lines for errors. (The data still loads exactly the same.)
# turn off HuggingFace download progress bars (only if not already set)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="IProgress not found.*")   # hide this specific harmless warning
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")   # hide the "no login" notice


def _quiet_hf() -> None:
    """Quiet down the HuggingFace libraries so their logs don't clutter the notebook."""
    try:                                        # 'try' = attempt this, but don't crash if it fails
        import datasets                         # the HuggingFace 'datasets' library
        datasets.disable_progress_bars()        # stop it from printing loading bars
        datasets.utils.logging.set_verbosity_error()   # only show errors, not chatty info logs
    except Exception:                           # if any of that failed (e.g. library missing)...
        pass                                    # ...just skip it silently
    try:                                        # second attempt, for the huggingface_hub library
        from huggingface_hub.utils import logging as _hf_logging   # its logging controls
        _hf_logging.set_verbosity_error()       # again, only show errors
    except Exception:                           # if that library isn't available...
        pass                                    # ...skip it too


import numpy as np                  # fast number arrays and math (the workhorse of data science)
import torch                         # PyTorch: builds and trains neural networks
import torch.nn as nn               # 'nn' holds ready-made network layers (Linear, Conv2d, ...)
from torch.utils.data import DataLoader, Dataset   # tools to feed data to a model in batches
import torchvision.transforms as T   # image preprocessing steps (resize, crop, flip, normalize)

# Binary "referable DR" task: does this eye need to see a doctor?
# Grades 0-1 (none/mild) = not referable; grades 2-4 (moderate+) = referable.
# This is the task actually deployed in clinics, and far more learnable (and
# motivating) than 5-class severity grading.
NUM_CLASSES = 2                      # two answers: referable or not referable
REFERABLE_THRESHOLD = 2              # DR grade 2 or higher counts as "referable" (needs a doctor)
HF_DATASET = "dreamxjei/aptos-mini"  # instructor-hosted pre-resized subset (labels 0-4)
NORM_MEAN = (0.485, 0.456, 0.406)    # average R,G,B of ImageNet photos; used to normalize inputs
NORM_STD = (0.229, 0.224, 0.225)     # spread (std dev) of R,G,B; pretrained models expect this scaling
GRADE_NAMES = ["not referable", "referable"]   # human-readable name for class 0 and class 1


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class _HFImageDataset(Dataset):
    """Wraps a HuggingFace split (columns: 'image' PIL, 'diagnosis' int) as a
    torch Dataset that applies a transform."""
    def __init__(self, hf_split, transform):   # runs once when we create the dataset
        self.ds = hf_split                     # remember the raw HuggingFace data split
        self.transform = transform             # remember the image-preprocessing steps to apply

    def __len__(self):                         # how many images are in this dataset
        return len(self.ds)                    # just the length of the underlying split

    def __getitem__(self, idx):                # fetch one sample by its position number
        row = self.ds[idx]                     # grab that row (has an 'image' and a 'diagnosis')
        img = row["image"].convert("RGB")      # make sure the image has 3 color channels (R,G,B)
        label = int(row["diagnosis"] >= REFERABLE_THRESHOLD)  # binary: referable DR
        return self.transform(img), label      # return the preprocessed image and its 0/1 label


def _transform(size: int, train: bool):
    """Build the ordered list of image preprocessing steps (different for train vs. val)."""
    # only during training: randomly flip and slightly rotate to teach the model variety
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(15)] if train else []
    # Resize the shorter edge then center-crop to a square: keeps the eye's true
    # proportions (no stretching) and matches how ResNet/ViT were pretrained.
    # Compose = run these steps in order: resize, crop, (augment), to-tensor, normalize
    return T.Compose(
        [T.Resize(size), T.CenterCrop(size), *aug, T.ToTensor(),
         T.Normalize(NORM_MEAN, NORM_STD)]   # rescale colors using the ImageNet mean/std above
    )


def get_loaders(size: int = 224, batch_size: int = 32):
    """Load the APTOS-mini subset from HuggingFace and return train/val loaders.

    `size` controls image resolution: use 64 for the flat-feature models
    (logreg, MLP) and 224 for the CNN/ResNet/ViT steps.
    """
    from datasets import load_dataset          # HuggingFace helper that downloads/loads datasets

    _quiet_hf()                                # silence the download chatter first
    ds = load_dataset(HF_DATASET)              # download (or load cached) the APTOS-mini dataset
    train = _HFImageDataset(ds["train"], _transform(size, train=True))    # training set, with augmentation
    val = _HFImageDataset(ds["validation"], _transform(size, train=False))   # validation set, no augmentation
    return (
        # training loader: hands out shuffled batches so the model doesn't memorize order
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),   # validation loader: fixed order, no shuffle needed
    )


def playground_setup(size: int = 224, batch_size: int = 32):
    """Rebuild what the Day-1 parameter playground needs, so those cells run even
    if you jump straight to them (e.g. after a kernel restart).

    Returns (nbfig, device, train_loader, val_loader).
    """
    import os                                  # file-path helpers
    import sys                                  # to adjust where Python looks for imports
    # add the shared '_shared' folder to the import path so 'import nbfig' below works
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
    import nbfig                                # our helper for pretty notebook figures
    nbfig.use()                                # apply its plot styling
    # pick the fastest hardware available: NVIDIA GPU (cuda), else Apple GPU (mps), else CPU
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader = get_loaders(size=size, batch_size=batch_size)   # build the data loaders
    return nbfig, device, train_loader, val_loader   # hand all four back to the notebook cell


def synthetic_loaders(size: int = 224, batch_size: int = 8, n: int = 32):
    """Random-noise loaders with the right shapes. For local smoke tests only,
    when the real HF dataset isn't downloaded."""
    class _Rand(Dataset):                       # a tiny fake dataset that makes up random images
        def __len__(self):                      # how many fake samples
            return n                            # n of them

        def __getitem__(self, i):               # make fake sample number i
            g = torch.Generator().manual_seed(i)   # seed so sample i is always the same random image
            # a random 3xHxW image, plus a random 0/1 label
            return torch.rand(3, size, size, generator=g), int(torch.randint(0, NUM_CLASSES, (1,), generator=g))

    return (
        DataLoader(_Rand(), batch_size=batch_size, shuffle=True),   # fake "train" loader
        DataLoader(_Rand(), batch_size=batch_size),   # fake "val" loader
    )


def flatten_for_classical(loader, max_n: Optional[int] = None):
    """Flatten image batches into (N, H*W*C) for sklearn models."""
    Xs, ys = [], []                            # empty lists to collect image data (X) and labels (y)
    seen = 0                                    # counter of how many images we've gathered so far
    for xb, yb in loader:                       # loop over batches: xb = images, yb = labels
        # flatten each image in the batch into one long row of numbers, and store as a numpy array
        Xs.append(xb.reshape(xb.size(0), -1).numpy())
        ys.append(np.asarray(yb))               # store this batch's labels as a numpy array
        seen += xb.size(0)                      # add this batch's size to the running count
        if max_n and seen >= max_n:             # if a cap was set and we've collected enough...
            break                               # ...stop looping early
    X = np.concatenate(Xs)                      # stack all image batches into one big array
    y = np.concatenate(ys)                      # stack all label batches into one big array
    if max_n:                                   # if a cap was set...
        X, y = X[:max_n], y[:max_n]             # ...trim to exactly max_n samples
    return X, y                                 # give back the flat features and their labels


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def make_mlp(in_features: int, num_classes: int = NUM_CLASSES, hidden=(256, 128)):
    """MLP on flattened pixels. Accepts image-shaped batches (N, C, H, W) and
    flattens them, so it drops into the same training loop as the CNN.
    `in_features` must equal C*H*W of the input images.
    """
    layers, d = [nn.Flatten()], in_features    # start with a flatten step; d = current input width
    for h in hidden:                            # for each hidden layer size we want...
        layers += [nn.Linear(d, h), nn.ReLU()]  # add a Linear layer of width h, then a ReLU nonlinearity
        d = h                                   # the next layer's input width is this layer's output width
    layers.append(nn.Linear(d, num_classes))    # final layer maps to one score per class
    return nn.Sequential(*layers)               # chain all the layers into one runnable model


def _activation(name: str):
    """Fresh activation module by name -- used by the Day-1 parameter playground."""
    # a lookup: text name -> the matching PyTorch activation class
    table = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh,
             "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "elu": nn.ELU}
    key = name.lower()                          # lowercase so "ReLU" and "relu" both work
    if key not in table:                        # if the requested name isn't in our table...
        raise ValueError(f"unknown activation {name!r}; try one of {list(table)}")   # ...error with hints
    return table[key]()                         # build and return a fresh activation module


def make_small_cnn(num_classes: int = NUM_CLASSES, dropout: float = 0.3, activation: str = "relu"):
    """A small from-scratch CNN with tunable knobs (for the parameter playground).

    dropout: regularization strength before the final layer (0 = none, 0.5 = heavy).
    activation: the nonlinearity after each conv ('relu', 'leaky_relu', 'tanh',
        'sigmoid', 'gelu', 'elu'). BatchNorm after each conv keeps training stable.
    """
    A = lambda: _activation(activation)   # a fresh module each call
    return nn.Sequential(
        # block 1: look for edges (3->32 feature maps), normalize, activate, halve the image size
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), A(), nn.MaxPool2d(2),
        # block 2: combine into richer patterns (32->64 maps), then shrink again
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), A(), nn.MaxPool2d(2),
        # block 3: deeper features (64->128 maps), then shrink again
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), A(), nn.MaxPool2d(2),
        # block 4: more features, then average each map down to a single number
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), A(), nn.AdaptiveAvgPool2d(1),
        # flatten to 128 numbers, randomly drop some to reduce overfitting, then map to class scores
        nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes),
    )


def make_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True):
    """Load a ResNet-50 (a famous 50-layer CNN) and adapt its head for our 2-class task."""
    import torchvision.models as tvm            # torchvision's collection of prebuilt models
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None   # learned ImageNet weights, or none
    m = tvm.resnet50(weights=weights)           # build the ResNet-50, optionally with those weights
    if freeze_backbone:                         # if we want to keep the pretrained body fixed...
        for p in m.parameters():                # ...loop over every weight in the model...
            p.requires_grad = False             # ...and mark it "don't update during training"
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # new head, trainable
    return m                                    # hand back the ready-to-train model


def make_vit_base(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = True):
    """Load a Vision Transformer (ViT) that treats an image as a grid of patches, for our 2-class task."""
    import timm                                 # 'timm' library with many pretrained image models
    m = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)   # ViT with a 2-class head
    if freeze_backbone:                         # if we want to keep the pretrained body fixed...
        for p in m.parameters():                # ...loop over every weight...
            p.requires_grad = False             # ...and freeze it (no updates)
        for p in m.get_classifier().parameters():  # train only the new head
            p.requires_grad = True              # unfreeze just the classifier head's weights
    return m                                    # hand back the model


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_model(model, train_loader, val_loader, epochs: int = 3, lr: float = 1e-3,
                device: str = "cpu", verbose: bool = True, weight_decay: float = 0.0):
    """Standard supervised loop.

    Returns list of (epoch, val_acc, train_loss) per epoch. The val_acc stays at
    index [1] so older callers (history[-1][1]) keep working; train_loss at [2]
    lets the notebook draw a learning curve.

    weight_decay: L2 regularization strength (0 = off). Another dial students can
    turn in the parameter playground to fight overfitting.
    """
    # free any GPU memory left by a previous (re-)run so repeated re-runs don't OOM
    import gc                                   # Python's garbage collector (frees unused memory)
    gc.collect()                                # ask it to clean up now
    if torch.cuda.is_available():               # if we're on an NVIDIA GPU...
        torch.cuda.empty_cache()                # ...release cached GPU memory from earlier runs
    model = model.to(device)                    # move the model onto the chosen device (GPU/CPU)
    # Adam optimizer: the algorithm that nudges the weights; only feeds it the trainable weights
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=weight_decay)   # lr = step size, weight_decay = L2 penalty
    loss_fn = nn.CrossEntropyLoss()             # measures how wrong the predictions are (for classification)
    history = []                                # will collect (epoch, val_acc, train_loss) each round
    for epoch in range(epochs):                 # repeat the whole dataset 'epochs' times
        model.train()                           # put model in training mode (dropout/batchnorm active)
        running, n = 0.0, 0                      # running = summed loss so far; n = images seen so far
        for xb, yb in train_loader:             # loop over training batches: xb images, yb labels
            xb, yb = xb.to(device), yb.to(device)   # move this batch to the same device as the model
            opt.zero_grad()                     # clear leftover gradients from the previous step
            loss = loss_fn(model(xb), yb)       # run the model and measure how wrong it was
            loss.backward()                     # compute gradients (how to tweak each weight)
            opt.step()                          # apply the tweak: update the weights
            running += float(loss) * len(xb); n += len(xb)   # accumulate loss and image count
        train_loss = running / max(n, 1)        # average training loss this epoch (avoid divide-by-zero)
        # measure accuracy on the validation set (argmax picks the higher-scoring class)
        acc = evaluate_classifier(lambda b: model(b).argmax(1), val_loader, device)["accuracy"]
        history.append((epoch + 1, acc, train_loss))   # record this epoch's numbers
        if verbose:                             # if printing is on...
            # ...show a tidy progress line for this epoch
            print(f"  epoch {epoch + 1:2d}/{epochs}  train_loss {train_loss:.3f}  val_acc {acc:.3f}")
    return history                              # give back the per-epoch results for plotting


def evaluate_classifier(predict_fn: Callable, loader, device: str = "cpu"):
    """Eval any model. `predict_fn(batch_on_device) -> predicted label tensor/array.`"""
    ys, ps = [], []                            # collect true labels (ys) and predicted labels (ps)
    was_training = None                         # placeholder (kept for compatibility; unused here)
    for xb, yb in loader:                       # loop over batches: xb images, yb true labels
        xb = xb.to(device) if hasattr(xb, "to") else xb   # move images to device if they're tensors
        with torch.no_grad():                   # turn off gradient tracking (faster, we're only predicting)
            pred = predict_fn(xb)               # ask the model for its predicted labels
        pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)   # to a plain numpy array
        ys.append(np.asarray(yb))               # store this batch's true labels
        ps.append(pred.reshape(-1))             # store predictions as a flat 1-D row
    y = np.concatenate(ys)                      # all true labels, stacked together
    p = np.concatenate(ps)                      # all predictions, stacked together
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)   # empty confusion matrix (rows=true, cols=pred)
    for t, q in zip(y, p):                       # for each (true label t, predicted label q) pair...
        conf[int(t), int(q)] += 1               # ...add one to that cell of the confusion matrix
    # per-class accuracy: for each class, fraction it got right (diagonal / row total)
    per_class = {GRADE_NAMES[c]: (conf[c, c] / conf[c].sum() if conf[c].sum() else 0.0) for c in range(NUM_CLASSES)}
    # bundle up overall accuracy, the confusion matrix, per-class scores, and the raw labels/preds
    return {"accuracy": float((y == p).mean()), "confusion": conf, "per_class": per_class, "y": y, "pred": p}


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _denorm(img_tensor):
    """Undo the Normalize step so an image looks natural again for display."""
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)   # the mean we subtracted, shaped to match the image
    std = torch.tensor(NORM_STD).view(3, 1, 1)     # the std we divided by, same shape
    return (img_tensor.cpu() * std + mean).clamp(0, 1)   # reverse it, then clamp colors into 0..1


def show_rgb_split(img_tensor):
    """Show an image and its R, G, B channels separately."""
    import matplotlib.pyplot as plt             # the plotting library
    img = _denorm(img_tensor)                   # undo normalization so colors look right
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))   # one row of 4 side-by-side plots
    axes[0].imshow(img.permute(1, 2, 0)); axes[0].set_title("RGB")   # full color image (reorder axes to H,W,C)
    # loop over the 3 channels, drawing each one alone in its own color scale
    for i, (name, cmap) in enumerate([("Red", "Reds"), ("Green", "Greens"), ("Blue", "Blues")]):
        axes[i + 1].imshow(img[i], cmap=cmap); axes[i + 1].set_title(name)   # channel i in its colormap
    for a in axes:                              # for every subplot...
        a.axis("off")                           # ...hide the x/y axis ticks
    plt.tight_layout()                          # tidy the spacing so titles don't overlap
    plt.show()  # display once; don't return the fig (Jupyter would echo it -> double image)


def show_pixel_histogram(img_tensor):
    """Plot how bright the pixels are in each color channel (a histogram per channel)."""
    import matplotlib.pyplot as plt             # the plotting library
    img = _denorm(img_tensor)                   # undo normalization first
    fig, ax = plt.subplots(figsize=(6, 3))      # one plot to hold all three histograms
    for i, c in enumerate(["red", "green", "blue"]):   # for each color channel...
        # draw a semi-transparent histogram of that channel's pixel values
        ax.hist(img[i].flatten().numpy(), bins=40, color=c, alpha=0.5, label=c)
    ax.set_title("Pixel intensity per channel"); ax.legend()   # title and a color legend
    plt.show()  # display once; don't return the fig (Jupyter would echo it -> double image)


def show_augmentations(pil_image, size: int = 224):
    """Show the same image under several augmentations side-by-side."""
    import matplotlib.pyplot as plt             # the plotting library
    base = T.Resize((size, size))(pil_image.convert("RGB"))   # make an RGB square starting image
    variants = {                                # a dictionary of name -> transformed image
        "original": base,                       # the untouched image
        "h-flip": T.functional.hflip(base),     # mirrored left-to-right
        "rotate 15": T.functional.rotate(base, 15),   # rotated 15 degrees
        "brighter": T.functional.adjust_brightness(base, 1.6),   # 1.6x brighter
    }
    fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3))   # one plot per variant
    for ax, (name, im) in zip(axes, variants.items()):   # pair each subplot with a named image...
        ax.imshow(im); ax.set_title(name); ax.axis("off")   # ...draw it, title it, hide the axes
    plt.tight_layout()                          # tidy the spacing
    plt.show()                                  # display the figure


def show_first_layer_filters(model):
    """Visualize the first conv layer's learned filters (for the from-scratch CNN)."""
    import matplotlib.pyplot as plt             # the plotting library
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))   # find the first Conv2d layer
    w = conv.weight.detach().cpu()              # grab its learned filter weights (as plain numbers)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)   # rescale to 0..1 so they display as images
    n = min(16, w.size(0))                      # show at most 16 filters
    fig, axes = plt.subplots(2, n // 2, figsize=(n, 3))   # arrange them in a 2-row grid
    for i, ax in enumerate(axes.flat):          # for each filter's subplot...
        ax.imshow(w[i].permute(1, 2, 0)); ax.axis("off")   # ...draw the filter (reorder to H,W,C), hide axes
    fig.suptitle("First-layer filters")         # overall title
    plt.show()                                  # display the figure


def show_feature_maps(model, img_tensor, device: str = "cpu", k: int = 4):
    """Show what the network 'sees' at THREE depths -- not just the first layer.

    Hooks an early, a middle, and a late Conv2d layer, runs one image through, and plots a few
    channels from each: early layers fire on fine edges, later ones on bigger, more abstract shapes.
    Works for the from-scratch CNN and the pretrained ResNet alike.
    """
    import numpy as np                          # number arrays
    from PIL import Image                        # image handling (for resizing the maps)
    import nbfig                                 # our notebook figure helper
    model = model.to(device).eval()             # move model to device and set evaluation mode
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]   # list every Conv2d layer in order
    # pick an early, a middle, and a late conv layer (or all of them if there are fewer than 3)
    picks = [convs[0], convs[len(convs) // 2], convs[-1]] if len(convs) >= 3 else convs
    acts = {}                                    # will store the output ("activation") of each picked layer
    # attach a "hook" to each picked layer that saves its output when the image passes through
    handles = [layer.register_forward_hook(
        (lambda key: lambda m, i, o: acts.__setitem__(key, o.detach()))(j))
        for j, layer in enumerate(picks)]
    with torch.no_grad():                        # no gradients needed, we're just observing
        model(img_tensor.unsqueeze(0).to(device))   # run one image (add a batch dimension) through the model
    for h in handles:                            # for each hook we attached...
        h.remove()                               # ...detach it so it doesn't linger

    def channels(a):                             # turn one layer's output into a few small preview images
        a = a[0].cpu()                           # take the first (only) image, move to CPU
        idx = a.flatten(1).var(1).argsort(descending=True)[:k]   # the busiest channels
        out = []                                 # collected preview images
        for c in idx:                            # for each chosen channel...
            m = a[c].numpy()                     # that channel's 2-D activation map as numbers
            m = (m - m.min()) / (np.ptp(m) + 1e-9)   # rescale to 0..1 (ptp = max minus min)
            # convert to a grayscale image, enlarge to 96x96 with blocky pixels, store it
            out.append(np.array(Image.fromarray((m * 255).astype("uint8")).resize((96, 96), Image.NEAREST)))
        while len(out) < k:                      # if we found fewer than k channels...
            out.append(np.zeros((96, 96), "uint8"))   # ...pad with black squares
        return np.stack(out)                     # stack the previews into one array

    mats = [channels(acts[j]) for j in range(len(picks))]   # build previews for each picked layer
    while len(mats) < 3:                          # make sure we have 3 sets (the plotter expects 3)...
        mats.append(mats[-1])                     # ...by repeating the last one if needed
    nbfig.show_feature_maps(mats[0], mats[1], mats[2]).show()   # draw the early/middle/late feature maps


def gradcam(model, img_tensor, target_class: Optional[int] = None, device: str = "cpu"):
    """Grad-CAM heatmap for a CNN/ResNet. Returns (heatmap HxW numpy, predicted_class).

    Hooks the last Conv2d layer. Works for make_small_cnn and make_resnet50.
    """
    model = model.to(device).eval()             # move model to device and set evaluation mode
    last_conv = [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]   # the final Conv2d layer
    acts, grads = {}, {}                         # will hold the layer's output and its gradients

    def fwd_hook(_, __, out):                    # runs when the image flows forward through the layer
        acts["v"] = out.detach()                # save the layer's output (the activations)

    def bwd_hook(_, gin, gout):                  # runs during backprop through the layer
        grads["v"] = gout[0].detach()           # save the gradient flowing back into the layer

    h1 = last_conv.register_forward_hook(fwd_hook)   # attach the forward hook
    h2 = last_conv.register_full_backward_hook(bwd_hook)   # attach the backward hook
    try:                                         # do the work, and always clean up hooks afterward
        # requires_grad on the input forces autograd to build the full backward
        # graph through the conv layers even when the backbone is frozen, so the
        # gradient hook on the last conv actually fires.
        x = img_tensor.unsqueeze(0).to(device).requires_grad_(True)   # one image, tracking gradients
        logits = model(x)                        # run it: logits = raw scores for each class
        cls = int(logits.argmax(1)) if target_class is None else target_class   # class to explain (top or given)
        model.zero_grad()                        # clear any old gradients
        logits[0, cls].backward()                # backprop from that class's score to fill the hooks
        a, g = acts["v"][0], grads["v"][0]       # the saved activations and gradients (drop batch dim)
        weights = g.mean(dim=(1, 2))             # importance of each channel = its average gradient
        cam = torch.relu((weights[:, None, None] * a).sum(0))   # weighted sum of maps, keep positives only
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)   # rescale the heatmap to 0..1
        return cam.cpu().numpy(), cls            # give back the heatmap and which class it explains
    finally:                                     # this always runs, even if something errored above
        h1.remove()                              # detach the forward hook
        h2.remove()                              # detach the backward hook
