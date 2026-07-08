"""Single-cell setup for Colab. Idempotent: students can rerun safely.

Usage in a notebook:
    import sys; sys.path.insert(0, "../_shared")
    import colab_setup
    colab_setup.ensure()      # installs anything missing
    colab_setup.gpu_check()   # reports whether a GPU is attached
"""
import importlib
import subprocess
import sys

# Package groups per day. Most are preinstalled on Colab; ensure() only installs
# what's actually missing, so on Colab this is fast (often just timm / tabpfn).
DAY1 = [
    "torch", "torchvision", "timm",
    "scikit-learn", "matplotlib", "seaborn", "pillow",
    "tqdm", "pandas", "numpy", "datasets", "ipywidgets",
]
# tabpfn is PINNED to 2.2.1: newer (8.x) releases gate the model-weight download
# behind a one-time license token, which fails on Colab's non-interactive runtime
# (TabPFNLicenseError). 2.2.1 downloads weights freely from HuggingFace -- no token,
# no per-student setup -- and gives identical results.
DAY2 = ["pandas", "numpy", "scipy", "scikit-learn", "scikit-image", "matplotlib", "pillow",
        "tabpfn==2.2.1", "catboost"]
# Day 3 capstone, tabular groups (heart / estrogen): sklearn model zoo + tabular foundation model.
DAY3_TABULAR = ["ucimlrepo", "catboost", "tabpfn==2.2.1", "scikit-learn", "pandas", "numpy", "matplotlib"]
# Day 3 capstone, CRISPR sequence group: a guide is a short string, so no torch/GPU needed.
DAY3_SEQ = ["scikit-learn", "catboost", "pandas", "numpy", "matplotlib"]

REQUIRED = DAY1  # default = the Day 1 set

# pip name -> import name, when they differ
IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    "huggingface_hub": "huggingface_hub",
}


def ensure(*pkgs):
    """Install any missing packages. Defaults to the full course requirement set.

    A pinned spec (e.g. 'tabpfn==2.2.1') is ALWAYS (re)installed so the exact
    version is guaranteed even if Colab preinstalled a different one; unpinned
    packages are only installed when missing.
    """
    targets = pkgs or REQUIRED
    missing = []
    for p in targets:
        if any(op in p for op in ("==", "<", ">", "~", "!")):
            missing.append(p)   # pinned -> force the exact version
            continue
        import_name = IMPORT_NAMES.get(p, p.replace("-", "_").lower())
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(p)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    print(f"setup ok: {len(targets)} packages ready")


def gpu_check():
    """Report whether a CUDA GPU is attached. Nudge the student if not."""
    try:
        import torch
    except ImportError:
        print("torch not installed yet -- run ensure() first")
        return
    if torch.cuda.is_available():
        print(f"GPU OK: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Go to Runtime -> Change runtime type -> T4 GPU, then rerun.")
