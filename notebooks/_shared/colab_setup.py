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

REQUIRED = [
    "torch", "torchvision", "timm",
    "scikit-learn", "matplotlib", "seaborn", "pillow",
    "tqdm", "pandas", "numpy",
    "transformers", "tabpfn",
    "pyradiomics", "SimpleITK",
]

# pip name -> import name, when they differ
IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "pyradiomics": "radiomics",
    "SimpleITK": "SimpleITK",
}


def ensure(*pkgs):
    """Install any missing packages. Defaults to the full course requirement set."""
    targets = pkgs or REQUIRED
    missing = []
    for p in targets:
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
