"""Build a small brain-CT npz for the Day 3 capstone (Group 3: head trauma / stroke).

Source: little-duck/brain_CT_transform on the HuggingFace Hub -- 7369 brain CT slices
labelled stroke-present vs stroke-absent (Turkish "inme var / inme yok"). The full set is
~30 GB of 512x512x4 float arrays, so we stream a handful of parquet shards, take a
balanced subset, downsample each slice to 64x64 grayscale, and save a ~10 MB npz that a
student can load in Colab in one second.

We FLIP the label so 1 = stroke present (the "disease" / positive class), matching the
intuition used everywhere else in the course (positive = the thing you are screening for).

IMPORTANT teaching note for Group 3: this dataset ships an image and a label and NOTHING
else -- no age, no sex, no scanner, no site. That absence is the point of their project:
you cannot audit a model for fairness across groups the dataset never records.

Run:  python datasets/build_brain_ct.py
Output: datasets/brain_ct.npz  (X uint8 [N,64,64], y int [N], class_names)
"""
import sys
from pathlib import Path

import numpy as np

N_PER_CLASS = 600          # balanced subset -> 1200 slices total
SIZE = 224
OUT = Path(__file__).resolve().parent / "brain_ct.npz"
N_TRAIN_SHARDS = 42


def _resize_gray(arr, size=SIZE):
    """(4,512,512) float in [-1,1] -> (size,size) uint8 [0,255] grayscale.

    Take the first channel (the brain-window image), rescale to 0..255, resize with PIL.
    """
    from PIL import Image
    ch = np.asarray(arr, dtype=np.float32)[0]          # (512,512), values in [-1,1]
    ch = (ch + 1.0) * 127.5                             # -> [0,255]
    im = Image.fromarray(np.clip(ch, 0, 255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def build():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    per_class = {0: [], 1: []}                          # keyed by FLIPPED label (1=stroke)
    for shard in range(N_TRAIN_SHARDS):
        if all(len(v) >= N_PER_CLASS for v in per_class.values()):
            break
        fn = f"data/train-{shard:05d}-of-{N_TRAIN_SHARDS:05d}.parquet"
        path = hf_hub_download("little-duck/brain_CT_transform", fn, repo_type="dataset")
        tbl = pq.read_table(path).to_pydict()
        for img, lab in zip(tbl["image"], tbl["label"]):
            y = 1 - int(lab)                            # 0="inme var"(stroke) -> 1 ; 1="yok" -> 0
            if len(per_class[y]) >= N_PER_CLASS:
                continue
            per_class[y].append(_resize_gray(img))
        print(f"shard {shard:02d}: stroke={len(per_class[1])}  normal={len(per_class[0])}", flush=True)

    n = min(len(per_class[0]), len(per_class[1]))
    X = np.stack(per_class[1][:n] + per_class[0][:n])                    # stroke first, then normal
    y = np.array([1] * n + [0] * n, dtype=np.int64)
    # shuffle deterministically so train/val/test splits downstream are mixed
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    np.savez_compressed(OUT, X=X, y=y, class_names=np.array(["normal", "stroke"]))
    mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT}  shape={X.shape}  stroke={int(y.sum())}  normal={int((y==0).sum())}  ({mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(build())
