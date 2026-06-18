"""Instructor-only, one-shot: build a small APTOS subset and push it to HuggingFace.

Takes the full APTOS-2019 download (datasets/raw/aptos/), resizes every image to
224x224, takes a class-balanced subsample, splits train/val, and pushes the result
to the HuggingFace Hub as a Dataset. Students then load it in one line with no
Kaggle token and no 9 GB download.

Usage (from repo root, in conda env outset):
    HF_TOKEN=hf_xxx python datasets/prep_aptos_subset.py \
        --per-class 180 --repo dreamxjei/aptos-mini --push

Without --push it just builds + saves locally to datasets/aptos_mini/ for inspection.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets/raw/aptos"
OUT_LOCAL = ROOT / "datasets/aptos_mini"  # gitignored local copy for inspection
IMG_SIZE = 224
VAL_FRAC = 0.2
SEED = 0


def build_dataframe(per_class: int) -> pd.DataFrame:
    df = pd.read_csv(RAW / "train.csv")  # columns: id_code, diagnosis
    # class-balanced subsample: up to `per_class` per grade (capped by availability)
    parts = []
    for grade, group in df.groupby("diagnosis"):
        n = min(per_class, len(group))
        parts.append(group.sample(n=n, random_state=SEED))
    bal = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print("class counts after balancing:")
    print(bal["diagnosis"].value_counts().sort_index().to_string())
    return bal


def square_resize(img: Image.Image, size: int) -> Image.Image:
    """Resize the shorter edge to `size`, then center-crop to size x size.

    Aspect-ratio preserving: the round retina stays round (a plain
    `.resize((size, size))` squashes the wide APTOS photos into ovals).
    APTOS images are wider than tall with the retina centered, so the
    center crop trims the black side margins and keeps the full disc.
    """
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def to_hf_dataset(df: pd.DataFrame):
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, ClassLabel, Value

    img_dir = RAW / "train_images"

    def gen():
        for _, row in df.iterrows():
            path = img_dir / f"{row['id_code']}.png"
            img = square_resize(Image.open(path).convert("RGB"), IMG_SIZE)
            yield {"image": img, "diagnosis": int(row["diagnosis"]), "id_code": row["id_code"]}

    features = Features({
        "image": HFImage(),
        "diagnosis": ClassLabel(names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"]),
        "id_code": Value("string"),
    })
    full = Dataset.from_generator(gen, features=features)
    split = full.train_test_split(test_size=VAL_FRAC, seed=SEED, stratify_by_column="diagnosis")
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=180, help="max images per DR grade")
    ap.add_argument("--repo", default="dreamxjei/aptos-mini", help="HF Hub dataset repo id")
    ap.add_argument("--push", action="store_true", help="push to HF Hub (needs HF_TOKEN)")
    args = ap.parse_args()

    if not (RAW / "train.csv").exists():
        raise SystemExit(f"APTOS not found at {RAW}. Run the Kaggle download first.")

    df = build_dataframe(args.per_class)
    ds = to_hf_dataset(df)
    print(ds)

    ds.save_to_disk(str(OUT_LOCAL))
    print(f"saved local copy to {OUT_LOCAL}")

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--push needs HF_TOKEN env var")
        ds.push_to_hub(args.repo, token=token, private=False)
        print(f"pushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
