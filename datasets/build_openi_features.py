"""Instructor-only, one-shot: build the Open-i multimodal feature table.

For ~N cases, extracts image features (scikit-image), pulls cached LLM features, adds
synthetic demographics, and writes one tidy table to datasets/openi_features.csv with
a `label` column. This small CSV is committed; students load it in the lab and never
need the 1.5 GB of raw chest X-rays.

Prereqs (instructor runs first):
  python datasets/download_openi.py
  python scripts/cache_openi_llm.py --mode anthropic   # (or --mode rules)
  python datasets/make_synthetic_demographics.py

Usage:  python datasets/build_openi_features.py --target Cardiomegaly --n 600
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/day2_multimodal"))
import common  # noqa: E402

OUT = ROOT / "datasets/openi_features.csv"
SAMPLE_DIR = ROOT / "notebooks/day2_multimodal/sample_images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Cardiomegaly")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-samples", type=int, default=3, help="sample CXRs to bundle for the demo")
    args = ap.parse_args()

    cases = common.list_cases(args.target, balanced=True)
    rows = []
    for i, (case_id, img_path, _rec, label) in enumerate(tqdm(cases, desc="features")):
        row = common.build_feature_row(case_id, img_path, use_text=True).to_dict()
        row["case_id"] = case_id
        row["label"] = label
        rows.append(row)
        # stash a few raw sample images for the notebook's extraction demo
        if i < args.n_samples:
            from PIL import Image
            SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
            Image.open(img_path).convert("L").resize((256, 256)).save(SAMPLE_DIR / f"{case_id}.png")

    df = pd.DataFrame(rows)
    # tidy column order: ids/label first, then image, text, demographic groups
    front = ["case_id", "label"]
    cols = front + [c for c in df.columns if c not in front]
    df[cols].to_csv(OUT, index=False)
    pos = int(df["label"].sum())
    print(f"wrote {len(df)} rows ({pos} positive / {len(df)-pos} negative) -> {OUT}")
    print(f"feature columns: {[c for c in df.columns if c not in front]}")
    print(f"bundled {args.n_samples} sample images -> {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
