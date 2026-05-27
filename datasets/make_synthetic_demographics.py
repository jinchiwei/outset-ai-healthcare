"""Instructor-only: generate synthetic demographics for Open-i cases.

Open-i ships images + reports but no patient demographics. We generate plausible,
clearly-synthetic demographics (age, sex, smoker) with a mild correlation to the
target finding, so the demographics channel carries a little real signal in the
multimodal lab. The header of the output CSV says it's synthetic.

Usage:  python datasets/make_synthetic_demographics.py --target Cardiomegaly
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/day2_multimodal"))
OUT = ROOT / "datasets/synthetic_demographics.csv"
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Cardiomegaly")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    import common  # noqa: E402

    rng = np.random.RandomState(SEED)
    rows = []
    for case_id, _img, _rec, label in common.list_cases(args.target, balanced=True):
        # Positive cases skew older, more likely smokers (mild, synthetic signal).
        base_age = 62 if label else 50
        age = int(np.clip(rng.normal(base_age, 12), 18, 92))
        smoker = int(rng.random() < (0.45 if label else 0.30))
        sex = "M" if rng.random() < 0.5 else "F"
        rows.append({"case_id": case_id, "age": age, "sex": sex, "smoker": smoker})

    df = pd.DataFrame(rows)
    with open(OUT, "w") as f:
        f.write("# SYNTHETIC demographics for Open-i cases (generated, not real patient data).\n")
        f.write(f"# age/smoker mildly correlated with {args.target} label. seed={SEED}.\n")
        df.to_csv(f, index=False)
    print(f"wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
