"""Day 2 helpers: Open-i chest X-ray multimodal stack.

The big idea of Day 2: every signal becomes a row of numbers, then a foundation
model (TabPFN) handles the table. Each modality contributes its own *vote*:

  image  -> a trained image model's probability  (late fusion / stacking; pre-cached)
  report -> structured findings via the Anthropic API (pre-cached to JSON)
  patient-> demographics (synthetic for Open-i)
  concat -> one tabular row -> TabPFN

The image vote (`img_pred`): rather than feed handcrafted radiomics or a raw 512-d
embedding into the table, we train an actual image classifier (transfer learning, like
Day 1) and use ITS single probability as the image's contribution. The instructor
pre-computes these out-of-fold (cross_val_predict) so each case is scored by a model
that never trained on it -- see scripts/cache_openi_image_preds.py. One honest number
per X-ray, cached to JSON.

We still keep `extract_image_features` below: it's the classic handcrafted-radiomics
way (first-order intensity + GLCM texture via scikit-image, since PyRadiomics won't
build on Python 3.12). The notebook shows it once to contrast "handcraft numbers" with
"let a trained model vote."
"""
from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

OPENI_ROOT = Path(__file__).resolve().parents[2] / "datasets/raw/openi"
LLM_CACHE_PATH = Path(__file__).resolve().parents[2] / "datasets/openi_llm_extractions.json"
IMG_PRED_CACHE_PATH = Path(__file__).resolve().parents[2] / "datasets/openi_image_preds.json"
DEMO_PATH = Path(__file__).resolve().parents[2] / "datasets/synthetic_demographics.csv"


# --------------------------------------------------------------------------- #
# Open-i reports + image pairing
# --------------------------------------------------------------------------- #
def parse_report(xml_path: Path) -> dict:
    """Pull findings, impression, linked image ids, and MeSH major topics."""
    root = ET.parse(xml_path).getroot()
    text = {}
    abstract = root.find(".//Abstract")
    if abstract is not None:
        for at in abstract.findall("AbstractText"):
            text[at.get("Label", "").lower()] = (at.text or "").strip()
    images = [img.get("id") for img in root.findall(".//parentImage")]
    mesh = []
    mesh_node = root.find(".//MeSH")
    if mesh_node is not None:
        for m in mesh_node.findall(".//major"):
            if m.text:
                mesh.append(m.text.strip())
    return {
        "findings": text.get("findings", ""),
        "impression": text.get("impression", ""),
        "images": images,
        "mesh_majors": mesh,
    }


def list_cases(target_label: str = "Cardiomegaly", n: int | None = None,
               balanced: bool = False, seed: int = 0):
    """Return (case_id, image_path, report_dict, label) tuples.

    label = 1 if `target_label` appears in the report's MeSH major topics.

    balanced=True: scan all reports, keep every positive case and an equal number of
    randomly-chosen negatives (deterministic given seed), so the binary task isn't
    swamped by the majority class. All pipeline scripts call this the same way, so
    they select the identical set of cases.
    """
    import random

    reports_dir = OPENI_ROOT / "reports/ecgen-radiology"
    images_dir = OPENI_ROOT / "images"
    scanned = []
    for xml in sorted(reports_dir.glob("*.xml")):
        rec = parse_report(xml)
        if not rec["images"]:
            continue
        img = images_dir / f"{rec['images'][0]}.png"
        if not img.exists():
            continue
        label = int(any(target_label.lower() in m.lower() for m in rec["mesh_majors"]))
        scanned.append((xml.stem, img, rec, label))
        if not balanced and n and len(scanned) >= n:
            break

    if not balanced:
        return scanned

    pos = [c for c in scanned if c[3] == 1]
    neg = [c for c in scanned if c[3] == 0]
    random.Random(seed).shuffle(neg)
    k = min(len(pos), len(neg))
    if n:
        k = min(k, n // 2)
    sel = pos[:k] + neg[:k]
    random.Random(seed).shuffle(sel)
    return sel


# --------------------------------------------------------------------------- #
# Image features (scikit-image; same families as PyRadiomics firstorder + glcm)
# --------------------------------------------------------------------------- #
def extract_image_features(image_path: Path, size: int = 256, levels: int = 32) -> dict:
    """Radiomics-style handcrafted features from a grayscale chest X-ray.

    First-order intensity stats + GLCM texture. Returns a flat dict of floats.
    """
    from PIL import Image
    from scipy.stats import skew, kurtosis
    from skimage.feature import graycomatrix, graycoprops

    g = np.asarray(Image.open(image_path).convert("L").resize((size, size)), dtype=np.float64)
    flat = g.ravel()

    feats = {
        "intensity_mean": float(flat.mean()),
        "intensity_std": float(flat.std()),
        "intensity_skew": float(skew(flat)),
        "intensity_kurtosis": float(kurtosis(flat)),
        "intensity_p10": float(np.percentile(flat, 10)),
        "intensity_p50": float(np.percentile(flat, 50)),
        "intensity_p90": float(np.percentile(flat, 90)),
    }
    # histogram entropy
    hist, _ = np.histogram(flat, bins=levels, range=(0, 255), density=True)
    hist = hist[hist > 0]
    feats["intensity_entropy"] = float(-(hist * np.log2(hist)).sum())

    # GLCM texture, averaged over 4 directions at distance 1
    q = (g / 256 * levels).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop).mean())

    return feats


# --------------------------------------------------------------------------- #
# Text features (pre-cached LLM extractions) + demographics
# --------------------------------------------------------------------------- #
def load_cached_llm_features(case_id: str) -> dict:
    cache = json.loads(LLM_CACHE_PATH.read_text())
    rec = cache.get(case_id, {})
    severity_map = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    feats = {}
    for k, v in rec.items():
        if isinstance(v, bool):
            feats[f"llm_{k}"] = int(v)
        elif isinstance(v, str):
            feats[f"llm_{k}_ord"] = severity_map.get(v.lower(), 0)
        elif isinstance(v, (int, float)):
            feats[f"llm_{k}"] = float(v)
    return feats


def load_cached_image_pred(case_id: str) -> dict:
    """The image model's out-of-fold probability for this case (late fusion vote).

    Pre-computed by scripts/cache_openi_image_preds.py. One honest number per X-ray.
    """
    cache = json.loads(IMG_PRED_CACHE_PATH.read_text())
    return {"img_pred": float(cache.get(case_id, 0.5))}


def load_demographics(case_id: str) -> dict:
    df = pd.read_csv(DEMO_PATH, comment="#").set_index("case_id")
    if case_id in df.index:
        row = df.loc[case_id]
        return {"age": int(row["age"]), "sex_male": int(row["sex"] == "M"), "smoker": int(row["smoker"])}
    return {"age": 50, "sex_male": 0, "smoker": 0}


def build_feature_row(case_id: str, image_path: Path, use_text: bool = True) -> pd.Series:
    """Combine the image vote + (optional) text + demographic features into one row.

    The image channel is the trained model's cached probability (`img_pred`), not
    handcrafted radiomics -- late fusion / stacking.
    `use_text=False` drops the LLM features -- used for the leakage ablation.
    """
    feats = {}
    feats.update(load_cached_image_pred(case_id))
    if use_text:
        feats.update(load_cached_llm_features(case_id))
    feats.update(load_demographics(case_id))
    return pd.Series(feats)
