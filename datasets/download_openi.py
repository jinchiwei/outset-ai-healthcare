"""Download the Open-i Indiana University Chest X-ray collection.

Real chest X-ray images + real radiologist reports + MeSH labels. No PhysioNet
credentialing, no Kaggle account, no auth. Instructor runs this once; the small
processed artifacts (cached LLM extractions, demographics) get committed so students
don't need the raw images for the lab.

Usage:  python datasets/download_openi.py
"""
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets/raw/openi"
IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"


def fetch(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest.name} present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[done] {dest.name}")


def extract(archive: Path, into: Path):
    if (into / ".extracted").exists():
        print(f"[skip] {into.name} already extracted")
        return
    into.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive.name} -> {into}")
    with tarfile.open(archive) as tf:
        tf.extractall(into)
    (into / ".extracted").touch()


def main():
    fetch(IMAGES_URL, ROOT / "NLMCXR_png.tgz")
    fetch(REPORTS_URL, ROOT / "NLMCXR_reports.tgz")
    extract(ROOT / "NLMCXR_png.tgz", ROOT / "images")
    extract(ROOT / "NLMCXR_reports.tgz", ROOT / "reports")
    n_img = len(list((ROOT / "images").rglob("*.png")))
    n_rep = len(list((ROOT / "reports").rglob("*.xml")))
    print(f"ok: {n_img} images, {n_rep} reports")


if __name__ == "__main__":
    main()
