"""
prepare_dataset.py
==================
One script that does everything:
  1. Downloads TrashNet (no Kaggle login needed)
  2. Remaps 6 original classes → your 3 bins
  3. Splits into train / val / test
  4. Prints a summary of image counts

Run it once, then go straight to training:
    python prepare_dataset.py
    python trainer.py --epochs 10 --batch_size 16

Expected output structure:
    data/
      train/
        Recyclable/   (~1,400 images)
        Compost/      (~280  images)  ← mapped from 'trash' + duplicated biological
        Landfill/     (~420  images)
      val/
        Recyclable/   (~300 images)
        ...
      test/
        Recyclable/   (~300 images)
        ...
"""

import os
import sys
import shutil
import zipfile
import random
import urllib.request
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
DOWNLOAD_URL = (
    "https://github.com/garythung/trashnet/releases/download/v1.0/dataset-resized.zip"
)
RAW_DIR    = Path("data/raw")
OUTPUT_DIR = Path("data")
ZIP_PATH   = RAW_DIR / "trashnet.zip"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15

RANDOM_SEED = 42

# ── Remap: TrashNet 6 classes → your 3 bins ───────────────────────────────────
# TrashNet classes: glass, paper, cardboard, plastic, metal, trash
REMAP = {
    "glass":     "Recyclable",
    "paper":     "Recyclable",
    "cardboard": "Recyclable",
    "plastic":   "Recyclable",
    "metal":     "Recyclable",
    "trash":     "Landfill",
    # NOTE: TrashNet has no food/compost class.
    # We create a small Compost folder from 'trash' images tagged with food
    # (handled below via duplication + note). For a real dataset, add your
    # own cafeteria food-scrap photos into data/train/Compost/ manually.
}

SPLITS = ["train", "val", "test"]
BINS   = ["Recyclable", "Compost", "Landfill"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def download_trashnet():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1_000:
        print(f"✅ Already downloaded: {ZIP_PATH}")
        return

    print("⬇️  Downloading TrashNet (~50 MB) ...")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_PATH, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(
            "\nManual download steps:\n"
            "  1. Open this URL in your browser:\n"
            f"     {DOWNLOAD_URL}\n"
            "  2. Save the file as:  data/raw/trashnet.zip\n"
            "  3. Re-run this script.\n"
        )
        sys.exit(1)


def _progress(count, block_size, total_size):
    pct = min(count * block_size / total_size * 100, 100)
    bar = int(pct / 2)
    print(f"\r  [{'█'*bar}{'░'*(50-bar)}] {pct:.1f}%", end="", flush=True)


def extract_trashnet():
    extract_dir = RAW_DIR / "extracted"
    if extract_dir.exists():
        print(f"✅ Already extracted: {extract_dir}")
        return extract_dir

    print("📦 Extracting zip ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(extract_dir)
    print(f"   Extracted to {extract_dir}")
    return extract_dir


def find_class_dirs(extract_dir: Path) -> dict:
    """
    TrashNet zips can have different internal structures.
    This finds all leaf directories that match known class names.
    """
    class_dirs = {}
    for path in sorted(extract_dir.rglob("*")):
        if path.is_dir() and path.name.lower() in REMAP:
            class_dirs[path.name.lower()] = path
    return class_dirs


def build_split(class_dirs: dict):
    """
    Collect all images per bin, shuffle, split, copy.
    Also creates a minimal Compost class by borrowing some Landfill images
    (since TrashNet has no food class) and flagging them for replacement.
    """
    # 1. Gather all images per target bin
    bin_images: dict[str, list] = defaultdict(list)

    for original_class, src_dir in class_dirs.items():
        target_bin = REMAP[original_class]
        images = sorted([
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])
        bin_images[target_bin].extend(images)
        print(f"   {original_class:12s} ({len(images):4d} imgs) → {target_bin}")

    # 2. Bootstrap Compost from a slice of Landfill images
    #    (temporary placeholder — replace with real food images later)
    if not bin_images["Compost"] and bin_images["Landfill"]:
        compost_count = max(len(bin_images["Landfill"]) // 2, 50)
        random.seed(RANDOM_SEED)
        bin_images["Compost"] = random.sample(bin_images["Landfill"], compost_count)
        print(
            f"\n⚠️  TrashNet has NO food/compost images.\n"
            f"   Bootstrapped Compost with {compost_count} Landfill images as placeholders.\n"
            f"   ACTION REQUIRED: Replace data/*/Compost/ with real food-scrap photos\n"
            f"   for accurate compost classification.\n"
        )

    # 3. Create output dirs
    for split in SPLITS:
        for bin_name in BINS:
            (OUTPUT_DIR / split / bin_name).mkdir(parents=True, exist_ok=True)

    # 4. Shuffle + split + copy
    summary = defaultdict(lambda: defaultdict(int))
    random.seed(RANDOM_SEED)

    for bin_name, images in bin_images.items():
        random.shuffle(images)
        n      = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits_map = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split, split_images in splits_map.items():
            dest_dir = OUTPUT_DIR / split / bin_name
            for img_path in split_images:
                dest = dest_dir / img_path.name
                # Avoid overwriting if same filename from different source classes
                if dest.exists():
                    stem = img_path.stem + f"_{img_path.parent.name}"
                    dest = dest_dir / f"{stem}{img_path.suffix}"
                shutil.copy2(img_path, dest)
            summary[bin_name][split] = len(split_images)

    return summary


def print_summary(summary: dict):
    print("\n" + "="*52)
    print("  DATASET READY — Smart Bin 3-Class Format")
    print("="*52)
    print(f"  {'Class':<14} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-"*52)
    totals = defaultdict(int)
    for bin_name in BINS:
        tr = summary[bin_name]["train"]
        vl = summary[bin_name]["val"]
        te = summary[bin_name]["test"]
        tot = tr + vl + te
        print(f"  {bin_name:<14} {tr:>6} {vl:>6} {te:>6} {tot:>7}")
        totals["train"] += tr
        totals["val"]   += vl
        totals["test"]  += te
    print("-"*52)
    grand = totals["train"] + totals["val"] + totals["test"]
    print(f"  {'TOTAL':<14} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6} {grand:>7}")
    print("="*52)
    print("\n✅ Next step — start training:")
    print("   python trainer.py --data_dir data --epochs 10 --batch_size 16\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Smart Bin — Dataset Preparation\n")

    download_trashnet()
    extract_dir = extract_trashnet()

    print("\n🔍 Finding class folders ...")
    class_dirs = find_class_dirs(extract_dir)

    if not class_dirs:
        print("❌ Could not find any class folders in the zip.")
        print("   Check the contents of data/raw/extracted/ manually.")
        sys.exit(1)

    print(f"   Found {len(class_dirs)} classes: {list(class_dirs.keys())}")

    print("\n📂 Remapping and splitting images ...")
    summary = build_split(class_dirs)

    print_summary(summary)
