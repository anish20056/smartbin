"""
prepare_realwaste.py
====================
Remaps RealWaste dataset (9 classes) into your 3 bins and splits into train/val/test.

Run:
    python prepare_realwaste.py

Make sure your folder structure is:
    realwaste-main/
        RealWaste/
            Cardboard/
            Food Organics/
            Glass/
            Metal/
            Miscellaneous Trash/
            Paper/
            Plastic/
            Textile Trash/
            Vegetation/
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
SRC_DIR    = Path("realwaste-main/RealWaste")   # path to your downloaded dataset
OUTPUT_DIR = Path("data")                        # output folder for training
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15

# ── Remap: RealWaste 9 classes → your 3 bins ──────────────────────────────────
REMAP = {
    "Cardboard":           "Recyclable",
    "Glass":               "Recyclable",
    "Metal":               "Recyclable",
    "Paper":               "Recyclable",
    "Plastic":             "Recyclable",
    "Food Organics":       "Compost",
    "Vegetation":          "Compost",
    "Miscellaneous Trash": "Landfill",
    "Textile Trash":       "Landfill",
}

# ── Waste item name mapping (for UI display) ───────────────────────────────────
# This tells the UI what to show as the specific item name
ITEM_NAMES = {
    "Cardboard":           "Cardboard / Box",
    "Glass":               "Glass Bottle / Jar",
    "Metal":               "Metal Can / Tin",
    "Paper":               "Paper / Newspaper",
    "Plastic":             "Plastic Bottle / Container",
    "Food Organics":       "Food Scraps / Organics",
    "Vegetation":          "Vegetation / Leaves",
    "Miscellaneous Trash": "Mixed / Miscellaneous Trash",
    "Textile Trash":       "Textile / Fabric",
}

BINS   = ["Recyclable", "Compost", "Landfill"]
SPLITS = ["train", "val", "test"]


def prepare():
    # ── Validate source ────────────────────────────────────────────────────────
    if not SRC_DIR.exists():
        print(f"❌ Source folder not found: {SRC_DIR}")
        print("Make sure you unzipped the dataset and the folder is named 'realwaste-main'")
        print("Place it in the same directory as this script.")
        return

    # ── Clear old data folder ──────────────────────────────────────────────────
    if OUTPUT_DIR.exists():
        print(f"🗑️  Removing old data folder...")
        shutil.rmtree(OUTPUT_DIR)

    # ── Create output dirs ─────────────────────────────────────────────────────
    for split in SPLITS:
        for bin_name in BINS:
            (OUTPUT_DIR / split / bin_name).mkdir(parents=True, exist_ok=True)

    # ── Gather images per bin ──────────────────────────────────────────────────
    print("\n📂 Reading source folders...")
    bin_images = defaultdict(list)
    item_label_map = {}  # filename → original class name (for waste name display)

    for original_class, bin_name in REMAP.items():
        src_class_dir = SRC_DIR / original_class
        if not src_class_dir.exists():
            print(f"  ⚠️  Folder not found: {src_class_dir} — skipping")
            continue

        images = sorted([
            p for p in src_class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])

        print(f"  {original_class:<22} → {bin_name:<12} ({len(images)} images)")

        for img in images:
            # Rename to include original class so we know the item name later
            item_label_map[img.name] = original_class
            bin_images[bin_name].append((img, original_class))

    # ── Shuffle + split + copy ─────────────────────────────────────────────────
    print("\n✂️  Splitting into train / val / test...")
    random.seed(RANDOM_SEED)
    summary = defaultdict(lambda: defaultdict(int))

    for bin_name in BINS:
        items = bin_images[bin_name]
        random.shuffle(items)

        n       = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        split_map = {
            "train": items[:n_train],
            "val":   items[n_train:n_train + n_val],
            "test":  items[n_train + n_val:],
        }

        for split, split_items in split_map.items():
            dest_dir = OUTPUT_DIR / split / bin_name
            for img_path, original_class in split_items:
                # Prefix filename with original class so item name is recoverable
                new_name = f"{original_class.replace(' ', '_')}_{img_path.name}"
                dest = dest_dir / new_name
                shutil.copy2(img_path, dest)
            summary[bin_name][split] = len(split_items)

    # ── Save item name mapping ─────────────────────────────────────────────────
    # This JSON is used by the API to show specific waste item names in the UI
    import json
    mapping = {
        "remap":      REMAP,
        "item_names": ITEM_NAMES,
    }
    with open("waste_label_map.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print("\n💾 Saved waste_label_map.json")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  DATASET READY — RealWaste → Smart Bin 3-Class Format")
    print("="*55)
    print(f"  {'Class':<14} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-"*55)
    totals = defaultdict(int)
    for bin_name in BINS:
        tr  = summary[bin_name]["train"]
        vl  = summary[bin_name]["val"]
        te  = summary[bin_name]["test"]
        tot = tr + vl + te
        print(f"  {bin_name:<14} {tr:>6} {vl:>6} {te:>6} {tot:>7}")
        totals["train"] += tr
        totals["val"]   += vl
        totals["test"]  += te
    print("-"*55)
    grand = totals['train'] + totals['val'] + totals['test']
    print(f"  {'TOTAL':<14} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6} {grand:>7}")
    print("="*55)
    print("\n✅ Next step — start training:")
    print("   python trainer.py --data_dir data --epochs 15 --batch_size 16 --num_workers 0\n")


if __name__ == "__main__":
    prepare()
