import os, shutil, random
from pathlib import Path

SRC    = Path("Validation")
OUTPUT = Path("data")
SEEDS  = 42
BINS   = ["Compost", "Landfill", "Recyclable"]

random.seed(42)

for bin_name in BINS:
    src_dir = SRC / bin_name
    images  = list(src_dir.glob("*.*"))
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    split_map = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }
s
    for split, files in split_map.items():
        dest = OUTPUT / split / bin_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)

    print(f"{bin_name}: {n_train} train / {n_val} val / {n - n_train - n_val} test")

print("\n✅ Done! Now run: python trainer.py --data_dir data --epochs 10 --batch_size 16")