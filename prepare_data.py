import os
import shutil
import random
from pathlib import Path

TRAIN_SRC = Path("train")
DATA_DST = Path("data/yolov12n")

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
SEED = 42

# Class remap:
# Original data: class 1 = pricetag (full), class 0 = inner elements
# New schema:    0 = pricetag, 1 = price_block (all inner as default)
CLASS_REMAP = {"1": "0", "0": "1"}

random.seed(SEED)

images_src = sorted(TRAIN_SRC.glob("images/*"))
random.shuffle(images_src)

split_idx = int(len(images_src) * TRAIN_RATIO)
train_files = images_src[:split_idx]
val_files = images_src[split_idx:]

print(f"Total: {len(images_src)}, Train: {len(train_files)}, Val: {len(val_files)}")

for split_name, files in [("train", train_files), ("val", val_files)]:
    img_dst = DATA_DST / "images" / split_name
    lbl_dst = DATA_DST / "labels" / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path in files:
        shutil.copy2(img_path, img_dst / img_path.name)

        lbl_path = TRAIN_SRC / "labels" / (img_path.stem + ".txt")
        if lbl_path.exists():
            out_lines = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = parts[0]
                        new_class = CLASS_REMAP.get(old_class, old_class)
                        out_lines.append(f"{new_class} " + " ".join(parts[1:5]))
            with open(lbl_dst / (img_path.stem + ".txt"), "w") as f:
                f.write("\n".join(out_lines))

print("Done. Remap: class 1->0 (pricetag), class 0->1 (price_block default)")
print("NOTE: inner element classes (2-6) need manual re-labeling for fine-grained detection.")
