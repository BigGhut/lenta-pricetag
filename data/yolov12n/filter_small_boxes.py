"""Filter price_block boxes that are too small for YOLO to detect.
Removes boxes with area < min_area (default 0.005 = ~25x25px at 640x640).
Keeps pricetag boxes unchanged.

Usage:
    cd Z:\Hakaton_project
    .venv_torch\Scripts\python.exe data/yolov12n/filter_small_boxes.py
"""

import glob
from pathlib import Path

MIN_AREA = 0.005  # ~25x25px at 640x640 — YOLO can detect this

DATA_DIR = Path(r"Z:\Hakaton_project\data\yolov12n")
SETS = ["train", "val"]

total_removed = 0
total_kept = 0
files_modified = 0

for set_name in SETS:
    labels_dir = DATA_DIR / "labels" / set_name
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = f.readlines()

        kept = []
        removed = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            w, h = float(parts[3]), float(parts[4])
            area = w * h

            # Only filter price_block (class 1)
            if cls == 1 and area < MIN_AREA:
                removed += 1
                continue
            kept.append(line)

        if removed > 0:
            total_removed += removed
            files_modified += 1

            if kept:
                with open(label_file, "w") as f:
                    f.writelines(kept)
            else:
                # Remove empty label and corresponding image
                label_file.unlink()
                images_dir = DATA_DIR / "images" / set_name
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    img = images_dir / (label_file.stem + ext)
                    if img.exists():
                        img.unlink()
                        break

        total_kept += len(kept)

print(f"Removed {total_removed} small price_block boxes from {files_modified} files")
print(f"Total remaining boxes: {total_kept}")

# Final stats
for set_name in SETS:
    n_labels = len(list((DATA_DIR / "labels" / set_name).glob("*.txt")))
    n_images = len(list((DATA_DIR / "images" / set_name).glob("*")))
    print(f"{set_name}: {n_labels} labels, {n_images} images")
