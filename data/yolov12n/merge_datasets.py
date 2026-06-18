"""
Merge video_yolo_tiled_v2 into yolov12n dataset.
- Converts class 0 (price) → class 0 (pricetag)
- Normalizes pixel coords to 0..1 if needed
- Filters small boxes
- Deduplicates across all sources
- Copies images + labels into yolov12n

Usage (from Windows PowerShell):
    cd Z:\\Hakaton_project
    .venv_torch\\Scripts\\python.exe data/yolov12n/merge_datasets.py
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

MIN_AREA = 0.0005

SRC_DIR = Path("data/video_yolo_tiled_v2")
DST_DIR = Path("data/yolov12n")
SETS = ["train", "val"]

stats = defaultdict(int)


def parse_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        c = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        return c, cx, cy, w, h
    except:
        return None


def normalize_if_needed(c, cx, cy, w, h):
    """If coords are in pixels (>1), normalize to 0..1 assuming 480x480 tiles."""
    if cx > 1 or cy > 1 or w > 1 or h > 1:
        # Assume 480x480 tile size based on filename pattern
        cx, w = cx / 480, w / 480
        cy, h = cy / 480, h / 480
        stats["normalized"] += 1
    return c, cx, cy, w, h


def is_valid(c, cx, cy, w, h):
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    if w * h < MIN_AREA:
        return False
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    if x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1:
        return False
    return True


def find_image_for_label(label_name, images_dir):
    """Find image file matching a label file stem."""
    stem = label_name.rsplit(".", 1)[0]
    # Try common extensions
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG"]:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def process():
    for set_name in SETS:
        src_labels = SRC_DIR / "labels" / set_name
        src_images = SRC_DIR / "images" / set_name
        dst_labels = DST_DIR / "labels" / set_name
        dst_images = DST_DIR / "images" / set_name

        if not src_labels.exists():
            print(f"  SKIP {set_name}: {src_labels} not found")
            continue

        dst_labels.mkdir(parents=True, exist_ok=True)
        dst_images.mkdir(parents=True, exist_ok=True)

        label_files = list(src_labels.glob("*.txt"))
        print(f"\nProcessing {set_name}: {len(label_files)} source label files")

        copied = 0
        skipped_dup = 0
        skipped_small = 0
        skipped_noimg = 0

        for label_file in label_files:
            # Read and parse source label
            with open(label_file) as f:
                raw_lines = f.readlines()

            cleaned = []
            for line in raw_lines:
                parsed = parse_line(line)
                if parsed is None:
                    continue
                c, cx, cy, w, h = normalize_if_needed(*parsed)

                # Map: src class 0 (price) → dst class 0 (pricetag)
                # If you want to map some as class 1 (price_block), adjust here
                new_c = 0

                if not is_valid(new_c, cx, cy, w, h):
                    skipped_small += 1
                    continue

                cleaned.append(f"{new_c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                stats[f"class_{new_c}"] += 1

            if not cleaned:
                continue

            # Find source image
            src_img = find_image_for_label(label_file.name, src_images)
            if src_img is None:
                skipped_noimg += 1
                continue

            # Generate unique destination name
            dst_stem = f"tiled_{set_name}_{label_file.stem}"
            dst_label_file = dst_labels / (dst_stem + ".txt")
            dst_img_file = dst_images / (dst_stem + ".jpg")

            # Handle duplicates
            counter = 0
            while dst_label_file.exists() or dst_img_file.exists():
                counter += 1
                dst_label_file = dst_labels / (dst_stem + f"_{counter}.txt")
                dst_img_file = dst_images / (dst_stem + f"_{counter}.jpg")

            # Write label
            with open(dst_label_file, "w") as f:
                f.writelines(cleaned)

            # Copy image
            suffix = src_img.suffix
            dst_img_file = dst_img_file.with_suffix(suffix)
            shutil.copy2(src_img, dst_img_file)
            copied += 1

        print(f"  Copied: {copied}")
        print(f"  Skipped (no image): {skipped_noimg}")
        print(f"  Skipped (too small): {skipped_small}")

    # Final count
    for set_name in SETS:
        train_count = len(list((DST_DIR / "labels" / set_name).glob("*.txt")))
        print(f"\n{set_name}: {train_count} total label files")

    print("\nClass distribution:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    process()
