"""
Dataset cleanup script for LENTA pricetag detection.
Removes:
  - Duplicate bounding boxes (same coords across files)
  - Boxes too small for YOLO to detect (area < min_area)
  - Empty label files (and corresponding images)
  - Label files with no valid boxes

Usage (from Windows PowerShell):
    cd Z:\Hakaton_project
    .venv_torch\Scripts\python.exe data/yolov12n/cleanup_dataset.py
"""

import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict

# === Config ===
MIN_AREA = 0.0005          # min box area (fraction of image) — ~20x20px at 640
MIN_WIDTH = 0.008          # min box width  — ~5px at 640
MIN_HEIGHT = 0.008         # min box height — ~5px at 640
REMOVE_EMPTY = True        # remove images with no valid labels after cleanup
BACKUP = True              # backup labels before modifying

DATA_DIR = Path(__file__).parent  # data/yolov12n
SETS = ["train", "val"]


def parse_label_line(line):
    """Parse YOLO label line: class cx cy w h"""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        return (cls, cx, cy, w, h)
    except ValueError:
        return None


def is_valid_box(cls, cx, cy, w, h):
    """Check if box is valid and large enough for YOLO"""
    area = w * h
    if area < MIN_AREA:
        return False
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    # Check bounds (YOLO format: 0..1)
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    # Box must be within image
    x1, x2 = cx - w/2, cx + w/2
    y1, y2 = cy - h/2, cy + h/2
    if x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1:
        return False
    return True


def round_box(cls, cx, cy, w, h, precision=6):
    """Round box coords for dedup comparison"""
    return (cls, round(cx, precision), round(cy, precision), round(w, precision), round(h, precision))


def process_set(set_name):
    """Process one dataset split (train/val)"""
    labels_dir = DATA_DIR / "labels" / set_name
    images_dir = DATA_DIR / "images" / set_name

    if not labels_dir.exists():
        print(f"  SKIP: {labels_dir} not found")
        return

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"\n{'='*60}")
    print(f"Processing {set_name}: {len(label_files)} label files")
    print(f"{'='*60}")

    # Backup
    if BACKUP:
        backup_dir = DATA_DIR / "labels_backup" / set_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in label_files:
            shutil.copy2(f, backup_dir / f.name)
        print(f"  Backed up to {backup_dir}")

    # Stats
    total_boxes = 0
    removed_small = 0
    removed_dup = 0
    removed_invalid = 0
    empty_after = 0
    class_counts = defaultdict(int)

    # First pass: collect all boxes for dedup
    all_boxes = defaultdict(list)  # rounded_box -> [file_path, ...]
    file_boxes = {}  # file_path -> [original_lines]

    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()

        file_boxes[label_file] = lines
        for line in lines:
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            cls, cx, cy, w, h = parsed
            total_boxes += 1
            rounded = round_box(cls, cx, cy, w, h)
            all_boxes[rounded].append(label_file)

    # Find duplicates (same box in 3+ files)
    duplicates = {box: files for box, files in all_boxes.items() if len(files) >= 3}
    dup_files = set()
    for box, files in duplicates.items():
        dup_files.update(files)

    print(f"\n  Total boxes: {total_boxes}")
    print(f"  Unique box signatures: {len(all_boxes)}")
    print(f"  Duplicate signatures (3+ files): {len(duplicates)}")
    print(f"  Files with duplicates: {len(dup_files)}")

    # Second pass: clean each file
    for label_file in label_files:
        lines = file_boxes[label_file]
        kept = []
        seen_local = set()

        for line in lines:
            parsed = parse_label_line(line)
            if parsed is None:
                removed_invalid += 1
                continue

            cls, cx, cy, w, h = parsed

            # Check validity
            if not is_valid_box(cls, cx, cy, w, h):
                removed_small += 1
                continue

            # Check global dedup (keep only first occurrence)
            rounded = round_box(cls, cx, cy, w, h)
            if rounded in duplicates:
                if rounded in seen_local:
                    removed_dup += 1
                    continue
                # Check if this is the first file with this box
                first_file = duplicates[rounded][0]
                if label_file != first_file:
                    removed_dup += 1
                    continue
                seen_local.add(rounded)

            kept.append(line)
            class_counts[cls] += 1

        # Write cleaned labels
        if kept:
            with open(label_file, "w") as f:
                f.writelines(kept)
        else:
            empty_after += 1
            if REMOVE_EMPTY:
                # Remove label file
                label_file.unlink()
                # Remove corresponding image
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    img_file = images_dir / (label_file.stem + ext)
                    if img_file.exists():
                        img_file.unlink()
                        break
                    # Also check without .rf.xxx suffix
                    stem = label_file.stem
                    if ".rf." in stem:
                        base = stem.split(".rf.")[0]
                        img_file = images_dir / (base + ext)
                        if img_file.exists():
                            img_file.unlink()
                            break

    # Print stats
    print(f"\n  Removed:")
    print(f"    Invalid lines:    {removed_invalid}")
    print(f"    Too small:        {removed_small}")
    print(f"    Duplicates:       {removed_dup}")
    print(f"    Empty files:      {empty_after}")
    print(f"\n  Remaining class distribution:")
    for cls in sorted(class_counts.keys()):
        print(f"    Class {cls}: {class_counts[cls]}")
    print(f"  Remaining boxes: {sum(class_counts.values())}")

    return class_counts


def main():
    print("=" * 60)
    print("LENTA Dataset Cleanup")
    print("=" * 60)
    print(f"MIN_AREA = {MIN_AREA} (~{int(MIN_AREA*640*640)}px at 640x640)")
    print(f"MIN_WIDTH = {MIN_WIDTH} (~{int(MIN_WIDTH*640)}px)")
    print(f"MIN_HEIGHT = {MIN_HEIGHT} (~{int(MIN_HEIGHT*640)}px)")

    total_counts = defaultdict(int)
    for set_name in SETS:
        counts = process_set(set_name)
        if counts:
            for cls, cnt in counts.items():
                total_counts[cls] += cnt

    print(f"\n{'='*60}")
    print("TOTAL remaining boxes:")
    for cls in sorted(total_counts.keys()):
        print(f"  Class {cls}: {total_counts[cls]}")
    print(f"  Total: {sum(total_counts.values())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
