"""
Merge video_yolo_tiled_v2 into yolov12n dataset.
- Maps all tiled boxes to class 0 (pricetag) — they are large, coherent price tags
- Normalizes pixel coords to 0..1 if needed
- Filters small boxes (area < 0.0005)
- Copies images + labels into yolov12n

Usage (from Windows PowerShell):
    cd Z:\Hakaton_project
    .venv_torch\Scripts\python.exe data/yolov12n/merge_tiled.py
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

SRC = Path(r"Z:\Hakaton_project\data\video_yolo_tiled_v2")
DST = Path(r"Z:\Hakaton_project\data\yolov12n")

MIN_AREA = 0.0005
MAP_CLASS = 0  # tiled class 0 'price' → 0 (pricetag)

stats = defaultdict(int)


def parse_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        c, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        return c, cx, cy, w, h
    except:
        return None


def normalize(c, cx, cy, w, h):
    if cx > 1 or cy > 1 or w > 1 or h > 1:
        cx, w = cx / 480, w / 480
        cy, h = cy / 480, h / 480
        stats["normalized"] += 1
    return c, cx, cy, w, h


def is_valid(cx, cy, w, h):
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    if w * h < MIN_AREA:
        return False
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    if x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1:
        return False
    return True


def find_image(stem, images_dir):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG"]:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def process_set(set_name):
    src_lbl = SRC / "labels" / set_name
    src_img = SRC / "images" / set_name
    dst_lbl = DST / "labels" / set_name
    dst_img = DST / "images" / set_name

    if not src_lbl.exists():
        print(f"  SKIP {set_name}: {src_lbl} not found")
        return

    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    files = list(src_lbl.glob("*.txt"))
    print(f"\n{set_name}: {len(files)} source files")

    copied = 0
    noimg = 0
    empty = 0

    for lf in files:
        with open(lf) as f:
            raw = f.readlines()

        cleaned = []
        for line in raw:
            parsed = parse_line(line)
            if parsed is None:
                continue
            c, cx, cy, w, h = normalize(*parsed)

            if not is_valid(cx, cy, w, h):
                continue

            cleaned.append(f"{MAP_CLASS} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            stats["boxes_added"] += 1

        if not cleaned:
            empty += 1
            continue

        img = find_image(lf.stem, src_img)
        if img is None:
            noimg += 1
            continue

        stem = f"tiled_{lf.stem}"
        out_lf = dst_lbl / (stem + ".txt")
        out_img = dst_img / (stem + img.suffix)

        # Avoid overwriting
        ctr = 0
        while out_lf.exists():
            ctr += 1
            out_lf = dst_lbl / (f"{stem}_{ctr}.txt")
            out_img = dst_img / (f"{stem}_{ctr}{img.suffix}")

        with open(out_lf, "w") as f:
            f.writelines(cleaned)

        shutil.copy2(img, out_img)
        copied += 1

    print(f"  Copied: {copied} | No image: {noimg} | Empty: {empty}")


def main():
    print("=" * 60)
    print("Merge video_yolo_tiled_v2 → yolov12n")
    print(f"Target class: {MAP_CLASS} (pricetag)")
    print("=" * 60)

    for s in ["train", "val"]:
        process_set(s)

    for s in ["train", "val"]:
        n = len(list((DST / "labels" / s).glob("*.txt")))
        ni = len(list((DST / "images" / s).glob("*")))
        print(f"\n{s}: {n} labels, {ni} images in yolov12n")

    print(f"\nStats: {dict(stats)}")


if __name__ == "__main__":
    main()
