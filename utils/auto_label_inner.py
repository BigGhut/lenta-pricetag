import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans

LABELS_SRC = Path("data/yolov12n/labels/train")
IMAGES_SRC = Path("data/yolov12n/images/train")
LABELS_DST = Path("data/yolov12n/labels_full/train")
IMAGES_DST = Path("data/yolov12n/images_full/train")

# Heuristic mapping rules:
# Each inner element gets a class based on position + shape inside pricetag
RULES = [
    # (condition, new_class)
    # Priority order: first match wins
]


def parse_yolo_label(filepath):
    boxes = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                boxes.append({"class": cls, "x_c": x_c, "y_c": y_c, "w": w, "h": h,
                              "aspect_ratio": w / h if h > 0 else 0})
    return boxes


def compute_relative_position(inner, pt):
    rel_x = (inner["x_c"] - pt["x_c"]) / pt["w"] + 0.5
    rel_y = (inner["y_c"] - pt["y_c"]) / pt["h"] + 0.5
    rel_w = inner["w"] / pt["w"]
    rel_h = inner["h"] / pt["h"]
    area_frac = (inner["w"] * inner["h"]) / (pt["w"] * pt["h"])
    return rel_x, rel_y, rel_w, rel_h, area_frac


def learn_clusters():
    """Learn KMeans clusters from existing data to define class mapping."""
    print("Learning position clusters from data...")

    features = []
    metadata = []

    for lf in sorted(LABELS_SRC.glob("*.txt")):
        boxes = parse_yolo_label(lf)
        pricetags = [b for b in boxes if b["class"] == 0]
        inners = [b for b in boxes if b["class"] >= 1]
        if not pricetags:
            continue
        pt = pricetags[0]
        for inner in inners:
            rel_x, rel_y, rel_w, rel_h, area_frac = compute_relative_position(inner, pt)
            features.append([rel_x, rel_y, rel_w, rel_h, inner["aspect_ratio"], area_frac])
            metadata.append((lf.stem, inner))

    features = np.array(features)
    n_clusters = 4  # 4 elements per pricetag

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Analyze each cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(features[i])

    print(f"\nLearned {n_clusters} clusters:")
    for c in sorted(clusters.keys()):
        pts = np.array(clusters[c])
        print(f"  Cluster {c}: n={len(pts)}, "
              f"pos=({pts[:,0].mean():.2f}, {pts[:,1].mean():.2f}), "
              f"size=({pts[:,2].mean():.3f}, {pts[:,3].mean():.3f}), "
              f"ar={pts[:,4].mean():.2f}, area_frac={pts[:,5].mean():.4f}")

    return kmeans


def heuristic_classify(rel_x, rel_y, rel_w, rel_h, aspect_ratio, area_fraction):
    """
    Heuristic classification of inner elements based on position/shape.
    Designed for typical ЛЕНТА pricetag layout.
    """
    # barcode: very narrow (tall), usually in bottom half
    if aspect_ratio < 0.4 and rel_y > 0.5:
        return 5  # barcode

    # price_digits: narrow, in upper/middle half
    if aspect_ratio < 0.6 and rel_y < 0.6:
        return 2  # price_digits

    # product_name: wider, in top portion
    if rel_y < 0.35 and aspect_ratio > 0.5:
        return 3  # product_name

    # qr_code: nearly square, small
    if 0.7 < aspect_ratio < 1.3 and area_fraction < 0.005:
        return 4  # qr_code

    # promo_sticker: could be overlaid, check if it's near edges
    if rel_x < 0.15 or rel_x > 0.85:
        return 6  # promo_sticker

    # Default: price_block (general price area)
    return 1  # price_block


def auto_label(kmeans=None, val_split=False):
    """Auto-label inner elements with heuristic + optional KMeans."""
    LABELS_DST.mkdir(parents=True, exist_ok=True)
    IMAGES_DST.mkdir(parents=True, exist_ok=True)

    class_counts = defaultdict(int)

    for lf in sorted(LABELS_SRC.glob("*.txt")):
        boxes = parse_yolo_label(lf)
        pricetags = [b for b in boxes if b["class"] == 0]
        inners = [b for b in boxes if b["class"] >= 1]

        if not pricetags:
            continue

        pt = pricetags[0]
        new_boxes = [{"class": 0, **{k: v for k, v in pt.items() if k != "class"}}]

        for inner in inners:
            rel_x, rel_y, rel_w, rel_h, area_frac = compute_relative_position(inner, pt)

            if kmeans is not None:
                feat = np.array([[rel_x, rel_y, rel_w, rel_h, inner["aspect_ratio"], area_frac]])
                cluster_id = int(kmeans.predict(feat)[0])
                new_class = cluster_id + 1
            else:
                new_class = heuristic_classify(rel_x, rel_y, rel_w, rel_h,
                                                inner["aspect_ratio"], area_frac)

            new_boxes.append({"class": new_class,
                              "x_c": inner["x_c"], "y_c": inner["y_c"],
                              "w": inner["w"], "h": inner["h"]})
            class_counts[new_class] += 1

        out_path = LABELS_DST / lf.name
        with open(out_path, "w") as f:
            for box in new_boxes:
                f.write(f"{box['class']} {box['x_c']:.6f} {box['y_c']:.6f} "
                       f"{box['w']:.6f} {box['h']:.6f}\n")

        img_src = IMAGES_SRC / (lf.stem + ".jpg")
        if img_src.exists():
            shutil.copy2(img_src, IMAGES_DST / img_src.name)

    print("\n=== Auto-labeling Complete ===")
    print(f"Output: {LABELS_DST}")
    total = sum(class_counts.values())
    for c in sorted(class_counts.keys()):
        print(f"  Class {c}: {class_counts[c]:5d} ({class_counts[c]/total*100:.1f}%)")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "heuristic"

    if mode == "cluster":
        print("Mode: KMeans clustering")
        kmeans = learn_clusters()
        auto_label(kmeans=kmeans)
    else:
        print("Mode: heuristic rules")
        auto_label(kmeans=None)
