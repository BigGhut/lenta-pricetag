import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

LABELS_DIR = Path("data/yolov12n/labels/train")
PLOTS_DIR = Path("utils/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_yolo_label(filepath):
    boxes = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                boxes.append({"class": cls, "x_c": x_c, "y_c": y_c, "w": w, "h": h,
                              "aspect_ratio": w / h if h > 0 else 0, "area": w * h})
    return boxes


def analyze():
    pricetag_stats = []
    inner_elements = []
    elements_per_tag = []

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    print(f"Analyzing {len(label_files)} label files...")

    for lf in label_files:
        boxes = parse_yolo_label(lf)
        pricetags = [b for b in boxes if b["class"] == 0]
        inners = [b for b in boxes if b["class"] >= 1]

        if not pricetags:
            continue

        pt = pricetags[0]
        for inner in inners:
            rel_x = (inner["x_c"] - pt["x_c"]) / pt["w"] + 0.5
            rel_y = (inner["y_c"] - pt["y_c"]) / pt["h"] + 0.5
            rel_w = inner["w"] / pt["w"]
            rel_h = inner["h"] / pt["h"]

            inner_elements.append({
                "rel_x": rel_x,
                "rel_y": rel_y,
                "rel_w": rel_w,
                "rel_h": rel_h,
                "aspect_ratio": inner["aspect_ratio"],
                "area_fraction": inner["area"] / pt["area"],
                "class": inner["class"],
            })

        elements_per_tag.append(len(inners))

    inner_elements = np.array([(e["rel_x"], e["rel_y"], e["rel_w"], e["rel_h"],
                                 e["aspect_ratio"], e["area_fraction"], e["class"])
                                for e in inner_elements])

    print(f"\n=== Statistics ===")
    print(f"Total pricetags analyzed: {len(elements_per_tag)}")
    print(f"Total inner elements: {len(inner_elements)}")
    print(f"Elements per pricetag: mean={np.mean(elements_per_tag):.1f}, "
          f"median={np.median(elements_per_tag):.0f}, "
          f"min={np.min(elements_per_tag)}, max={np.max(elements_per_tag)}")

    print("\n--- Position (relative to pricetag) ---")
    print(f"  X center: mean={np.mean(inner_elements[:,0]):.3f}, "
          f"std={np.std(inner_elements[:,0]):.3f}")
    print(f"  Y center: mean={np.mean(inner_elements[:,1]):.3f}, "
          f"std={np.std(inner_elements[:,1]):.3f}")
    print(f"  Width:    mean={np.mean(inner_elements[:,2]):.3f}, "
          f"std={np.std(inner_elements[:,2]):.3f}")
    print(f"  Height:   mean={np.mean(inner_elements[:,3]):.3f}, "
          f"std={np.std(inner_elements[:,3]):.3f}")

    print("\n--- Aspect Ratio ---")
    print(f"  mean={np.mean(inner_elements[:,4]):.2f}, median={np.median(inner_elements[:,4]):.2f}")
    print(f"  <1.0 (tall): {np.sum(inner_elements[:,4] < 1.0)} ({np.mean(inner_elements[:,4] < 1.0)*100:.1f}%)")
    print(f"  >3.0 (wide): {np.sum(inner_elements[:,4] > 3.0)} ({np.mean(inner_elements[:,4] > 3.0)*100:.1f}%)")

    print("\n--- Area Fraction ---")
    print(f"  mean={np.mean(inner_elements[:,5]):.4f}, median={np.median(inner_elements[:,5]):.4f}")
    print(f"  >0.3 (large): {np.sum(inner_elements[:,5] > 0.3)} ({np.mean(inner_elements[:,5] > 0.3)*100:.1f}%)")

    print("\n--- Class Distribution ---")
    for c in sorted(set(inner_elements[:,6])):
        count = np.sum(inner_elements[:,6] == c)
        print(f"  Class {int(c)}: {count} ({count / len(inner_elements) * 100:.1f}%)")

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(inner_elements[:, 0], inner_elements[:, 1],
                         s=inner_elements[:, 5] * 500, alpha=0.4, c=inner_elements[:, 6],
                         cmap="tab10")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("Relative X position")
    ax.set_ylabel("Relative Y position")
    ax.set_title("Inner Element Positions (size=area fraction)")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Class ID")

    ax = axes[0, 1]
    ax.hist(inner_elements[:, 1], bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Relative Y position")
    ax.set_ylabel("Count")
    ax.set_title("Distribution by Y position")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.33, color="red", ls="--", alpha=0.5, label="y=0.33")
    ax.axvline(0.66, color="red", ls="--", alpha=0.5, label="y=0.66")
    ax.legend()

    ax = axes[0, 2]
    ax.hist(inner_elements[:, 4], bins=50, alpha=0.7, color="coral", edgecolor="white")
    ax.set_xlabel("Aspect Ratio (w/h)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution by Aspect Ratio")
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color="red", ls="--", alpha=0.5)
    ax.axvline(3.0, color="red", ls="--", alpha=0.5)

    ax = axes[1, 0]
    ax.hist(inner_elements[:, 5], bins=50, alpha=0.7, color="forestgreen", edgecolor="white")
    ax.set_xlabel("Area Fraction (relative to pricetag)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution by Area Fraction")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(elements_per_tag, bins=range(0, max(elements_per_tag) + 2),
            alpha=0.7, color="goldenrod", edgecolor="white", align="left")
    ax.set_xlabel("Number of inner elements per pricetag")
    ax.set_ylabel("Count")
    ax.set_title("Elements per Pricetag")

    ax = axes[1, 2]
    classes = sorted(set(int(c) for c in inner_elements[:, 6]))
    counts = [np.sum(inner_elements[:, 6] == c) for c in classes]
    labels = [f"Class {int(c)}" for c in classes]
    bars = ax.bar(labels, counts, color="mediumpurple", edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")
    plt.close()

    # CSV dump of raw stats
    csv_path = PLOTS_DIR / "inner_elements_stats.csv"
    with open(csv_path, "w") as f:
        f.write("rel_x,rel_y,rel_w,rel_h,aspect_ratio,area_fraction,class\n")
        for e in inner_elements:
            f.write(f"{e[0]:.4f},{e[1]:.4f},{e[2]:.4f},{e[3]:.4f},{e[4]:.4f},{e[5]:.6f},{int(e[6])}\n")
    print(f"Stats CSV saved: {csv_path}")

    # Heuristic suggestions based on analysis
    print("\n=== Heuristic Suggestions ===")
    y_third = np.percentile(inner_elements[:, 1], [33, 66])
    print(f"  Y thirds: bottom={y_third[0]:.2f}, middle=0.{int(y_third[0]*100)}-{int(y_third[1]*100)}, top={y_third[1]:.2f}")

    tall_mask = inner_elements[:, 4] < 0.5
    wide_mask = inner_elements[:, 4] > 5.0
    large_mask = inner_elements[:, 5] > 0.3
    print(f"  Tall elements (ar<0.5, likely barcode): {np.sum(tall_mask)} ({np.mean(tall_mask)*100:.1f}%)")
    print(f"  Wide elements (ar>5.0, likely price_block): {np.sum(wide_mask)} ({np.mean(wide_mask)*100:.1f}%)")
    print(f"  Large elements (area>30%, likely price_digits): {np.sum(large_mask)} ({np.mean(large_mask)*100:.1f}%)")


if __name__ == "__main__":
    analyze()
