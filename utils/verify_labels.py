import cv2
import random
from pathlib import Path

LABELS_DIRS = {
    "original": Path("data/yolov12n/labels/train"),
    "heuristic": Path("data/yolov12n/labels_full/train"),
}
IMAGES_DIR = Path("data/yolov12n/images/train")
OUTPUT_DIR = Path("utils/plots/verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    0: (0, 255, 0),    # pricetag - green
    1: (255, 0, 0),    # price_block - blue
    2: (0, 255, 255),  # price_digits - yellow
    3: (255, 255, 0),  # product_name - cyan
    4: (255, 0, 255),  # qr_code - magenta
    5: (0, 128, 255),  # barcode - orange
    6: (0, 0, 255),    # promo_sticker - red
}

CLASS_NAMES = {
    0: "pricetag",
    1: "price_block",
    2: "price_digits",
    3: "product_name",
    4: "qr_code",
    5: "barcode",
    6: "promo_sticker",
}


def draw_labels(image_path, labels_path, label_set_name):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if not labels_path.exists():
        return img

    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c, y_c, bw, bh = map(float, parts[1:5])

            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            color = CLASS_COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = CLASS_NAMES.get(cls, f"class_{cls}")
            cv2.putText(img, label, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img


def verify():
    label_files = sorted(LABELS_DIRS["original"].glob("*.txt"))
    sample = random.sample(label_files, min(20, len(label_files)))

    print(f"Verifying {len(sample)} samples...")

    for i, lf in enumerate(sample):
        img_path = IMAGES_DIR / (lf.stem + ".jpg")
        if not img_path.exists():
            continue

        panels = []
        for name, ld in LABELS_DIRS.items():
            lbl_path = ld / lf.name
            panel = draw_labels(img_path, lbl_path, name)
            if panel is not None:
                h, w = panel.shape[:2]
                cv2.putText(panel, f"[{name}]", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                panels.append(panel)

        if len(panels) == 2:
            combined = cv2.hconcat(panels)
            out_path = OUTPUT_DIR / f"verify_{lf.stem}.png"
            cv2.imwrite(str(out_path), combined)

    print(f"Verification images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    verify()
