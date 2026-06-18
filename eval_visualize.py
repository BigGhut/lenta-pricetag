import cv2
import numpy as np
from utils.tiled_inference import TiledYOLO
from utils.candidate_detector import detect_pricetag_candidates, auto_calibrate_hsv
from utils.cascade_pipeline import _nms_merge

IMG_PATH = "frame_00001.jpg"
MODEL_PATH = "models/yolov12n/yolov12n.pt"
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]
print(f"Image: {W}x{H}")


def draw_boxes(vis, boxes, color_map):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        conf = b[4]
        tag_type = b[5] if len(b) > 5 else "yolo"
        color = color_map.get(tag_type, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        label = f"{tag_type} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


color_map = {
    "yolo": (255, 0, 0),       # Blue = YOLO
    "pricetag": (0, 255, 0),   # Green = white pricetag
    "promo": (0, 0, 255),      # Red = promo/red tag
    "yellow": (0, 255, 255),   # Yellow = yellow tag
}

# ─── 1. TiledYOLO only ───
print("\n--- 1. TiledYOLO (SAHI + multi-scale) ---")
tiled = TiledYOLO(MODEL_PATH, tile_size=640, stride=320, conf=0.15, iou=0.45,
                   scales=[0.75, 1.0, 1.25])
yolo_boxes = tiled.detect(img)
yolo_fmt = [(x1, y1, x2, y2, conf, "yolo") for x1, y1, x2, y2, conf in yolo_boxes]
print(f"  YOLO detections: {len(yolo_fmt)}")
vis_yolo = img.copy()
draw_boxes(vis_yolo, yolo_fmt, color_map)
cv2.imwrite("eval_yolo_only.jpg", vis_yolo)
print("  Saved: eval_yolo_only.jpg")

# ─── 2. Color detection only ───
print("\n--- 2. Color detection ---")
hsv_calib = auto_calibrate_hsv(img)
color_boxes = detect_pricetag_candidates(img, hsv_calib=hsv_calib) or []
print(f"  Color detections: {len(color_boxes)}")
vis_color = img.copy()
draw_boxes(vis_color, color_boxes, color_map)
cv2.imwrite("eval_color_only.jpg", vis_color)
print("  Saved: eval_color_only.jpg")

# ─── 3. Merged (YOLO + Color) ───
print("\n--- 3. YOLO + Color merged ---")
merged = _nms_merge(yolo_fmt, color_boxes, iou_thresh=0.45)
print(f"  Merged detections: {len(merged)}")
vis_merged = img.copy()
draw_boxes(vis_merged, merged, color_map)
legend_y = 30
for name, rgb in [("YOLO", (255, 0, 0)), ("White pricetag", (0, 255, 0)),
                   ("Red promo", (0, 0, 255))]:
    cv2.putText(vis_merged, name, (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)
    legend_y += 25
cv2.imwrite("eval_merged.jpg", vis_merged)
print("  Saved: eval_merged.jpg")

# ─── 4. Confidence heatmap (all detections by confidence) ───
print("\n--- 4. Confidence heatmap ---")
vis_heat = img.copy()
for b in yolo_fmt:
    x1, y1, x2, y2 = map(int, b[:4])
    conf = b[4]
    intensity = int(255 * conf)
    color = (0, intensity, 255 - intensity)  # Red (low) → Green (high)
    cv2.rectangle(vis_heat, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis_heat, f"{conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imwrite("eval_heatmap.jpg", vis_heat)
print("  Saved: eval_heatmap.jpg")

# ─── Summary ───
print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"  YOLO only:      {len(yolo_fmt):3d} boxes")
print(f"  Color only:     {len(color_boxes):3d} boxes")
print(f"  Merged:         {len(merged):3d} boxes (suppressed: {len(yolo_fmt)+len(color_boxes)-len(merged)})")

# Size distribution
for name, boxes in [("YOLO", yolo_fmt), ("Color", color_boxes), ("Merged", merged)]:
    small = sum(1 for b in boxes if (b[2]-b[0])*(b[3]-b[1]) < 10000)
    medium = sum(1 for b in boxes if 10000 <= (b[2]-b[0])*(b[3]-b[1]) < 40000)
    large = sum(1 for b in boxes if (b[2]-b[0])*(b[3]-b[1]) >= 40000)
    if boxes:
        avg_conf = np.mean([b[4] for b in boxes])
        print(f"  {name+':':12s} small={small} medium={medium} large={large} avg_conf={avg_conf:.3f}")
