"""
Сравнение старой и новой модели на тестовом кадре.
Запуск: python eval_compare.py

Для сравнения с новой моделью:
  1. Обучи модель: python run_train.py
  2. Запусти: python eval_compare.py --new models/yolov12n/yolov12n.pt
"""
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO


def run_tiled_yolo(model_path, img, conf=0.15, iou=0.45, tile_size=640, stride=320):
    model = YOLO(model_path)
    H, W = img.shape[:2]
    all_boxes = []
    t0 = time.time()

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x1, y1 = x, y
            x2 = min(W, x + tile_size)
            y2 = min(H, y + tile_size)
            tile = img[y1:y2, x1:x2]
            results = model(tile, conf=conf, iou=iou, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            for b in boxes:
                tx1, ty1, tx2, ty2 = b.xyxy[0].tolist()
                conf_val = b.conf[0].item()
                all_boxes.append((x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2, conf_val))

    elapsed = time.time() - t0

    # NMS
    if all_boxes:
        boxes_list = [[b[0], b[1], b[2], b[3]] for b in all_boxes]
        scores = [b[4] for b in all_boxes]
        idxs = cv2.dnn.NMSBoxes(boxes_list, scores, conf, iou)
        if len(idxs) > 0:
            idxs = idxs.flatten()
            final = [all_boxes[i] for i in idxs]
        else:
            final = []
    else:
        final = []

    return final, elapsed


def draw_boxes(img, boxes, color=(0, 255, 0), label_prefix=""):
    vis = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        conf = b[4]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        txt = f"{label_prefix}{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return vis


def compute_metrics(boxes, ref_boxes, iou_thresh=0.45):
    """Простая оценка: сколько боксов совпало с референсными."""
    if not ref_boxes:
        return {"tp": 0, "fp": len(boxes), "fn": 0, "precision": 0, "recall": 0}

    tp = 0
    used_ref = set()
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        for ri, rb in enumerate(ref_boxes):
            if ri in used_ref:
                continue
            rx1, ry1, rx2, ry2 = rb[:4]
            ix1 = max(x1, rx1); iy1 = max(y1, ry1)
            ix2 = min(x2, rx2); iy2 = min(y2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (x2 - x1) * (y2 - y1) + (rx2 - rx1) * (ry2 - ry1) - inter
            if inter / (union + 1e-6) >= iou_thresh:
                tp += 1
                used_ref.add(ri)
                break

    fp = len(boxes) - tp
    fn = len(ref_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "total_detections": len(boxes),
        "total_reference": len(ref_boxes),
    }


def format_metrics(name, m):
    return (
        f"  {name:12s} | {m['total_detections']:3d} det | "
        f"TP={m['tp']:2d} FP={m['fp']:2d} FN={m['fn']:2d} | "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
    )


def print_table(metrics_list):
    print(f"\n{'='*65}")
    print(f"  {'Модель':<20} {'Detections':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'='*65}")
    for name, m in metrics_list:
        print(f"  {name:<20} {m['total_detections']:>10d} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare old vs new YOLO model on test frame")
    p.add_argument("--image", default="frame_00001.jpg", help="Test image path")
    p.add_argument("--old", default="models/yolov12n/yolov12n.pt", help="Old model path")
    p.add_argument("--new", default=None, help="New (trained) model path")
    p.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Cannot load {args.image}")
        exit(1)

    H, W = img.shape[:2]
    print(f"Image: {W}x{H}")
    print(f"Models: old={args.old}", end="")
    if args.new:
        print(f", new={args.new}")
    else:
        print(" (no new model — use --new to compare)")

    results = []

    # --- OLD model ---
    print(f"\n{'─'*50}")
    print(f"OLD model: {Path(args.old).name}")
    print(f"{'─'*50}")
    old_boxes, old_time = run_tiled_yolo(args.old, img, args.conf, args.iou)
    print(f"  Detections: {len(old_boxes)}, Time: {old_time:.2f}s")
    for i, b in enumerate(sorted(old_boxes, key=lambda x: x[4], reverse=True)[:10]):
        w, h = b[2]-b[0], b[3]-b[1]
        print(f"    #{i+1}: conf={b[4]:.3f} [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {w:.0f}x{h:.0f}")
    if len(old_boxes) > 10:
        print(f"    ... and {len(old_boxes)-10} more")

    vis_old = draw_boxes(img, old_boxes, color=(0, 255, 0))
    cv2.imwrite("eval_old_model.jpg", vis_old)
    print(f"  Saved: eval_old_model.jpg")

    # --- NEW model (if provided) ---
    if args.new and Path(args.new).exists():
        print(f"\n{'─'*50}")
        print(f"NEW model: {Path(args.new).name}")
        print(f"{'─'*50}")
        new_boxes, new_time = run_tiled_yolo(args.new, img, args.conf, args.iou)
        print(f"  Detections: {len(new_boxes)}, Time: {new_time:.2f}s")
        for i, b in enumerate(sorted(new_boxes, key=lambda x: x[4], reverse=True)[:10]):
            w, h = b[2]-b[0], b[3]-b[1]
            print(f"    #{i+1}: conf={b[4]:.3f} [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {w:.0f}x{h:.0f}")
        if len(new_boxes) > 10:
            print(f"    ... and {len(new_boxes)-10} more")

        vis_new = draw_boxes(img, new_boxes, color=(255, 0, 0))
        cv2.imwrite("eval_new_model.jpg", vis_new)
        print(f"  Saved: eval_new_model.jpg")

        # Side-by-side comparison
        h_border = np.ones((H, 20, 3), dtype=np.uint8) * 255
        side_by_side = np.hstack([vis_old, h_border, vis_new])
        # Add labels
        cv2.putText(side_by_side, "OLD model", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(side_by_side, "NEW model", (W + 50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(side_by_side, f"Old: {len(old_boxes)} det ({old_time:.1f}s)", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(side_by_side, f"New: {len(new_boxes)} det ({new_time:.1f}s)", (W + 50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imwrite("eval_side_by_side.jpg", side_by_side)
        print(f"  Saved: eval_side_by_side.jpg ({side_by_side.shape[1]}x{side_by_side.shape[0]})")

        # Metrics (using old as reference - assumes old model is the baseline)
        m_old = compute_metrics(old_boxes, old_boxes)
        m_old["total_reference"] = len(old_boxes)
        m_new = compute_metrics(new_boxes, old_boxes)

        print(f"\n{'='*65}")
        print(f"  СРАВНЕНИЕ (reference = old model detections)")
        print(f"  Время: old={old_time:.2f}s, new={new_time:.2f}s")
        print_table([("OLD", m_old), ("NEW", m_new)])
    else:
        if args.new:
            print(f"\nWarning: New model not found at {args.new}")
        print(f"\nDone. Run after training: python eval_compare.py --new models/yolov12n/yolov12n.pt")
