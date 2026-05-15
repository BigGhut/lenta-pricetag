import cv2
import numpy as np
from utils.config import get

CFG = get("candidate_detector")


def detect_pricetag_candidates(frame):
    """
    Simple color-based pricetag detection.
    Returns (x1, y1, x2, y2, confidence, type).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color masks
    white_mask = cv2.inRange(hsv, np.array(CFG["white"]["lower"]), np.array(CFG["white"]["upper"]))
    r1 = cv2.inRange(hsv, np.array(CFG["red"]["lower1"]), np.array(CFG["red"]["upper1"]))
    r2 = cv2.inRange(hsv, np.array(CFG["red"]["lower2"]), np.array(CFG["red"]["upper2"]))
    red_mask = r1 | r2
    yellow_mask = cv2.inRange(hsv, np.array(CFG["yellow"]["lower"]), np.array(CFG["yellow"]["upper"]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CFG["morph_kernel"], CFG["morph_kernel"]))

    def get_boxes(mask, min_w, min_h):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw >= min_w and bh >= min_h:
                res.append((x, y, x + bw, y + bh, bw, bh))
        return res

    white = get_boxes(white_mask, 40, 20)
    red = get_boxes(red_mask, 15, 15)
    yellow = get_boxes(yellow_mask, 40, 20)

    candidates = []

    # --- Merge red + white (захватывает ценник целиком) ---
    merged_white = set()
    for rx1, ry1, rx2, ry2, rbw, rbh in red:
        rcx = (rx1 + rx2) / 2
        rcy = (ry1 + ry2) / 2
        best_w, best_d = None, 250
        for wx1, wy1, wx2, wy2, wbw, wbh in white:
            d = np.sqrt((rcx - (wx1 + wx2) / 2) ** 2 + (rcy - (wy1 + wy2) / 2) ** 2)
            if d < best_d:
                best_d, best_w = d, (wx1, wy1, wx2, wy2, wbw, wbh)
        if best_w:
            wx1, wy1, wx2, wy2, wbw, wbh = best_w
            merged_white.add((wx1, wy1, wx2, wy2, wbw, wbh))
            cx1, cy1 = min(rx1, wx1), min(ry1, wy1)
            cx2, cy2 = max(rx2, wx2), max(ry2, wy2)
            if (cx2 - cx1) * (cy2 - cy1) < 80000:
                conf = 0.6 + 0.3 * min((cx2 - cx1) / 300, 1.0)
                candidates.append((cx1, cy1, cx2, cy2, min(conf, 0.95), "promo"))

    # --- Red-only (если белый рядом не нашёлся) ---
    for rx1, ry1, rx2, ry2, rbw, rbh in red:
        rcx = (rx1 + rx2) / 2
        rcy = (ry1 + ry2) / 2
        has_white = False
        for wx1, wy1, wx2, wy2, wbw, wbh in white:
            if np.sqrt((rcx - (wx1+wx2)/2)**2 + (rcy - (wy1+wy2)/2)**2) < 250:
                has_white = True
                break
        if not has_white and rbh > 30 and rbw * rbh < 40000:
            candidates.append((rx1, ry1, rx2, ry2, 0.5, "promo"))

    # --- White-only (обычные белые ценники) ---
    for wx1, wy1, wx2, wy2, wbw, wbh in white:
        if (wx1, wy1, wx2, wy2, wbw, wbh) in merged_white:
            continue
        ar = wbw / wbh if wbh else 0
        if 0.4 < ar < 6.0 and wbw * wbh < 60000:
            conf = 0.5 + 0.3 * min(wbw / 250, 1.0)
            candidates.append((wx1, wy1, wx2, wy2, min(conf, 0.85), "pricetag"))

    # --- Yellow ---
    for yx1, yy1, yx2, yy2, ybw, ybh in yellow:
        ar = ybw / ybh if ybh else 0
        if 0.4 < ar < 5.0 and ybw * ybh < 50000:
            candidates.append((yx1, yy1, yx2, yy2, 0.55, "promo"))

    # --- NMS ---
    candidates.sort(key=lambda c: c[4], reverse=True)
    keep = []
    while candidates:
        best = candidates.pop(0)
        keep.append(best)
        candidates = [c for c in candidates if _iou(best[:4], c[:4]) < 0.3]

    return keep


def expand_crop(bbox, frame_w, frame_h, target_size=640, expand_factor=None):
    if expand_factor is None:
        expand_factor = get("cascade.crop_expand_factor", 5)
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    expand = max(target_size, bw * expand_factor, bh * expand_factor)
    return (int(max(0, cx - expand / 2)),
            int(max(0, cy - expand / 2)),
            int(min(frame_w, cx + expand / 2)),
            int(min(frame_h, cy + expand / 2)))


def _iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)
