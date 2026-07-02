"""Color-based pricetag candidate detection."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from utils.config import get
from utils.iou import iou

logger = logging.getLogger("pricetag.candidate_detector")

_CFG: Optional[dict] = None


def _get_cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = get("candidate_detector") or {}
    return _CFG


def auto_calibrate_hsv(frame: np.ndarray) -> Optional[dict]:
    """Auto-calibrate HSV thresholds from frame statistics."""
    if not get("adaptive_hsv.enabled", False) or not HAS_CV2:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    v_low = max(np.percentile(v, 5), 30)
    v_high = min(np.percentile(v, 95), 255)
    s_low = np.percentile(s, 10)
    s_high = min(np.percentile(s, 90), 60)

    return {
        "white": {
            "lower": [0, int(s_low), int(v_low)],
            "upper": [180, int(s_high), int(v_high)],
        },
        "red": {
            "lower1": [0, max(30, int(np.percentile(s, 20))), 50],
            "upper1": [10, 255, 255],
            "lower2": [170, max(30, int(np.percentile(s, 20))), 50],
            "upper2": [180, 255, 255],
        },
        "yellow": {
            "lower": [20, max(40, int(np.percentile(s, 30))), max(120, int(v_high * 0.6))],
            "upper": [35, 255, 255],
        },
    }


def detect_pricetag_candidates(
    frame: np.ndarray,
    hsv_calib: Optional[dict] = None,
) -> list[tuple[int, int, int, int, float, str]]:
    """
    Color-based pricetag detection.
    Returns list of (x1, y1, x2, y2, confidence, type).
    """
    h, w = frame.shape[:2]

    if not HAS_CV2:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cfg = hsv_calib if hsv_calib is not None else _get_cfg()

    # Build masks
    white_mask = cv2.inRange(
        hsv,
        np.array(cfg["white"]["lower"]),
        np.array(cfg["white"]["upper"]),
    )
    r1 = cv2.inRange(
        hsv,
        np.array(cfg["red"]["lower1"]),
        np.array(cfg["red"]["upper1"]),
    )
    r2 = cv2.inRange(
        hsv,
        np.array(cfg["red"]["lower2"]),
        np.array(cfg["red"]["upper2"]),
    )
    red_mask = r1 | r2
    yellow_mask = cv2.inRange(
        hsv,
        np.array(cfg["yellow"]["lower"]),
        np.array(cfg["yellow"]["upper"]),
    )

    morph_kernel_size = cfg.get("morph_kernel", 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))

    def _get_boxes(mask: np.ndarray, color_key: str) -> list[tuple[int, int, int, int, int, int]]:
        """Extract filtered bounding boxes from a binary mask."""
        wcfg = cfg.get(color_key, {})
        min_w = wcfg.get("min_w", 40)
        min_h = wcfg.get("min_h", 20)
        max_area = wcfg.get("max_area", 60000)
        ar_min = wcfg.get("ar_min", 0.4)
        ar_max = wcfg.get("ar_max", 6.0)

        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            ar = bw / bh if bh else 0
            if bw >= min_w and bh >= min_h and area <= max_area and ar_min < ar < ar_max:
                boxes.append((x, y, x + bw, y + bh, bw, bh))
        return boxes

    white = _get_boxes(white_mask, "white")
    red = _get_boxes(red_mask, "red")
    yellow = _get_boxes(yellow_mask, "yellow")

    merge_radius = cfg.get("merge_radius", 150)
    nms_iou_thresh = cfg.get("nms_iou", 0.2)
    candidates: list[tuple[int, int, int, int, float, str]] = []

    # --- Merge red + white ---
    merged_white: set[tuple[int, int, int, int, int, int]] = set()
    for rx1, ry1, rx2, ry2, rbw, rbh in red:
        rcx = (rx1 + rx2) / 2
        rcy = (ry1 + ry2) / 2
        best_w: Optional[tuple[int, int, int, int, int, int]] = None
        best_d = merge_radius
        for wx1, wy1, wx2, wy2, wbw, wbh in white:
            d = np.sqrt((rcx - (wx1 + wx2) / 2) ** 2 + (rcy - (wy1 + wy2) / 2) ** 2)
            if d < best_d:
                best_d, best_w = d, (wx1, wy1, wx2, wy2, wbw, wbh)
        if best_w is not None:
            wx1, wy1, wx2, wy2, wbw, wbh = best_w
            merged_white.add((wx1, wy1, wx2, wy2, wbw, wbh))
            cx1, cy1 = min(rx1, wx1), min(ry1, wy1)
            cx2, cy2 = max(rx2, wx2), max(ry2, wy2)
            if (cx2 - cx1) * (cy2 - cy1) < 80000:
                conf = 0.6 + 0.3 * min((cx2 - cx1) / 300, 1.0)
                candidates.append((cx1, cy1, cx2, cy2, min(conf, 0.95), "promo"))

    # --- Red-only ---
    for rx1, ry1, rx2, ry2, rbw, rbh in red:
        rcx = (rx1 + rx2) / 2
        rcy = (ry1 + ry2) / 2
        has_white = any(
            np.sqrt((rcx - (wx1 + wx2) / 2) ** 2 + (rcy - (wy1 + wy2) / 2) ** 2) < merge_radius
            for wx1, wy1, wx2, wy2, wbw, wbh in white
        )
        if not has_white and rbh > 30 and rbw * rbh < 30000:
            candidates.append((rx1, ry1, rx2, ry2, 0.5, "promo"))

    # --- White-only ---
    for wx1, wy1, wx2, wy2, wbw, wbh in white:
        if (wx1, wy1, wx2, wy2, wbw, wbh) in merged_white:
            continue
        ar = wbw / wbh if wbh else 0
        wcfg = cfg.get("white", {})
        if wcfg.get("ar_min", 0.4) < ar < wcfg.get("ar_max", 6.0):
            conf = 0.5 + 0.3 * min(wbw / 250, 1.0)
            candidates.append((wx1, wy1, wx2, wy2, min(conf, 0.85), "pricetag"))

    # --- Yellow ---
    for yx1, yy1, yx2, yy2, ybw, ybh in yellow:
        candidates.append((yx1, yy1, yx2, yy2, 0.55, "promo"))

    # --- NMS ---
    candidates.sort(key=lambda c: c[4], reverse=True)
    keep: list[tuple[int, int, int, int, float, str]] = []
    while candidates:
        best = candidates.pop(0)
        keep.append(best)
        candidates = [c for c in candidates if iou(best[:4], c[:4]) < nms_iou_thresh]
    return keep


def expand_crop(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    target_size: int = 640,
    expand_factor: Optional[int] = None,
) -> tuple[int, int, int, int]:
    """Expand bbox by factor but keep within frame. Adds padding for OCR context."""
    if expand_factor is None:
        expand_factor = get("cascade.crop_expand_factor", 2) or 2
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # Expand: 2x bbox size, with min 64px padding per side
    pad = max(64, int(max(bw, bh) * (expand_factor - 1) / 2))
    return (
        int(max(0, x1 - pad)),
        int(max(0, y1 - pad)),
        int(min(frame_w, x2 + pad)),
        int(min(frame_h, y2 + pad)),
    )
