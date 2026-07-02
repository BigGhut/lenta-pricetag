"""IoU computation utilities."""

from __future__ import annotations

import numpy as np


def iou(box1: tuple[float, ...] | np.ndarray, box2: tuple[float, ...] | np.ndarray) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    # Optimization: unpack to avoid repeated indexing and float casting overhead
    b1_x1, b1_y1, b1_x2, b1_y2 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

    # Optimization: calculate intersection width and height first
    inter_w = min(b1_x2, b2_x2) - max(b1_x1, b2_x1)
    # Optimization: early return if no intersection, saving area calculations and division
    if inter_w <= 0:
        return 0.0

    inter_h = min(b1_y2, b2_y2) - max(b1_y1, b2_y1)
    if inter_h <= 0:
        return 0.0

    inter = inter_w * inter_h
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter / (area1 + area2 - inter + 1e-6)
