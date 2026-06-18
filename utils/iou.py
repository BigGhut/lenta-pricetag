"""IoU computation utilities."""

from __future__ import annotations

import numpy as np


def iou(box1: tuple[float, ...] | np.ndarray, box2: tuple[float, ...] | np.ndarray) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
    area2 = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))
    return inter / (area1 + area2 - inter + 1e-6)
