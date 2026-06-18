import numpy as np
from utils.iou import iou


def weighted_boxes_fusion(boxes, scores, iou_thr=0.45, skip_box_thr=0.15):
    if boxes is None or len(boxes) == 0:
        return None

    boxes = np.array(boxes)
    scores = np.array(scores)

    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    scores = scores[order]

    fused = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue

        cluster = [i]
        used.add(i)

        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            iou_val = iou(boxes[i], boxes[j])
            if iou_val >= iou_thr:
                cluster.append(j)
                used.add(j)

        if len(cluster) == 1:
            if scores[cluster[0]] >= skip_box_thr:
                fused.append([*boxes[cluster[0]], scores[cluster[0]]])
            continue

        weighted_box = np.zeros(4)
        total_conf = 0.0
        for idx in cluster:
            conf = scores[idx]
            weighted_box += boxes[idx] * conf
            total_conf += conf

        weighted_box /= total_conf
        avg_conf = np.mean([scores[idx] for idx in cluster])

        if avg_conf >= skip_box_thr:
            fused.append([*weighted_box, avg_conf])

    return np.array(fused) if fused else None



