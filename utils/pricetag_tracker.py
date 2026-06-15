import numpy as np
from collections import OrderedDict

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def center_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    avg_size = ((box1[2] - box1[0]) + (box1[3] - box1[1]) +
                (box2[2] - box2[0]) + (box2[3] - box2[1])) / 4
    return dist / (avg_size + 1e-6)


def size_ratio(box1, box2):
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return min(a1, a2) / (max(a1, a2) + 1e-6)


class TrackedTag:
    __slots__ = ("tag_id", "bbox", "ocr_data", "last_seen", "first_seen",
                 "confidence", "tag_type", "frames_missed", "color",
                 "_stale_threshold")

    def __init__(self, tag_id, bbox, confidence=0.0, tag_type="white", frame_idx=0):
        self.tag_id = tag_id
        self.bbox = bbox
        self.ocr_data = {}
        self.last_seen = frame_idx
        self.first_seen = frame_idx
        self.confidence = confidence
        self.tag_type = tag_type
        self.frames_missed = 0
        self.color = None
        self._stale_threshold = 5  # default; tracker может переопределить

    def update(self, bbox, confidence, frame_idx):
        self.bbox = bbox
        self.confidence = max(self.confidence, confidence)
        self.last_seen = frame_idx
        self.frames_missed = 0

    def mark_missed(self):
        self.frames_missed += 1

    @property
    def is_stale(self):
        # Порог по умолчанию; трекер использует своё значение через _cleanup/active_tags.
        return self.frames_missed > self._stale_threshold

    @property
    def age(self):
        return self.last_seen - self.first_seen


class PricetagTracker:
    def __init__(self, center_dist_thresh=2.0, size_ratio_thresh=0.25, max_frames_missed=5):
        self.center_dist_thresh = center_dist_thresh
        self.size_ratio_thresh = size_ratio_thresh
        self.max_frames_missed = max_frames_missed
        self._tags = OrderedDict()
        self._next_id = 0

    @property
    def active_tags(self):
        return [t for t in self._tags.values() if not t.frames_missed > self.max_frames_missed]

    def _match_score(self, tb, cb):
        cd = center_distance(tb, cb)
        sr = size_ratio(tb, cb)
        if cd > self.center_dist_thresh or sr < self.size_ratio_thresh:
            return -1.0
        return sr * (1.0 - cd / self.center_dist_thresh)

    def update(self, candidates, frame_idx):
        if not candidates:
            for t in self.active_tags:
                t.mark_missed()
            self._cleanup()
            return [], list(self.active_tags)

        cand_bboxes = np.array([c[:4] for c in candidates])
        cand_confs = np.array([c[4] for c in candidates])
        cand_types = [c[5] for c in candidates]

        tracked = self.active_tags
        if not tracked:
            return self._create_new(candidates, frame_idx), []

        scores = np.zeros((len(tracked), len(candidates)))
        for ti, t in enumerate(tracked):
            for ci, cb in enumerate(cand_bboxes):
                scores[ti, ci] = self._match_score(t.bbox, cb)

        if HAS_SCIPY and scores.size > 0:
            t_idx, c_idx = linear_sum_assignment(-scores)
        else:
            t_idx, c_idx = self._greedy(scores)

        matched = set()
        for ti, ci in zip(t_idx, c_idx):
            if scores[ti, ci] > 0:
                tracked[ti].update(cand_bboxes[ci], cand_confs[ci], frame_idx)
                tracked[ti].tag_type = cand_types[ci]
                matched.add(ci)

        unmatched_t = set(range(len(tracked))) - set(t_idx)
        for ti in unmatched_t:
            tracked[ti].mark_missed()

        new_tags = self._create_new(
            [candidates[i] for i in range(len(candidates)) if i not in matched],
            frame_idx,
        )

        self._cleanup()
        return new_tags, list(self.active_tags)

    def _create_new(self, candidates, frame_idx):
        tags = []
        for c in candidates:
            tag = TrackedTag(self._next_id, tuple(map(int, c[:4])), c[4], c[5], frame_idx)
            tag._stale_threshold = self.max_frames_missed
            self._tags[self._next_id] = tag
            self._next_id += 1
            tags.append({"tag": tag, "bbox": c[:4], "confidence": c[4], "tag_type": c[5]})
        return tags

    def _greedy(self, scores):
        rows, cols = [], []
        used_r, used_c = set(), set()
        while True:
            idx = np.unravel_index(np.argmax(scores), scores.shape)
            if scores[idx] <= 0:
                break
            if idx[0] not in used_r and idx[1] not in used_c:
                rows.append(idx[0])
                cols.append(idx[1])
                used_r.add(idx[0])
                used_c.add(idx[1])
            scores[idx] = -1
        return np.array(rows), np.array(cols)

    def _cleanup(self):
        stale = [tid for tid, t in self._tags.items() if t.frames_missed > self.max_frames_missed]
        for tid in stale:
            del self._tags[tid]

    def reset(self):
        self._tags.clear()
        self._next_id = 0
