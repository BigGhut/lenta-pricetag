"""Tests for utils.candidate_detector — expand_crop and _iou helper functions.

detect_pricetag_candidates requires cv2 image input and is better suited for
integration tests; here we test the pure geometry functions.
"""

import pytest
from utils.candidate_detector import expand_crop, _iou


# ── expand_crop ────────────────────────────────────────────────


class TestExpandCrop:
    def test_small_box_expanded(self):
        # Small box in large frame → should expand to target_size
        bbox = (480, 270, 520, 310)
        result = expand_crop(bbox, frame_w=960, frame_h=540, expand_factor=5)
        x1, y1, x2, y2 = result
        # Crop should be centered on bbox center (500, 290)
        assert x1 < 480
        assert y1 < 270
        assert x2 > 520
        assert y2 > 310

    def test_crop_clamped_to_frame(self):
        # Box at top-left corner → should not go negative
        bbox = (0, 0, 50, 50)
        result = expand_crop(bbox, frame_w=640, frame_h=480, expand_factor=3)
        x1, y1, x2, y2 = result
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480

    def test_crop_centered(self):
        # Verify crop is roughly centered on bbox center when there's room
        # to expand on both sides (bbox far from frame edges).
        bbox = (760, 440, 840, 540)  # center (800, 490), big frame
        result = expand_crop(bbox, frame_w=1920, frame_h=1080, expand_factor=2)
        bbox_cx = (760 + 840) / 2
        bbox_cy = (440 + 540) / 2
        crop_cx = (result[0] + result[2]) / 2
        crop_cy = (result[1] + result[3]) / 2
        assert abs(bbox_cx - crop_cx) < 1
        assert abs(bbox_cy - crop_cy) < 1

    def test_expand_factor_in_config(self):
        # Default expand_factor=5 (from config)
        bbox = (460, 260, 500, 300)
        result = expand_crop(bbox, frame_w=960, frame_h=540)
        width = result[2] - result[0]
        assert width > 500  # expanded significantly from 40px bbox

    def test_uses_target_size_minimum(self):
        # When bbox is centered in a large frame, expand >= target_size on
        # both sides → crop width should be at least target_size.
        bbox = (950, 520, 970, 540)  # small box near center of 1920x1080
        result = expand_crop(bbox, frame_w=1920, frame_h=1080, target_size=640)
        width = result[2] - result[0]
        assert width >= 640  # at least target_size


# ── _iou ───────────────────────────────────────────────────────


class TestIoU:
    def test_identical_boxes_one(self):
        box = (0, 0, 100, 100)
        assert _iou(box, box) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap_zero(self):
        box1 = (0, 0, 100, 100)
        box2 = (200, 200, 300, 300)
        assert _iou(box1, box2) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        # Intersection: (50,50)-(100,100) = 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert _iou(box1, box2) == pytest.approx(expected, abs=1e-6)

    def test_contained_box(self):
        box1 = (0, 0, 100, 100)
        box2 = (25, 25, 75, 75)
        # box2 fully inside box1 → IoU = area2 / area1
        assert _iou(box1, box2) == pytest.approx(0.25, abs=1e-6)

    def test_touching_edges_no_overlap(self):
        box1 = (0, 0, 100, 100)
        box2 = (100, 0, 200, 100)  # touching at x=100
        assert _iou(box1, box2) == pytest.approx(0.0, abs=1e-6)
