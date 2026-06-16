"""Tests for utils.pricetag_tracker — distance functions, TrackedTag lifecycle, PricetagTracker matching.
"""

import pytest
from utils.pricetag_tracker import (
    center_distance,
    size_ratio,
    TrackedTag,
    PricetagTracker,
)


# ── center_distance ────────────────────────────────────────────


class TestCenterDistance:
    def test_identical_boxes_zero(self):
        box = (0, 0, 100, 100)
        assert center_distance(box, box) == pytest.approx(0.0, abs=1e-6)

    def test_same_center_different_sizes(self):
        # Same center, one wider → distance = 0 (centers match)
        box1 = (0, 0, 100, 100)
        box2 = (25, 25, 75, 75)  # same center at (50, 50)
        assert center_distance(box1, box2) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_box_positive_distance(self):
        box1 = (0, 0, 100, 100)   # center (50, 50)
        box2 = (100, 0, 200, 100)  # center (150, 50)
        dist = center_distance(box1, box2)
        assert dist > 0

    def test_large_shift_normalized(self):
        # Distance is normalized by average box size
        box1 = (0, 0, 100, 100)
        box2 = (200, 200, 300, 300)  # center shifted by ~283
        dist = center_distance(box1, box2)
        # Should be > 1 (shift larger than box)
        assert dist > 1.0

    def test_tiny_boxes_big_shift(self):
        box1 = (0, 0, 10, 10)
        box2 = (100, 100, 110, 110)
        dist = center_distance(box1, box2)
        assert dist > 10.0


# ── size_ratio ─────────────────────────────────────────────────


class TestSizeRatio:
    def test_identical_boxes_one(self):
        box = (0, 0, 100, 100)
        assert size_ratio(box, box) == pytest.approx(1.0, abs=1e-6)

    def test_one_twice_other(self):
        box1 = (0, 0, 100, 100)   # area = 10000
        box2 = (0, 0, 70, 70)     # area = 4900
        assert size_ratio(box1, box2) == pytest.approx(0.49, abs=1e-6)

    def test_order_independent(self):
        box1 = (0, 0, 100, 100)
        box2 = (0, 0, 50, 50)
        assert size_ratio(box1, box2) == pytest.approx(size_ratio(box2, box1), abs=1e-6)

    def test_zero_area_box(self):
        # Degenerate box with zero size → min/max gives 0/area
        box1 = (0, 0, 0, 0)
        box2 = (0, 0, 100, 100)
        assert size_ratio(box1, box2) == pytest.approx(0.0, abs=1e-6)


# ── TrackedTag ─────────────────────────────────────────────────


class TestTrackedTag:
    def test_creation_defaults(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.8, "pricetag", 10)
        assert tag.tag_id == 1
        assert tag.bbox == (0, 0, 100, 100)
        assert tag.confidence == 0.8
        assert tag.tag_type == "pricetag"
        assert tag.ocr_data == {}
        assert tag.frames_missed == 0
        assert tag.first_seen == 10
        assert tag.last_seen == 10
        assert tag.color is None

    def test_update_increases_confidence(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5, "pricetag", 0)
        tag.update((10, 10, 110, 110), 0.9, 5)
        assert tag.confidence == 0.9  # max(0.5, 0.9)
        assert tag.bbox == (10, 10, 110, 110)
        assert tag.last_seen == 5
        assert tag.frames_missed == 0

    def test_update_resets_missed(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5)
        tag.mark_missed()
        tag.mark_missed()
        assert tag.frames_missed == 2
        tag.update((0, 0, 100, 100), 0.5, 3)
        assert tag.frames_missed == 0

    def test_mark_missed_increments(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5)
        tag.mark_missed()
        assert tag.frames_missed == 1
        tag.mark_missed()
        assert tag.frames_missed == 2

    def test_is_stale_below_threshold(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5)
        tag._stale_threshold = 5
        for _ in range(5):
            tag.mark_missed()
        assert tag.is_stale is False  # 5 == 5, not > 5

    def test_is_stale_above_threshold(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5)
        tag._stale_threshold = 5
        for _ in range(6):
            tag.mark_missed()
        assert tag.is_stale is True

    def test_is_stale_custom_threshold(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5)
        tag._stale_threshold = 3
        for _ in range(4):
            tag.mark_missed()
        assert tag.is_stale is True

    def test_age(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5, frame_idx=10)
        tag.update((0, 0, 100, 100), 0.5, 20)
        assert tag.age == 10  # 20 - 10

    def test_slots_prevent_extra_attrs(self):
        tag = TrackedTag(1, (0, 0, 100, 100), 0.5)
        with pytest.raises(AttributeError):
            tag.nonexistent_attr = "value"


# ── PricetagTracker ────────────────────────────────────────────


class TestPricetagTracker:
    def test_empty_update_no_candidates(self):
        tracker = PricetagTracker(max_frames_missed=3)
        new, tracked = tracker.update([], frame_idx=0)
        assert new == []
        assert tracked == []
        assert tracker.active_tags == []

    def test_first_frame_creates_new(self):
        tracker = PricetagTracker()
        candidates = [
            (0, 0, 100, 100, 0.8, "pricetag"),
            (200, 200, 300, 300, 0.7, "promo"),
        ]
        new, tracked = tracker.update(candidates, frame_idx=0)
        assert len(new) == 2
        assert tracked == []
        assert len(tracker.active_tags) == 2

    def test_match_existing_tag(self):
        tracker = PricetagTracker(center_dist_thresh=3.0, size_ratio_thresh=0.2)
        candidates = [(0, 0, 100, 100, 0.8, "pricetag")]
        tracker.update(candidates, frame_idx=0)

        # Same bbox next frame → should match
        candidates = [(5, 5, 105, 105, 0.85, "pricetag")]
        new, tracked = tracker.update(candidates, frame_idx=1)
        assert len(new) == 0
        assert len(tracked) == 1

    def test_stale_tag_removed(self):
        tracker = PricetagTracker(max_frames_missed=2, center_dist_thresh=3.0)
        candidates = [(0, 0, 100, 100, 0.8, "pricetag")]
        tracker.update(candidates, frame_idx=0)

        # No candidates for 3 frames → stale
        tracker.update([], frame_idx=1)
        tracker.update([], frame_idx=2)
        tracker.update([], frame_idx=3)
        assert tracker.active_tags == []

    def test_new_candidate_after_stale_cleanup(self):
        tracker = PricetagTracker(max_frames_missed=1, center_dist_thresh=3.0)
        candidates = [(0, 0, 100, 100, 0.8, "pricetag")]
        tracker.update(candidates, frame_idx=0)

        # Miss two frames → tag stale (frames_missed > max_frames_missed)
        tracker.update([], frame_idx=1)  # frames_missed = 1
        tracker.update([], frame_idx=2)  # frames_missed = 2 > 1 → cleaned
        candidates = [(0, 0, 100, 100, 0.8, "pricetag")]
        new, tracked = tracker.update(candidates, frame_idx=3)
        # Old tag was cleaned up, so this is a new tag
        assert len(new) == 1
        assert len(tracked) == 0

    def test_reset_clears_all(self):
        tracker = PricetagTracker()
        candidates = [(0, 0, 100, 100, 0.8, "pricetag")]
        tracker.update(candidates, frame_idx=0)
        tracker.reset()
        assert tracker.active_tags == []
        assert len(tracker._tags) == 0

    def test_multiple_new_candidates(self):
        tracker = PricetagTracker()
        candidates = [
            (0, 0, 100, 100, 0.8, "pricetag"),
            (200, 200, 300, 300, 0.7, "promo"),
            (400, 0, 500, 80, 0.6, "pricetag"),
        ]
        new, tracked = tracker.update(candidates, frame_idx=0)
        assert len(new) == 3
        # Verify each new tag has the correct bbox and type
        assert new[0]["bbox"] == (0, 0, 100, 100)
        assert new[1]["tag_type"] == "promo"
        assert new[2]["confidence"] == 0.6
