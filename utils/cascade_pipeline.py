"""Cascade pipeline: color detection → tracking → YOLO → OCR → CSV."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

from utils.candidate_detector import detect_pricetag_candidates, expand_crop, auto_calibrate_hsv
from utils.pricetag_tracker import PricetagTracker, TrackedTag
from utils.csv_writer import PricetagCSV, make_row, update_from_yolo_detection, classify_color
from utils.config import get
from utils.tiled_inference import TiledYOLO
from utils.iou import iou

logger = logging.getLogger("pricetag.cascade_pipeline")

# OCR engine setup — EasyOCR only (GPU via torch)
from infer.infer_easyocr import (
    recognize_full_pricetag,
    extract_price,
    validate_ean13,
)
HAS_OCR = True

from utils.qr_parser import parse_qr_content
from utils.product_db import lookup as lookup_product


def _nms_merge(
    yolo_boxes: list[list[float]],
    color_candidates: list[list[float]],
    iou_thresh: float = 0.3,
) -> list[list[float]]:
    """Merge YOLO and color candidates with NMS."""
    yolo_priority: list[list[float]] = []
    for b in yolo_boxes:
        yolo_priority.append([*b[:4], b[4], 0.0])  # 0.0 = yolo type encoded as float

    color: list[list[float]] = []
    for c in color_candidates:
        color.append([*c[:4], c[4], 1.0])  # 1.0 = color type

    if not yolo_priority and not color:
        return []

    # YOLO boxes suppress overlapping color boxes
    for yb in yolo_priority:
        color = [c for c in color if iou(yb[:4], c[:4]) < iou_thresh]

    # NMS within color only
    color.sort(key=lambda x: x[4], reverse=True)
    kept_color: list[list[float]] = []
    while color:
        best = color.pop(0)
        kept_color.append(best)
        color = [c for c in color if iou(best[:4], c[:4]) < iou_thresh]

    return yolo_priority + kept_color


class CascadePipeline:
    """Main detection pipeline with tracking and OCR caching."""

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self.tiled_yolo: Optional[TiledYOLO] = None
        sahi_cfg = get("sahi", {})
        ms_cfg = get("multiscale", {})
        if sahi_cfg.get("enabled", False):
            try:
                self.tiled_yolo = TiledYOLO(
                    model_path=get("model.trained"),
                    tile_size=sahi_cfg.get("tile_size", 640),
                    stride=sahi_cfg.get("stride", 320),
                    conf=sahi_cfg.get("conf_threshold", 0.15),
                    iou=sahi_cfg.get("iou_threshold", 0.45),
                    scales=ms_cfg.get("scales", [1.0]) if ms_cfg.get("enabled", False) else [1.0],
                )
                logger.info(
                    "TiledYOLO loaded: tile=%d, stride=%d, scales=%s",
                    sahi_cfg.get("tile_size", 640),
                    sahi_cfg.get("stride", 320),
                    self.tiled_yolo.scales,
                )
            except Exception as e:
                logger.error("TiledYOLO init failed: %s", e)

        self.csv = PricetagCSV(csv_path)
        tcfg = get("tracker")
        self.tracker = PricetagTracker(
            center_dist_thresh=tcfg["center_dist_thresh"],
            size_ratio_thresh=tcfg["size_ratio_thresh"],
            max_frames_missed=tcfg["max_frames_missed"],
        )
        self._hsv_calib: Optional[dict] = None
        self._calib_frames: list[np.ndarray] = []
        self._first_frame = True
        self._ocr_count = 0
        self._tracked_hits = 0
        self._all_rows: list[dict[str, str]] = []

        # OCR interval for tracked tags: re-OCR every N frames
        self._ocr_interval = get("cascade.ocr_interval", 5)

        # Adaptive YOLO: skip YOLO when tracker is confident
        self._adaptive_yolo = get("cascade.adaptive_yolo", True)
        self._yolo_skip_frames = get("cascade.yolo_skip_frames", 5)  # skip YOLO for N frames when tracking
        self._yolo_force_interval = get("cascade.yolo_force_interval", 15)  # force YOLO every N frames
        self._yolo_cooldown = 0  # frames since last YOLO run

        acfg = get("adaptive_conf", {})
        self._conf_adaptive = acfg.get("enabled", False)
        self._conf_base = acfg.get("base_threshold", 0.5)
        self._conf_current = self._conf_base
        self._conf_min = acfg.get("min_threshold", 0.25)
        self._conf_max = acfg.get("max_threshold", 0.7)
        self._detection_history: deque[int] = deque(maxlen=acfg.get("window", 30))
        self._density_low = acfg.get("density_low", 3)
        self._density_high = acfg.get("density_high", 12)
        self._temporal_boost = get("temporal_boost.enabled", False)
        self._temporal_max_boost = get("temporal_boost.max_boost", 1.3)
        self._temporal_boost_per_frame = get("temporal_boost.per_frame", 0.03)

    def _calibrate(self, frame: np.ndarray) -> None:
        calib_frames = get("adaptive_hsv.calibration_frames", 5)
        self._calib_frames.append(frame)
        if len(self._calib_frames) >= calib_frames:
            ref = np.mean(self._calib_frames, axis=0).astype(np.uint8)
            self._hsv_calib = auto_calibrate_hsv(ref)
            self._calib_frames.clear()
            if self._hsv_calib:
                logger.info(
                    "HSV calibrated: white(V=[%d, %d])",
                    self._hsv_calib["white"]["lower"][2],
                    self._hsv_calib["white"]["upper"][2],
                )

    def _update_threshold(self, detection_count: int) -> None:
        if not self._conf_adaptive:
            return
        self._detection_history.append(detection_count)
        if len(self._detection_history) < 5:
            return
        median = float(np.median(self._detection_history))
        if median < self._density_low:
            new_conf = max(self._conf_min, self._conf_base - 0.15)
        elif median > self._density_high:
            new_conf = min(self._conf_max, self._conf_base + 0.15)
        else:
            new_conf = self._conf_base
        self._conf_current = self._conf_current * 0.7 + new_conf * 0.3

    def _apply_temporal_boost(self, tag: object) -> float:
        if not self._temporal_boost:
            return tag.confidence
        boost = 1.0 + min(tag.age * self._temporal_boost_per_frame, self._temporal_max_boost - 1.0)
        return min(tag.confidence * boost, 1.0)

    def process_frame(
        self,
        frame: np.ndarray,
        filename: str = "camera",
        frame_timestamp: int = 0,
    ) -> list[dict]:
        """Process a single frame through the cascade pipeline."""
        h, w = frame.shape[:2]

        if self._first_frame:
            self._calibrate(frame)
            self._first_frame = False
        elif len(self._calib_frames) > 0:
            self._calibrate(frame)

        # --- Adaptive YOLO: decide whether to run YOLO this frame ---
        active_count = len(self.tracker.active_tags)
        should_run_yolo = True
        if self._adaptive_yolo:
            # Force YOLO periodically (new tags may appear)
            if self._yolo_cooldown < self._yolo_force_interval:
                # Skip YOLO if we have confident tracks and cooldown not expired
                has_confident_tracks = any(
                    t.age >= 3 and t.frames_missed == 0
                    for t in self.tracker.active_tags
                )
                if has_confident_tracks and self._yolo_cooldown < self._yolo_skip_frames:
                    should_run_yolo = False
            self._yolo_cooldown += 1

        # YOLO detection (tiled) — conditionally skipped
        yolo_candidates: list[tuple[int, int, int, int, float, str]] = []
        if self.tiled_yolo is not None and should_run_yolo:
            try:
                yolo_boxes = self.tiled_yolo.detect(frame)
                yolo_candidates = [
                    (int(x1), int(y1), int(x2), int(y2), float(conf), "yolo")
                    for x1, y1, x2, y2, conf in yolo_boxes
                ]
                self._yolo_cooldown = 0  # reset cooldown after YOLO run
            except Exception as e:
                logger.error("TiledYOLO error: %s", e)

        # Color-based detection (always runs — cheap)
        color_candidates = detect_pricetag_candidates(frame, hsv_calib=self._hsv_calib) or []

        # Merge with NMS
        yolo_as_list: list[list[float]] = [[float(v) for v in c[:5]] for c in yolo_candidates]
        color_as_list: list[list[float]] = [[float(v) for v in c[:5]] + [0.0] for c in color_candidates]
        merged = _nms_merge(yolo_as_list, color_as_list, iou_thresh=0.45)

        # Update tracker — convert merged boxes back to tuples
        merged_tuples = [
            (int(m[0]), int(m[1]), int(m[2]), int(m[3]), m[4], "yolo" if m[5] == 0.0 else "color")
            for m in merged
        ]
        new_tags, tracked = self.tracker.update(merged_tuples, frame_timestamp)

        self._update_threshold(len(merged) + len(tracked))

        # Process tracked tags (with OCR interval + batching)
        results = self._process_tracked(tracked, frame, filename, frame_timestamp, h, w)
        # Process new tags (always OCR + batching)
        results += self._process_new(new_tags, frame, filename, frame_timestamp, h, w)
        return results

    def _process_tracked(
        self,
        tags: list[TrackedTag],
        frame: Optional[np.ndarray],
        filename: str,
        frame_timestamp: int,
        h: int,
        w: int,
    ) -> list[dict]:
        """Process tracked tags — OCR only every N frames, with batching."""
        results: list[dict] = []
        # Collect crops that need OCR this frame
        ocr_batch: list[tuple[TrackedTag, np.ndarray]] = []

        for t in tags:
            self._tracked_hits += 1
            boosted_conf = self._apply_temporal_boost(t)

            # Re-OCR only if no OCR data yet (first time seeing this tag)
            should_ocr = (
                frame is not None
                and HAS_OCR
                and t.frames_missed == 0
                and not t.ocr_data
            )
            if should_ocr:
                expand_factor = get("cascade.crop_expand_factor", 5)
                cx1, cy1, cx2, cy2 = expand_crop(t.bbox, w, h, expand_factor=expand_factor)
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                if cx2 > cx1 and cy2 > cy1:
                    pt_crop = frame[cy1:cy2, cx1:cx2]
                    if pt_crop.size > 0:
                        ocr_batch.append((t, pt_crop))

            # Append result (OCR may be filled below)
            results.append({
                "bbox": t.bbox,
                "confidence": boosted_conf,
                "ocr": t.ocr_data,
                "type": t.tag_type,
                "tracked": True,
                "tracker_id": t.tag_id,
            })

        # Run batched OCR for all collected crops
        if ocr_batch:
            self._run_easyocr_batch(ocr_batch)

        return results

    def _run_batched_ocr(
        self,
        batch: list[tuple[TrackedTag, np.ndarray]],
    ) -> None:
        """Run OCR on a batch of crops using EasyOCR batch processing."""
        if not batch:
            return

        # Process each crop individually but efficiently
        for tag, crop in batch:
            ocr_info = recognize_full_pricetag(crop)
            if ocr_info:
                tag._merge_ocr(ocr_info)
                self._ocr_count += 1

    def _run_easyocr_batch(
        self,
        batch: list[tuple[TrackedTag, np.ndarray]],
    ) -> None:
        """Run EasyOCR on each crop individually (resized for speed)."""
        if not batch:
            return
        for tag, crop in batch:
            ocr_info = recognize_full_pricetag(crop)
            if ocr_info:
                tag._merge_ocr(ocr_info)
                self._ocr_count += 1

    def _texts_to_ocr_info(self, texts: list[str]) -> dict:
        """Convert raw OCR texts to structured info (simplified version of recognize_full_pricetag)."""
        import re
        info: dict = {}
        all_text = " ".join(str(t) for t in texts)

        # Extract prices
        prices = extract_price(texts)
        if prices:
            info["price_default"] = prices[0]
            if len(prices) > 1:
                info["price_card"] = prices[1]
            if len(prices) > 2:
                info["price_discount"] = prices[2]

        # Barcode
        m = re.search(r"\b(\d{12,13})\b", all_text)
        if m and validate_ean13(m.group(1)):
            info["barcode"] = m.group(1)

        # Product name
        long_texts = [
            t for t in texts
            if isinstance(t, str) and len(t) > 10
            and not re.search(r"\d{6,}", t)
            and not re.fullmatch(r"[\d.,%/\\-]+", t)
        ]
        if long_texts:
            info["product_name"] = max(long_texts, key=len)

        return info

    def _process_new(
        self,
        new_tags: list[dict],
        frame: np.ndarray,
        filename: str,
        frame_timestamp: int,
        h: int,
        w: int,
    ) -> list[dict]:
        """Process new tags — always run OCR."""
        results: list[dict] = []
        for nt in new_tags:
            tag = nt["tag"]
            cand = nt["bbox"][:4]
            is_yolo = nt.get("tag_type") == "yolo"

            if is_yolo:
                ax1, ay1, ax2, ay2 = [int(v) for v in cand]
                # Expand YOLO bbox for OCR (tight box misses text context)
                expand_factor = get("cascade.crop_expand_factor", 5)
                cx1, cy1, cx2, cy2 = expand_crop(cand, w, h, expand_factor=expand_factor)
                # Color from original tight bbox
                ax1, ay1 = max(0, ax1), max(0, ay1)
                ax2, ay2 = min(w, ax2), min(h, ay2)
                # OCR from expanded crop
                pt_crop = frame[cy1:cy2, cx1:cx2]
            else:
                expand_factor = get("cascade.crop_expand_factor", 5)
                cx1, cy1, cx2, cy2 = expand_crop(cand, w, h, expand_factor=expand_factor)
                # Color from original tight bbox
                ax1, ay1, ax2, ay2 = [int(v) for v in cand]
                ax1, ay1 = max(0, ax1), max(0, ay1)
                ax2, ay2 = min(w, ax2), min(h, ay2)
                # OCR from expanded crop (more context around pricetag)
                pt_crop = frame[cy1:cy2, cx1:cx2]

            if pt_crop.size == 0:
                continue

            # Color from original tight bbox, OCR from (possibly expanded) crop
            color_crop = frame[ay1:ay2, ax1:ax2]
            if color_crop.size > 0:
                tag.color = np.mean(color_crop, axis=(0, 1))
            ocr_info: dict = {}
            if HAS_OCR:
                ocr_info = recognize_full_pricetag(pt_crop)

            tag.ocr_data = ocr_info
            self._ocr_count += 1

            row = make_row()
            row["tracker_id"] = str(tag.tag_id)
            update_from_yolo_detection(row, [*tag.bbox, tag.confidence, 0], filename, frame_timestamp)
            self.csv.update_from_ocr(row, list(ocr_info.values()), tag.color)
            for k, v in ocr_info.items():
                if k in row and v and not row.get(k):
                    row[k] = v

            qr_raw = ocr_info.get("qr_code_raw")
            if qr_raw:
                qr_parsed = parse_qr_content(qr_raw)
                self.csv.update_from_qr(row, qr_parsed)

            # Look up product name from barcode database
            barcode = row.get("barcode") or ocr_info.get("barcode")
            if barcode and not row.get("product_name"):
                product_name = lookup_product(str(barcode))
                if product_name:
                    row["product_name"] = product_name

            if nt["tag_type"] == "promo":
                existing = row.get("special_symbols", "")
                if "red" not in existing:
                    row["special_symbols"] = (existing + ",red").strip(",")

            self._all_rows.append(row)
            results.append({
                "bbox": tag.bbox,
                "confidence": tag.confidence,
                "ocr": ocr_info,
                "type": nt["tag_type"],
                "tracked": False,
                "tracker_id": tag.tag_id,
            })
        return results

    def finalize_csv(self) -> None:
        """Deduplicate and write all buffered rows to CSV."""
        if not self._all_rows:
            self.csv.close()
            return

        by_tracker: dict[int, list[dict[str, str]]] = {}
        no_tracker: list[dict[str, str]] = []
        for row in self._all_rows:
            tid = row.get("tracker_id")
            if tid is not None:
                tid_int = int(tid)
                if tid_int not in by_tracker:
                    by_tracker[tid_int] = []
                by_tracker[tid_int].append(row)
            else:
                no_tracker.append(row)

        for tid, rows in by_tracker.items():
            best = max(rows, key=lambda r: sum(1 for v in r.values() if v and v != ""))
            self.csv.write_row(best)

        for row in no_tracker:
            self.csv.write_row(row)

        unique_count = len(by_tracker) + len(no_tracker)
        logger.info("CSV: %d raw -> %d unique (by tracker_id)", len(self._all_rows), unique_count)
        self.csv.close()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "ocr_calls": self._ocr_count,
            "tracked_hits": self._tracked_hits,
            "active_tags": len(self.tracker.active_tags),
        }

    def release(self) -> None:
        self.finalize_csv()

    def process_video_frame(
        self,
        frame: np.ndarray,
        filename: str,
        frame_timestamp: int,
    ) -> list[dict]:
        return self.process_frame(frame, filename, frame_timestamp)
