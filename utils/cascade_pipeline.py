import cv2
import numpy as np
from utils.candidate_detector import detect_pricetag_candidates, expand_crop
from utils.pricetag_tracker import PricetagTracker
from utils.csv_writer import PricetagCSV, make_row, update_from_yolo_detection, classify_color
from utils.config import get

try:
    from infer.infer_yolo_rknn import YOLOv12nRKNN
    HAS_RKNN = True
except ImportError:
    print("RKNN not available — YOLO refine disabled")
    HAS_RKNN = False
    YOLOv12nRKNN = None

try:
    from infer.infer_paddleocr import recognize_full_pricetag, recognize_qr_code
    HAS_OCR = True
except ImportError:
    try:
        from infer.infer_easyocr import recognize_full_pricetag, recognize_qr_code
        HAS_OCR = True
    except ImportError:
        print("OCR not available — text recognition disabled")
        HAS_OCR = False


class CascadePipeline:
    def __init__(self, csv_path=None):
        self.yolo = YOLOv12nRKNN() if HAS_RKNN else None
        self.csv = PricetagCSV(csv_path)
        tcfg = get("tracker")
        self.tracker = PricetagTracker(
            center_dist_thresh=tcfg["center_dist_thresh"],
            size_ratio_thresh=tcfg["size_ratio_thresh"],
            max_frames_missed=tcfg["max_frames_missed"],
        )
        self._ocr_count = 0
        self._tracked_hits = 0
        self._all_rows = []

    def process_frame(self, frame, filename="camera", frame_timestamp=0):
        h, w = frame.shape[:2]
        candidates = detect_pricetag_candidates(frame) or []

        new_tags, tracked = self.tracker.update(candidates, frame_timestamp)
        results = self._process_tracked(tracked)
        results += self._process_new(new_tags, frame, filename, frame_timestamp, h, w)
        return results

    def _process_tracked(self, tags):
        for t in tags:
            self._tracked_hits += 1
        return [{"bbox": t.bbox, "confidence": t.confidence, "ocr": t.ocr_data,
                 "type": t.tag_type, "tracked": True} for t in tags]

    def _process_new(self, new_tags, frame, filename, frame_timestamp, h, w):
        results = []
        for nt in new_tags:
            tag = nt["tag"]
            cand = nt["bbox"][:4]
            expand_factor = get("cascade.crop_expand_factor", 5)
            cx1, cy1, cx2, cy2 = expand_crop(cand, w, h, expand_factor=expand_factor)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            refined = self.yolo.refine_pricetag(crop) if self.yolo else None
            if refined is not None:
                r_h, r_w = crop.shape[:2]
                rx1, ry1, rx2, ry2, r_conf, _ = refined
                ax1 = cx1 + int(rx1 * (cx2 - cx1) / r_w)
                ay1 = cy1 + int(ry1 * (cy2 - cy1) / r_h)
                ax2 = cx1 + int(rx2 * (cx2 - cx1) / r_w)
                ay2 = cy1 + int(ry2 * (cy2 - cy1) / r_h)
                tag.bbox = (ax1, ay1, ax2, ay2)
                tag.confidence = max(tag.confidence, r_conf)
                pt_crop = frame[ay1:ay2, ax1:ax2]
            else:
                ax1, ay1, ax2, ay2 = map(int, cand)
                pt_crop = frame[ay1:ay2, ax1:ax2]

            if pt_crop.size == 0:
                continue

            tag.color = np.mean(pt_crop, axis=(0, 1))
            ocr_info = {}
            if HAS_OCR:
                ocr_info = recognize_full_pricetag(pt_crop)
                qr = recognize_qr_code(pt_crop)
                if qr:
                    ocr_info["qr_code_raw"] = qr

            tag.ocr_data = ocr_info
            self._ocr_count += 1

            row = make_row()
            update_from_yolo_detection(row, [*tag.bbox, tag.confidence, 0], filename, frame_timestamp)
            self.csv.update_from_ocr(row, list(ocr_info.values()), tag.color)
            for k, v in ocr_info.items():
                if k in row and v and not row.get(k):
                    row[k] = v
            if nt["tag_type"] == "red":
                existing = row.get("special_symbols", "")
                if "red" not in existing:
                    row["special_symbols"] = (existing + ",red").strip(",")

            self._all_rows.append(row)
            results.append({"bbox": tag.bbox, "confidence": tag.confidence,
                            "ocr": ocr_info, "type": nt["tag_type"], "tracked": False})
        return results

    def finalize_csv(self):
        if not self._all_rows:
            self.csv.close()
            return

        points = []
        valid = []
        for row in self._all_rows:
            try:
                x1, y1, x2, y2 = map(int, (row["x_min"], row["y_min"], row["x_max"], row["y_max"]))
                valid.append(row)
                points.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            except (ValueError, TypeError):
                valid.append(row)

        if len(points) < 2:
            for r in self._all_rows:
                self.csv.write_row(r)
            self.csv.close()
            return

        points = np.array(points)
        used = set()
        clusters = []
        dcfg = get("cluster")

        for i in range(len(points)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(points)):
                if j in used:
                    continue
                dy = abs(points[j, 1] - points[i, 1])
                dx = abs(points[j, 0] - points[i, 0])
                if dy < dcfg["dy_threshold"] and dx < dcfg["dx_threshold"]:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)

        for cluster in clusters:
            best = max(cluster, key=lambda idx: sum(1 for v in valid[idx].values() if v and v != ""))
            self.csv.write_row(valid[best])

        print(f"CSV: {len(self._all_rows)} raw → {len(clusters)} unique")
        self.csv.close()

    @property
    def stats(self):
        return {"ocr_calls": self._ocr_count, "tracked_hits": self._tracked_hits,
                "active_tags": len(self.tracker.active_tags)}

    def release(self):
        self.finalize_csv()
        if self.yolo:
            self.yolo.release()

    def process_video_frame(self, frame, filename, frame_timestamp):
        return self.process_frame(frame, filename, frame_timestamp)
