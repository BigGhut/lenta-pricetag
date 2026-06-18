"""Tiled YOLO inference with parallel tile processing."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils.wbf import weighted_boxes_fusion

logger = logging.getLogger("pricetag.tiled_inference")


class TiledYOLO:
    """YOLO inference with tiling and optional multi-scale + parallel execution."""

    def __init__(
        self,
        model_path: str,
        tile_size: int = 640,
        stride: int = 320,
        conf: float = 0.15,
        iou: float = 0.45,
        scales: Optional[list[float]] = None,
        max_workers: int = 4,
    ) -> None:
        self.model = YOLO(model_path)
        # Ensure model runs on GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda:0")
                logger.info("YOLO model moved to CUDA")
        except Exception:
            pass
        self.tile_size = tile_size
        self.stride = stride
        self.conf = conf
        self.iou = iou
        self.scales = scales or [1.0]
        self.max_workers = max_workers
        logger.info(
            "TiledYOLO: tile=%d, stride=%d, scales=%s, workers=%d",
            tile_size, stride, self.scales, max_workers,
        )

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Run detection on frame. On GPU: single pass (no tiling). On CPU/NPU: tiled."""
        H, W = frame.shape[:2]

        # Fast path: if model is on GPU, run single-pass inference (much faster)
        if next(self.model.model.parameters()).device.type == "cuda":
            return self._detect_single(frame)

        # Tiled path for CPU/NPU
        return self._detect_tiled(frame)

    def _detect_single(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Single-pass inference on full frame (GPU fast path)."""
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        output: list[tuple[int, int, int, int, float]] = []
        for b in boxes:
            tx1, ty1, tx2, ty2 = b.xyxy[0].tolist()
            conf_val = b.conf[0].item()
            output.append((int(tx1), int(ty1), int(tx2), int(ty2), float(conf_val)))
        return output

    def _detect_tiled(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Tiled inference for CPU/NPU (slow but memory-efficient)."""
        H, W = frame.shape[:2]
        all_boxes: list[tuple[float, ...]] = []

        for scale in self.scales:
            if scale != 1.0:
                scaled_w = int(W * scale)
                scaled_h = int(H * scale)
                scaled = cv2.resize(frame, (scaled_w, scaled_h))
            else:
                scaled = frame
                scaled_w, scaled_h = W, H

            sH, sW = scaled.shape[:2]

            # Collect tile positions
            tiles: list[tuple[int, int, np.ndarray]] = []
            for y in range(0, sH - self.tile_size + 1, self.stride):
                for x in range(0, sW - self.tile_size + 1, self.stride):
                    tiles.append((x, y, scaled[y:y + self.tile_size, x:x + self.tile_size]))

            # Edge tiles: right column and bottom row (avoid duplicates at corner)
            right_x = max(0, sW - self.tile_size)
            bottom_y = max(0, sH - self.tile_size)
            if sW % self.tile_size != 0:
                for y in range(0, sH - self.tile_size + 1, self.stride):
                    tiles.append((right_x, y, scaled[y:y + self.tile_size, right_x:right_x + self.tile_size]))
            if sH % self.tile_size != 0:
                for x in range(0, sW - self.tile_size + 1, self.stride):
                    tiles.append((x, bottom_y, scaled[bottom_y:bottom_y + self.tile_size, x:x + self.tile_size]))
            # Bottom-right corner tile (if both dimensions need it and not already added)
            if sW % self.tile_size != 0 and sH % self.tile_size != 0:
                tiles.append((right_x, bottom_y,
                              scaled[bottom_y:bottom_y + self.tile_size, right_x:right_x + self.tile_size]))

            # Process tiles in parallel
            scale_boxes = self._process_tiles_parallel(tiles, scale)
            all_boxes.extend(scale_boxes)

        return self._merge(all_boxes)

    def _process_tiles_parallel(
        self,
        tiles: list[tuple[int, int, np.ndarray]],
        scale: float,
    ) -> list[tuple[float, ...]]:
        """Process tiles sequentially (thread-unsafe for YOLO model)."""
        results: list[tuple[float, ...]] = []
        for x, y, tile in tiles:
            try:
                results.extend(self._infer_tile(x, y, tile, scale))
            except Exception as e:
                logger.warning("Tile (%d, %d) failed: %s", x, y, e)
        return results

    def _infer_tile(
        self,
        x: int,
        y: int,
        tile: np.ndarray,
        scale: float,
    ) -> list[tuple[float, ...]]:
        """Run YOLO on a single tile and return boxes in original coordinates."""
        results = self.model(tile, conf=self.conf, iou=self.iou, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        output: list[tuple[float, ...]] = []
        for b in boxes:
            tx1, ty1, tx2, ty2 = b.xyxy[0].tolist()
            conf_val = b.conf[0].item()
            if scale != 1.0:
                output.append((
                    (x + tx1) / scale, (y + ty1) / scale,
                    (x + tx2) / scale, (y + ty2) / scale,
                    conf_val,
                ))
            else:
                output.append((x + tx1, y + ty1, x + tx2, y + ty2, conf_val))
        return output

    def _merge(
        self,
        boxes: list[tuple[float, ...]],
    ) -> list[tuple[int, int, int, int, float]]:
        """Merge overlapping boxes using WBF."""
        if len(boxes) < 2:
            if boxes:
                b = boxes[0]
                return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4]))]
            return []

        boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in boxes])
        scores = np.array([b[4] for b in boxes])
        fused = weighted_boxes_fusion(boxes_arr, scores, iou_thr=self.iou, skip_box_thr=self.conf)
        if fused is None or len(fused) == 0:
            return []
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])) for b in fused]
