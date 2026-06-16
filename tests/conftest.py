"""Shared fixtures for all tests.

Mocks config.yaml so tests never depend on a real config file on disk.
"""

import sys
import os
import types

import pytest


# Ensure project root is on sys.path so `from utils.xxx import ...` works
# whether running `pytest` from the repo root or from tests/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Minimal config matching config.yaml structure (enough for all modules).
DEFAULT_CONFIG = {
    "paths": {
        "data": "data/yolov12n",
        "models": "models/yolov12n",
        "ocr_models": "models/ocr/easyocr_model",
        "results": "results",
        "export": "export",
    },
    "model": {
        "pretrained": "yolo12n.pt",
        "trained": "models/yolov12n/yolov12n.pt",
        "trained_onnx": "models/yolov12n/yolov12n.onnx",
        "export_onnx": "export/yolov12n.onnx",
        "rknn": "models/yolov12n/yolov12n.rknn",
    },
    "data": {"yaml": "data/yolov12n/data_2class.yaml", "imgsz": 640, "nc": 2},
    "training": {"epochs": 300, "batch": 16, "device": 0, "workers": 4,
                "patience": 10, "delta": 0.001, "monitor": "val/box_loss"},
    "export": {"simplify": True, "dynamic": False},
    "rknn": {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]],
             "target_platform": "rk3588", "optimization_level": 3},
    "cascade": {"conf_threshold": 0.5, "cascade_conf": 0.15,
                "iou_threshold": 0.45, "min_ocr_width": 200,
                "crop_expand_factor": 5},
    "candidate_detector": {
        "min_size": 40,
        "white": {"lower": [0, 0, 150], "upper": [180, 40, 255]},
        "red": {
            "lower1": [0, 50, 50], "upper1": [10, 255, 255],
            "lower2": [170, 50, 50], "upper2": [180, 255, 255],
        },
        "yellow": {"lower": [20, 80, 180], "upper": [35, 255, 255]},
        "morph_kernel": 15,
    },
    "tracker": {"center_dist_thresh": 2.0, "size_ratio_thresh": 0.25,
                "max_frames_missed": 5},
    "ocr": {"engine": "paddleocr", "languages": ["ru", "en"], "gpu": True},
    "video": {"skip_frames": 3},
    "cluster": {"dy_threshold": 80, "dx_threshold": 600},
}


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    """Inject DEFAULT_CONFIG into utils.config so get() works in tests."""
    # We import lazily to avoid triggering module-level side-effects.
    import utils.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_CONFIG", DEFAULT_CONFIG)
