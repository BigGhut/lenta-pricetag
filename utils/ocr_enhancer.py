import cv2
import numpy as np
from utils.config import get

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


def _min_ocr_width():
    return get("cascade.min_ocr_width", 200)


def enhance_crop(crop):
    """Apply CLAHE + sharpen + adaptive upscale."""
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]

    # Upscale if too small
    scale = 1.0
    min_w = _min_ocr_width()
    if w < min_w:
        scale = max(2.0, min(4.0, min_w / w))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE on LAB luminance
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced, scale


def decode_barcodes(crop):
    """Decode barcodes using pyzbar. Returns list of (data, type, rect)."""
    if not HAS_PYZBAR or crop.size == 0:
        return []

    results = []
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        decoded = pyzbar_decode(gray)
        for d in decoded:
            rect = d.rect
            results.append({
                "data": d.data.decode("utf-8", errors="replace"),
                "type": d.type,
                "rect": (rect.left, rect.top, rect.width, rect.height),
            })
    except Exception:
        pass

    return results


def decode_barcodes_multi(crop):
    """Try multiple preprocessing strategies for barcode decoding."""
    results = decode_barcodes(crop)
    if results:
        return results

    # Try with CLAHE
    enhanced, _ = enhance_crop(crop)
    results = decode_barcodes(enhanced)
    if results:
        return results

    # Try large upscale for really small barcodes
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    results = decode_barcodes(big)
    if results:
        return results

    return []


def is_ocr_reliable(crop):
    """Check if crop is large enough for meaningful OCR."""
    h, w = crop.shape[:2]
    min_w = _min_ocr_width()
    return w >= min_w and h >= min_w * 0.5
