import cv2
import numpy as np
import os
import contextlib
from utils.config import get


def validate_ean13(code):
    digits = [int(d) for d in code if d.isdigit()]
    if len(digits) not in (12, 13):
        return False
    checksum = sum(digits[i] * (3 if i % 2 else 1) for i in range(len(digits) - 1)) % 10
    checksum = (10 - checksum) % 10
    return digits[-1] == checksum

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


MIN_OCR_WIDTH = get("cascade.min_ocr_width", 200)


def enhance_crop(crop):
    """Apply CLAHE + sharpen + adaptive upscale."""
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]

    # Upscale if too small
    scale = 1.0
    if w < MIN_OCR_WIDTH:
        scale = max(2.0, min(4.0, MIN_OCR_WIDTH / w))
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
        # Suppress zbar stderr spam
        with open(os.devnull, 'w') as devnull:
            old_stderr = os.dup(2)
            os.dup2(devnull.fileno(), 2)
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
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
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


def is_ocr_reliable(crop, min_width=None, min_height=None, min_blur=80.0):
    """Check if crop is large enough and sharp enough for meaningful OCR."""
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    min_w = min_width or get("cascade.min_ocr_width", 200)
    min_h = min_height or int(min_w * 0.5)
    if w < min_w or h < min_h:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur >= min_blur


def preprocess_for_ocr(crop, strategy="default"):
    """Apply different preprocessing strategies for OCR multi-pass."""
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    if strategy == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    elif strategy == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    elif strategy == "denoise":
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    elif strategy == "deskew":
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 100:
            return crop
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # default: CLAHE + sharpen
    enhanced, _ = enhance_crop(crop)
    return enhanced
