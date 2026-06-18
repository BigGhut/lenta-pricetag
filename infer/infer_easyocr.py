import re
import cv2
import numpy as np
from utils.config import get
from utils.ocr_enhancer import enhance_crop, decode_barcodes_multi, is_ocr_reliable, validate_ean13, preprocess_for_ocr

_reader = None
_OCR_CONF_THRESHOLD = 0.5
_QR_DETECTOR = None


def _get_qr_detector():
    global _QR_DETECTOR
    if _QR_DETECTOR is None:
        _QR_DETECTOR = cv2.QRCodeDetector()
    return _QR_DETECTOR


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        import os
        os.environ["PYTHONIOENCODING"] = "utf-8"
        _reader = easyocr.Reader(
            lang_list=get("ocr.languages"),
            gpu=get("ocr.gpu"),
            model_storage_directory=get("paths.ocr_models"),
        )
    return _reader


def _ocr_crop_with_conf(crop, paragraph=True):
    if crop.size == 0:
        return []
    r = get_reader()

    h, w = crop.shape[:2]

    # Scale UP small crops — EasyOCR CRAFT needs minimum size to detect text
    min_side = 400
    if max(h, w) < min_side:
        scale = min_side / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # Try original (resized) crop first, then OTSU thresholding
    for processed in [crop] + _preprocess_variants(crop):
        results = r.readtext(processed, detail=1, paragraph=False)
        if results:
            filtered = [(text, float(conf)) for bbox, text, conf in results if conf >= _OCR_CONF_THRESHOLD]
            if filtered:
                return filtered
    return []


def _preprocess_variants(crop):
    """Generate a small number of preprocessing variants for OCR retry."""
    variants = []
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        # OTSU thresholding only — helps with low-contrast pricetags
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
    except Exception:
        pass
    return variants


def _ocr_crop(crop, paragraph=True):
    results = _ocr_crop_with_conf(crop, paragraph)
    return [text for text, conf in results]


def _is_date_like(text):
    return bool(re.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", text.strip()))


def _is_time_like(text):
    return bool(re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", text.strip()))


def _is_sku_like(text):
    digits = re.sub(r"\D", "", text)
    return len(digits) >= 6 and not re.search(r"[.,]", text)


def extract_price(texts):
    pattern = re.compile(r"(\d{1,5})[.,](\d{1,2})|(\d{1,5})")
    candidates = []
    for text in texts:
        if not isinstance(text, str):
            continue
        cleaned = text.strip()
        if _is_date_like(cleaned) or _is_time_like(cleaned) or _is_sku_like(cleaned):
            continue
        match = pattern.search(cleaned)
        if match:
            value = float(f"{match.group(1)}.{match.group(2)}") if match.group(1) else float(match.group(3))
            if 10 <= value <= 999999:
                candidates.append(value)
    return sorted(set(candidates), reverse=True) if candidates else None


def recognize_price(crop):
    prices = extract_price(_ocr_crop(crop, paragraph=False))
    return prices[0] if prices else None


def recognize_product_name(crop):
    results = _ocr_crop(crop, paragraph=False)
    if not results:
        return None
    exclude = re.compile(r"(\d{1,5})[.,](\d{1,2})|(\d{1,5})|(\d{12,13})")
    candidates = [t for t in results if isinstance(t, str) and not exclude.search(t)]
    return " ".join(candidates) if candidates else None


def recognize_barcode_ean(crop):
    barcodes = decode_barcodes_multi(crop)
    for b in barcodes:
        digits = re.sub(r"\D", "", b["data"])
        if len(digits) >= 12 and validate_ean13(digits):
            return digits
    results = _ocr_crop(crop)
    for t in results:
        m = re.search(r"\b(\d{12,13})\b", str(t))
        if m and validate_ean13(m.group(1)):
            return m.group(1)
    return None


def recognize_qr_code(crop):
    try:
        detector = _get_qr_detector()
        data, _, _ = detector.detectAndDecode(crop)
        if data:
            return data
        big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        data, _, _ = detector.detectAndDecode(big)
        return data if data else None
    except Exception:
        return None


def recognize_all(crop):
    return {
        "price": recognize_price(crop),
        "product_name": recognize_product_name(crop),
        "barcode": recognize_barcode_ean(crop),
        "qr_code": recognize_qr_code(crop),
    }


def recognize_full_pricetag(crop):
    if crop.size == 0:
        return {}

    results = _ocr_crop(crop, paragraph=True)
    if not results:
        return {}

    info = {}
    all_text = " ".join(str(t) for t in results)

    prices = extract_price(results)
    if prices:
        info["price_default"] = prices[0]
        if len(prices) > 1:
            info["price_card"] = prices[1]
        if len(prices) > 2:
            info["price_discount"] = prices[2]

    m = re.search(r"(-?\d+)\s*%", all_text)
    if m:
        info["discount_amount"] = m.group(0)

    m = re.search(r"\b(\d{12,13})\b", all_text)
    if m and validate_ean13(m.group(1)):
        info["barcode"] = m.group(1)

    m = re.search(r"(\d{2}[./]\d{2}[./]\d{2,4})", all_text)
    if m:
        info["print_datetime"] = m.group(1)

    long_texts = [t for t in results if isinstance(t, str) and len(t) > 10
                  and not re.search(r"\d{6,}", t)
                  and not re.fullmatch(r"[\d.,%/\-]+", t)]
    if long_texts:
        info["product_name"] = max(long_texts, key=len)

    m = re.search(r"\b(\d{9,15})\b", all_text)
    if m and "barcode" not in info:
        info["id_sku"] = m.group(1)

    special_chars = []
    for pat, ch in [(r"[%]", "%"), (r"[Шш]", "Ш"), (r"[★☆]", "★"), (r"[№#]", "№")]:
        if re.search(pat, all_text):
            special_chars.append(ch)
    if special_chars:
        info["special_symbols"] = ",".join(special_chars)

    additional = [t for t in results if isinstance(t, str) and 3 < len(t) < 30
                  and t not in (info.get("product_name"), "")
                  and not re.fullmatch(r"[\d.,%/\-]+", t)]
    if additional:
        info["additional_info"] = " | ".join(additional[:3])

    return info
