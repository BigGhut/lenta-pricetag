"""
PaddleOCR inference with automatic fallback to EasyOCR when models unavailable.
Priority: PaddleOCR > EasyOCR (fallback)
"""

import re
import cv2
from utils.ocr_enhancer import enhance_crop, decode_barcodes_multi, is_ocr_reliable

_reader = None
_ENGINE = None


def get_reader():
    global _reader, _ENGINE
    if _reader is not None:
        return _reader

    # Try PaddleOCR first
    try:
        from paddleocr import PaddleOCR
        _reader = PaddleOCR(lang="ru", use_angle_cls=False, show_log=False)
        _ENGINE = "paddleocr"
        print("OCR engine: PaddleOCR")
        return _reader
    except Exception as e:
        print(f"PaddleOCR unavailable: {e}")

    # Fallback to EasyOCR
    try:
        from infer.infer_easyocr import get_reader as easy_reader
        _reader = easy_reader()
        _ENGINE = "easyocr"
        print("OCR engine: EasyOCR (fallback)")
        return _reader
    except ImportError:
        raise RuntimeError("No OCR engine available (tried PaddleOCR, EasyOCR)")


def _ocr_crop(crop, paragraph=False):
    if crop.size == 0:
        return []

    if _ENGINE == "paddleocr":
        r = get_reader()
        results = r.ocr(crop, cls=False)
        if not results or not results[0]:
            enhanced, _ = enhance_crop(crop)
            results = r.ocr(enhanced, cls=False)
        if results and results[0]:
            return [line[1][0] for line in results[0]]
        return []

    # EasyOCR path
    enhanced, _ = enhance_crop(crop)
    r = get_reader()
    results = r.readtext(enhanced, detail=0, paragraph=paragraph)
    if not results:
        results = r.readtext(crop, detail=0, paragraph=paragraph)
    return results


def extract_price(texts):
    pattern = re.compile(r"(\d{1,5})[.,](\d{1,2})|(\d{1,5})")
    candidates = []
    for text in texts:
        if not isinstance(text, str):
            continue
        cleaned = re.sub(r"[^\d.,]", "", text)
        match = pattern.search(cleaned)
        if match:
            value = float(f"{match.group(1)}.{match.group(2)}") if match.group(1) else float(match.group(3))
            if 10 <= value <= 1000000:
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
        if len(digits) >= 12:
            return digits
    results = _ocr_crop(crop)
    for t in results:
        m = re.search(r"\b(\d{12,13})\b", str(t))
        if m:
            return m.group(1)
    return None


def recognize_qr_code(crop):
    try:
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(crop)
        if data:
            return data
        big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(big)
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

    if not is_ocr_reliable(crop):
        info = {}
        for b in decode_barcodes_multi(crop):
            digits = re.sub(r"\D", "", b["data"])
            if len(digits) >= 12:
                info["barcode"] = digits
        return info

    results = _ocr_crop(crop, paragraph=True)
    if not results:
        for b in decode_barcodes_multi(crop):
            return {"barcode": re.sub(r"\D", "", b["data"])}
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
    if m:
        info["barcode"] = m.group(1)
    else:
        for b in decode_barcodes_multi(crop):
            digits = re.sub(r"\D", "", b["data"])
            if len(digits) >= 12:
                info["barcode"] = digits
                break

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
