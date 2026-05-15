import re
import json


def decode_qr(crop):
    """Decode QR code from image crop using OpenCV."""
    import cv2
    detector = cv2.QRCodeDetector()
    data, pts, _ = detector.detectAndDecode(crop)
    return data if data else None


def parse_qr_content(raw):
    """
    Parse QR code content. Format unknown for ЛЕНТА.
    Tries JSON first, then key=value, then raw string.
    """
    if not raw:
        return {}

    result = {}

    # Try JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try key=value pairs (semicolon or newline separated)
    pairs = re.findall(r'(\w+)\s*[=:]\s*([^;\n]+)', raw)
    if pairs:
        for k, v in pairs:
            result[k.strip()] = v.strip()
        return result

    # Try pipe-separated
    fields = raw.split("|")
    if len(fields) > 1:
        for i, f in enumerate(fields):
            result[f"field_{i}"] = f.strip()
        return result

    # Return as-is
    return {"raw": raw}
