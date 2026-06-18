"""CSV export for pricetag detection results."""

from __future__ import annotations

import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Any, Optional

from utils.ocr_enhancer import validate_ean13

logger = logging.getLogger("pricetag.csv_writer")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


CSV_COLUMNS = [
    "tracker_id",
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
    "raw_qr_data",
]

COLOR_MAP = {
    "red": "красный",
    "yellow": "жёлтый",
    "white": "белый",
    "orange": "оранжевый",
}

_BATCH_FLUSH_SIZE = 50  # flush to disk every N rows


def classify_color(mean_bgr: np.ndarray) -> str:
    """Classify mean BGR color to a named category."""
    import numpy as np
    b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
    if r > 150 and g < 100 and b < 100:
        return "red"
    if r > 200 and g > 150 and b < 80:
        return "yellow"
    if r > 150 and g > 100 and b < 50:
        return "orange"
    return "white"


def _str(t: Any) -> str:
    return str(t) if not isinstance(t, str) else t


def extract_discount(texts: list[Any]) -> Optional[str]:
    for t in texts:
        m = re.search(r"(-?\d+)\s*%", _str(t))
        if m:
            return m.group(0)
    return None


def extract_date(texts: list[Any]) -> Optional[str]:
    for t in texts:
        m = re.search(r"(\d{2}[./]\d{2}[./]\d{2,4})", _str(t))
        if m:
            return m.group(1)
    return None


def is_barcode_like(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    digits = re.sub(r"\D", "", text)
    return len(digits) >= 12


def extract_price_value(texts: list[Any]) -> list[float]:
    prices: list[float] = []
    for t in texts:
        t_str = str(t) if not isinstance(t, str) else t
        if is_barcode_like(t_str):
            continue
        cleaned = re.sub(r"[^\d.,]", "", t_str)
        if len(cleaned) >= 12:
            continue
        m = re.search(r"(\d{1,5})[.,](\d{1,2})|(\d{1,5})", cleaned)
        if m:
            if m.group(1) and m.group(2):
                val = float(f"{m.group(1)}.{m.group(2)}")
            elif m.group(3):
                val = float(m.group(3))
            else:
                continue
            if 10 <= val <= 1000000:
                prices.append(val)
    return sorted(set(prices), reverse=True)


def extract_barcode(texts: list[Any]) -> Optional[str]:
    for t in texts:
        m = re.search(r"\b(\d{12,13})\b", _str(t))
        if m and validate_ean13(m.group(1)):
            return m.group(1)
    return None


def extract_special_symbols(texts: list[Any], mean_color: Any) -> Optional[str]:
    symbols: list[str] = []
    color = classify_color(mean_color) if mean_color is not None else "white"
    if color == "red":
        symbols.append("red")
    if color == "yellow":
        symbols.append("yellow")

    for t in texts:
        s = _str(t)
        if re.search(r"[Шш]", s):
            symbols.append("Ш")
        if re.search(r"[★☆★]", s):
            symbols.append("★")
        if re.search(r"[%]", s):
            symbols.append("%")

    return ",".join(set(symbols)) if symbols else None


def make_row(**kwargs: Any) -> dict[str, str]:
    row: dict[str, str] = {col: "" for col in CSV_COLUMNS}
    row.update(kwargs)
    return row


class PricetagCSV:
    """Buffered CSV writer for pricetag results."""

    def __init__(self, filename: Optional[str | Path] = None) -> None:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = RESULTS_DIR / f"pricetags_{ts}.csv"
        self.filename = Path(filename)
        self._file = open(self.filename, "w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        self._writer.writeheader()
        self._buffer: list[dict[str, str]] = []
        self._written = 0
        logger.info("CSV opened: %s", self.filename)

    def make_row(self, **kwargs: Any) -> dict[str, str]:
        return make_row(**kwargs)

    def update_from_ocr(
        self,
        row: dict[str, str],
        ocr_texts: list[Any],
        mean_color: Any = None,
    ) -> dict[str, str]:
        if not ocr_texts:
            return row

        prices = extract_price_value(ocr_texts)
        if prices:
            row["price_default"] = f"{prices[0]:.2f}"
            if len(prices) > 1:
                row["price_card"] = f"{prices[1]:.2f}"
            if len(prices) > 2:
                row["price_discount"] = f"{prices[2]:.2f}"

        discount = extract_discount(ocr_texts)
        if discount:
            row["discount_amount"] = discount

        barcode = extract_barcode(ocr_texts)
        if barcode:
            row["barcode"] = barcode

        date_val = extract_date(ocr_texts)
        if date_val:
            row["print_datetime"] = date_val

        long_texts = [
            t for t in ocr_texts
            if isinstance(t, str) and len(t) > 15
            and not re.search(r"\d{6,}", t)
            and not re.search(r"^\d+[.,]?\d*$", t)
        ]
        if long_texts and not row["product_name"]:
            row["product_name"] = long_texts[0]

        special = extract_special_symbols(ocr_texts, mean_color)
        if special:
            row["special_symbols"] = special

        if mean_color is not None:
            row["color"] = classify_color(mean_color)

        return row

    def update_from_qr(self, row: dict[str, str], qr_parsed: dict[str, Any]) -> dict[str, str]:
        if not qr_parsed:
            return row

        raw = qr_parsed.get("raw", "")
        if raw:
            row["raw_qr_data"] = raw

        if "barcode" in qr_parsed and not row.get("barcode"):
            row["qr_code_barcode"] = qr_parsed["barcode"]

        prices: list[str] = []
        for key, val in qr_parsed.items():
            if "price" in key.lower() or "cost" in key.lower():
                m = re.search(r"(\d+[.,]\d+)", str(val))
                if m:
                    prices.append(m.group(1).replace(",", "."))

        if not prices:
            prices = re.findall(r"(\d+[.,]\d+)", raw)

        qr_price_fields = ["price1_qr", "price2_qr", "price3_qr", "price4_qr"]
        for i, price in enumerate(prices[:4]):
            row[qr_price_fields[i]] = price

        sku = qr_parsed.get("sku") or qr_parsed.get("id") or qr_parsed.get("code")
        if sku and not row.get("id_sku"):
            digits = re.sub(r"\D", "", str(sku))
            if 9 <= len(digits) <= 15:
                row["id_sku"] = digits
        elif not row.get("id_sku"):
            sku_match = re.search(r"(\d{9,15})", raw.replace(".", ""))
            if sku_match:
                row["id_sku"] = sku_match.group(1)

        for key in [
            "wholesale_level_1_count", "wholesale_level_1_price",
            "wholesale_level_2_count", "wholesale_level_2_price",
            "action_price_qr", "action_code_qr",
        ]:
            if key in qr_parsed and not row.get(key):
                row[key] = str(qr_parsed[key])

        return row

    def write_row(self, row: dict[str, str]) -> None:
        """Buffer a row and flush to disk periodically."""
        # Validate price
        if row.get("price_default"):
            try:
                price = float(row["price_default"])
                if price <= 0 or price > 9999999:
                    row["price_default"] = ""
            except (ValueError, TypeError):
                row["price_default"] = ""

        # Validate barcode
        if row.get("barcode"):
            barcode = re.sub(r"\D", "", str(row["barcode"]))
            if len(barcode) < 8 or len(barcode) > 14:
                row["barcode"] = ""
            else:
                row["barcode"] = barcode

        # Validate bbox
        if row.get("x_min") and row.get("x_max"):
            try:
                x1, x2 = int(float(row["x_min"])), int(float(row["x_max"]))
                y1, y2 = int(float(row["y_min"])), int(float(row["y_max"]))
                if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                    row["x_min"] = row["y_min"] = row["x_max"] = row["y_max"] = ""
            except (ValueError, TypeError):
                row["x_min"] = row["y_min"] = row["x_max"] = row["y_max"] = ""

        self._buffer.append(row)
        if len(self._buffer) >= _BATCH_FLUSH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._file.flush()
        self._written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        self._flush()
        self._file.close()
        size = os.path.getsize(self.filename)
        logger.info("CSV saved: %s (%d rows, %d bytes)", self.filename, self._written, size)


def update_from_yolo_detection(
    row: dict[str, str],
    bbox: list[float],
    filename: str,
    frame_timestamp: int,
) -> dict[str, str]:
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    row["filename"] = filename
    row["frame_timestamp"] = str(frame_timestamp)
    row["x_min"] = str(x1)
    row["y_min"] = str(y1)
    row["x_max"] = str(x2)
    row["y_max"] = str(y2)
    return row
