import csv
import re
import os
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


CSV_COLUMNS = [
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


def classify_color(mean_bgr):
    b, g, r = mean_bgr
    if r > 150 and g < 100 and b < 100:
        return "red"
    if r > 200 and g > 150 and b < 80:
        return "yellow"
    if r > 150 and g > 100 and b < 50:
        return "orange"
    return "white"


def _str(t):
    return str(t) if not isinstance(t, str) else t


def extract_discount(texts):
    for t in texts:
        m = re.search(r"(-?\d+)\s*%", _str(t))
        if m:
            return m.group(0)
    return None


def extract_date(texts):
    for t in texts:
        m = re.search(r"(\d{2}[./]\d{2}[./]\d{2,4})", _str(t))
        if m:
            return m.group(1)
    return None


def is_barcode_like(text):
    if not isinstance(text, str):
        return False
    digits = re.sub(r"\D", "", text)
    return len(digits) >= 12


def extract_price_value(texts):
    prices = []
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


def extract_barcode(texts):
    for t in texts:
        m = re.search(r"\b(\d{12,13})\b", _str(t))
        if m:
            return m.group(1)
    return None


def extract_special_symbols(texts, mean_color):
    symbols = []
    color = classify_color(mean_color)
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


def make_row(**kwargs):
    row = {col: "" for col in CSV_COLUMNS}
    row.update(kwargs)
    return row


class PricetagCSV:
    def __init__(self, filename=None):
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = RESULTS_DIR / f"pricetags_{ts}.csv"
        self.filename = Path(filename)
        self.file = open(self.filename, "w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        self.writer.writeheader()
        print(f"CSV: {self.filename}")

    def make_row(self, **kwargs):
        return make_row(**kwargs)

    def update_from_ocr(self, row, ocr_texts, mean_color=None):
        if not ocr_texts:
            return row

        prices = extract_price_value(ocr_texts)
        if prices:
            row["price_default"] = prices[0]
            if len(prices) > 1:
                row["price_card"] = prices[1]
            if len(prices) > 2:
                row["price_discount"] = prices[2]

        discount = extract_discount(ocr_texts)
        if discount:
            row["discount_amount"] = discount

        barcode = extract_barcode(ocr_texts)
        if barcode:
            row["barcode"] = barcode

        date_val = extract_date(ocr_texts)
        if date_val:
            row["print_datetime"] = date_val

        long_texts = [t for t in ocr_texts
                      if isinstance(t, str) and len(t) > 15
                      and not re.search(r"\d{6,}", t)
                      and not re.search(r"^\d+[.,]?\d*$", t)]
        if long_texts and not row["product_name"]:
            row["product_name"] = long_texts[0]

        special = extract_special_symbols(ocr_texts, mean_color)
        if special:
            row["special_symbols"] = special

        if mean_color is not None:
            row["color"] = classify_color(mean_color)

        return row

    def update_from_qr(self, row, qr_raw):
        if not qr_raw:
            return row

        row["raw_qr_data"] = qr_raw

        row["qr_code_barcode"] = qr_raw

        prices = re.findall(r"(\d+[.,]\d+)", qr_raw)
        qr_price_fields = ["price1_qr", "price2_qr", "price3_qr", "price4_qr"]
        for i, price in enumerate(prices[:4]):
            row[qr_price_fields[i]] = price.replace(",", ".")

        sku = re.search(r"(\d{9,15})", qr_raw.replace(".", ""))
        if sku:
            row["id_sku"] = sku.group(1)

        return row

    def write_row(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
        print(f"CSV saved: {self.filename} ({os.path.getsize(self.filename)} bytes)")


def update_from_yolo_detection(row, bbox, filename, frame_timestamp):
    x1, y1, x2, y2 = map(int, bbox[:4])
    row["filename"] = filename
    row["frame_timestamp"] = str(frame_timestamp)
    row["x_min"] = x1
    row["y_min"] = y1
    row["x_max"] = x2
    row["y_max"] = y2
    return row
