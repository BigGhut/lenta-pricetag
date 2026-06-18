"""Product database lookup from CSV catalog."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pricetag.product_db")

_DB: dict[str, str] = {}
_LOADED = False


def load_db(path: str | Path = "db_hack.csv", encoding: str = "windows-1251") -> dict[str, str]:
    """Load product CSV into memory as {barcode: fullname} dict."""
    global _DB, _LOADED
    path = Path(path)
    if not path.exists():
        logger.warning("Product DB not found: %s", path)
        return {}

    count = 0
    with open(path, encoding=encoding, errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            fullname = row[0].strip()
            code = row[1].strip()
            if code and fullname:
                _DB[code] = fullname
                count += 1

    _LOADED = True
    logger.info("Product DB loaded: %d entries from %s", count, path)
    return _DB


def lookup(barcode: str) -> Optional[str]:
    """Look up product name by barcode. Auto-loads DB on first call."""
    global _LOADED
    if not _LOADED:
        load_db()
    # Try exact match first
    if barcode in _DB:
        return _DB[barcode]
    # Try without leading zeros
    stripped = barcode.lstrip("0")
    if stripped in _DB:
        return _DB[stripped]
    # Try with leading zeros (13-digit EAN)
    padded = barcode.zfill(13)
    if padded in _DB:
        return _DB[padded]
    return None


def reload(path: str | Path = "db_hack.csv", encoding: str = "windows-1251") -> dict[str, str]:
    """Force reload the database."""
    global _DB, _LOADED
    _DB = {}
    _LOADED = False
    return load_db(path, encoding)
