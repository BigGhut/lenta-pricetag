"""Configuration loader with dot-notation access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("pricetag.config")

_CONFIG: dict[str, Any] | None = None


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    global _CONFIG
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file not found: %s, using empty config", path)
        _CONFIG = {}
        return _CONFIG
    with open(config_path) as f:
        data = yaml.safe_load(f)
        _CONFIG = data if isinstance(data, dict) else {}
    logger.info("Config loaded from %s (%d top-level keys)", path, len(_CONFIG))
    return _CONFIG


def get_config() -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    assert _CONFIG is not None
    return _CONFIG


def get(key: str, default: Any = None) -> Any:
    """Get config value by dot-notation key (e.g. 'tracker.center_dist_thresh')."""
    cfg = get_config()
    parts = key.split(".")
    val: Any = cfg
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return default
    return val if val is not None else default
