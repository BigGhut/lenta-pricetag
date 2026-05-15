import yaml
from pathlib import Path

_CONFIG = None


def load_config(path="config.yaml"):
    global _CONFIG
    with open(path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def get(key: str, default=None):
    cfg = get_config()
    parts = key.split(".")
    val = cfg
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return default
    return val if val is not None else default
