"""Parse pixelrag.yaml with parameter forwarding."""

import os
from pathlib import Path

import yaml

from .sources import SOURCES

DEFAULT_CONFIG = {
    "ingest": {"backend": "cdp", "quality": 85, "tile_height": 8192},
    "embed": {"model": "Qwen/Qwen3-VL-Embedding-2B", "device": "cuda"},
    "output": "./index",
}


def load_config(path=None):
    if path is None:
        for c in [Path("pixelrag.yaml"), Path("pixelrag.yml")]:
            if c.exists():
                path = str(c)
                break
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    merged = {**DEFAULT_CONFIG, **config}
    for section in ("ingest", "embed"):
        if section in DEFAULT_CONFIG and section in config and isinstance(config[section], dict):
            merged[section] = {**DEFAULT_CONFIG[section], **config[section]}
    return merged


def make_source(config):
    source_config = dict(config.get("source", {}))
    source_type = source_config.pop("type", "local")
    # Expand ~ in any string values that look like paths
    for k, v in source_config.items():
        if isinstance(v, str) and ("/" in v or "\\" in v or "~" in v):
            source_config[k] = str(Path(v).expanduser())
    return SOURCES[source_type](**source_config)
