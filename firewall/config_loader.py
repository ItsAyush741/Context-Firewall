"""
Context Rot Firewall — Central Configuration Loader
Reads config.yaml and exposes a singleton `cfg` dict.
"""
import os
import yaml
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Singleton — import this anywhere
cfg = load_config()
