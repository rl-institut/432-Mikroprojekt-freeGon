# ── cfg.py ──────────────────────────────────────────────────────────────
"""
Global, read-only access to the YAML configuration file.

Usage
-----
    import cfg

    cfg.PATHS.input.dlr_file
    cfg.MATCHING.dlr.buffer_distance
    cfg.VIS.map.tiles
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import yaml


# ----------------------------------------------------------------------
# 1.  locate the YAML file
# ----------------------------------------------------------------------
# project root = directory that *contains* cfg.py
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_YAML = ROOT_DIR / "default_config.yaml"

CONFIG_FILE = Path(
    os.environ.get("GRID_MATCH_CFG", DEFAULT_YAML)  # env-var override
).expanduser().resolve()

if not CONFIG_FILE.is_file():
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")


# ----------------------------------------------------------------------
# 2.  load & convert yaml → nested SimpleNamespaces  (dot-access)
# ----------------------------------------------------------------------
def _to_namespace(obj):
    """Recursively convert dicts to SimpleNamespace so we can do cfg.foo.bar."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


with CONFIG_FILE.open("r", encoding="utf-8") as f:
    _raw_cfg = yaml.safe_load(f)

# top-level names
PATHS       = _to_namespace(_raw_cfg.get("paths", {}))
MATCHING    = _to_namespace(_raw_cfg.get("matching", {}))
VIS         = _to_namespace(_raw_cfg.get("visualization", {}))
LOGGING     = _to_namespace(_raw_cfg.get("logging", {}))

# keep a dict copy around in case you ever need to iterate
AS_DICT = _raw_cfg.copy()

# nice shortcut: expose root dir to the rest of the program
ROOT = ROOT_DIR
