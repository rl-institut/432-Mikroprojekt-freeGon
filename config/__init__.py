# config/__init__.py
"""
Make   import cfg   work everywhere, even though cfg.py
lives inside the  config/  package.
"""

import sys
from importlib import import_module

_cfg = import_module(".cfg", package=__name__)
sys.modules["cfg"] = _cfg            # register alias

# (optional) re-export attributes so one can also do  import config as cfg
from .cfg import *                   # noqa: F401, F403
