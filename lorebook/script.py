import sys as _sys, types as _types
from pathlib import Path as _Path

_PKG  = "extensions.lorebook"
_DIR  = str(_Path(__file__).resolve().parent)

if "extensions" not in _sys.modules:
    _ns = _types.ModuleType("extensions")
    _ns.__path__ = []
    _sys.modules["extensions"] = _ns

if _PKG not in _sys.modules:
    _pkg = _types.ModuleType(_PKG)
    _pkg.__path__    = [_DIR]
    _pkg.__package__ = _PKG
    _sys.modules[_PKG] = _pkg

__package__ = _PKG
del _sys, _types, _Path, _PKG, _DIR

from . import _chat_hook  # noqa: F401
from .config import params  # noqa: F401
from . import summary as _summary_module  # noqa: F401

from .ui import ui, custom_css  # noqa: F401
from .injection import (  # noqa: F401
    state_modifier,
    chat_input_modifier,
    custom_generate_reply,
)
