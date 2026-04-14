import json
import logging
import re

from .config import (
    params,
    LOREBOOKS_DIR,
    _ACTIVE_STATE_FILE,
    _PARAMS_FILE,
    _PARAMS_PERSIST_KEYS,
)
from .state import _st, _bump_active_version

logger = logging.getLogger(__name__)


def _safe_stem(name):
    stem = re.sub(r'[^\w\s\-]', '', name).strip()
    return re.sub(r'\s+', '_', stem) or "lorebook"


def get_lorebook_files():
    return sorted(p.stem for p in LOREBOOKS_DIR.glob("*.json"))


def load_lorebook(stem):
    path = LOREBOOKS_DIR / f"{stem}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load lorebook %s", stem)
        return None


def save_lorebook_file(stem, data):
    try:
        (LOREBOOKS_DIR / f"{stem}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        logger.exception("Failed to save lorebook %s", stem)
        return False


def delete_lorebook_file(stem):
    path = LOREBOOKS_DIR / f"{stem}.json"
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception:
            logger.exception("Failed to delete lorebook %s", stem)
            return False
    return False


def _save_active_state():
    try:
        with _st.state_lock:
            keys = list(_st.active_lorebooks.keys())
        _ACTIVE_STATE_FILE.write_text(
            json.dumps({"active": keys}, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save active lorebook state")


def _load_active_state():
    if not _ACTIVE_STATE_FILE.exists():
        return
    try:
        stems = json.loads(_ACTIVE_STATE_FILE.read_text(encoding="utf-8")).get("active", [])
        with _st.state_lock:
            for stem in stems:
                lb = load_lorebook(stem)
                if lb:
                    _st.active_lorebooks[stem] = lb
            _bump_active_version()
    except Exception:
        logger.exception("Failed to load active lorebook state")


def _save_params():
    try:
        data = {k: v for k, v in params.items() if k in _PARAMS_PERSIST_KEYS}
        _PARAMS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save params")


def _load_params():
    if not _PARAMS_FILE.exists():
        return
    try:
        data = json.loads(_PARAMS_FILE.read_text(encoding="utf-8"))
        for k, v in data.items():
            if k in _PARAMS_PERSIST_KEYS:
                params[k] = v
    except Exception:
        logger.exception("Failed to load params")


_load_active_state()
_load_params()
