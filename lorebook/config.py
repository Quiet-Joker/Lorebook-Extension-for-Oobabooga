import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def _cached_token_count(text: str, model_name: str) -> int:
    try:
        from modules.text_generation import get_encoded_length
        return get_encoded_length(text)
    except Exception:
        return max(1, len(text) // 4)


def _count_tokens(text: str) -> int:
    try:
        import modules.shared as _shared
        current_model = getattr(_shared, "model_name", "") or ""
    except Exception:
        current_model = ""
    return _cached_token_count(text, current_model)


params = {
    "activate": True,
    "is_tab": True,
    "display_name": "Lorebook",
    "scan_depth": -1,
    "token_budget": 1024,
    "injection_prefix": "\n[World Info]\n",
    "injection_suffix": "\n[/World Info]",
    "_st.current_lorebook": "",
    "constant_entries": True,
    "recursive_scan": True,
    "max_recursion_steps": 3,
    "mid_gen_interrupt": False,
    "max_interrupts": 3,
    "position_override_enabled": False,
    "position_override_value": "after_context",
    "chat_only_scan": False,
}

_MAX_GATHER_DEPTH = 20
EXT_DIR = Path(__file__).parent
LOREBOOKS_DIR  = EXT_DIR / "lorebooks"
TEMPLATES_DIR  = EXT_DIR / "Auto Summary Prompt Templates"
_ACTIVE_STATE_FILE = EXT_DIR / "active_state.json"
_PARAMS_FILE       = EXT_DIR / "params.json"
LOREBOOKS_DIR.mkdir(parents=True, exist_ok=True)

_PARAMS_PERSIST_KEYS = {
    "activate", "scan_depth", "token_budget", "injection_prefix", "injection_suffix",
    "constant_entries", "recursive_scan", "max_recursion_steps", "mid_gen_interrupt",
    "max_interrupts", "position_override_enabled", "position_override_value",
    "chat_only_scan",
}

_ORIG_CTX = "_lb_orig_ctx"
_HIST_MSGS = "_lb_hist_msgs"

_ST_SELECTIVE_INT_TO_STR = {0: "AND ANY", 1: "NOT ALL", 2: "NOT ANY", 3: "AND ALL"}
_ST_SELECTIVE_STR_TO_INT = {v: k for k, v in _ST_SELECTIVE_INT_TO_STR.items()}

_ST_POSITION = {0: "before_context", 1: "after_context", 2: "before_reply"}
_OUR_POSITION = {"before_context": 0, "after_context": 1, "before_reply": 2, "notebook": 2}
