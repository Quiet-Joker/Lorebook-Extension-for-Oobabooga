import copy
import json
import random
import re
import tempfile
import threading
from pathlib import Path

import gradio as gr


def _count_tokens(text):
    """Count tokens using the loaded model's tokenizer.
    Falls back to a char-based estimate if the model isn't loaded yet."""
    try:
        from modules.text_generation import get_encoded_length
        return get_encoded_length(text)
    except Exception:
        return max(1, len(text) // 4)

params = {
    "activate": True,
    "is_tab": True,
    "display_name": "Lorebook",
    "scan_depth": -1,
    "token_budget": 1024,
    "injection_prefix": "\n[World Info]\n",
    "injection_suffix": "\n[/World Info]",
    "current_lorebook": "",
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
LOREBOOKS_DIR = EXT_DIR / "lorebooks"
_ACTIVE_STATE_FILE = EXT_DIR / "active_state.json"
_PARAMS_FILE       = EXT_DIR / "params.json"
LOREBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Keys that are user-configurable and should survive restarts.
# Internal/session keys (is_tab, display_name, current_lorebook) are excluded.
_PARAMS_PERSIST_KEYS = {
    "activate", "scan_depth", "token_budget", "injection_prefix", "injection_suffix",
    "constant_entries", "recursive_scan", "max_recursion_steps", "mid_gen_interrupt",
    "max_interrupts", "position_override_enabled", "position_override_value",
    "chat_only_scan",
}

current_lorebook: dict | None = None
_next_uid: int = 1
_active_lorebooks: dict = {}
_cur_injected: set = set()

_ORIG_CTX = "_lb_orig_ctx"
_HIST_MSGS = "_lb_hist_msgs"

_last_injection_lock = threading.Lock()
_last_injection_info: dict = {"entries": [], "interrupts": 0, "total_tokens": 0}

_last_notebook_injection_lock = threading.Lock()
_last_notebook_injection_info: dict = {"entries": [], "interrupts": 0, "total_tokens": 0}

# Per-session injection history (newest first, capped at 60 turns each).
# Tracks new / repeat / returned / dropped status across turns so the UI
# can show a full audit trail of what fired and when.
_injection_history_lock = threading.Lock()
_injection_history: list = []           # chat / instruct
_injection_history_notebook: list = []  # notebook
_prev_chat_labels: set = set()          # labels fired last chat turn
_prev_notebook_labels: set = set()
_all_chat_labels: set = set()           # every label ever fired this session (chat)
_all_notebook_labels: set = set()
_chat_turn_counter: int = 0
_notebook_turn_counter: int = 0

# FIX #10: A dedicated lock for all mutations to the global lorebook state
# (current_lorebook, _next_uid, _active_lorebooks). The original only locked
# _last_injection_info, leaving the rest unprotected across concurrent Gradio events.
_state_lock = threading.RLock()  # RLock so _save_active_state can be called from inside locked blocks

_ST_SELECTIVE_INT_TO_STR = {0: "AND ANY", 1: "NOT ALL", 2: "NOT ANY", 3: "AND ALL"}
_ST_SELECTIVE_STR_TO_INT = {v: k for k, v in _ST_SELECTIVE_INT_TO_STR.items()}
# FIX #1: In SillyTavern, position 0 = before_context, 1 = after_context, 2 = AT_DEPTH.
# The original had 0 and 1 swapped. Key 2 (AT_DEPTH) now maps to before_reply — our new
# "Between User and Assistant" mode — since AT_DEPTH=0 in ST is the closest equivalent
# (inject just before the last reply at depth 0).
_ST_POSITION = {0: "before_context", 1: "after_context", 2: "before_reply"}
# FIX 3: Was {"after_context": 0, "before_context": 1} which is the inverse of _ST_POSITION,
# causing all exported entries to have their positions flipped on round-trip.
# before_reply and notebook both export as ST position 2 (AT_DEPTH) — no direct ST
# equivalent exists for either. On re-import they'll come back as before_reply, which
# is the closest semantic match. Users should re-set to notebook manually if needed.
_OUR_POSITION = {"before_context": 0, "after_context": 1, "before_reply": 2, "notebook": 2}


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
        return None

def save_lorebook_file(stem, data):
    try:
        (LOREBOOKS_DIR / f"{stem}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False

def delete_lorebook_file(stem):
    # FIX #12: Wrap unlink in try/except so filesystem errors don't bubble up as unhandled exceptions.
    path = LOREBOOKS_DIR / f"{stem}.json"
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception:
            return False
    return False

def _save_active_state():
    try:
        with _state_lock:
            keys = list(_active_lorebooks.keys())
        _ACTIVE_STATE_FILE.write_text(
            json.dumps({"active": keys}, indent=2), encoding="utf-8")
    except Exception:
        pass

def _load_active_state():
    if not _ACTIVE_STATE_FILE.exists():
        return
    try:
        for stem in json.loads(_ACTIVE_STATE_FILE.read_text(encoding="utf-8")).get("active", []):
            lb = load_lorebook(stem)
            if lb:
                _active_lorebooks[stem] = lb
    except Exception:
        pass

def _save_params():
    try:
        data = {k: v for k, v in params.items() if k in _PARAMS_PERSIST_KEYS}
        _PARAMS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

def _load_params():
    if not _PARAMS_FILE.exists():
        return
    try:
        data = json.loads(_PARAMS_FILE.read_text(encoding="utf-8"))
        for k, v in data.items():
            if k in _PARAMS_PERSIST_KEYS:
                params[k] = v
    except Exception:
        pass

_load_active_state()
_load_params()


def import_from_sillytavern(raw_bytes):
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        return {}, f"Could not parse file: {exc}"
    raw_entries = data.get("entries", {})
    if isinstance(raw_entries, dict):
        raw_entries = list(raw_entries.values())
    our_entries = []
    for e in raw_entries:
        keys = [k for k in (e.get("keys") or e.get("key") or []) if k]
        sec = [k for k in (e.get("secondary_keys") or e.get("keysecondary") or []) if k]
        comment = e.get("name") or e.get("comment") or ""
        enabled = bool(e["enabled"]) if "enabled" in e else not bool(e.get("disable", False))
        position = _ST_POSITION.get(e.get("position", 0), "after_context")
        # FIX #4: Use explicit None checks so 0 is preserved as a valid value.
        # The original used `or` chains which turned priority=0 into 10.
        _prio_raw = next((e.get(k) for k in ("insertion_order", "order", "priority") if e.get(k) is not None), None)
        priority = _prio_raw if _prio_raw is not None else 10
        _depth_raw = next((e.get(k) for k in ("depth", "scan_depth") if e.get(k) is not None), None)
        scan_depth = _depth_raw if _depth_raw is not None else 0
        sel_logic = _ST_SELECTIVE_INT_TO_STR.get(e.get("selectiveLogic", 0), "AND ANY")
        probability = int(e.get("probability", 100))
        inclusion_group = (e.get("group") or "").strip()
        constant = bool(e.get("constant", False))
        if not keys and not constant:
            comment = (comment + " [NO TRIGGER WORDS — add keys to activate]").strip()
        _uid_raw = e.get("uid") if e.get("uid") is not None else e.get("id")
        uid_val  = _uid_raw if _uid_raw is not None else len(our_entries) + 1
        our_entries.append({
            "uid": uid_val,
            "enabled": enabled,
            "comment": comment,
            "keys": keys,
            "secondary_keys": sec,
            "selective_logic": sel_logic,
            "content": e.get("content", ""),
            "case_sensitive": bool(e.get("case_sensitive", False)),
            # FIX 6: Read match_whole_words and use_regex from source data rather than
            # hardcoding True/False. SillyTavern stores these as "extensions.regex" or
            # "use_regex" / "match_whole_words". Fall back to safe defaults if absent.
            "match_whole_words": bool(e.get("match_whole_words", e.get("extensions", {}).get("match_whole_words", True))),
            "use_regex": bool(e.get("use_regex", e.get("extensions", {}).get("regex", False))),
            "priority": int(priority),
            "position": position,
            "scan_depth": int(scan_depth),
            "probability": probability,
            "inclusion_group": inclusion_group,
            "constant": constant,
        })
    lorebook = {
        "name": data.get("name", "Imported Lorebook"),
        "description": data.get("description", ""),
        "entries": our_entries,
        "_source": "sillytavern",
        "_import_stats": {
            "total": len(our_entries),
            "constant": sum(1 for e in our_entries if e.get("constant")),
            "no_keys": sum(1 for e in our_entries if not e["keys"] and not e.get("constant")),
        },
    }
    return lorebook, ""


def export_to_sillytavern(lorebook):
    st_entries = {}
    for i, e in enumerate(lorebook.get("entries", []), 1):
        uid = e.get("uid", i)
        keys = e.get("keys", [])
        priority = e.get("priority", 10)
        # FIX #3: Use explicit None check so scan_depth=0 is preserved on export.
        # The original `or 4` turned a real 0 into 4.
        _raw_depth = e.get("scan_depth")
        depth = _raw_depth if _raw_depth is not None else 4
        enabled = e.get("enabled", True)
        sel_int = _ST_SELECTIVE_STR_TO_INT.get(e.get("selective_logic", "AND ANY"), 0)
        prob = e.get("probability", 100)
        st_entries[str(uid)] = {
            "uid": uid, "id": uid,
            "key": keys, "keys": keys,
            "keysecondary": e.get("secondary_keys", []),
            "secondary_keys": e.get("secondary_keys", []),
            "comment": e.get("comment", ""), "name": e.get("comment", ""),
            "content": e.get("content", ""),
            "constant": e.get("constant", False),
            "selective": bool(e.get("secondary_keys")),
            "selectiveLogic": sel_int,
            "group": e.get("inclusion_group", ""),
            "order": priority, "insertion_order": priority, "priority": priority,
            "position": _OUR_POSITION.get(e.get("position", "after_context"), 0),
            "disable": not enabled, "enabled": enabled,
            "probability": prob, "useProbability": True,
            "addMemo": True, "excludeRecursion": True,
            "displayIndex": i,
            "case_sensitive": e.get("case_sensitive", False),
            # FIX 6: Preserve match_whole_words and use_regex on export so the round-trip
            # is lossless. Previously these were never written, so regex entries became
            # whole-word non-regex entries after an import → export cycle.
            "match_whole_words": e.get("match_whole_words", True),
            "use_regex": e.get("use_regex", False),
            "depth": depth, "characterFilter": None,
            "extensions": {
                "depth": depth, "weight": priority, "addMemo": True,
                "probability": prob, "displayIndex": i, "selectiveLogic": sel_int,
                "useProbability": True, "characterFilter": None, "excludeRecursion": True,
                "match_whole_words": e.get("match_whole_words", True),
                "regex": e.get("use_regex", False),
            },
        }
    return json.dumps({
        "name": lorebook.get("name", "Exported Lorebook"),
        "description": lorebook.get("description", ""),
        "is_creation": False,
        "scan_depth": params.get("scan_depth", 1),
        "token_budget": params.get("token_budget", 1024),
        "recursive_scanning": params.get("recursive_scan", True),
        "extensions": {},
        "entries": st_entries,
    }, indent=2, ensure_ascii=False).encode("utf-8")


def _gather_messages_list(history, orig_ctx=None):
    """Collect user-side messages from history for keyword scanning.

    When chat_only_scan is ON and orig_ctx is provided, any history slot whose
    text IS or CONTAINS the character persona/context is silently dropped.
    In some oobabooga template modes the persona ends up embedded in the very
    first internal history entry's user slot; without this guard, every persona
    keyword would fire lorebook entries on every turn regardless of what the
    user actually typed.
    """
    msgs = []
    ctx_strip = orig_ctx.strip() if orig_ctx else None
    for pair in history.get("internal", [])[-_MAX_GATHER_DEPTH:]:
        user_text = pair[0]
        if not user_text or user_text == "<|BEGIN-VISIBLE-CHAT|>":
            continue
        # Drop any slot that contains the persona text so it can never trigger
        # entries.  Only active when the user has turned on "Only trigger words
        # in chat"; at all other times every slot is included as before.
        if ctx_strip and params.get("chat_only_scan") and ctx_strip in str(user_text):
            continue
        msgs.append(str(user_text))
    return msgs


def _hit_key(raw_key, text, entry):
    k = raw_key.strip()
    if not k:
        return False
    case = entry.get("case_sensitive", False)
    flags = 0 if case else re.IGNORECASE
    if entry.get("use_regex", False):
        try:
            return bool(re.search(k, text, flags))
        except re.error:
            return False
    haystack = text if case else text.lower()
    needle = k if case else k.lower()
    if entry.get("match_whole_words", True):
        # FIX #9: Replace \b with lookarounds. \b fails when the key starts or ends with
        # punctuation because \b requires a word/non-word boundary, but punctuation is
        # already non-word, so no boundary exists. (?<!\w)/(?!\w) check for the absence
        # of an adjacent alphanumeric character, which works correctly regardless of punctuation.
        return bool(re.search(r'(?<!\w)' + re.escape(needle) + r'(?!\w)', haystack))
    return needle in haystack


def _entry_matches(current_text, history_msgs, entry):
    if not entry.get("enabled", True):
        return False
    prob = entry.get("probability", 100)
    # FIX V: Clamp probability to 0–100 before the roll check. Without the clamp,
    # a malformed entry with probability=150 satisfies `150 < 100 == False` and always
    # fires, ignoring the intended probability entirely.
    prob = max(0, min(100, prob))
    if prob <= 0:
        return False
    if prob < 100 and random.randint(1, 100) > prob:
        return False
    keys = entry.get("keys", [])
    if not keys:
        return False
    # FIX #2: Use explicit None check so scan_depth=0 ("current message only") is honoured.
    # The original `entry.get("scan_depth") or ...` treated 0 as falsy and fell back to the
    # global setting, making per-entry depth-0 entries impossible to use correctly.
    _entry_depth = entry.get("scan_depth")
    if _entry_depth is not None:
        depth = _entry_depth
    elif params["scan_depth"] >= 0:
        depth = params["scan_depth"]
    else:
        depth = 1
    scan_text = (current_text + "\n" + "\n".join(history_msgs[-depth:])) if depth > 0 else current_text
    if not any(_hit_key(k, scan_text, entry) for k in keys):
        return False
    sec = entry.get("secondary_keys", [])
    logic = entry.get("selective_logic", "AND ANY")
    if sec:
        hits = [_hit_key(k, scan_text, entry) for k in sec]
        if logic == "AND ANY" and not any(hits):
            return False
        elif logic == "AND ALL" and not all(hits):
            return False
        elif logic == "NOT ANY" and any(hits):
            return False
        elif logic == "NOT ALL" and all(hits):
            return False
    return True


def _apply_inclusion_groups(matched):
    groups = {}
    ungrouped = []
    for e in matched:
        group = (e.get("inclusion_group") or "").strip()
        if group:
            groups.setdefault(group, []).append(e)
        else:
            ungrouped.append(e)
    result = list(ungrouped)
    for group_entries in groups.values():
        result.append(max(group_entries, key=lambda e: e.get("priority", 0)))
    return result


def _trim_to_budget(entries):
    budget_tokens = int(params["token_budget"])
    kept, used = [], 0
    for e in entries:
        remaining = budget_tokens - used
        if remaining <= 0:
            break
        size = _count_tokens(e.get("content", ""))
        if size <= remaining:
            kept.append(e)
            used += size
        else:
            truncated = copy.copy(e)
            # FIX #11: Snap truncation back to the nearest word boundary so the AI
            # doesn't receive half-words. rfind(' ') returns -1 if no space is found,
            # in which case we fall back to the hard character limit.
            hard_cut = e["content"][:remaining]
            snap = hard_cut.rfind(' ')
            truncated["content"] = hard_cut[:snap] if snap > 0 else hard_cut
            kept.append(truncated)
            break
    return kept


def _all_active_entries():
    # FIX 2: Snapshot _active_lorebooks under the lock before iterating. The mutation
    # side is already locked (FIX A), but reads on the generation thread were not,
    # leaving a window for "dictionary changed size during iteration" RuntimeErrors.
    # FIX 4 (new): Deep-copy each lorebook so the generation thread owns its own
    # entries list. Without this, _active_lorebooks[stem] and current_lorebook point
    # to the same dict object, meaning UI-thread mutations (pop/append/replace on
    # entries) can corrupt the list the generation thread is currently iterating —
    # even though the dict snapshot itself is safe.
    # FIX UID-COLLISION: UIDs are only unique within a single lorebook file. When
    # multiple lorebooks are active, raw-UID deduplication across them causes entries
    # with colliding UIDs to silently shadow each other — constant entries from one
    # lorebook suppress trigger entries from another, or vice-versa. Stamp each entry
    # with its lorebook stem so callers can use (stem, uid) as the composite key.
    with _state_lock:
        active_items = list(_active_lorebooks.items())
    active_lbs = [(stem, copy.deepcopy(lb)) for stem, lb in active_items]
    combined = []
    for stem, lb in active_lbs:
        for e in lb.get("entries", []):
            e["_lb_stem"] = stem
            combined.append(e)
    return combined


def _eff_pos(e):
    if params.get("position_override_enabled"):
        return params.get("position_override_value", "after_context")
    return e.get("position", "after_context")


def _eid(e):
    """Composite entry identity: (lorebook_stem, uid).
    UIDs are only unique within one lorebook file; raw-UID deduplication across
    multiple active lorebooks causes silent cross-lorebook shadowing. Using the
    stem-qualified pair guarantees global uniqueness regardless of how many
    lorebooks are active simultaneously."""
    return (e.get("_lb_stem", ""), e.get("uid"))


def _find_active_matches(current_text, history_msgs):
    all_entries = _all_active_entries()
    seen_eids = set()
    matched_list = []

    def _do_pass(text_to_scan, msgs, candidates):
        newly = []
        for e in candidates:
            eid = _eid(e)
            if eid in seen_eids:
                continue
            if _entry_matches(text_to_scan, msgs, e):
                seen_eids.add(eid)
                newly.append(e)
        return newly

    wave = _do_pass(current_text, history_msgs, [e for e in all_entries if not e.get("constant")])
    matched_list.extend(wave)

    if params.get("recursive_scan", True):
        for _ in range(int(params.get("max_recursion_steps", 3))):
            if not wave:
                break
            wave = _do_pass(
                " ".join(e.get("content", "") for e in wave), [],
                [e for e in all_entries if _eid(e) not in seen_eids]
            )
            matched_list.extend(wave)

    if params.get("constant_entries", True):
        for e in all_entries:
            if not (e.get("constant") and e.get("enabled", True)):
                continue
            if _eid(e) in seen_eids:
                continue
            prob = e.get("probability", 100)
            # FIX B: Clamp probability here too. FIX V only patched _entry_matches;
            # this inline check for constant entries had the same unclamped bug.
            prob = max(0, min(100, prob))
            if prob <= 0 or (prob < 100 and random.randint(1, 100) > prob):
                continue
            seen_eids.add(_eid(e))
            matched_list.append(e)

    matched_list = _apply_inclusion_groups(matched_list)
    # Sort by priority DESC; for ties use reverse discovery order so the last-triggered
    # entry comes first into _trim_to_budget (gets the most budget) and ends up freshest
    # (closest to reply) after the final reversal.  enumerate gives each entry its stable
    # discovery index; negating it puts later-discovered entries first on ties.
    matched_list = [e for _, e in sorted(
        enumerate(matched_list),
        key=lambda pair: (pair[1].get("priority", 0), pair[0]),
        reverse=True,
    )]
    trimmed = _trim_to_budget(matched_list)
    # Reverse so highest-priority / last-activated entry ends up closest to reply (freshest).
    return list(reversed(trimmed)), matched_list


def _format_injection(entries):
    parts = []
    for e in entries:
        content = e.get("content", "").strip()
        if not content:
            continue
        comment = e.get("comment", "").strip()
        parts.append(f"[{comment}]\n{content}" if comment else content)
    return "\n\n".join(parts)


def _strip_wi_block(ctx):
    pref = params["injection_prefix"]
    suf = params["injection_suffix"]
    # FIX I (CRITICAL): Guard against empty prefix/suffix. In Python, "any_string".find("")
    # always returns 0, so an empty pref/suf causes start=0 and end=0 every iteration,
    # ctx never changes, and the while True loop spins forever, locking up the server.
    if not pref or not suf:
        return ctx
    while True:
        start = ctx.find(pref)
        if start == -1:
            break
        end = ctx.find(suf, start)
        if end == -1:
            break
        block_end = end + len(suf)
        if block_end < len(ctx) and ctx[block_end] == "\n":
            block_end += 1
        ctx = ctx[:start] + ctx[block_end:]
    return ctx


def _do_wi_injection(scan_text, history_msgs, state):
    """Shared WI matching and context injection logic used by both state_modifier
    (regenerate path) and chat_input_modifier (normal send path). Operates directly
    on the state dict in place. orig_ctx must already be set in state[_ORIG_CTX]."""
    global _cur_injected
    orig_ctx = state.get(_ORIG_CTX)
    ctx_k = _ctx_key(state)
    if orig_ctx is None:
        # state_modifier may not have run (e.g. extension load order edge case).
        # Compute the clean context on the fly rather than KeyError.
        orig_ctx = _strip_wi_block(state.get(ctx_k, ""))
        state[_ORIG_CTX] = orig_ctx

    # "Only trigger words in chat" mode: strip the character persona from the current
    # user message before scanning. History slots that contained the persona were already
    # dropped at collection time by _gather_messages_list, so only scan_text needs cleaning.
    if params.get("chat_only_scan") and orig_ctx:
        scan_text = scan_text.replace(orig_ctx, "").strip()
    matched = _find_active_matches(scan_text, history_msgs)
    matched, all_matched = matched  # unpack (injected_reversed, full_pre_budget_list)
    matched = [e for e in matched if _eff_pos(e) != "notebook"]
    all_matched = [e for e in all_matched if _eff_pos(e) != "notebook"]
    if matched:
        pref = params["injection_prefix"]
        suf  = params["injection_suffix"]
        before_entries       = [e for e in matched if _eff_pos(e) == "before_context"]
        after_entries        = [e for e in matched if _eff_pos(e) == "after_context"]
        before_reply_entries = [e for e in matched if _eff_pos(e) == "before_reply"]
        ctx = orig_ctx
        if before_entries:
            block = (pref + _format_injection(before_entries) + suf).rstrip("\n") + "\n"
            ctx = block + ctx if ctx else block.lstrip("\n")
        if after_entries:
            ctx = ctx + "\n" + (pref + _format_injection(after_entries) + suf).rstrip("\n")
        state[ctx_k] = ctx
        state["_lb_before_reply_entries"] = before_reply_entries
    else:
        state[ctx_k] = orig_ctx
        state["_lb_before_reply_entries"] = []
        all_matched = []
    _cur_injected = {_eid(e) for e in matched} if matched else set()
    return matched, all_matched


def state_modifier(state):
    state = dict(state)
    ctx_k = _ctx_key(state)
    if _ORIG_CTX not in state:
        state[_ORIG_CTX] = _strip_wi_block(state.get(ctx_k, ""))
    # Pass orig_ctx so persona slots are filtered out during collection when
    # chat_only_scan is ON, rather than being stripped after the fact.
    state[_HIST_MSGS] = _gather_messages_list(state.get("history", {}), state[_ORIG_CTX])

    # Regenerate fix: apply_extensions('chat_input') is only called by oobabooga inside
    # `if not (regenerate or _continue)` in chatbot_wrapper, so chat_input_modifier
    # never runs on regenerate. state_modifier IS called on every path, so we do the
    # full WI scan here using the last user message from history — which on regenerate
    # is always internal[-1][0] (the message the user already sent).
    #
    # On normal sends this result is immediately overwritten by chat_input_modifier
    # (which runs next with the correct current text), so there is no conflict.
    if not params["activate"] or not _active_lorebooks:
        return state
    internal = state.get("history", {}).get("internal", [])
    if not internal:
        return state
    last_user = internal[-1][0]
    if not last_user or last_user == "<|BEGIN-VISIBLE-CHAT|>":
        return state
    # FIX: On regenerate, chat_input_modifier never runs, so _lb_chat_matched /
    # _lb_chat_all_matched are never written to state.  The mid_gen ON path in
    # custom_generate_reply reads _cur_injected directly and is unaffected, but
    # the mid_gen OFF early-return path reads state["_lb_chat_matched"] and
    # gets [] every time — so the history showed nothing even when entries fired.
    # Store the results here, mirroring what chat_input_modifier does on normal
    # sends, so custom_generate_reply always has the data it needs.
    _regen_matched, _regen_all = _do_wi_injection(last_user, state[_HIST_MSGS], state)
    state["_lb_chat_matched"]     = _regen_matched or []
    state["_lb_chat_all_matched"] = _regen_all     or []
    return state


def chat_input_modifier(text, visible_text, state):
    # Only called on normal sends (not regenerate). Overwrites whatever state_modifier
    # injected (which used the previous user message) with the correct current text.
    if not params["activate"] or not _active_lorebooks:
        global _cur_injected
        _cur_injected = set()
        return text, visible_text
    # FIX 1 (CRITICAL): Do NOT make a local copy of state. Oobabooga reads the original
    # state dict after this function returns. The previous `state = dict(state)` rebound
    # the local name to a new dict, so all context mutations were silently discarded and
    # lorebooks never actually injected anything into any prompt.
    history_msgs = state.get(_HIST_MSGS, _gather_messages_list(state.get("history", {}), state.get(_ORIG_CTX)))
    matched, all_matched = _do_wi_injection(text or "", history_msgs, state)
    # Store both lists in state so custom_generate_reply can write the preview
    # using is_chat, which is the only reliable signal for which UI tab is active.
    # Writing the preview here (before is_chat is known) caused notebook generations
    # to contaminate the chat store whenever an instruct template was loaded.
    state["_lb_chat_matched"]     = matched or []
    state["_lb_chat_all_matched"] = all_matched or []
    # FIX IV: _cur_injected is read on the generation thread in custom_generate_reply.
    # We don't need a lock around the set-comprehension itself (it's a single atomic
    # rebind in CPython), but document the dependency here so it's explicit. The real
    # protection is that custom_generate_reply immediately copies it: `already = set(_cur_injected)`,
    # which is safe because set() construction from another set is atomic in CPython.
    # For strict multi-interpreter safety the correct fix would be to pass it via state,
    # but that requires oobabooga API changes outside this file's scope.
    return text, visible_text


def _find_new_trigger_entries(text, already_injected, notebook=False):
    newly = []
    for e in _all_active_entries():
        eid = _eid(e)
        if eid in already_injected or not e.get("enabled", True) or e.get("constant"):
            continue
        # In notebook mode only scan notebook-position entries; in chat mode skip them.
        pos = _eff_pos(e)
        if notebook and pos != "notebook":
            continue
        if not notebook and pos == "notebook":
            continue
        keys = e.get("keys", [])
        if not keys:
            continue
        prob = e.get("probability", 100)
        # FIX C: Same clamp as FIX V / FIX B — prob > 100 bypasses the roll otherwise.
        prob = max(0, min(100, prob))
        if prob <= 0 or (prob < 100 and random.randint(1, 100) > prob):
            continue
        if any(_hit_key(k, text, e) for k in keys):
            # Honour secondary keys / selective logic — same as _entry_matches.
            sec = e.get("secondary_keys", [])
            logic = e.get("selective_logic", "AND ANY")
            if sec:
                hits = [_hit_key(k, text, e) for k in sec]
                if logic == "AND ANY" and not any(hits):
                    continue
                elif logic == "AND ALL" and not all(hits):
                    continue
                elif logic == "NOT ANY" and any(hits):
                    continue
                elif logic == "NOT ALL" and all(hits):
                    continue
            newly.append(e)
    newly.sort(key=lambda e: e.get("priority", 0), reverse=True)
    # FIX #8: Apply inclusion group filtering here too, consistent with _find_active_matches.
    # Without this, two entries from the same group could both fire mid-generation.
    newly = _apply_inclusion_groups(newly)
    # Apply the same (priority DESC, reverse-discovery) sort so equal-priority entries
    # triggered later during generation end up freshest after budget trimming + reversal.
    newly = [e for _, e in sorted(
        enumerate(newly),
        key=lambda pair: (pair[1].get("priority", 0), pair[0]),
        reverse=True,
    )]
    return _trim_to_budget(newly)


def _replace_world_info_block(prompt, all_entries):
    pref = params["injection_prefix"]
    suf = params["injection_suffix"]
    # FIX I: Same empty-string guard as _strip_wi_block — avoid find("") == 0 logic errors.
    if not pref or not suf:
        return prompt, []

    # Sort by priority DESC; for ties, later position in all_entries = more recently triggered
    # = should be freshest.  Negate the index so later entries sort first (get most budget),
    # then reverse at the end so they end up last (freshest) in injection order.
    trimmed = list(reversed(_trim_to_budget([e for _, e in sorted(
        enumerate(all_entries),
        key=lambda pair: (pair[1].get("priority", 0), pair[0]),
        reverse=True,
    )])))
    before_entries       = [e for e in trimmed if _eff_pos(e) == "before_context"]
    after_entries        = [e for e in trimmed if _eff_pos(e) == "after_context"]
    before_reply_entries = [e for e in trimmed if _eff_pos(e) == "before_reply"]
    notebook_entries     = [e for e in trimmed if _eff_pos(e) == "notebook"]

    # FIX (Gemini #2): Strip ALL existing WI blocks then re-inject, so orphaned
    # blocks from prior interrupts don't accumulate in the prompt.
    clean_prompt = _strip_wi_block(prompt)

    ctx = clean_prompt
    if before_entries:
        block = (pref + _format_injection(before_entries) + suf).rstrip("\n") + "\n"
        ctx = block + ctx if ctx else block.lstrip("\n")
    if after_entries:
        # FIX 3: Insert before the bot prefix line, not after it.
        # rstrip trailing newlines so rfind finds the newline *before* the
        # assistant header rather than the one that is part of it.
        block = (pref + _format_injection(after_entries) + suf).rstrip("\n")
        ctx_stripped = ctx.rstrip("\n")
        trailing     = ctx[len(ctx_stripped):]
        last_nl      = ctx_stripped.rfind("\n")
        if last_nl != -1:
            ctx = ctx_stripped[:last_nl] + "\n" + block + ctx_stripped[last_nl:] + trailing
        else:
            ctx = block + "\n" + ctx_stripped + trailing
    if before_reply_entries:
        # before_reply also lives just above the bot prefix — same insertion point as
        # after_context during mid-gen, but semantically distinct: it belongs to the
        # conversation layer, not the system prompt layer. Insert after any after_context
        # block (i.e. still using rfind so it ends up immediately above the bot prefix).
        block = (pref + _format_injection(before_reply_entries) + suf).rstrip("\n")
        ctx_stripped = ctx.rstrip("\n")
        trailing     = ctx[len(ctx_stripped):]
        last_nl      = ctx_stripped.rfind("\n")
        if last_nl != -1:
            ctx = ctx_stripped[:last_nl] + "\n" + block + ctx_stripped[last_nl:] + trailing
        else:
            ctx = block + "\n" + ctx_stripped + trailing
    if notebook_entries:
        # Notebook entries: prepend the WI block before the prompt body so the story
        # text stays at the end of context where the model continues it naturally.
        # This mirrors the initial notebook injection in custom_generate_reply and
        # ensures notebook entries survive every mid-gen interrupt re-injection.
        block = (pref + _format_injection(notebook_entries) + suf).rstrip("\n")
        ctx_stripped = ctx.rstrip("\n")
        trailing     = ctx[len(ctx_stripped):]
        if trailing:
            # Template-formatted prompt: insert before the assistant header.
            last_nl = ctx_stripped.rfind("\n")
            if last_nl != -1:
                ctx = ctx_stripped[:last_nl] + "\n" + block + ctx_stripped[last_nl:] + trailing
            else:
                ctx = block + "\n" + ctx_stripped + trailing
        else:
            ctx = block + "\n" + ctx

    return ctx, trimmed


def _update_injection_preview(all_fired, interrupt_count, notebook=False, budget_dropped=None):
    import datetime
    global _prev_chat_labels, _prev_notebook_labels
    global _all_chat_labels, _all_notebook_labels
    global _chat_turn_counter, _notebook_turn_counter

    rows = []
    total_tokens = 0
    labels_this_turn: set = set()

    for e in all_fired:
        label = e.get("comment", "") or ", ".join(e.get("keys", []))[:30] or f"UID {e.get('uid')}"
        toks = _count_tokens(e.get("content", ""))
        total_tokens += toks
        rows.append((label, toks, e.get("priority", 0)))
        labels_this_turn.add(label)

    # Choose the correct session sets for this path.
    prev_labels = _prev_notebook_labels if notebook else _prev_chat_labels
    all_labels  = _all_notebook_labels  if notebook else _all_chat_labels

    # Classify each entry relative to previous turns.
    entry_records = []
    for label, tokens, priority in rows:
        if label not in all_labels:
            status = "new"
        elif label in prev_labels:
            status = "repeat"
        else:
            status = "returned"
        entry_records.append({"label": label, "tokens": tokens, "status": status, "priority": priority})

    # Entries that triggered keywords but were cut by the token budget.
    budget_dropped_records = []
    if budget_dropped:
        injected_labels = labels_this_turn
        for e in budget_dropped:
            label = e.get("comment", "") or ", ".join(e.get("keys", []))[:30] or f"UID {e.get('uid')}"
            if label in injected_labels:
                continue  # made it in — don't double-count
            tokens = _count_tokens(e.get("content", ""))
            budget_dropped_records.append({"label": label, "tokens": tokens, "status": "budget_dropped", "priority": e.get("priority", 0)})

    # Entries present last turn that are gone this turn (keyword no longer matches).
    dropped = [lbl for lbl in prev_labels if lbl not in labels_this_turn]

    # Advance session state.
    all_labels.update(labels_this_turn)
    if notebook:
        _all_notebook_labels    = all_labels
        _prev_notebook_labels   = labels_this_turn
        _notebook_turn_counter += 1
        turn_num = _notebook_turn_counter
    else:
        _all_chat_labels   = all_labels
        _prev_chat_labels  = labels_this_turn
        _chat_turn_counter += 1
        turn_num = _chat_turn_counter

    record = {
        "turn":           turn_num,
        "time":           datetime.datetime.now().strftime("%H:%M:%S"),
        "entries":        entry_records,   # injected (priority order, highest = freshest)
        "budget_dropped": budget_dropped_records,  # triggered but cut by token budget
        "dropped":        dropped,
        "total_tokens":   total_tokens,
        "interrupts":     interrupt_count,
    }

    if notebook:
        with _last_notebook_injection_lock:
            _last_notebook_injection_info["entries"] = rows
            _last_notebook_injection_info["interrupts"] = interrupt_count
            _last_notebook_injection_info["total_tokens"] = total_tokens
        with _injection_history_lock:
            _injection_history_notebook.insert(0, record)
            if len(_injection_history_notebook) > 60:
                _injection_history_notebook.pop()
    else:
        with _last_injection_lock:
            _last_injection_info["entries"] = rows
            _last_injection_info["interrupts"] = interrupt_count
            _last_injection_info["total_tokens"] = total_tokens
        with _injection_history_lock:
            _injection_history.insert(0, record)
            if len(_injection_history) > 60:
                _injection_history.pop()


def _find_notebook_matches(question_text):
    """Scan the raw notebook/text-completion prompt for notebook-position entry keywords.

    Unlike chat mode there is no structured history — the entire textbox content is
    the scan target. Only entries with effective position == 'notebook' are considered;
    all other positions are ignored in text-completion mode by design.
    """
    all_entries = _all_active_entries()
    notebook_candidates = [e for e in all_entries if _eff_pos(e) == "notebook"]
    seen_eids_nb = set()
    matched = []

    for e in notebook_candidates:
        eid = _eid(e)
        if eid in seen_eids_nb or not e.get("enabled", True):
            continue
        prob = max(0, min(100, e.get("probability", 100)))
        if prob <= 0 or (prob < 100 and random.randint(1, 100) > prob):
            continue
        if e.get("constant"):
            seen_eids_nb.add(eid)
            matched.append(e)
            continue
        keys = e.get("keys", [])
        if not keys:
            continue
        if any(_hit_key(k, question_text, e) for k in keys):
            seen_eids_nb.add(eid)
            matched.append(e)

    matched = _apply_inclusion_groups(matched)
    matched = [e for _, e in sorted(
        enumerate(matched),
        key=lambda pair: (pair[1].get("priority", 0), pair[0]),
        reverse=True,
    )]
    trimmed = _trim_to_budget(matched)
    return list(reversed(trimmed)), matched


def custom_generate_reply(question, original_question, state,
                          stopping_strings=None, is_chat=False):
    import modules.shared as shared
    from modules.text_generation import generate_reply_HF, generate_reply_custom

    def _base_gen(q, oq, st):
        if shared.model.__class__.__name__ in ["LlamaServer", "Exllamav3Model", "TensorRTLLMModel"]:
            yield from generate_reply_custom(q, oq, st, stopping_strings, is_chat=is_chat)
        else:
            yield from generate_reply_HF(q, oq, st, stopping_strings, is_chat=is_chat)

    # Always inject "Between User and Assistant" entries into the prompt, even when
    # mid-gen interrupt is disabled. chat_input_modifier stored these in state rather
    # than the system message; we insert them here just above the bot prefix line so
    # the model sees them at maximum recency, immediately before its response.
    nb_for_interrupt = []   # notebook entries already injected; feeds the interrupt loop
    if params["activate"] and _active_lorebooks:
        pref = params["injection_prefix"]
        suf  = params["injection_suffix"]
        if pref and suf:
            if not is_chat:
                # ── Notebook / text-completion mode ──────────────────────────────
                # chat_input_modifier never runs here, so we scan the raw prompt
                # directly for notebook-position entries and insert the WI block.
                nb_matched, nb_all_matched = _find_notebook_matches(question)
                nb_for_interrupt = nb_matched  # carry into interrupt loop below
                if nb_matched:
                    block = (pref + _format_injection(nb_matched) + suf).rstrip("\n")
                    # FIX: When an instruct-template model is used in notebook mode
                    # the prompt still ends with e.g. "<|im_start|>assistant\n".
                    # A blind append puts the WI *inside* the assistant turn body,
                    # making the model treat it as already-generated output and emit
                    # EOS immediately (0 tokens out) — the same bug as the chat
                    # before_reply path.
                    # Guard: if the prompt ends with trailing newlines the assistant
                    # header is there; use rstrip+rfind to land just before it.
                    # If there are no trailing newlines it's a raw text-completion
                    # prompt with no template — fall back to a plain append.
                    q_stripped = question.rstrip("\n")
                    trailing   = question[len(q_stripped):]
                    if trailing:
                        # Template-formatted prompt: insert before the assistant header.
                        last_nl = q_stripped.rfind("\n")
                        if last_nl != -1:
                            question = q_stripped[:last_nl] + "\n" + block + q_stripped[last_nl:] + trailing
                        else:
                            question = block + "\n" + q_stripped + trailing
                    else:
                        # Raw text completion: prepend WI block so the story text
                        # remains at the end and the model continues it naturally.
                        # Appending the block caused [/World Info] to be the last
                        # token in context, which made the model emit EOS immediately
                        # (0 tokens generated) — the WI fired but generation never ran.
                        question = block + "\n" + question
                    original_question = question
                    # Update _cur_injected so the mid-gen interrupt loop's `already` set
                    # is seeded with the notebook entries that were just injected.
                    # Without this, _cur_injected holds stale chat-path data (or nothing),
                    # all_injected_entries starts empty, last_trimmed ends empty, and every
                    # notebook entry shows as DROPPED rather than REPEAT/BUDGET CUT.
                    global _cur_injected
                    _cur_injected = {_eid(e) for e in nb_matched}
                    # Build budget_dropped = entries that matched but didn't fit.
                    injected_eids = {_eid(e) for e in nb_matched}
                    nb_budget_dropped = [e for e in nb_all_matched if _eid(e) not in injected_eids]
                    _update_injection_preview(nb_matched, 0, notebook=True, budget_dropped=nb_budget_dropped)
                elif nb_all_matched:
                    # Everything matched but nothing fit in budget.
                    _cur_injected = set()
                    _update_injection_preview([], 0, notebook=True, budget_dropped=nb_all_matched)
            else:
                # ── Chat mode ────────────────────────────────────────────────────
                before_reply_entries = state.get("_lb_before_reply_entries", [])
                if before_reply_entries:
                    block   = (pref + _format_injection(before_reply_entries) + suf).rstrip("\n")
                    # FIX: Strip trailing newlines before rfind so the insertion
                    # point lands *before* the assistant turn header, not after it.
                    # In instruct mode the prompt ends with e.g. "<|im_start|>assistant\n";
                    # without rstrip, rfind finds that trailing \n and inserts the block
                    # inside the assistant turn body, making the model see it as
                    # already-generated text and emit EOS immediately (0 tokens out).
                    q_stripped = question.rstrip("\n")
                    trailing   = question[len(q_stripped):]
                    last_nl    = q_stripped.rfind("\n")
                    if last_nl != -1:
                        question = q_stripped[:last_nl] + "\n" + block + q_stripped[last_nl:] + trailing
                    else:
                        question = block + "\n" + q_stripped + trailing
                    original_question = question

    if not params["activate"] or not params["mid_gen_interrupt"] or not _active_lorebooks:
        # Write the chat preview here — chat_input_modifier no longer does it so that
        # notebook generations (is_chat=False) can never contaminate the chat store.
        if is_chat:
            chat_matched     = state.get("_lb_chat_matched", [])
            chat_all_matched = state.get("_lb_chat_all_matched", [])
            injected_eids    = {_eid(e) for e in chat_matched}
            chat_budget_dropped = [e for e in chat_all_matched if _eid(e) not in injected_eids]
            _update_injection_preview(chat_matched, 0, notebook=False, budget_dropped=chat_budget_dropped)
        yield from _base_gen(question, original_question, state)
        return

    max_ints = max(0, int(params["max_interrupts"]))
    if is_chat:
        already = set(_cur_injected)
        all_injected_entries = [e for e in _all_active_entries() if _eid(e) in already]
    else:
        # notebook: _cur_injected is never set for this path; use the entries we just injected
        already = {_eid(e) for e in nb_for_interrupt}
        all_injected_entries = list(nb_for_interrupt)
    cur_question = question
    cur_original = original_question
    cumulative_text = ""
    gen_offset = ""
    interrupts = 0
    word_buf = ""
    last_trimmed = list(all_injected_entries)

    while True:
        # FIX FORCE-STOP: If the user pressed stop while the previous generator was
        # being closed and the loop was about to restart, honour that signal here
        # before spawning a new generator. Without this check, the interrupt logic
        # closes the current gen, loops back, and immediately starts a fresh one —
        # so a stop that coincides with an interrupt is silently swallowed.
        # shared.stop_everything is set by oobabooga's UI stop button; we must check
        # it at the restart boundary, not just rely on the inner generator to see it.
        try:
            if getattr(shared, "stop_everything", False):
                break
        except Exception:
            pass
        gen = _base_gen(cur_question, cur_original, state)
        interrupted = False
        for reply in gen:
            new_text = reply[len(cumulative_text) - len(gen_offset):]
            cumulative_text = gen_offset + reply
            word_buf += new_text
            yield cumulative_text
            if interrupts < max_ints and re.search(r"\s", new_text):
                # FIX #13: Strip the trailing incomplete word before scanning.
                # When the AI has streamed "The world" but not yet emitted the next
                # character, "world" sits at the end of word_buf with nothing after it,
                # so (?!\w) sees end-of-string and passes — a false positive for any key
                # that is a prefix of a longer word (e.g. "world" matching mid-"worldview").
                # We only scan the *committed* portion: text up to the last word boundary.
                # The trailing \w+ tail is kept in word_buf so it still accumulates.
                m_tail   = re.search(r'\w+$', word_buf)
                scan_buf = word_buf[:m_tail.start()] if m_tail else word_buf
                pending  = word_buf[m_tail.start():] if m_tail else ""
                new_entries = _find_new_trigger_entries(scan_buf, already, notebook=not is_chat) if scan_buf else []
                if new_entries:
                    # FIX #6: Only clear word_buf when an interrupt actually fires.
                    # The original always wiped it on whitespace, so multi-word keys like
                    # "ancient dragon" could never accumulate enough text to match.
                    # Preserve the pending tail — it's a real partial word, not noise.
                    word_buf = pending
                    interrupts += 1
                    already |= {_eid(e) for e in new_entries}
                    all_injected_entries.extend(new_entries)
                    cur_question, last_trimmed = _replace_world_info_block(cur_question, all_injected_entries)
                    cur_question = cur_question + cumulative_text
                    cur_original = cur_question
                    gen_offset = cumulative_text
                    interrupted = True
                    try:
                        gen.close()
                    except Exception:
                        pass
                    break
                else:
                    # Cap buffer to prevent unbounded memory growth while still allowing
                    # multi-word phrases to accumulate across streaming chunks.
                    word_buf = word_buf[-2000:]
        if not interrupted:
            break

    injected_eids = {_eid(e) for e in last_trimmed}
    budget_dropped = [e for e in all_injected_entries if _eid(e) not in injected_eids]
    _update_injection_preview(last_trimmed, interrupts, notebook=not is_chat, budget_dropped=budget_dropped)
    yield cumulative_text


# ── UI helpers ────────────────────────────────────────────────────────────────

def _get_active_keys():
    """Thread-safe snapshot of active lorebook keys for UI components."""
    with _state_lock:
        return list(_active_lorebooks.keys())


def _ctx_key(state):
    """Return the state key that holds the system message for the current mode.

    In instruct mode oobabooga builds the prompt from state['custom_system_message'],
    not state['context'].  In chat and chat-instruct mode it uses state['context'].
    The lorebook must read and write the correct key so its injections actually reach
    the model.
    """
    return "custom_system_message" if state.get("mode") == "instruct" else "context"


def _sync_active_lorebook():
    """If the currently open lorebook is active, push the in-memory edits into
    _active_lorebooks so generation immediately uses the updated version.
    Must be called while holding _state_lock."""
    stem = params.get("current_lorebook", "")
    if stem and stem in _active_lorebooks and current_lorebook is not None:
        _active_lorebooks[stem] = current_lorebook


def _entry_label(e, i):
    uid = e.get("uid", i)
    comment = (e.get("comment", "") or f"Entry {uid}")[:32]
    status = "🔵" if e.get("constant") else ("✅" if e.get("enabled", True) else "❌")
    return f"{status} [{uid}] {comment}"

def _entry_choices():
    if not current_lorebook:
        return []
    return [_entry_label(e, i) for i, e in enumerate(current_lorebook.get("entries", []))]

def _uid_from_choice(choice):
    m = re.search(r'\[(\d+)\]', choice or "")
    return int(m.group(1)) if m else None

def _idx_from_uid(uid):
    if not current_lorebook:
        return -1
    for i, e in enumerate(current_lorebook.get("entries", [])):
        if e.get("uid") == uid:
            return i
    return -1


def _build_stats_html():
    if not params.get("activate", True):
        return ('<div style="padding:10px 14px;border-radius:8px;font-size:13px;font-weight:500;'
                'background:rgba(239,68,68,.08);border:0.5px solid rgba(239,68,68,.4);'
                'color:var(--color-text-danger,#c0392b)">'
                '⚠ Lorebook system is <strong>OFF</strong> — enable it in Settings.</div>')
    with _last_injection_lock:
        info = dict(_last_injection_info)
    # FIX II: Snapshot _active_lorebooks under _state_lock before reading, consistent
    # with the pattern used in _all_active_entries(). Unlocked reads can still see a
    # mid-mutation dict and produce wrong counts or a RuntimeError in Python 3.12+.
    with _state_lock:
        active_lbs = list(_active_lorebooks.values())
        n_active = len(active_lbs)
    n_entries = sum(len(lb.get("entries", [])) for lb in active_lbs)
    tok = info["total_tokens"]
    ints = info["interrupts"]

    def _card(val, lbl):
        return (f'<div style="background:var(--color-background-secondary,rgba(0,0,0,.04));'
                f'border-radius:8px;padding:8px 10px">'
                f'<div style="font-size:20px;font-weight:500;color:var(--color-text-primary)">{val}</div>'
                f'<div style="font-size:11px;color:var(--color-text-secondary);margin-top:2px">{lbl}</div></div>')

    return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
            f'{_card(n_active, "active lorebooks")}'
            f'{_card(n_entries, "total entries")}'
            f'{_card("~" + str(tok) if tok else "—", "tokens last turn")}'
            f'{_card(ints, "interrupts")}'
            f'</div>')


def _build_history_html(notebook=False):
    """Render the full session injection history as a scrollable HTML block.

    Each card shows one generation turn:
      • Numbered rows in priority order — #1 = oldest in context (lowest priority);
        the last row = freshest (highest priority, closest to the reply).
      • NEW      = first time this label has fired this session.
      • REPEAT   = also fired the immediately preceding turn.
      • RETURNED = fired before but skipped ≥1 turn in between.
      • BUDGET ✂  = keyword matched but entry was cut by the token budget.
      • DROPPED  = were injected last turn but keyword no longer matches this turn.
    """
    with _injection_history_lock:
        history = list(_injection_history_notebook if notebook else _injection_history)

    if not history:
        msg = "use the Notebook tab first" if notebook else "send a message first"
        return (f'<p style="color:var(--body-text-color-subdued);font-size:13px;padding:8px 4px">'
                f'No history yet — {msg}.</p>')

    _S = {
        # status: (emoji, bg, border-color, text-color)
        "new":           ("🆕", "rgba(34,197,94,.14)",  "rgba(34,197,94,.55)",  "#15803d"),
        "repeat":        ("🔁", "rgba(99,102,241,.10)", "rgba(99,102,241,.40)", "var(--body-text-color-subdued)"),
        "returned":      ("↩️", "rgba(245,158,11,.13)", "rgba(245,158,11,.50)", "#b45309"),
        "budget_dropped":("✂",  "rgba(251,146,60,.13)", "rgba(251,146,60,.50)", "#c2410c"),
    }

    cards = []
    for rec in history:
        entries         = rec["entries"]   # [{label, tokens, status}]
        budget_dropped  = rec.get("budget_dropped", [])
        dropped         = rec["dropped"]
        total           = len(entries)
        ints            = rec["interrupts"]

        # ── header ──
        int_note = f" &nbsp;·&nbsp; {ints} interrupt{'s' if ints != 1 else ''}" if ints else ""
        header = (
            f'<div style="padding:5px 10px 5px 10px;'
            f'background:var(--color-background-secondary,rgba(0,0,0,.05));'
            f'border-bottom:1px solid var(--border-color-primary,rgba(0,0,0,.08));'
            f'font-size:11px;font-weight:600;display:flex;justify-content:space-between;align-items:center">'
            f'<span>Turn {rec["turn"]}{int_note}</span>'
            f'<span style="font-weight:400;color:var(--body-text-color-subdued)">'
            f'~{rec["total_tokens"]} tokens &nbsp;·&nbsp; {rec["time"]}</span>'
            f'</div>'
        )

        # ── entry rows ──
        row_html = []
        for i, e in enumerate(entries):
            # Position label: first = oldest in context, last = freshest (closest to reply)
            if total == 1:
                pos = "#1"
            elif i == 0:
                pos = "#1 &nbsp;<span title='Lowest priority — appears deepest in context; first to be forgotten as context grows' style='opacity:.55;font-size:10px'>oldest</span>"
            elif i == total - 1:
                pos = f"#{total} &nbsp;<span title='Highest priority — closest to the reply; freshest in the model attention window' style='opacity:.55;font-size:10px'>freshest</span>"
            else:
                pos = f"#{i + 1}"

            prio_val = e.get("priority", 0)
            prio_badge = (f'<span style="font-size:10px;padding:1px 6px;border-radius:10px;'
                          f'background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.35);'
                          f'color:var(--body-text-color-subdued);white-space:nowrap">p{prio_val}</span>')
            em, bg, bc, tc = _S.get(e["status"], _S["repeat"])
            badge = (f'<span style="font-size:10px;padding:1px 7px;border-radius:10px;'
                     f'background:{bg};border:1px solid {bc};color:{tc};white-space:nowrap">'
                     f'{em} {e["status"].upper()}</span>')
            row_html.append(
                f'<tr style="border-bottom:1px solid var(--border-color-primary,rgba(0,0,0,.06))">'
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued);'
                f'white-space:nowrap;vertical-align:middle">{pos}</td>'
                f'<td style="padding:4px 8px;font-size:12px;vertical-align:middle">{e["label"]}</td>'
                f'<td style="padding:4px 8px;text-align:center;vertical-align:middle">{prio_badge}</td>'
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued);'
                f'text-align:right;white-space:nowrap;vertical-align:middle">~{e["tokens"]} tok</td>'
                f'<td style="padding:4px 8px;text-align:right;vertical-align:middle">{badge}</td>'
                f'</tr>'
            )

        # ── budget-cut rows (triggered keyword but didn't fit in the token budget) ──
        for e in budget_dropped:
            prio_val = e.get("priority", 0)
            prio_badge = (f'<span style="font-size:10px;padding:1px 6px;border-radius:10px;'
                          f'background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.35);'
                          f'color:var(--body-text-color-subdued);white-space:nowrap">p{prio_val}</span>')
            em, bg, bc, tc = _S["budget_dropped"]
            badge = (f'<span style="font-size:10px;padding:1px 7px;border-radius:10px;'
                     f'background:{bg};border:1px solid {bc};color:{tc};white-space:nowrap">'
                     f'{em} BUDGET CUT</span>')
            row_html.append(
                f'<tr style="opacity:.55;border-bottom:1px solid var(--border-color-primary,rgba(0,0,0,.06))">'
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued)">—</td>'
                f'<td style="padding:4px 8px;font-size:12px;font-style:italic">{e["label"]}</td>'
                f'<td style="padding:4px 8px;text-align:center">{prio_badge}</td>'
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued);'
                f'text-align:right;white-space:nowrap">~{e["tokens"]} tok</td>'
                f'<td style="padding:4px 8px;text-align:right">{badge}</td>'
                f'</tr>'
            )

        # ── keyword-dropped rows (no longer matching) ──
        for lbl in dropped:
            row_html.append(
                f'<tr style="opacity:.40;border-bottom:1px solid var(--border-color-primary,rgba(0,0,0,.06))">'
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued)">—</td>'
                f'<td style="padding:4px 8px;font-size:12px;text-decoration:line-through">{lbl}</td>'
                f'<td style="padding:4px 8px"></td>'
                f'<td style="padding:4px 8px"></td>'
                f'<td style="padding:4px 8px;text-align:right">'
                f'<span style="font-size:10px;padding:1px 7px;border-radius:10px;'
                f'background:rgba(239,68,68,.10);border:1px solid rgba(239,68,68,.40);'
                f'color:#dc2626;white-space:nowrap">⬇ DROPPED</span></td>'
                f'</tr>'
            )

        table = f'<table style="width:100%;border-collapse:collapse">{"".join(row_html)}</table>'
        cards.append(
            f'<div style="margin-bottom:7px;border:1px solid var(--border-color-primary,rgba(0,0,0,.1));'
            f'border-radius:8px;overflow:hidden">{header}{table}</div>'
        )

    return (f'<div style="max-height:360px;overflow-y:auto;padding:2px 0">'
            + "".join(cards) + '</div>')


def custom_css():
    return """
/* ── Buttons ── use oobabooga's own refresh-button sizing as base */
.lb-btn-primary{border-color:rgba(139,92,246,.8)!important;background:rgba(139,92,246,.15)!important}
.lb-btn-primary:hover{background:rgba(139,92,246,.28)!important;border-color:rgba(139,92,246,1)!important}
.lb-btn-danger{border-color:rgba(239,68,68,.7)!important;background:rgba(239,68,68,.08)!important}
.lb-btn-danger:hover{border-color:rgba(239,68,68,1)!important;background:rgba(239,68,68,.18)!important}

/* ── Active lorebook pills ── keep native checkbox, just wrap labels nicely */
#lb-active-pills .wrap{display:flex!important;flex-wrap:wrap!important;gap:6px!important;padding:4px 0!important}
#lb-active-pills .wrap label{display:inline-flex!important;align-items:center!important;gap:5px!important;padding:3px 12px 3px 8px!important;border-radius:20px!important;border:1px solid var(--border-color-primary)!important;background:var(--button-secondary-background-fill)!important;font-size:12px!important;cursor:pointer!important;white-space:nowrap!important}
#lb-active-pills .wrap label:hover{border-color:rgba(139,92,246,.7)!important}
#lb-active-pills input[type=checkbox]{accent-color:#8b5cf6!important;width:13px!important;height:13px!important}
#lb-active-pills .gap{display:none!important}

/* ── Fix checkbox alignment in rows ── remove the dead space above checkboxes */
#lorebook-tab .row > .checkbox-wrap,
#lorebook-tab .row > div > .checkbox-wrap{margin-top:auto!important}

/* ── Section label */
.lb-section-label p{font-size:12px!important;font-weight:600!important;color:var(--body-text-color-subdued)!important;margin:10px 0 2px!important}

/* ── Misc */
#lorebook-tab textarea{resize:vertical!important}
"""


def ui():
    global current_lorebook, _next_uid

    with gr.Row():

        # ── LEFT: lorebook editor + entry editor ──────────────────────────────
        with gr.Column(scale=3):

            # Lorebook selector bar
            with gr.Row():
                lb_dropdown = gr.Dropdown(
                    choices=get_lorebook_files(), label="Lorebook file",
                    elem_classes="slim-dropdown", scale=4, interactive=True,
                    info="All .json files in the lorebooks/ folder.",
                )
                lb_new_btn    = gr.Button("New",    elem_classes="refresh-button")
                lb_save_btn   = gr.Button("Save",   variant="primary", elem_classes="lb-btn-primary refresh-button")
                lb_delete_btn = gr.Button("Delete", variant="stop",    elem_classes="lb-btn-danger refresh-button")

            lb_name_input = gr.Textbox(
                label="Name", placeholder="My Fantasy World",
                info="Displayed in the active lorebooks tab.",
            )
            lb_desc_input = gr.Textbox(
                label="Description", placeholder="Optional notes — the AI never reads this.",
            )

            lb_status = gr.Markdown("*No lorebook open — select one above or click New.*")

            with gr.Accordion("Entries", open=False, elem_classes="tgw-accordion"):

              gr.HTML('<div style="margin-top:6px"></div>')
              with gr.Row():
                entry_radio = gr.Dropdown(
                    choices=[], label="Entry",
                    elem_classes="slim-dropdown", interactive=True,
                    info="✅ enabled  🔵 constant  ❌ disabled",
                    scale=4,
                )
                entry_new_btn    = gr.Button("+ Add",  variant="primary", elem_classes="lb-btn-primary refresh-button", scale=1)
                entry_delete_btn = gr.Button("Remove", variant="stop",    elem_classes="lb-btn-danger refresh-button",  scale=1)

              entry_comment = gr.Textbox(label="Entry name", placeholder="e.g. Dragon Lore",
                                            info="Label only — the AI never reads this.")

              with gr.Row():
                  entry_enabled  = gr.Checkbox(label="Enabled",               value=True)
                  entry_constant = gr.Checkbox(label="Constant (always on)", value=False)
                  entry_case     = gr.Checkbox(label="Case sensitive",         value=False)
                  entry_whole    = gr.Checkbox(label="Whole words only",       value=True)
                  entry_regex    = gr.Checkbox(label="Use regex",              value=False)

              with gr.Row():
                  entry_keys     = gr.Textbox(label="Trigger words",           placeholder="dragon, drake, wyrm", scale=2,
                                              info="Comma-separated. ANY one fires this entry.")
                  entry_sec_keys = gr.Textbox(label="Also require (optional)", placeholder="ancient, fire", scale=2,
                                              info="Secondary keys evaluated against the logic below.")

              entry_selective_logic = gr.Dropdown(
                  choices=["AND ANY", "AND ALL", "NOT ANY", "NOT ALL"],
                  value="AND ANY", label="Secondary key logic",
                  elem_classes="slim-dropdown",
                  info="AND ANY = at least one must match.  AND ALL = all must match.  NOT ANY = none may match.  NOT ALL = suppressed only when all match.",
              )

              entry_content = gr.Textbox(
                  label="Content", lines=5,
                  placeholder="e.g. Aurelion is an ancient fire-breathing dragon who guards the Sunken Vault...",
                  info="This is what the AI reads when the entry fires.",
              )

              with gr.Row():
                  entry_priority   = gr.Number(label="Priority",   value=10, step=1, minimum=0, scale=1,
                                               info="Higher priority fires first.")
                  entry_scan_depth = gr.Number(label="Scan depth", value=0,  step=1, minimum=0, scale=1,
                                               info="Messages back to scan for this entry.")
              entry_position = gr.Dropdown(
                                           choices=[
                                               ("After System Prompt",  "after_context"),
                                               ("Before System Prompt", "before_context"),
                                           ],
                                           value="after_context",
                                           label="Position", elem_classes="slim-dropdown",
                                           info="After System Prompt = below character context (default).  Before System Prompt = above it (grounding).  Use the override in Settings to force all entries to Between User and Assistant or Notebook Mode.")

              entry_probability = gr.Slider(minimum=0, maximum=100, value=100, step=1,
                                            label="Trigger %",
                                            info="Chance this entry fires when triggered. 100 = always, 50 = coin flip.")
              entry_inclusion_group = gr.Textbox(label="Inclusion group", value="",
                                                 placeholder="e.g. weather_mood",
                                                 info="Only the highest-priority entry in a group fires. Leave blank to ignore.")

              entry_save_btn = gr.Button("Save entry", variant="primary", elem_classes="lb-btn-primary")
              entry_save_status = gr.Markdown("")


        # ── RIGHT: tabs ───────────────────────────────────────────────────────
        with gr.Column(scale=2):

            with gr.Tabs():

                with gr.Tab("Lorebooks"):
                    gr.Markdown("Check to turn on, uncheck to turn off. Multiple can be active at once.")
                    active_pills = gr.CheckboxGroup(
                        choices=get_lorebook_files(),
                        value=_get_active_keys(),
                        label="",
                        show_label=False,
                        elem_id="lb-active-pills",
                    )
                    active_refresh_btn = gr.Button("Refresh list", elem_classes="refresh-button")
                    active_status = gr.Markdown("")

                with gr.Tab("Overview"):
                    stats_html = gr.HTML(_build_stats_html())
                    stats_refresh_btn = gr.Button("Refresh stats", elem_classes="refresh-button")
                    with gr.Tabs():
                        with gr.Tab("Chat / Instruct"):
                            gr.Markdown("**Last injection** — entries in priority order. #1 = lowest priority (deepest in context, oldest). Last # = highest priority (closest to reply, freshest).")
                            preview_box = gr.Markdown("*No generation yet — send a message first.*")
                            preview_refresh_btn = gr.Button("Refresh injection", elem_classes="refresh-button")
                            gr.Markdown("**Session history** — tracks what fired, what was new, what got dropped each turn.")
                            history_box = gr.HTML(_build_history_html(notebook=False))
                            history_clear_btn = gr.Button("Clear history", elem_classes="refresh-button")
                        with gr.Tab("Notebook"):
                            gr.Markdown("**Last injection** — entries in priority order. #1 = lowest priority (deepest in context, oldest). Last # = highest priority (closest to generation point, freshest).")
                            notebook_preview_box = gr.Markdown("*No generation yet — use the Notebook tab first.*")
                            notebook_preview_refresh_btn = gr.Button("Refresh injection", elem_classes="refresh-button")
                            gr.Markdown("**Session history** — tracks what fired, what was new, what got dropped each turn.")
                            notebook_history_box = gr.HTML(_build_history_html(notebook=True))
                            notebook_history_clear_btn = gr.Button("Clear history", elem_classes="refresh-button")

                with gr.Tab("All entries"):
                    gr.Markdown("Quick table of all entries in the currently open lorebook.")
                    overview_btn = gr.Button("Refresh", elem_classes="refresh-button")
                    overview_box = gr.Markdown("")

                with gr.Tab("Settings"):
                    gr.Markdown("These settings apply to all active lorebooks.")

                    activate_cb = gr.Checkbox(
                        label="Lorebook system active", value=params["activate"],
                        info="Master on/off switch. Turn off to disable all lorebooks at once.",
                    )

                    with gr.Row():
                        scan_depth_n   = gr.Number(label="Scan depth override", value=params["scan_depth"], step=1, minimum=-1,
                                                   info="-1 = disabled, 0 = current message only.")
                        token_budget_n = gr.Number(label="Token budget",        value=params["token_budget"], step=64, minimum=64,
                                                   info="Max world info tokens to inject per turn.")

                    with gr.Row():
                        max_recursion_n   = gr.Number(label="Max recursion steps", value=params["max_recursion_steps"], step=1, minimum=1, maximum=10,
                                                      info="Caps recursive passes to prevent infinite loops.")

                    with gr.Row():
                        constant_entries_cb = gr.Checkbox(label="Inject constant entries", value=params["constant_entries"],
                                                          info="Constant entries are injected every turn regardless of trigger words.")
                        recursive_scan_cb   = gr.Checkbox(label="Recursive scanning",      value=params["recursive_scan"],
                                                          info="Matched entries can trigger further entries via keywords in their content.")

                    chat_only_scan_cb = gr.Checkbox(label="Only trigger words in chat", value=params["chat_only_scan"],
                                                    info="When ON, trigger scanning ignores the character persona/context — only actual chat messages can fire entries. Prevents persona keywords from accidentally triggering entries every turn.")

                    inj_prefix = gr.Textbox(label="Injection prefix", value=params["injection_prefix"], lines=2,
                                            info="Added before the world info block in the prompt.")
                    inj_suffix = gr.Textbox(label="Injection suffix", value=params["injection_suffix"], lines=2,
                                            info="Added after the world info block in the prompt.")

                    gr.HTML('<div style="margin:14px 0 6px;padding:8px 12px;border-left:3px solid rgba(139,92,246,.7);background:rgba(139,92,246,.06);border-radius:0 6px 6px 0"><span style="font-weight:600;font-size:13px;color:var(--body-text-color)">Mid-generation interrupt</span><span style="font-size:12px;color:var(--body-text-color-subdued)"> — pauses on new trigger words in model output, expands WI block, then resumes.</span></div>')
                    mid_gen_interrupt_cb = gr.Checkbox(label="Enable mid-gen interrupt", value=params["mid_gen_interrupt"],
                                                       info="Requires stream mode to be on in generation settings.")
                    max_interrupts_n = gr.Number(label="Max interrupts per reply", value=params["max_interrupts"],
                                                 step=1, minimum=1, maximum=100,
                                                 info="How many times generation can be interrupted in a single reply.")

                    gr.HTML('<div style="margin:14px 0 6px;padding:8px 12px;border-left:3px solid rgba(139,92,246,.7);background:rgba(139,92,246,.06);border-radius:0 6px 6px 0"><span style="font-weight:600;font-size:13px;color:var(--body-text-color)">Context position override</span><span style="font-size:12px;color:var(--body-text-color-subdued)"> — force all entries to one side, ignoring their individual position setting.</span></div>')
                    position_override_cb = gr.Checkbox(label="Enable override", value=params["position_override_enabled"],
                                                       info="OFF = each entry uses its own setting.  ON = all entries are forced to the chosen position.")
                    position_override_dd = gr.Dropdown(
                                                       choices=[
                                                           ("After System Prompt",        "after_context"),
                                                           ("Before System Prompt",       "before_context"),
                                                           ("Between User and Assistant", "before_reply"),
                                                           ("Notebook Mode (Text Completion)", "notebook"),
                                                       ],
                                                       value=params["position_override_value"],
                                                       label="Force all entries to",
                                                       elem_classes="slim-dropdown",
                                                       interactive=params["position_override_enabled"],
                                                       info="After System Prompt = below character context (stays fresh).  Before System Prompt = above it (grounding).  Between User and Assistant = just before bot reply (maximum recency).  Notebook Mode = text completion only, appended at end of prompt.")

                with gr.Tab("Import / Export"):
                    gr.Markdown("#### Import from SillyTavern")
                    gr.Markdown("Drop a SillyTavern world-info `.json` file below and click **Import**. Selective logic, probability, and inclusion groups are preserved.")
                    with gr.Row():
                        st_import_file = gr.File(label="SillyTavern .json", file_types=[".json"], scale=3)
                        st_import_btn  = gr.Button("Import", scale=1, variant="primary", elem_classes="lb-btn-primary")

                    st_import_status = gr.Markdown("")

                    gr.Markdown("#### Export to SillyTavern")
                    gr.Markdown("Export the currently open lorebook as a SillyTavern world-info file.")
                    st_export_btn    = gr.Button("Export current lorebook", variant="secondary", elem_classes="refresh-button")
                    st_export_file   = gr.File(label="Download", visible=False, interactive=False)
                    st_export_status = gr.Markdown("")

    # ── Event handlers ────────────────────────────────────────────────────────

    def _do_load(name):
        global current_lorebook, _next_uid
        if not name:
            return gr.update(), gr.update(), gr.update(value="*No lorebook selected.*"), gr.update(choices=[])
        lb = load_lorebook(name)
        if lb is None:
            return gr.update(), gr.update(), gr.update(value=f"❌ Could not load **{name}.json** — file may be corrupt."), gr.update(choices=[])
        with _state_lock:  # FIX #10
            current_lorebook = lb
            params["current_lorebook"] = name
            _next_uid = max((e.get("uid", 0) for e in lb.get("entries", [])), default=0) + 1
        n = len(lb.get("entries", []))
        return (gr.update(value=lb.get("name", name)),
                gr.update(value=lb.get("description", "")),
                gr.update(value=f"✅ **{name}** loaded — {n} entr{'y' if n == 1 else 'ies'} ready."),
                gr.update(choices=_entry_choices(), value=None))

    def _do_new_lb():
        global current_lorebook, _next_uid
        with _state_lock:  # FIX #10
            current_lorebook = {"name": "New Lorebook", "description": "", "entries": []}
            _next_uid = 1
        return (gr.update(value="New Lorebook"), gr.update(value=""),
                gr.update(value="New lorebook created! Give it a name and click **Save**."),
                gr.update(choices=[], value=None))

    def _do_save_lb(name, desc):
        global current_lorebook
        old_stem = params.get("current_lorebook", "")
        with _state_lock:  # FIX #10
            if not current_lorebook:
                current_lorebook = {"name": name, "description": desc, "entries": []}
            else:
                current_lorebook["name"] = name
                current_lorebook["description"] = desc
            # FIX 2 (new): Capture a reference to current_lorebook while still inside
            # the lock. Without this, another Gradio event (_do_new_lb, _do_load) could
            # swap the global to None or a different dict between the lock release and
            # the save_lorebook_file() call, causing the wrong data (or a crash) to
            # be written to disk.
            lb_snapshot = current_lorebook
        stem = _safe_stem(name)
        ok = save_lorebook_file(stem, lb_snapshot)
        if ok:
            params["current_lorebook"] = stem
            with _state_lock:
                # FIX B: refresh the active copy on save.
                # FIX 5: Also handle renames — if the stem changed, remove the old active
                # entry and add the new one. Without this, the old stem stays active with
                # stale data and the new stem never gets added to active generation.
                if old_stem and old_stem != stem and old_stem in _active_lorebooks:
                    _active_lorebooks.pop(old_stem)
                    _active_lorebooks[stem] = lb_snapshot
                    _save_active_state()
                elif stem in _active_lorebooks:
                    _active_lorebooks[stem] = lb_snapshot
        return (gr.update(value=f"✅ Saved as **{stem}.json**" if ok else "❌ Save failed — check folder permissions."),
                gr.update(choices=get_lorebook_files(), value=stem if ok else None))

    def _do_delete_lb(name):
        global current_lorebook
        if not name:
            return gr.update(), gr.update(value="❌ No lorebook selected.")
        ok = delete_lorebook_file(name)
        with _state_lock:
            # FIX #5: The original only cleared current_lorebook. It never removed the
            # deleted lorebook from _active_lorebooks or called _save_active_state(), so
            # the deleted lorebook stayed active in memory and kept affecting generation.
            if params.get("current_lorebook") == name:
                current_lorebook = None
                params["current_lorebook"] = ""
            if name in _active_lorebooks:
                _active_lorebooks.pop(name)
                _save_active_state()
        return (gr.update(choices=get_lorebook_files(), value=None),
                gr.update(value=f"Deleted **{name}**." if ok else f"❌ Could not delete **{name}**."))

    def _do_select_entry(choice):
        BLANK = ("", "", "", "", "AND ANY", "", True, False, True, False, 10, "after_context", 0, 100, False)
        if not choice or not current_lorebook:
            return BLANK
        uid = _uid_from_choice(choice)
        if uid is None:
            return BLANK
        idx = _idx_from_uid(uid)
        if idx < 0:
            return BLANK
        e = current_lorebook["entries"][idx]
        return (e.get("comment", ""), ", ".join(e.get("keys", [])), ", ".join(e.get("secondary_keys", [])),
                e.get("content", ""), e.get("selective_logic", "AND ANY"), e.get("inclusion_group", ""),
                e.get("enabled", True), e.get("case_sensitive", False), e.get("match_whole_words", True),
                e.get("use_regex", False), float(e.get("priority", 10)), e.get("position", "after_context"),
                float(e.get("scan_depth", 0)), float(e.get("probability", 100)), e.get("constant", False))

    def _do_new_entry():
        global _next_uid
        if not current_lorebook:
            # Return enough values for the expanded output list below; all no-ops
            return (gr.update(), gr.update(value="❌ Load or create a lorebook first."),
                    *[gr.update()] * 15)
        with _state_lock:  # FIX #10
            e = {"uid": _next_uid, "enabled": True, "constant": False, "comment": f"New Entry {_next_uid}",
                 "keys": [], "secondary_keys": [], "selective_logic": "AND ANY", "content": "",
                 "case_sensitive": False, "match_whole_words": True, "use_regex": False, "priority": 10,
                 "position": "after_context", "scan_depth": 0, "probability": 100, "inclusion_group": ""}
            _next_uid += 1
            current_lorebook["entries"].append(e)
            _sync_active_lorebook()  # FIX VI
        choices = _entry_choices()
        # FIX 9: Programmatic gr.update() on a Dropdown does not fire its .change event in
        # recent Gradio versions, so _do_select_entry never runs and the form retains the
        # previous entry's data. Return the new entry's field values directly here instead.
        return (
            gr.update(choices=choices, value=choices[-1] if choices else None),
            gr.update(value="New entry added! Fill in the fields and click **Save entry**."),
            gr.update(value=e["comment"]),           # entry_comment
            gr.update(value=""),                     # entry_keys
            gr.update(value=""),                     # entry_sec_keys
            gr.update(value=""),                     # entry_content
            gr.update(value="AND ANY"),              # entry_selective_logic
            gr.update(value=""),                     # entry_inclusion_group
            gr.update(value=True),                   # entry_enabled
            gr.update(value=False),                  # entry_case
            gr.update(value=True),                   # entry_whole
            gr.update(value=False),                  # entry_regex
            gr.update(value=10),                     # entry_priority
            gr.update(value="after_context"),        # entry_position
            gr.update(value=0),                      # entry_scan_depth
            gr.update(value=100),                    # entry_probability
            gr.update(value=False),                  # entry_constant
        )

    def _do_delete_entry(choice):
        if not choice or not current_lorebook:
            return (gr.update(), gr.update(value="❌ No entry selected."),
                    *[gr.update()] * 15)
        uid = _uid_from_choice(choice)
        idx = _idx_from_uid(uid) if uid is not None else -1
        if idx < 0:
            return (gr.update(), gr.update(value="❌ Entry not found."),
                    *[gr.update()] * 15)
        with _state_lock:  # FIX #10
            current_lorebook["entries"].pop(idx)
            _sync_active_lorebook()  # FIX VI
        # FIX D: Clear all form fields so the deleted entry's data doesn't linger.
        # The wiring below must include all 15 field outputs to match this return.
        return (
            gr.update(choices=_entry_choices(), value=None),
            gr.update(value="Entry removed."),
            gr.update(value=""),           # entry_comment
            gr.update(value=""),           # entry_keys
            gr.update(value=""),           # entry_sec_keys
            gr.update(value=""),           # entry_content
            gr.update(value="AND ANY"),    # entry_selective_logic
            gr.update(value=""),           # entry_inclusion_group
            gr.update(value=True),         # entry_enabled
            gr.update(value=False),        # entry_case
            gr.update(value=True),         # entry_whole
            gr.update(value=False),        # entry_regex
            gr.update(value=10),           # entry_priority
            gr.update(value="after_context"), # entry_position
            gr.update(value=0),            # entry_scan_depth
            gr.update(value=100),          # entry_probability
            gr.update(value=False),        # entry_constant
        )

    def _do_save_entry(choice, comment, keys_str, sec_keys_str, content, selective_logic,
                       inclusion_group, enabled, case_sensitive, whole_words, use_regex,
                       priority, position, scan_depth, probability, constant):
        global _next_uid
        if not current_lorebook:
            return gr.update(value="❌ No lorebook loaded."), gr.update()
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        sec_keys = [k.strip() for k in sec_keys_str.split(",") if k.strip()]
        uid = _uid_from_choice(choice) if choice else None
        idx = _idx_from_uid(uid) if uid is not None else -1
        with _state_lock:  # FIX #10
            # FIX VII: Assign uid inside the lock. The original read _next_uid at line
            # `"uid": uid if idx >= 0 else _next_uid` before acquiring the lock, so two
            # concurrent Gradio events could both read the same _next_uid and create
            # entries with duplicate UIDs. Building entry_data inside the lock closes this.
            entry_data = {
                "uid": uid if idx >= 0 else _next_uid,
                "enabled": bool(enabled), "constant": bool(constant),
                "comment": comment.strip(), "keys": keys, "secondary_keys": sec_keys,
                "selective_logic": selective_logic, "content": content.strip(),
                "case_sensitive": bool(case_sensitive), "match_whole_words": bool(whole_words),
                "use_regex": bool(use_regex), "priority": int(priority), "position": position,
                "scan_depth": int(scan_depth), "probability": int(probability),
                "inclusion_group": (inclusion_group or "").strip(),
            }
            if idx >= 0:
                current_lorebook["entries"][idx] = entry_data
                status = "✅ Entry updated! *(Remember to also **Save** the lorebook to write to disk.)*"
            else:
                _next_uid += 1
                current_lorebook["entries"].append(entry_data)
                status = "✅ New entry saved! *(Remember to also **Save** the lorebook to write to disk.)*"
            # FIX VI: Push entry changes into the active copy so generation sees them
            # immediately without requiring a full lorebook save/reload cycle.
            _sync_active_lorebook()
        choices = _entry_choices()
        new_choice = next((c for c in choices if f"[{entry_data['uid']}]" in c), None)
        return gr.update(value=status), gr.update(choices=choices, value=new_choice)

    def _do_overview():
        if not current_lorebook:
            return gr.update(value="*No lorebook loaded.*")
        entries = current_lorebook.get("entries", [])
        if not entries:
            return gr.update(value="*No entries yet — click **+ Add** to create one.*")
        lines = ["| ID | | Name | Trigger words | Pri | Prob | Group |",
                 "|----|--|------|---------------|-----|------|-------|"]
        for e in entries:
            st = ("🔵" if e.get("constant") else "✅") if e.get("enabled", True) else "❌"
            keys = ("CONSTANT" if e.get("constant") else ", ".join(e.get("keys", [])) or "—")[:40]
            lines.append(f"| {e.get('uid','?')} | {st} | {(e.get('comment','') or '—')[:30]} | {keys} | {e.get('priority',10)} | {e.get('probability',100)}% | {(e.get('inclusion_group') or '—')[:18]} |")
        return gr.update(value="\n".join(lines))

    def _do_toggle_active(selected):
        selected_set = set(selected or [])
        # FIX A: Wrap all _active_lorebooks mutations in _state_lock.
        # Merged into a single lock block: read, mutate, snapshot, and save
        # now happen atomically so there is no window between releases where
        # another thread can see a half-mutated dict.
        with _state_lock:
            current_set = set(_active_lorebooks.keys())
            for stem in current_set - selected_set:
                _active_lorebooks.pop(stem, None)
            for stem in selected_set - current_set:
                lb = load_lorebook(stem)
                if lb:
                    _active_lorebooks[stem] = lb
            actual_active = list(_active_lorebooks.keys())
        _save_active_state()
        n = len(actual_active)
        msg = f"{n} lorebook{'s' if n != 1 else ''} active." if n else "No lorebooks active."
        return (gr.update(value=_build_stats_html()),
                gr.update(value=msg),
                gr.update(value=actual_active))

    def _do_save_lb_and_refresh(name, desc):
        r = _do_save_lb(name, desc)
        files = get_lorebook_files()
        with _state_lock:
            active_keys = list(_active_lorebooks.keys())
        return (r[0], r[1],
                gr.update(choices=files, value=active_keys),
                gr.update(value=_build_stats_html()))

    def _do_delete_lb_and_refresh(name):
        r = _do_delete_lb(name)
        files = get_lorebook_files()
        with _state_lock:
            active_keys = list(_active_lorebooks.keys())
        return (r[0], r[1],
                gr.update(choices=files, value=active_keys),
                gr.update(value=_build_stats_html()),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(choices=[], value=None))

    def _do_st_import(file_obj):
        global current_lorebook, _next_uid
        if file_obj is None:
            # FIX 1 (new): Gradio wiring expects 6 outputs from _do_st_import.
            # All error-path early returns must match or Gradio throws ValueError.
            return gr.update(value="❌ No file selected."), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        try:
            # FIX C: Gradio's gr.File returns a file-like object in some versions and a
            # plain string path in others. getattr fallback handles both cases safely.
            file_path = getattr(file_obj, "name", file_obj)
            raw = Path(file_path).read_bytes()
        except Exception as exc:
            return gr.update(value=f"❌ Could not read file: {exc}"), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        lb, err = import_from_sillytavern(raw)
        if err:
            return gr.update(value=f"❌ {err}"), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        stem = base_stem = _safe_stem(lb.get("name", "imported"))
        counter = 1
        while (LOREBOOKS_DIR / f"{stem}.json").exists():
            stem = f"{base_stem}_{counter}"
            counter += 1
        # FIX 4: Strip internal keys from the saved file. Previously save_lorebook_file()
        # was called with the raw `lb` dict which still contained "_source" and "_import_stats".
        # FIX F (previous round) only cleaned current_lorebook in memory but the file on disk
        # was already written with the dirty dict. Now we clean first, then save and assign.
        clean_lb = {k: v for k, v in lb.items() if not k.startswith("_")}
        # Pull stats from the original lb before it's cleaned, for the UI message below.
        stats = lb.get("_import_stats", {})
        if not save_lorebook_file(stem, clean_lb):
            return gr.update(value="❌ Imported but could not save — check folder permissions."), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        # FIX III: Wrap mutations to current_lorebook and _next_uid in _state_lock,
        # consistent with every other handler that touches these globals.
        with _state_lock:
            current_lorebook = clean_lb
            params["current_lorebook"] = stem
            _next_uid = max((e.get("uid", 0) for e in clean_lb.get("entries", [])), default=0) + 1
        n = stats.get("total", len(clean_lb.get("entries", [])))
        n_const = stats.get("constant", 0)
        n_nokey = stats.get("no_keys", 0)
        notes = []
        if n_const:
            notes.append(f"{n_const} constant entr{'y' if n_const == 1 else 'ies'} (always-on 🔵)")
        if n_nokey:
            notes.append(f"{n_nokey} entr{'y' if n_nokey == 1 else 'ies'} with no trigger words — open them and add keys")
        msg = f"✅ Imported **{clean_lb.get('name', stem)}** — {n} entr{'y' if n == 1 else 'ies'} saved as **{stem}.json**."
        if notes:
            msg += "  \n\n**Note:** " + ";  ".join(notes) + "."
        files = get_lorebook_files()
        with _state_lock:
            active_keys = list(_active_lorebooks.keys())
        return (gr.update(value=msg),
                gr.update(choices=files, value=stem),
                gr.update(choices=files, value=active_keys),
                # FIX E: Populate the editor fields from the imported lorebook so the
                # user sees the imported data immediately without manually reselecting.
                gr.update(value=clean_lb.get("name", stem)),
                gr.update(value=clean_lb.get("description", "")),
                gr.update(choices=_entry_choices(), value=None))

    def _do_st_export():
        if not current_lorebook:
            return gr.update(value="❌ No lorebook loaded. Load one first."), gr.update(visible=False)
        raw = export_to_sillytavern(current_lorebook)
        stem = _safe_stem(current_lorebook.get("name", "export"))
        out = Path(tempfile.gettempdir()) / f"{stem}_sillytavern.json"
        out.write_bytes(raw)
        n = len(current_lorebook.get("entries", []))
        return (gr.update(value=f"✅ Exported **{current_lorebook.get('name', stem)}** — {n} entr{'y' if n == 1 else 'ies'}."),
                gr.update(value=str(out), visible=True))

    def _do_preview_refresh():
        with _last_injection_lock:
            info = dict(_last_injection_info)
        if not info["entries"]:
            table_md = "*No injection recorded yet — send a message first.*"
        else:
            total = len(info["entries"])
            lines = ["| # | Entry | Pri | Tokens |", "|---|-------|-----|--------|"]
            for i, (label, toks, prio) in enumerate(info["entries"]):
                if total == 1:
                    pos = "#1"
                elif i == 0:
                    pos = "#1 *(oldest)*"
                elif i == total - 1:
                    pos = f"#{total} *(freshest)*"
                else:
                    pos = f"#{i + 1}"
                lines.append(f"| {pos} | {label} | p{prio} | {toks} |")
            lines.append(f"\n**Total: ~{info['total_tokens']} tokens** | **Interrupts: {info['interrupts']}**")
            table_md = "\n".join(lines)
        return gr.update(value=table_md), gr.update(value=_build_history_html(notebook=False))

    def _do_preview_refresh_notebook():
        with _last_notebook_injection_lock:
            info = dict(_last_notebook_injection_info)
        if not info["entries"]:
            table_md = "*No injection recorded yet — use the Notebook tab first.*"
        else:
            total = len(info["entries"])
            lines = ["| # | Entry | Pri | Tokens |", "|---|-------|-----|--------|"]
            for i, (label, toks, prio) in enumerate(info["entries"]):
                if total == 1:
                    pos = "#1"
                elif i == 0:
                    pos = "#1 *(oldest)*"
                elif i == total - 1:
                    pos = f"#{total} *(freshest)*"
                else:
                    pos = f"#{i + 1}"
                lines.append(f"| {pos} | {label} | p{prio} | {toks} |")
            lines.append(f"\n**Total: ~{info['total_tokens']} tokens**")
            table_md = "\n".join(lines)
        return gr.update(value=table_md), gr.update(value=_build_history_html(notebook=True))

    def _do_clear_history():
        global _injection_history, _prev_chat_labels, _all_chat_labels, _chat_turn_counter
        with _injection_history_lock:
            _injection_history.clear()
        _prev_chat_labels = set()
        _all_chat_labels  = set()
        _chat_turn_counter = 0
        return gr.update(value=_build_history_html(notebook=False))

    def _do_clear_history_notebook():
        global _injection_history_notebook, _prev_notebook_labels, _all_notebook_labels, _notebook_turn_counter
        with _injection_history_lock:
            _injection_history_notebook.clear()
        _prev_notebook_labels  = set()
        _all_notebook_labels   = set()
        _notebook_turn_counter = 0
        return gr.update(value=_build_history_html(notebook=True))

    def _set_activate(x):
        params["activate"] = x
        _save_params()
        return gr.update(value=_build_stats_html())

    def _set_position_override(enabled):
        params["position_override_enabled"] = enabled
        _save_params()
        return gr.update(interactive=enabled)

    # ── Wire events ───────────────────────────────────────────────────────────

    # FIX 7: Safe conversion helpers so settings callbacks don't crash when Gradio
    # sends None (e.g. when a numeric field is cleared). Falls back to current value.
    # Each helper also persists params to disk so settings survive restarts.
    def _safe_int(key, x):
        try:
            params[key] = int(x)
            _save_params()
        except (TypeError, ValueError):
            pass

    def _safe_float(key, x):
        try:
            params[key] = float(x)
            _save_params()
        except (TypeError, ValueError):
            pass

    def _set_param(key, x):
        params[key] = x
        _save_params()

    # Lorebook file management
    lb_dropdown.change(_do_load,   [lb_dropdown], [lb_name_input, lb_desc_input, lb_status, entry_radio])
    lb_new_btn.click(  _do_new_lb, [],            [lb_name_input, lb_desc_input, lb_status, entry_radio])

    lb_save_btn.click(
        _do_save_lb_and_refresh, [lb_name_input, lb_desc_input],
        [lb_status, lb_dropdown, active_pills, stats_html],
    )
    lb_delete_btn.click(
        _do_delete_lb_and_refresh, [lb_dropdown],
        # FIX D: Added lb_name_input, lb_desc_input, entry_radio so they're cleared on delete.
        [lb_dropdown, lb_status, active_pills, stats_html, lb_name_input, lb_desc_input, entry_radio],
    )

    # Active pills toggle
    # FIX VIII: Added active_pills as a third output so the UI deselects pills whose
    # lorebook files failed to load (corrupt/missing), keeping display in sync with reality.
    active_pills.change(_do_toggle_active, [active_pills], [stats_html, active_status, active_pills])
    active_refresh_btn.click(
        lambda: gr.update(choices=get_lorebook_files(), value=_get_active_keys()),
        [], [active_pills]
    )

    # Entry management
    entry_radio.change(
        _do_select_entry, [entry_radio],
        [entry_comment, entry_keys, entry_sec_keys, entry_content,
         entry_selective_logic, entry_inclusion_group,
         entry_enabled, entry_case, entry_whole, entry_regex,
         entry_priority, entry_position, entry_scan_depth,
         entry_probability, entry_constant],
    )
    entry_new_btn.click(
        _do_new_entry, [],
        # FIX 9: Also update all form fields directly so the editor shows the new
        # entry's blank defaults instead of the previously selected entry's data.
        [entry_radio, entry_save_status,
         entry_comment, entry_keys, entry_sec_keys, entry_content,
         entry_selective_logic, entry_inclusion_group,
         entry_enabled, entry_case, entry_whole, entry_regex,
         entry_priority, entry_position, entry_scan_depth,
         entry_probability, entry_constant],
    )
    entry_delete_btn.click(
        _do_delete_entry, [entry_radio],
        # FIX D: Added all form fields so they're cleared when an entry is deleted.
        [entry_radio, entry_save_status,
         entry_comment, entry_keys, entry_sec_keys, entry_content,
         entry_selective_logic, entry_inclusion_group,
         entry_enabled, entry_case, entry_whole, entry_regex,
         entry_priority, entry_position, entry_scan_depth,
         entry_probability, entry_constant],
    )
    entry_save_btn.click(
        _do_save_entry,
        [entry_radio, entry_comment, entry_keys, entry_sec_keys, entry_content,
         entry_selective_logic, entry_inclusion_group,
         entry_enabled, entry_case, entry_whole, entry_regex,
         entry_priority, entry_position, entry_scan_depth,
         entry_probability, entry_constant],
        [entry_save_status, entry_radio],
    )

    # Settings
    # FIX E: Wrap all bare component references in lists.
    # FIX 7: Use _safe_int/_safe_float so cleared/None fields don't crash the handler.
    activate_cb.change(         _set_activate,                                                          [activate_cb],          [stats_html])
    scan_depth_n.change(        lambda x: _safe_int("scan_depth",          x),                         [scan_depth_n],         None)
    token_budget_n.change(      lambda x: _safe_int("token_budget",        x),                         [token_budget_n],       None)
    constant_entries_cb.change( lambda x: _set_param("constant_entries",       x),                     [constant_entries_cb],  None)
    recursive_scan_cb.change(   lambda x: _set_param("recursive_scan",         x),                     [recursive_scan_cb],    None)
    chat_only_scan_cb.change(   lambda x: _set_param("chat_only_scan",         x),                     [chat_only_scan_cb],    None)
    max_recursion_n.change(     lambda x: _safe_int("max_recursion_steps", x),                         [max_recursion_n],      None)
    inj_prefix.change(          lambda x: _set_param("injection_prefix",       x),                     [inj_prefix],           None)
    inj_suffix.change(          lambda x: _set_param("injection_suffix",       x),                     [inj_suffix],           None)
    mid_gen_interrupt_cb.change(lambda x: _set_param("mid_gen_interrupt",      x),                     [mid_gen_interrupt_cb], None)
    max_interrupts_n.change(    lambda x: _safe_int("max_interrupts",      x),                         [max_interrupts_n],     None)
    position_override_cb.change(_set_position_override, [position_override_cb], [position_override_dd])
    position_override_dd.change(lambda x: _set_param("position_override_value", x),                    [position_override_dd], None)

    # Import / Export
    # FIX E: Added lb_name_input, lb_desc_input, entry_radio so the editor populates
    # immediately after a SillyTavern import without requiring a manual reselect.
    st_import_btn.click(_do_st_import, [st_import_file],
                        [st_import_status, lb_dropdown, active_pills,
                         lb_name_input, lb_desc_input, entry_radio])
    st_export_btn.click(_do_st_export, [],               [st_export_status, st_export_file])

    # Overview + stats + preview refresh
    overview_btn.click(   _do_overview,                                   [],  [overview_box])
    stats_refresh_btn.click(lambda: gr.update(value=_build_stats_html()), [],  [stats_html])
    preview_refresh_btn.click(        _do_preview_refresh,          [], [preview_box,          history_box])
    notebook_preview_refresh_btn.click(_do_preview_refresh_notebook, [], [notebook_preview_box, notebook_history_box])
    history_clear_btn.click(          _do_clear_history,            [], [history_box])
    notebook_history_clear_btn.click( _do_clear_history_notebook,   [], [notebook_history_box])
