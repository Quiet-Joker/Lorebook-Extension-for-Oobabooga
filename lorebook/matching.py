import copy
import random
import re
from functools import lru_cache

from .config import params, _MAX_GATHER_DEPTH, _count_tokens
from .state import _st


@lru_cache(maxsize=512)
def _compile_pattern(pattern: str, flags: int) -> re.Pattern:
    return re.compile(pattern, flags)


def _eid(e):
    # UIDs are only unique within one lorebook; use (stem, uid) for global uniqueness.
    return (e.get("_lb_stem", ""), e.get("uid"))


def _all_active_entries():
    with _st.state_lock:
        if _st.active_entries_cache_version == _st.active_entries_version:
            return _st.active_entries_cache
        active_lbs = [(stem, copy.deepcopy(lb)) for stem, lb in _st.active_lorebooks.items()]
        combined = []
        for stem, lb in active_lbs:
            for e in lb.get("entries", []):
                e["_lb_stem"] = stem
                combined.append(e)
        _st.active_entries_cache = combined
        _st.active_entries_cache_version = _st.active_entries_version
    return combined


def _eff_pos(e):
    if params.get("position_override_enabled"):
        return params.get("position_override_value", "after_context")
    return e.get("position", "after_context")


def _eff_depth(entry) -> int:
    entry_depth = entry.get("scan_depth")
    if entry_depth is not None and entry_depth >= 0:
        return int(entry_depth)
    global_depth = params["scan_depth"]
    return int(global_depth) if global_depth >= 0 else 1


def _hit_key(raw_key, text, entry):
    k = raw_key.strip()
    if not k:
        return False
    case = entry.get("case_sensitive", False)
    flags = 0 if case else re.IGNORECASE
    if entry.get("use_regex", False):
        try:
            return bool(_compile_pattern(k, flags).search(text))
        except re.error:
            return False
    haystack = text if case else text.lower()
    needle = k if case else k.lower()
    if entry.get("match_whole_words", True):
        pattern = r'(?<!\w)' + re.escape(needle) + r'(?!\w)'
        return bool(_compile_pattern(pattern, 0 if not case else flags).search(haystack))
    return needle in haystack


def _secondary_keys_pass(entry, scan_text) -> bool:
    sec = entry.get("secondary_keys", [])
    if not sec:
        return True
    logic = entry.get("selective_logic", "AND ANY")
    hits = [_hit_key(k, scan_text, entry) for k in sec]
    if logic == "AND ANY":
        return any(hits)
    if logic == "AND ALL":
        return all(hits)
    if logic == "NOT ANY":
        return not any(hits)
    if logic == "NOT ALL":
        return not all(hits)
    return True


def _prob_pass(entry) -> bool:
    prob = max(0, min(100, entry.get("probability", 100)))
    return prob > 0 and (prob >= 100 or random.randint(1, 100) <= prob)


def _gather_messages_list(history, orig_ctx=None):
    msgs = []
    ctx_strip = orig_ctx.strip() if orig_ctx else None
    for pair in history.get("internal", [])[-_MAX_GATHER_DEPTH:]:
        user_text = pair[0]
        if not user_text or user_text == "<|BEGIN-VISIBLE-CHAT|>":
            continue
        if ctx_strip and params.get("chat_only_scan") and ctx_strip in str(user_text):
            continue
        msgs.append(str(user_text))
    return msgs


def _build_scan_text(current_text: str, history_msgs: list, depth: int) -> str:
    if depth <= 0:
        return current_text
    return current_text + "\n" + "\n".join(history_msgs[-depth:])


def _entry_matches(current_text, history_msgs, entry, scan_text: str | None = None):
    if not entry.get("enabled", True):
        return False
    if not _prob_pass(entry):
        return False
    keys = entry.get("keys", [])
    if not keys:
        return False
    if scan_text is None:
        scan_text = _build_scan_text(current_text, history_msgs, _eff_depth(entry))
    if not any(_hit_key(k, scan_text, entry) for k in keys):
        return False
    return _secondary_keys_pass(entry, scan_text)


def _sort_by_priority(entries: list) -> list:
    return [e for _, e in sorted(
        enumerate(entries),
        key=lambda pair: (-pair[1].get("priority", 0), pair[0]),
    )]


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
            # Estimate char limit at 4 chars/token, snap to word boundary,
            # then tighten in a loop to handle token-dense text.
            truncated = copy.copy(e)
            char_limit = remaining * 4
            hard_cut = e["content"][:char_limit]
            snap = hard_cut.rfind(' ')
            candidate = hard_cut[:snap] if snap > 0 else hard_cut
            for _ in range(20):
                if _count_tokens(candidate) <= remaining:
                    break
                candidate = candidate[:max(1, int(len(candidate) * 0.9))]
                snap = candidate.rfind(' ')
                if snap > 0:
                    candidate = candidate[:snap]
            truncated["content"] = candidate
            kept.append(truncated)
            break
    return kept


def _entry_short_label(e) -> str:
    return (e.get("comment", "") or
            ", ".join(e.get("keys", []))[:30] or
            f"UID {e.get('uid')}")


def _find_active_matches(current_text, history_msgs):
    all_entries = _all_active_entries()
    seen_eids = set()
    matched_list = []

    _scan_cache: dict[int, str] = {}

    def _get_scan_text(entry) -> str:
        depth = _eff_depth(entry)
        if depth not in _scan_cache:
            _scan_cache[depth] = _build_scan_text(current_text, history_msgs, depth)
        return _scan_cache[depth]

    def _do_pass(candidates, get_text=_get_scan_text):
        newly = []
        for e in candidates:
            eid = _eid(e)
            if eid in seen_eids:
                continue
            if _entry_matches(current_text, history_msgs, e, scan_text=get_text(e)):
                seen_eids.add(eid)
                newly.append(e)
        return newly

    wave = _do_pass([e for e in all_entries if not e.get("constant")])
    matched_list.extend(wave)

    if params.get("recursive_scan", True):
        for _ in range(int(params.get("max_recursion_steps", 3))):
            if not wave:
                break
            recursive_text = " ".join(e.get("content", "") for e in wave)
            wave = _do_pass(
                [e for e in all_entries if _eid(e) not in seen_eids],
                get_text=lambda e, t=recursive_text: t,
            )
            matched_list.extend(wave)

    if params.get("constant_entries", True):
        for e in all_entries:
            if not (e.get("constant") and e.get("enabled", True)):
                continue
            if _eid(e) in seen_eids:
                continue
            if not _prob_pass(e):
                continue
            seen_eids.add(_eid(e))
            matched_list.append(e)

    matched_list = _apply_inclusion_groups(matched_list)
    matched_list = _sort_by_priority(matched_list)
    trimmed = _trim_to_budget(matched_list)

    return list(reversed(trimmed)), matched_list
