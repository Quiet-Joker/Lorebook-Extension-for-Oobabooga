import json

from .config import (
    params,
    _ST_SELECTIVE_INT_TO_STR,
    _ST_SELECTIVE_STR_TO_INT,
    _ST_POSITION,
    _OUR_POSITION,
)


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
        # ST position 1 = after_context (the correct default; 0 = before_context)
        position = _ST_POSITION.get(e.get("position", 1), "after_context")

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
        uid_val = _uid_raw if _uid_raw is not None else len(our_entries) + 1
        our_entries.append({
            "uid": uid_val,
            "enabled": enabled,
            "comment": comment,
            "keys": keys,
            "secondary_keys": sec,
            "selective_logic": sel_logic,
            "content": e.get("content", ""),
            "case_sensitive": bool(e.get("case_sensitive", False)),
            "match_whole_words": bool(e.get("match_whole_words", e.get("extensions", {}).get("match_whole_words", True))),
            "use_regex": bool(e.get("use_regex", e.get("extensions", {}).get("regex", False))),
            "priority": int(priority),
            "position": position,
            "scan_depth": int(scan_depth),
            "probability": probability,
            "inclusion_group": inclusion_group,
            "constant": constant,
        })
    # Resolve UID collisions from source entries that shared the same uid/id.
    seen_uids: set = set()
    next_uid = max((e["uid"] for e in our_entries), default=0) + 1
    for entry in our_entries:
        if entry["uid"] in seen_uids:
            entry["uid"] = next_uid
            next_uid += 1
        seen_uids.add(entry["uid"])

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
        "scan_depth": max(1, params.get("scan_depth", 2)),
        "token_budget": params.get("token_budget", 1024),
        "recursive_scanning": params.get("recursive_scan", True),
        "extensions": {},
        "entries": st_entries,
    }, indent=2, ensure_ascii=False).encode("utf-8")
