import copy
import tempfile
from pathlib import Path

import gradio as gr

from .config import params
from .state import _st, _bump_active_version
from .storage import (
    get_lorebook_files,
    load_lorebook,
    save_lorebook_file,
    delete_lorebook_file,
    _save_active_state,
    _save_params,
    _safe_stem,
    LOREBOOKS_DIR,
)
from .sillytavern import import_from_sillytavern, export_to_sillytavern
from .injection import _get_active_keys, _sync_active_lorebook
from .matching import _hit_key, _secondary_keys_pass
from .ui_helpers import (
    _entry_choices,
    _uid_from_choice,
    _idx_from_uid,
    _build_stats_html,
    _build_history_html,
    _budget_bar_html,
    _build_import_preview_html,
)


def do_load(name):
    if not name:
        return gr.update(), gr.update(), gr.update(value="*No lorebook selected.*"), gr.update(choices=[])
    lb = load_lorebook(name)
    if lb is None:
        return gr.update(), gr.update(), gr.update(value=f"❌ Could not load **{name}.json** — file may be corrupt."), gr.update(choices=[])
    with _st.state_lock:
        _st.current_lorebook = lb
        params["_st.current_lorebook"] = name
        _st.next_uid = max((e.get("uid", 0) for e in lb.get("entries", [])), default=0) + 1
    n = len(lb.get("entries", []))
    return (gr.update(value=lb.get("name", name)),
            gr.update(value=lb.get("description", "")),
            gr.update(value=f"✅ **{name}** loaded — {n} entr{'y' if n == 1 else 'ies'} ready."),
            gr.update(choices=_entry_choices(), value=None))


def do_new_lb():
    with _st.state_lock:
        _st.current_lorebook = {"name": "New Lorebook", "description": "", "entries": []}
        _st.next_uid = 1
    return (gr.update(value="New Lorebook"), gr.update(value=""),
            gr.update(value="New lorebook created! Give it a name and click **Save**."),
            gr.update(choices=[], value=None))


def _do_save_lb(name, desc):
    old_stem = params.get("_st.current_lorebook", "")
    original_name = None
    with _st.state_lock:
        if not _st.current_lorebook:
            _st.current_lorebook = {"name": name, "description": desc, "entries": []}
        else:
            original_name = _st.current_lorebook.get("name", "")
            _st.current_lorebook["name"] = name
            _st.current_lorebook["description"] = desc
        lb_snapshot = copy.deepcopy(_st.current_lorebook)
    # Save to the original file when the display name hasn't changed.
    # Some lorebooks (e.g. _summaries.json / "Auto Story Summaries") have a
    # display name that doesn't round-trip through _safe_stem back to their
    # file stem, so we only derive a new stem when the user actually renames it.
    if old_stem and original_name is not None and name == original_name:
        stem = old_stem
    else:
        stem = _safe_stem(name)
    ok = save_lorebook_file(stem, lb_snapshot)
    if ok:
        params["_st.current_lorebook"] = stem
        with _st.state_lock:
            if old_stem and old_stem != stem and old_stem in _st.active_lorebooks:
                _st.active_lorebooks.pop(old_stem)
                _st.active_lorebooks[stem] = lb_snapshot
                _bump_active_version()
                _save_active_state()
            elif stem in _st.active_lorebooks:
                _st.active_lorebooks[stem] = lb_snapshot
                _bump_active_version()
    return (gr.update(value=f"✅ Saved as **{stem}.json**" if ok else "❌ Save failed — check folder permissions."),
            gr.update(choices=get_lorebook_files(), value=stem if ok else None))


def do_save_lb_and_refresh(name, desc):
    r = _do_save_lb(name, desc)
    files = get_lorebook_files()
    with _st.state_lock:
        active_keys = list(_st.active_lorebooks.keys())
    return (r[0], r[1],
            gr.update(choices=files, value=active_keys),
            gr.update(value=_build_stats_html()))


def _do_delete_lb(name):
    if not name:
        return gr.update(), gr.update(value="❌ No lorebook selected.")
    ok = delete_lorebook_file(name)
    with _st.state_lock:
        if params.get("_st.current_lorebook") == name:
            _st.current_lorebook = None
            params["_st.current_lorebook"] = ""
        if name in _st.active_lorebooks:
            _st.active_lorebooks.pop(name)
            _bump_active_version()
    _save_active_state()
    return (gr.update(choices=get_lorebook_files(), value=None),
            gr.update(value=f"Deleted **{name}**." if ok else f"❌ Could not delete **{name}**."))


def do_delete_lb_and_refresh(name):
    r = _do_delete_lb(name)
    files = get_lorebook_files()
    with _st.state_lock:
        active_keys = list(_st.active_lorebooks.keys())
    return (r[0], r[1],
            gr.update(choices=files, value=active_keys),
            gr.update(value=_build_stats_html()),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(choices=[], value=None))


def do_toggle_active(selected):
    selected_set = set(selected or [])
    with _st.state_lock:
        current_set = set(_st.active_lorebooks.keys())
        for stem in current_set - selected_set:
            _st.active_lorebooks.pop(stem, None)
        for stem in selected_set - current_set:
            lb = load_lorebook(stem)
            if lb:
                _st.active_lorebooks[stem] = lb
        _bump_active_version()
        actual_active = list(_st.active_lorebooks.keys())
    _save_active_state()
    n = len(actual_active)
    msg = f"{n} lorebook{'s' if n != 1 else ''} active." if n else "No lorebooks active."
    return (gr.update(value=_build_stats_html()),
            gr.update(value=msg),
            gr.update(value=actual_active))


_BLANK_ENTRY = ("", "", "", "", "AND ANY", "", True, False, True, False, 10, "after_context", 0, 100, False)


def do_select_entry(choice):
    if not choice or not _st.current_lorebook:
        return _BLANK_ENTRY
    uid = _uid_from_choice(choice)
    if uid is None:
        return _BLANK_ENTRY
    idx = _idx_from_uid(uid)
    if idx < 0:
        return _BLANK_ENTRY
    e = _st.current_lorebook["entries"][idx]
    return (e.get("comment", ""), ", ".join(e.get("keys", [])), ", ".join(e.get("secondary_keys", [])),
            e.get("content", ""), e.get("selective_logic", "AND ANY"), e.get("inclusion_group", ""),
            e.get("enabled", True), e.get("case_sensitive", False), e.get("match_whole_words", True),
            e.get("use_regex", False), float(e.get("priority", 10)), e.get("position", "after_context"),
            float(e.get("scan_depth", 0)), float(e.get("probability", 100)), e.get("constant", False))


def do_new_entry():
    if not _st.current_lorebook:
        return (gr.update(), gr.update(value="❌ Load or create a lorebook first."), *[gr.update()] * 15)
    with _st.state_lock:
        e = {
            "uid": _st.next_uid, "enabled": True, "constant": False,
            "comment": f"New Entry {_st.next_uid}",
            "keys": [], "secondary_keys": [], "selective_logic": "AND ANY", "content": "",
            "case_sensitive": False, "match_whole_words": True, "use_regex": False,
            "priority": 10, "position": "after_context", "scan_depth": 0,
            "probability": 100, "inclusion_group": "",
        }
        _st.next_uid += 1
        _st.current_lorebook["entries"].append(e)
        _sync_active_lorebook()
    choices = _entry_choices()
    return (
        gr.update(choices=choices, value=choices[-1] if choices else None),
        gr.update(value="New entry added! Fill in the fields and click **Save entry**."),
        gr.update(value=e["comment"]),
        gr.update(value=""), gr.update(value=""), gr.update(value=""),
        gr.update(value="AND ANY"), gr.update(value=""),
        gr.update(value=True), gr.update(value=False), gr.update(value=True), gr.update(value=False),
        gr.update(value=10), gr.update(value="after_context"),
        gr.update(value=0), gr.update(value=100), gr.update(value=False),
    )


def do_clone_entry(choice):
    if not choice or not _st.current_lorebook:
        return (gr.update(), gr.update(value="❌ No entry selected to clone."), *[gr.update()] * 15)
    uid = _uid_from_choice(choice)
    idx = _idx_from_uid(uid) if uid is not None else -1
    if idx < 0:
        return (gr.update(), gr.update(value="❌ Entry not found."), *[gr.update()] * 15)
    with _st.state_lock:
        clone = copy.deepcopy(_st.current_lorebook["entries"][idx])
        clone["uid"] = _st.next_uid
        _st.next_uid += 1
        clone["comment"] = (clone.get("comment", "") + " (copy)").strip()
        _st.current_lorebook["entries"].append(clone)
        _sync_active_lorebook()
    choices = _entry_choices()
    new_choice = next((c for c in choices if f"[{clone['uid']}]" in c), None)
    return (
        gr.update(choices=choices, value=new_choice),
        gr.update(value=f"✅ Cloned as **{clone['comment']}**. Edit fields then click **Save entry**."),
        gr.update(value=clone["comment"]),
        gr.update(value=", ".join(clone.get("keys", []))),
        gr.update(value=", ".join(clone.get("secondary_keys", []))),
        gr.update(value=clone.get("content", "")),
        gr.update(value=clone.get("selective_logic", "AND ANY")),
        gr.update(value=clone.get("inclusion_group", "")),
        gr.update(value=clone.get("enabled", True)),
        gr.update(value=clone.get("case_sensitive", False)),
        gr.update(value=clone.get("match_whole_words", True)),
        gr.update(value=clone.get("use_regex", False)),
        gr.update(value=float(clone.get("priority", 10))),
        gr.update(value=clone.get("position", "after_context")),
        gr.update(value=float(clone.get("scan_depth", 0))),
        gr.update(value=float(clone.get("probability", 100))),
        gr.update(value=clone.get("constant", False)),
    )


def do_delete_entry(choice):
    if not choice or not _st.current_lorebook:
        return (gr.update(), gr.update(value="❌ No entry selected."), *[gr.update()] * 15)
    uid = _uid_from_choice(choice)
    idx = _idx_from_uid(uid) if uid is not None else -1
    if idx < 0:
        return (gr.update(), gr.update(value="❌ Entry not found."), *[gr.update()] * 15)
    with _st.state_lock:
        _st.current_lorebook["entries"].pop(idx)
        _sync_active_lorebook()
    return (
        gr.update(choices=_entry_choices(), value=None),
        gr.update(value="Entry removed."),
        gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value=""),
        gr.update(value="AND ANY"), gr.update(value=""),
        gr.update(value=True), gr.update(value=False), gr.update(value=True), gr.update(value=False),
        gr.update(value=10), gr.update(value="after_context"),
        gr.update(value=0), gr.update(value=100), gr.update(value=False),
    )


def do_save_entry(choice, comment, keys_str, sec_keys_str, content, selective_logic,
                  inclusion_group, enabled, case_sensitive, whole_words, use_regex,
                  priority, position, scan_depth, probability, constant):
    if not _st.current_lorebook:
        return gr.update(value="❌ No lorebook loaded."), gr.update()
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    sec_keys = [k.strip() for k in sec_keys_str.split(",") if k.strip()]
    uid = _uid_from_choice(choice) if choice else None
    idx = _idx_from_uid(uid) if uid is not None else -1
    with _st.state_lock:
        entry_data = {
            "uid": uid if idx >= 0 else _st.next_uid,
            "enabled": bool(enabled), "constant": bool(constant),
            "comment": comment.strip(), "keys": keys, "secondary_keys": sec_keys,
            "selective_logic": selective_logic, "content": content.strip(),
            "case_sensitive": bool(case_sensitive), "match_whole_words": bool(whole_words),
            "use_regex": bool(use_regex), "priority": int(priority), "position": position,
            "scan_depth": int(scan_depth), "probability": int(probability),
            "inclusion_group": (inclusion_group or "").strip(),
        }
        if idx >= 0:
            _st.current_lorebook["entries"][idx] = entry_data
            status = "✅ Entry updated! *(Remember to also **Save** the lorebook to write to disk.)*"
        else:
            _st.next_uid += 1
            _st.current_lorebook["entries"].append(entry_data)
            status = "✅ New entry saved! *(Remember to also **Save** the lorebook to write to disk.)*"
        _sync_active_lorebook()
    choices = _entry_choices()
    new_choice = next((c for c in choices if f"[{entry_data['uid']}]" in c), None)
    return gr.update(value=status), gr.update(choices=choices, value=new_choice)


def do_filter_entries(filter_text, current_choice):
    all_choices = _entry_choices()
    if not filter_text or not filter_text.strip():
        keep = current_choice if current_choice in all_choices else None
        return gr.update(choices=all_choices, value=keep)
    query = filter_text.strip().lower()
    filtered = [c for c in all_choices if query in c.lower()]
    keep = current_choice if current_choice in filtered else None
    return gr.update(choices=filtered, value=keep)


def do_test_match(choice, test_text):
    if not choice or not _st.current_lorebook:
        return gr.update(value="*Select an entry first.*")
    if not (test_text or "").strip():
        return gr.update(value="*Paste some sample text above and click **Test**.*")
    uid = _uid_from_choice(choice)
    idx = _idx_from_uid(uid) if uid is not None else -1
    if idx < 0:
        return gr.update(value="❌ Entry not found.")
    e = _st.current_lorebook["entries"][idx]

    if e.get("constant"):
        return gr.update(value="🔵 **Constant** — this entry always fires; no keyword check needed.")
    keys = e.get("keys", [])
    if not keys:
        return gr.update(value="⚠ This entry has **no trigger words** — it will never fire from keywords.")

    hit_keys = [k for k in keys if _hit_key(k, test_text, e)]
    if not hit_keys:
        tried = ", ".join(f"`{k}`" for k in keys[:8])
        extra = f" *(+{len(keys)-8} more)*" if len(keys) > 8 else ""
        return gr.update(value=f"❌ **No match.** Keys tried: {tried}{extra}")

    hit_str = ", ".join(f"`{k}`" for k in hit_keys)
    sec_pass = _secondary_keys_pass(e, test_text)
    if not sec_pass:
        logic = e.get("selective_logic", "AND ANY")
        sec_keys = e.get("secondary_keys", [])
        return gr.update(
            value=f"⚠ Primary matched ({hit_str}) but **secondary check failed** "
                  f"(`{logic}` across {len(sec_keys)} secondary key{'s' if len(sec_keys) != 1 else ''})."
        )

    prob = e.get("probability", 100)
    prob_note = f"  \n*(Trigger probability is **{prob}%** — fires randomly {prob}% of the time.)*" if prob < 100 else ""
    return gr.update(value=f"✅ **Match!** Keys fired: {hit_str}.{prob_note}")


def do_overview():
    if not _st.current_lorebook:
        return gr.update(value="*No lorebook loaded.*")
    entries = _st.current_lorebook.get("entries", [])
    if not entries:
        return gr.update(value="*No entries yet — click **+ Add** to create one.*")
    lines = ["| ID | | Name | Trigger words | Pri | Prob | Group |",
             "|----|--|------|---------------|-----|------|-------|"]
    for e in entries:
        st = ("🔵" if e.get("constant") else "✅") if e.get("enabled", True) else "❌"
        keys = ("CONSTANT" if e.get("constant") else ", ".join(e.get("keys", [])) or "—")[:40]
        lines.append(
            f"| {e.get('uid','?')} | {st} | {(e.get('comment','') or '—')[:30]} "
            f"| {keys} | {e.get('priority',10)} | {e.get('probability',100)}% "
            f"| {(e.get('inclusion_group') or '—')[:18]} |"
        )
    return gr.update(value="\n".join(lines))


def _make_preview_table(info, notebook=False):
    if not info["entries"]:
        msg = "use the Notebook tab first" if notebook else "send a message first"
        return f"*No injection recorded yet — {msg}.*"
    total = len(info["entries"])
    lines = ["| # | Entry | Pri | Tokens |", "|---|-------|-----|--------|"]
    for i, (eid, label, toks, prio) in enumerate(info["entries"]):
        if total == 1:
            pos = "#1"
        elif i == 0:
            pos = "#1 *(oldest)*"
        elif i == total - 1:
            pos = f"#{total} *(freshest)*"
        else:
            pos = f"#{i + 1}"
        lines.append(f"| {pos} | {label} | p{prio} | {toks} |")
    suffix = f"\n**Total: ~{info['total_tokens']} tokens**"
    if not notebook:
        suffix += f" | **Interrupts: {info['interrupts']}**"
    lines.append(suffix)
    return "\n".join(lines)


def do_preview_refresh():
    with _st.last_injection_lock:
        info = dict(_st.last_injection_info)
    return (
        gr.update(value=_make_preview_table(info, notebook=False)),
        gr.update(value=_build_history_html(notebook=False)),
        gr.update(value=_budget_bar_html()),
    )


def do_preview_refresh_notebook():
    with _st.last_notebook_injection_lock:
        info = dict(_st.last_notebook_injection_info)
    return (
        gr.update(value=_make_preview_table(info, notebook=True)),
        gr.update(value=_build_history_html(notebook=True)),
        gr.update(value=_budget_bar_html()),
    )


def do_clear_history():
    with _st.injection_history_lock:
        _st.injection_history.clear()
    with _st.label_tracking_lock:
        _st.prev_chat_labels     = set()
        _st.all_chat_labels      = set()
        _st.prev_chat_eid_labels = {}
        _st.chat_turn_counter    = 0
    return gr.update(value=_build_history_html(notebook=False))


def do_clear_history_notebook():
    with _st.injection_history_lock:
        _st.injection_history_notebook.clear()
    with _st.label_tracking_lock:
        _st.prev_notebook_labels     = set()
        _st.all_notebook_labels      = set()
        _st.prev_notebook_eid_labels = {}
        _st.notebook_turn_counter    = 0
    return gr.update(value=_build_history_html(notebook=True))


def do_st_import_preview(file_obj):
    _hidden = gr.update(visible=False)
    if file_obj is None:
        return (gr.update(value="❌ No file selected."),
                _hidden, None, _hidden, _hidden)
    try:
        file_path = getattr(file_obj, "name", file_obj)
        raw = Path(file_path).read_bytes()
    except Exception as exc:
        return (gr.update(value=f"❌ Could not read file: {exc}"),
                _hidden, None, _hidden, _hidden)
    lb, err = import_from_sillytavern(raw)
    if err:
        return (gr.update(value=f"❌ {err}"),
                _hidden, None, _hidden, _hidden)

    stats = lb.get("_import_stats", {})
    preview_html = _build_import_preview_html(lb, stats)
    return (
        gr.update(value=""),
        gr.update(value=preview_html, visible=True),
        lb,
        gr.update(visible=True),
        gr.update(visible=True),
    )


def do_st_import_confirm(pending):
    _hidden = gr.update(visible=False)
    if not pending:
        return (gr.update(value="❌ Nothing to import — upload a file first."),
                _hidden, None, _hidden, _hidden,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    lb = pending
    stem = base_stem = _safe_stem(lb.get("name", "imported"))
    counter = 1
    while (LOREBOOKS_DIR / f"{stem}.json").exists():
        stem = f"{base_stem}_{counter}"
        counter += 1

    clean_lb = {k: v for k, v in lb.items() if not k.startswith("_")}
    stats = lb.get("_import_stats", {})
    if not save_lorebook_file(stem, clean_lb):
        return (gr.update(value="❌ Could not save — check folder permissions."),
                _hidden, None, _hidden, _hidden,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    with _st.state_lock:
        _st.current_lorebook = clean_lb
        params["_st.current_lorebook"] = stem
        _st.next_uid = max((e.get("uid", 0) for e in clean_lb.get("entries", [])), default=0) + 1

    n = stats.get("total", len(clean_lb.get("entries", [])))
    n_const = stats.get("constant", 0)
    n_nokey = stats.get("no_keys", 0)
    notes = []
    if n_const:
        notes.append(f"{n_const} constant entr{'y' if n_const == 1 else 'ies'} (always-on 🔵)")
    if n_nokey:
        notes.append(f"{n_nokey} entr{'y' if n_nokey == 1 else 'ies'} with no trigger words — add keys to activate")
    msg = f"✅ Imported **{clean_lb.get('name', stem)}** — {n} entr{'y' if n == 1 else 'ies'} saved as **{stem}.json**."
    if notes:
        msg += "  \n\n**Note:** " + ";  ".join(notes) + "."

    files = get_lorebook_files()
    with _st.state_lock:
        active_keys = list(_st.active_lorebooks.keys())
    return (
        gr.update(value=msg),
        gr.update(value="", visible=False),
        None,
        _hidden,
        _hidden,
        gr.update(choices=files, value=stem),
        gr.update(choices=files, value=active_keys),
        gr.update(value=clean_lb.get("name", stem)),
        gr.update(value=clean_lb.get("description", "")),
        gr.update(choices=_entry_choices(), value=None),
    )


def do_st_import_cancel():
    _hidden = gr.update(visible=False)
    return (
        gr.update(value="*Import cancelled.*"),
        gr.update(value="", visible=False),
        None,
        _hidden,
        _hidden,
    )


def do_st_export():
    if not _st.current_lorebook:
        return gr.update(value="❌ No lorebook loaded. Load one first."), gr.update(visible=False)
    raw = export_to_sillytavern(_st.current_lorebook)
    stem = _safe_stem(_st.current_lorebook.get("name", "export"))
    out = Path(tempfile.gettempdir()) / f"{stem}_sillytavern.json"
    out.write_bytes(raw)
    n = len(_st.current_lorebook.get("entries", []))
    return (gr.update(value=f"✅ Exported **{_st.current_lorebook.get('name', stem)}** — {n} entr{'y' if n == 1 else 'ies'}."),
            gr.update(value=str(out), visible=True))


def set_activate(x):
    params["activate"] = x
    _save_params()
    return gr.update(value=_build_stats_html())


def set_position_override(enabled):
    params["position_override_enabled"] = enabled
    _save_params()
    return gr.update(interactive=enabled)


def safe_int(key, x):
    try:
        params[key] = int(x)
        _save_params()
    except (TypeError, ValueError):
        pass


def set_param(key, x):
    params[key] = x
    _save_params()


def do_force_summary():
    try:
        from . import summary as _summary_mod
    except ImportError as exc:
        yield gr.update(value=f"❌ summary.py not available: {exc}")
        return
    yield gr.update(value="⏳ Generating story summary…")
    _summary_mod.force_summary()
    yield gr.update(value=_summary_mod.summary_status())


def do_refresh_summary_status():
    try:
        from .summary import summary_status
        return gr.update(value=summary_status())
    except ImportError:
        return gr.update(value="summary.py not available.")
