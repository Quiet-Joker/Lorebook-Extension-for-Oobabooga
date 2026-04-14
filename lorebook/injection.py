import datetime
import logging
import re

from .config import params, _ORIG_CTX, _HIST_MSGS, _count_tokens
from .state import _st
from .matching import (
    _all_active_entries,
    _eid,
    _eff_pos,
    _find_active_matches,
    _gather_messages_list,
    _hit_key,
    _secondary_keys_pass,
    _prob_pass,
    _apply_inclusion_groups,
    _sort_by_priority,
    _trim_to_budget,
    _entry_short_label,
)

logger = logging.getLogger(__name__)


def _ctx_key(state):
    return "custom_system_message" if state.get("mode") == "instruct" else "context"


def _get_active_keys():
    with _st.state_lock:
        return list(_st.active_lorebooks.keys())


def _sync_active_lorebook():
    stem = params.get("_st.current_lorebook", "")
    if stem and stem in _st.active_lorebooks and _st.current_lorebook is not None:
        _st.active_lorebooks[stem] = _st.current_lorebook
        from .state import _bump_active_version
        _bump_active_version()


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


def _insert_block_before_trailing(ctx: str, block: str) -> str:
    stripped = ctx.rstrip("\n")
    trailing = ctx[len(stripped):]
    last_nl = stripped.rfind("\n")
    if last_nl != -1:
        return stripped[:last_nl] + "\n" + block + stripped[last_nl:] + trailing
    return block + "\n" + stripped + trailing


def _replace_world_info_block(prompt, all_entries):
    pref = params["injection_prefix"]
    suf = params["injection_suffix"]

    if not pref or not suf:
        return prompt, []

    trimmed = list(reversed(_trim_to_budget(_sort_by_priority(all_entries))))
    before_entries       = [e for e in trimmed if _eff_pos(e) == "before_context"]
    after_entries        = [e for e in trimmed if _eff_pos(e) == "after_context"]
    before_reply_entries = [e for e in trimmed if _eff_pos(e) == "before_reply"]
    notebook_entries     = [e for e in trimmed if _eff_pos(e) == "notebook"]

    clean_prompt = _strip_wi_block(prompt)
    ctx = clean_prompt
    if before_entries:
        block = (pref + _format_injection(before_entries) + suf).rstrip("\n") + "\n"
        ctx = block + ctx if ctx else block.lstrip("\n")
    if after_entries:
        block = (pref + _format_injection(after_entries) + suf).rstrip("\n")
        ctx = _insert_block_before_trailing(ctx, block)
    if before_reply_entries:
        block = (pref + _format_injection(before_reply_entries) + suf).rstrip("\n")
        ctx = _insert_block_before_trailing(ctx, block)
    if notebook_entries:
        block = (pref + _format_injection(notebook_entries) + suf).rstrip("\n")
        ctx = _insert_block_before_trailing(ctx, block) if ctx.rstrip("\n") != ctx else block + "\n" + ctx

    return ctx, trimmed


def _update_injection_preview(all_fired, interrupt_count, notebook=False, budget_dropped=None):
    rows = []
    total_tokens = 0
    eids_this_turn: set = set()
    eid_to_label: dict = {}

    for e in all_fired:
        eid = _eid(e)
        label = _entry_short_label(e)
        toks = _count_tokens(e.get("content", ""))
        total_tokens += toks
        rows.append((eid, label, toks, e.get("priority", 0)))
        eids_this_turn.add(eid)
        eid_to_label[eid] = label

    with _st.label_tracking_lock:
        prev_eids = _st.prev_notebook_labels if notebook else _st.prev_chat_labels
        all_eids = (_st.all_notebook_labels if notebook else _st.all_chat_labels).copy()

        entry_records = []
        for eid, label, tokens, priority in rows:
            if eid not in all_eids:
                status = "new"
            elif eid in prev_eids:
                status = "repeat"
            else:
                status = "returned"
            entry_records.append({"label": label, "tokens": tokens, "status": status, "priority": priority})

        budget_dropped_records = []
        if budget_dropped:
            for e in budget_dropped:
                eid = _eid(e)
                if eid in eids_this_turn:
                    continue
                label = _entry_short_label(e)
                tokens = _count_tokens(e.get("content", ""))
                budget_dropped_records.append({"label": label, "tokens": tokens, "status": "budget_dropped", "priority": e.get("priority", 0)})

        prev_eid_labels = _st.prev_notebook_eid_labels if notebook else _st.prev_chat_eid_labels
        dropped = [prev_eid_labels.get(eid, str(eid)) for eid in prev_eids if eid not in eids_this_turn]

        all_eids.update(eids_this_turn)
        if notebook:
            _st.all_notebook_labels = all_eids
            _st.prev_notebook_labels = eids_this_turn
            _st.prev_notebook_eid_labels = eid_to_label
            _st.notebook_turn_counter += 1
            turn_num = _st.notebook_turn_counter
        else:
            _st.all_chat_labels = all_eids
            _st.prev_chat_labels = eids_this_turn
            _st.prev_chat_eid_labels = eid_to_label
            _st.chat_turn_counter += 1
            turn_num = _st.chat_turn_counter

    record = {
        "turn":           turn_num,
        "time":           datetime.datetime.now().strftime("%H:%M:%S"),
        "entries":        entry_records,
        "budget_dropped": budget_dropped_records,
        "dropped":        dropped,
        "total_tokens":   total_tokens,
        "interrupts":     interrupt_count,
    }

    if notebook:
        with _st.last_notebook_injection_lock:
            _st.last_notebook_injection_info["entries"] = rows
            _st.last_notebook_injection_info["interrupts"] = interrupt_count
            _st.last_notebook_injection_info["total_tokens"] = total_tokens
        with _st.injection_history_lock:
            _st.injection_history_notebook.appendleft(record)
    else:
        with _st.last_injection_lock:
            _st.last_injection_info["entries"] = rows
            _st.last_injection_info["interrupts"] = interrupt_count
            _st.last_injection_info["total_tokens"] = total_tokens
        with _st.injection_history_lock:
            _st.injection_history.appendleft(record)


def _do_wi_injection(scan_text, history_msgs, state):
    orig_ctx = state.get(_ORIG_CTX)
    ctx_k = _ctx_key(state)
    if orig_ctx is None:
        orig_ctx = _strip_wi_block(state.get(ctx_k, ""))
        state[_ORIG_CTX] = orig_ctx

    if params.get("chat_only_scan") and orig_ctx:
        scan_text = scan_text.replace(orig_ctx, "").strip()
    matched, all_matched = _find_active_matches(scan_text, history_msgs)
    matched = [e for e in matched if _eff_pos(e) != "notebook"]
    all_matched = [e for e in all_matched if _eff_pos(e) != "notebook"]
    if matched:
        before_reply_entries = [e for e in matched if _eff_pos(e) == "before_reply"]
        non_reply = [e for e in matched if _eff_pos(e) != "before_reply"]
        if non_reply:
            new_ctx, _ = _replace_world_info_block(orig_ctx, non_reply)
        else:
            new_ctx = orig_ctx
        state[ctx_k] = new_ctx
        state["_lb_before_reply_entries"] = before_reply_entries
    else:
        state[ctx_k] = orig_ctx
        state["_lb_before_reply_entries"] = []
    with _st.cur_injected_lock:
        _st.cur_injected = {_eid(e) for e in matched} if matched else set()
    return matched, all_matched


def state_modifier(state):
    state = dict(state)
    ctx_k = _ctx_key(state)
    if _ORIG_CTX not in state:
        state[_ORIG_CTX] = _strip_wi_block(state.get(ctx_k, ""))

    state[_HIST_MSGS] = _gather_messages_list(state.get("history", {}), state[_ORIG_CTX])

    try:
        from .summary import capture_state_for_force
        capture_state_for_force(state)
    except Exception:
        logger.debug("Auto-summary: capture_state_for_force failed", exc_info=True)

    if params.get("auto_summary_enabled") and params.get("activate"):
        try:
            from .summary import _ensure_summary_lorebook, _conv_key
            char_stem, conv_id = _conv_key(state)
            if conv_id:
                _ensure_summary_lorebook(char_stem, conv_id)
        except Exception:
            logger.exception("Auto-summary lorebook registration failed in state_modifier")

    if not params["activate"] or not _st.active_lorebooks:
        return state
    internal = state.get("history", {}).get("internal", [])
    if not internal:
        return state
    last_user = internal[-1][0]
    if not last_user or last_user == "<|BEGIN-VISIBLE-CHAT|>":
        return state

    _regen_matched, _regen_all = _do_wi_injection(last_user, state[_HIST_MSGS], state)
    state["_lb_chat_matched"]     = _regen_matched or []
    state["_lb_chat_all_matched"] = _regen_all     or []
    return state


def chat_input_modifier(text, visible_text, state):
    if not params["activate"] or not _st.active_lorebooks:
        with _st.cur_injected_lock:
            _st.cur_injected = set()
        return text, visible_text

    history_msgs = state.get(_HIST_MSGS, _gather_messages_list(state.get("history", {}), state.get(_ORIG_CTX)))
    matched, all_matched = _do_wi_injection(text or "", history_msgs, state)

    state["_lb_chat_matched"]     = matched or []
    state["_lb_chat_all_matched"] = all_matched or []

    return text, visible_text


def _find_new_trigger_entries(text, already_injected, notebook=False):
    newly = []
    for e in _all_active_entries():
        eid = _eid(e)
        if eid in already_injected or not e.get("enabled", True) or e.get("constant"):
            continue
        pos = _eff_pos(e)
        if notebook and pos != "notebook":
            continue
        if not notebook and pos == "notebook":
            continue
        keys = e.get("keys", [])
        if not keys:
            continue
        if not _prob_pass(e):
            continue
        if any(_hit_key(k, text, e) for k in keys):
            if not _secondary_keys_pass(e, text):
                continue
            newly.append(e)
    newly = _apply_inclusion_groups(newly)
    newly = _sort_by_priority(newly)
    return _trim_to_budget(newly)


def _find_notebook_matches(question_text):
    all_entries = _all_active_entries()
    notebook_candidates = [e for e in all_entries if _eff_pos(e) == "notebook"]
    seen_eids_nb = set()
    matched = []

    for e in notebook_candidates:
        eid = _eid(e)
        if eid in seen_eids_nb or not e.get("enabled", True):
            continue
        if not _prob_pass(e):
            continue
        if e.get("constant"):
            seen_eids_nb.add(eid)
            matched.append(e)
            continue
        keys = e.get("keys", [])
        if not keys:
            continue
        if any(_hit_key(k, question_text, e) for k in keys):
            if not _secondary_keys_pass(e, question_text):
                continue
            seen_eids_nb.add(eid)
            matched.append(e)

    matched = _apply_inclusion_groups(matched)
    matched = _sort_by_priority(matched)
    trimmed = _trim_to_budget(matched)
    return list(reversed(trimmed)), matched


def custom_generate_reply(question, original_question, state,
                          stopping_strings=None, is_chat=False):
    import modules.shared as shared
    from modules.text_generation import generate_reply_HF, generate_reply_custom

    _CUSTOM_BACKENDS = {
        "LlamaServer",
        "Exllamav2Model",
        "Exllamav3Model",
        "TensorRTLLMModel",
        "CtransformersModel",
        "RWKVModel",
        "llamacppmodel",
        "LlamaCppModel",
    }

    def _base_gen(q, oq, st):
        if shared.model.__class__.__name__ in _CUSTOM_BACKENDS:
            yield from generate_reply_custom(q, oq, st, stopping_strings, is_chat=is_chat)
        else:
            try:
                yield from generate_reply_HF(q, oq, st, stopping_strings, is_chat=is_chat)
            except ModuleNotFoundError as _e:
                if "torch" in str(_e):
                    logger.warning(
                        "torch not found for model class %r — falling back to "
                        "generate_reply_custom.  Add this class name to "
                        "_CUSTOM_BACKENDS in injection.py if generation fails.",
                        shared.model.__class__.__name__,
                    )
                    yield from generate_reply_custom(q, oq, st, stopping_strings, is_chat=is_chat)
                else:
                    raise

    _gen_mode = state.get("mode", "chat")
    if params.get("auto_summary_enabled") and params.get("activate") and (is_chat or _gen_mode == "instruct"):
        try:
            from .summary import should_summarise, run_auto_summary
            if should_summarise(state):
                yield "*(📖 Summarizing story so far… please wait.)*"
                run_auto_summary(state)
        except Exception:
            logger.exception("Auto-summary hook failed in custom_generate_reply")

    nb_for_interrupt = []
    if params["activate"] and _st.active_lorebooks:
        pref = params["injection_prefix"]
        suf  = params["injection_suffix"]
        if pref and suf:
            if not is_chat:
                nb_matched, nb_all_matched = _find_notebook_matches(question)
                nb_for_interrupt = nb_matched
                if nb_matched:
                    block = (pref + _format_injection(nb_matched) + suf).rstrip("\n")
                    question = _insert_block_before_trailing(question, block) if question.rstrip("\n") != question else block + "\n" + question
                    original_question = question

                    with _st.cur_injected_lock:
                        _st.cur_injected = {_eid(e) for e in nb_matched}

                    injected_eids = {_eid(e) for e in nb_matched}
                    nb_budget_dropped = [e for e in nb_all_matched if _eid(e) not in injected_eids]
                    _update_injection_preview(nb_matched, 0, notebook=True, budget_dropped=nb_budget_dropped)
                else:
                    with _st.cur_injected_lock:
                        _st.cur_injected = set()
                    _update_injection_preview(
                        [], 0, notebook=True,
                        budget_dropped=nb_all_matched if nb_all_matched else []
                    )
            else:
                before_reply_entries = state.get("_lb_before_reply_entries", [])
                if before_reply_entries:
                    block = (pref + _format_injection(before_reply_entries) + suf).rstrip("\n")
                    question = _insert_block_before_trailing(question, block)
                    original_question = question

    if not params["activate"] or not params["mid_gen_interrupt"] or not _st.active_lorebooks:
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
        with _st.cur_injected_lock:
            already = set(_st.cur_injected)
        all_injected_entries = [e for e in _all_active_entries() if _eid(e) in already]
    else:
        already = {_eid(e) for e in nb_for_interrupt}
        all_injected_entries = list(nb_for_interrupt)
    cur_question = question
    cur_original = original_question
    cumulative_text = ""
    gen_offset = ""
    interrupts = 0
    word_buf = ""
    last_trimmed = list(all_injected_entries)

    try:
        while True:
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
                    m_tail   = re.search(r'\w+$', word_buf)
                    scan_buf = word_buf[:m_tail.start()] if m_tail else word_buf
                    pending  = word_buf[m_tail.start():] if m_tail else ""
                    new_entries = _find_new_trigger_entries(scan_buf, already, notebook=not is_chat) if scan_buf else []
                    if new_entries:
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
                        word_buf = word_buf[-8000:]
            if not interrupted:
                break
    except (GeneratorExit, KeyboardInterrupt):
        with _st.cur_injected_lock:
            _st.cur_injected = set()
        raise

    injected_eids = {_eid(e) for e in last_trimmed}
    budget_dropped = [e for e in all_injected_entries if _eid(e) not in injected_eids]
    _update_injection_preview(last_trimmed, interrupts, notebook=not is_chat, budget_dropped=budget_dropped)
    yield cumulative_text
