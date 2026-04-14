import copy
import hashlib
import json
import logging
import threading

from .config import params, _count_tokens, LOREBOOKS_DIR, TEMPLATES_DIR
from .state import _st, _bump_active_version
from .storage import (
    _safe_stem,
    load_lorebook,
    save_lorebook_file,
    _save_active_state,
)

logger = logging.getLogger(__name__)

_AUTO_SUMMARY_LABEL   = "📖 Story Summary"
_SUMMARIES_STEM       = "_summaries"
_SUMMARY_ENTRY_PREFIX = "📖 Story Summary — "

_CUSTOM_BACKENDS: frozenset = frozenset({
    "LlamaServer",
    "Exllamav2Model",
    "Exllamav3Model",
    "TensorRTLLMModel",
    "CtransformersModel",
    "RWKVModel",
    "llamacppmodel",
    "LlamaCppModel",
})

_DEFAULT_FULL_PROMPT = (
    "You are a meticulous story chronicler. Your task is to write a detailed, "
    "chronological summary of an ongoing collaborative roleplay story based on "
    "the conversation transcript below.\n\n"
    "Your summary MUST include, in order:\n"
    "1. The setting and situation at the start of the story.\n"
    "2. Every significant event that has occurred, listed chronologically.\n"
    "3. Every decision made by the user/player and the direct consequences of "
    "those decisions.\n"
    "4. Key character moments: emotions, relationships, rivalries, alliances.\n"
    "5. Important world details, locations, and lore that have been established.\n"
    "6. Any unresolved plot threads, mysteries, or pending choices.\n\n"
    "Rules:\n"
    "- Write in plain, dense prose. No bullet points, no headers.\n"
    "- Be specific — use character names, place names, and exact events.\n"
    "- Do NOT editorialize or add your own commentary.\n"
    "- Do NOT start with 'Sure', 'Here is', or any preamble. "
    "Begin the summary immediately.\n\n"
    "--- STORY TRANSCRIPT ---\n"
    "{conversation}\n"
    "--- END TRANSCRIPT ---\n\n"
    "Detailed chronological story summary:"
)

_DEFAULT_DELTA_PROMPT = (
    "You are a meticulous story chronicler maintaining a running summary of "
    "an ongoing collaborative roleplay.\n\n"
    "Below is the EXISTING summary of the story up to a certain point, "
    "followed by the NEW events that have happened since then.\n\n"
    "Your task: rewrite the summary to incorporate the new events. "
    "Preserve everything important from the existing summary. "
    "Add the new events in chronological order at the end.\n\n"
    "Include:\n"
    "- Every significant event in chronological order.\n"
    "- Every decision made by the user/player and its consequences.\n"
    "- Key character moments, relationships, and emotional beats.\n"
    "- Important world details, lore, and locations established.\n"
    "- All unresolved plot threads, mysteries, and pending choices.\n\n"
    "Rules:\n"
    "- Write in plain, dense prose. No bullet points, no headers.\n"
    "- Be specific — use character names, place names, exact events.\n"
    "- Do NOT editorialize. Do NOT start with 'Sure' or any preamble.\n"
    "- Begin the updated summary immediately.\n\n"
    "--- EXISTING SUMMARY ---\n"
    "{previous_summary}\n"
    "--- END EXISTING SUMMARY ---\n\n"
    "--- NEW EVENTS ---\n"
    "{new_events}\n"
    "--- END NEW EVENTS ---\n\n"
    "Updated story summary:"
)

_STRUCTURED_SHORTHAND_FULL_PROMPT = (
    "You are a story tracker. Summarize the roleplay transcript below using ONLY "
    "these labeled tags — one line each, terse and specific. No prose, no preamble.\n\n"
    "[SETTING] Current location and situation in one line.\n"
    "[EVENTS] Chronological chain of key events, each prefixed with →\n"
    "[CHARACTERS] Each significant character — name, role, current status/emotional state.\n"
    "[TENSION] The central conflict or power dynamic right now.\n"
    "[OPEN] Unresolved threads, pending choices, unanswered questions.\n\n"
    "Rules:\n"
    "- Every item must be specific: use character names, places, exact actions.\n"
    "- No filler words. No sentences that could be cut without losing information.\n"
    "- Do NOT start with 'Sure', 'Here is', or any preamble. "
    "Output the tags immediately.\n\n"
    "--- STORY TRANSCRIPT ---\n"
    "{conversation}\n"
    "--- END TRANSCRIPT ---\n\n"
    "[SETTING]"
)

_STRUCTURED_SHORTHAND_DELTA_PROMPT = (
    "You are a story tracker maintaining a running shorthand log.\n\n"
    "Update the existing summary below by incorporating the new events. "
    "Keep all five tags. Replace or extend each tag's content as needed — "
    "preserve important history, add new developments.\n\n"
    "Rules:\n"
    "- Keep each tag to one line (EVENTS may be a chain of → items).\n"
    "- Specific names and actions only. No filler.\n"
    "- Do NOT start with 'Sure' or any preamble. "
    "Output the updated tags immediately.\n\n"
    "--- EXISTING SUMMARY ---\n"
    "{previous_summary}\n"
    "--- END EXISTING SUMMARY ---\n\n"
    "--- NEW EVENTS ---\n"
    "{new_events}\n"
    "--- END NEW EVENTS ---\n\n"
    "[SETTING]"
)

_MARKDOWN_BULLETS_FULL_PROMPT = (
    "You are a story chronicler. Summarize the roleplay transcript below using "
    "the exact markdown structure shown. Be specific — use character names, "
    "places, and exact events. No prose paragraphs. No preamble.\n\n"
    "**Setting:** One-line description of current location and situation.\n\n"
    "**Events:**\n"
    "- [chronological event]\n"
    "- [chronological event]\n\n"
    "**Characters:**\n"
    "- [Name] — role, current state or relationship to protagonist\n\n"
    "**Tension:** One line on the central conflict or power dynamic.\n\n"
    "**Unresolved:**\n"
    "- [open thread or pending choice]\n\n"
    "Rules:\n"
    "- Every bullet must be specific and earned — cut anything vague.\n"
    "- Do NOT start with 'Sure', 'Here is', or any preamble.\n\n"
    "--- STORY TRANSCRIPT ---\n"
    "{conversation}\n"
    "--- END TRANSCRIPT ---\n\n"
    "**Setting:**"
)

_MARKDOWN_BULLETS_DELTA_PROMPT = (
    "You are a story chronicler maintaining a running summary.\n\n"
    "Rewrite the summary below to incorporate the new events. "
    "Keep the exact same markdown structure. "
    "Update each section as needed — preserve important history, "
    "add new developments in chronological order.\n\n"
    "Rules:\n"
    "- Specific names, places, and actions only.\n"
    "- Do NOT start with 'Sure' or any preamble. "
    "Output the updated summary immediately.\n\n"
    "--- EXISTING SUMMARY ---\n"
    "{previous_summary}\n"
    "--- END EXISTING SUMMARY ---\n\n"
    "--- NEW EVENTS ---\n"
    "{new_events}\n"
    "--- END NEW EVENTS ---\n\n"
    "**Setting:**"
)

# ---------------------------------------------------------------------------
# Template file helpers
# ---------------------------------------------------------------------------
_TEMPLATE_SEPARATOR = "\n===DELTA PROMPT===\n"

_DEFAULT_TEMPLATE_FILES: dict = {
    "Dense Prose": (_DEFAULT_FULL_PROMPT, _DEFAULT_DELTA_PROMPT),
    "Structured Shorthand [TAGS]": (_STRUCTURED_SHORTHAND_FULL_PROMPT, _STRUCTURED_SHORTHAND_DELTA_PROMPT),
    "Markdown Bullets": (_MARKDOWN_BULLETS_FULL_PROMPT, _MARKDOWN_BULLETS_DELTA_PROMPT),
}


def _write_default_templates() -> None:
    """Seed the templates folder with the built-in presets (skips files that already exist)."""
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    for name, (full, delta) in _DEFAULT_TEMPLATE_FILES.items():
        path = TEMPLATES_DIR / f"{name}.txt"
        if not path.exists():
            try:
                path.write_text(full + _TEMPLATE_SEPARATOR + delta, encoding="utf-8")
            except Exception:
                logger.exception("Auto-summary: could not write default template %r", name)


def get_template_names() -> list:
    """Return a sorted list of template names (file stems) from the templates folder."""
    if not TEMPLATES_DIR.exists():
        return []
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.txt"))


def load_template(name: str):
    """Load a template by name. Returns (full_prompt, delta_prompt) or None."""
    path = TEMPLATES_DIR / f"{name}.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Auto-summary: could not read template %r", name)
        return None
    parts = text.split(_TEMPLATE_SEPARATOR, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # Fallback: treat the whole file as the full prompt, empty delta
    return text.strip(), ""


_AUTO_SUMMARY_PARAM_DEFAULTS: dict = {
    "auto_summary_enabled":           False,
    "auto_summary_interval":          2000,
    "auto_summary_max_new_tokens":    512,
    "auto_summary_history_turns":     40,
    "auto_summary_full_prompt":       _MARKDOWN_BULLETS_FULL_PROMPT,
    "auto_summary_delta_prompt":      _MARKDOWN_BULLETS_DELTA_PROMPT,
    "auto_summary_include_char_card": False,
}

_AUTO_SUMMARY_PERSIST_KEYS: frozenset = frozenset(_AUTO_SUMMARY_PARAM_DEFAULTS)


def _register_params() -> None:
    from .config import _PARAMS_PERSIST_KEYS as _ppk, _PARAMS_FILE

    for k, v in _AUTO_SUMMARY_PARAM_DEFAULTS.items():
        params.setdefault(k, v)

    _ppk.update(_AUTO_SUMMARY_PERSIST_KEYS)

    if _PARAMS_FILE.exists():
        try:
            data = json.loads(_PARAMS_FILE.read_text(encoding="utf-8"))
            for k in _AUTO_SUMMARY_PERSIST_KEYS:
                if k in data:
                    params[k] = data[k]
        except Exception:
            logger.exception("Failed to reload auto-summary params from disk")

    _write_default_templates()


_register_params()

_summary_lock = threading.Lock()
_summary_status = ""

_summary_char_state: dict = {}

_last_seen_history: dict = {}
_last_seen_history_lock  = threading.Lock()

_last_seen_state: dict = {}
_last_seen_state_lock = threading.Lock()

_last_any_state: dict = {}
_last_any_state_lock = threading.Lock()

_active_summary_stem: str = ""
_active_char_stem: str = ""
_active_conv_id:   str = ""
_active_char_raw:  str = ""


def _char_stem(state: dict) -> str:
    mode = state.get("mode", "chat")
    if mode == "instruct":
        return "instruct"
    if mode == "notebook":
        return "notebook"
    raw = state.get("character_menu", "") or "default"
    return _safe_stem(str(raw)) or "default"


def _conv_fingerprint(history: dict) -> str:
    for u, a in history.get("internal", [])[:10]:
        u = (u or "").strip()
        if u and u != "<|BEGIN-VISIBLE-CHAT|>":
            raw = u[:200].encode("utf-8", errors="replace")
            return hashlib.md5(raw).hexdigest()[:8]
    return ""


def _conv_key(state: dict) -> tuple:
    char    = _char_stem(state)
    conv_id = state.get("unique_id", "") or ""
    if not conv_id:
        conv_id = _conv_fingerprint(state.get("history", {}))
    return char, conv_id


def _conv_state_key(char_stem: str, conv_id: str) -> str:
    return f"{char_stem}__{conv_id}"


def summary_status() -> str:
    return _summary_status or "No summary generated yet this session."


def capture_state_for_force(state: dict) -> None:
    global _last_any_state
    with _last_any_state_lock:
        _last_any_state = state


def set_active_char_raw(char_raw: str) -> None:
    global _active_char_raw
    _active_char_raw = char_raw or ""


def _apply_instruction_template(prompt: str, live_state: dict | None = None) -> str:
    try:
        import modules.shared as _shared
        tokenizer = getattr(_shared, "tokenizer", None)
        if (
            tokenizer is not None
            and hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None)
        ):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted
    except Exception:
        logger.debug("Auto-summary: HF chat template unavailable", exc_info=True)

    def _try_jinja(template_str: str, source_label: str) -> str | None:
        if not template_str:
            return None
        try:
            from jinja2 import Template, TemplateError
            try:
                tmpl = Template(template_str)
                bos = (live_state or {}).get("bos_token", "") or ""
                eos = (live_state or {}).get("eos_token", "") or ""
                formatted = tmpl.render(
                    messages=[{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    bos_token=bos,
                    eos_token=eos,
                )
                return formatted
            except TemplateError:
                return None
        except ImportError:
            return None

    live_tmpl_str = (live_state or {}).get("instruction_template_str", "")
    result = _try_jinja(live_tmpl_str, "live UI instruction_template_str")
    if result is not None:
        return result

    try:
        import modules.shared as _shared
        settings_tmpl_str = (_shared.settings or {}).get("instruction_template_str", "")
        result = _try_jinja(settings_tmpl_str, "settings.yaml instruction_template_str")
        if result is not None:
            return result
    except Exception:
        pass

    return prompt


def _entry_comment(char_stem: str, conv_id: str) -> str:
    return f"{_SUMMARY_ENTRY_PREFIX}{char_stem} / {conv_id}"


def _ensure_summary_lorebook(char_stem: str, conv_id: str) -> str:
    global _active_summary_stem, _active_char_stem, _active_conv_id

    comment = _entry_comment(char_stem, conv_id)
    lb_path = LOREBOOKS_DIR / f"{_SUMMARIES_STEM}.json"

    needs_disk        = False
    needs_save_active = False

    with _st.state_lock:
        lb = _st.active_lorebooks.get(_SUMMARIES_STEM)

        if lb is not None and not lb_path.exists():
            lb = None

        if lb is None:
            lb = load_lorebook(_SUMMARIES_STEM) or {
                "name":        "Auto Story Summaries",
                "description": "Auto-managed story summaries for all conversations.",
                "entries": [],
            }
            _st.active_lorebooks[_SUMMARIES_STEM] = lb
            _bump_active_version()
            needs_save_active = True
            needs_disk        = True

        found = False
        for e in lb.get("entries", []):
            if not e.get("comment", "").startswith(_SUMMARY_ENTRY_PREFIX):
                continue
            if e.get("comment") == comment:
                found = True
                if not e.get("enabled") or not e.get("constant"):
                    e["enabled"]  = True
                    e["constant"] = True
                    needs_disk    = True
            else:
                if e.get("enabled") or e.get("constant"):
                    e["enabled"]  = False
                    e["constant"] = False
                    needs_disk    = True

        if not found:
            new_uid = max(
                (e.get("uid", 0) for e in lb.get("entries", [])), default=0
            ) + 1
            lb["entries"].append({
                "uid":               new_uid,
                "enabled":           True,
                "comment":           comment,
                "char_stem":         char_stem,
                "conv_id":           conv_id,
                "keys":              [],
                "secondary_keys":    [],
                "selective_logic":   "AND ANY",
                "content": (
                    "(Story summary not yet generated — "
                    "will appear after the first summary cycle.)"
                ),
                "case_sensitive":    False,
                "match_whole_words": True,
                "use_regex":         False,
                "priority":          999,
                "position":          "before_context",
                "scan_depth":        0,
                "probability":       100,
                "inclusion_group":   "",
                "constant":          True,
            })
            needs_disk = True

        if needs_disk:
            _bump_active_version()

        _active_summary_stem = _SUMMARIES_STEM
        _active_char_stem    = char_stem
        _active_conv_id      = conv_id

    if needs_disk:
        save_lorebook_file(_SUMMARIES_STEM, lb)
    if needs_save_active:
        _save_active_state()

    return _SUMMARIES_STEM


def _get_current_summary(char_stem: str, conv_id: str) -> str:
    comment = _entry_comment(char_stem, conv_id)
    with _st.state_lock:
        lb = (
            _st.active_lorebooks.get(_SUMMARIES_STEM)
            or load_lorebook(_SUMMARIES_STEM)
            or {}
        )
    for e in lb.get("entries", []):
        if e.get("comment") == comment:
            return e.get("content", "")
    return ""


def _update_summary_entry(char_stem: str, conv_id: str, new_content: str) -> None:
    comment  = _entry_comment(char_stem, conv_id)
    lb_path  = LOREBOOKS_DIR / f"{_SUMMARIES_STEM}.json"
    lb_to_save = None

    with _st.state_lock:
        lb = _st.active_lorebooks.get(_SUMMARIES_STEM)
        if lb is None:
            return

        if not lb_path.exists():
            logger.warning("Auto-summary: _summaries.json missing — recreating from in-memory state.")

        for e in lb.get("entries", []):
            if e.get("comment") == comment:
                e["content"] = new_content
                _bump_active_version()
                lb_to_save = lb
                break

    if lb_to_save is not None:
        save_lorebook_file(_SUMMARIES_STEM, lb_to_save)


def _build_full_summary_prompt(history_snapshot: dict) -> str:
    n     = int(params.get("auto_summary_history_turns", 40))
    turns = history_snapshot.get("internal", [])[-n:]
    lines: list[str] = []
    for u, a in turns:
        u = (u or "").strip()
        a = (a or "").strip()
        if u and u != "<|BEGIN-VISIBLE-CHAT|>":
            lines.append(f"[User]: {u}")
        if a:
            lines.append(f"[Assistant]: {a}")
    template = params.get("auto_summary_full_prompt", _DEFAULT_FULL_PROMPT)
    return template.replace("{conversation}", "\n".join(lines))


def _build_delta_summary_prompt(previous_summary: str,
                                history_snapshot: dict,
                                last_turn: int) -> str:
    all_turns = history_snapshot.get("internal", [])
    new_turns = all_turns[last_turn:]
    max_new   = int(params.get("auto_summary_history_turns", 40))
    new_turns = new_turns[-max_new:]

    lines: list[str] = []
    for u, a in new_turns:
        u = (u or "").strip()
        a = (a or "").strip()
        if u and u != "<|BEGIN-VISIBLE-CHAT|>":
            lines.append(f"[User]: {u}")
        if a:
            lines.append(f"[Assistant]: {a}")
    template = params.get("auto_summary_delta_prompt", _DEFAULT_DELTA_PROMPT)
    return (template
            .replace("{previous_summary}", previous_summary)
            .replace("{new_events}", "\n".join(lines)))


def _run_summary_inline(history_snapshot: dict,
                        char_stem: str,
                        conv_id: str,
                        force_full: bool = False,
                        generation_state: dict | None = None) -> str:
    global _summary_status
    _summary_status = "⏳ Generating story summary…"

    csk = _conv_state_key(char_stem, conv_id)

    with _summary_lock:
        cs        = _summary_char_state.setdefault(csk, {"last_turn": 0})
        last_turn = cs.get("last_turn", 0)

    previous_summary   = _get_current_summary(char_stem, conv_id)
    is_placeholder     = previous_summary.startswith("(Story summary not yet")
    current_turn_count = len(history_snapshot.get("internal", []))

    if force_full or last_turn == 0 or is_placeholder:
        raw_prompt = _build_full_summary_prompt(history_snapshot)
        mode_label = "full"
    else:
        raw_prompt = _build_delta_summary_prompt(
            previous_summary, history_snapshot, last_turn
        )
        mode_label = f"delta (+{current_turn_count - last_turn} turns)"

    if generation_state and isinstance(generation_state, dict):
        summary_state = copy.deepcopy(generation_state)
    else:
        try:
            import modules.shared as _shared
            summary_state = copy.deepcopy(_shared.settings)
        except Exception:
            summary_state = {}

    summary_state.update(dict(
        max_new_tokens        = int(params.get("auto_summary_max_new_tokens", 512)),
        auto_max_new_tokens   = False,
        mode                  = "instruct",
        custom_system_message = "",
        context               = "",
        history               = {"internal": [], "visible": []},
        stream                = True,
        reasoning_effort      = "low",
    ))

    if summary_state.get("enable_thinking"):
        summary_state["max_new_tokens"] = summary_state["max_new_tokens"] * 2

    _STOP_SEQS = ["[User]:", "\n\n[User]:", "\n\n[Assistant]:", "story summary:"]
    try:
        import copy as _copy_ss
        from modules.chat import get_stopping_strings as _get_ss
        _derived_stops = _get_ss(_copy_ss.deepcopy(summary_state))
        if _derived_stops:
            _STOP_SEQS = _derived_stops + _STOP_SEQS
    except Exception:
        pass

    summary = ""

    try:
        import modules.shared as _shared
        model_class = _shared.model.__class__.__name__

        if model_class == "LlamaServer":
            formatted_prompt = _apply_instruction_template(raw_prompt, live_state=summary_state)
            summary_state.setdefault("sampler_priority", "")
            summary_state.setdefault("logit_bias", {})
            summary_state.setdefault("logprobs", 0)

            full_text = ""
            for chunk in _shared.model.generate_with_streaming(formatted_prompt, summary_state):
                full_text = chunk
                for stop in _STOP_SEQS:
                    if stop in full_text:
                        full_text = full_text[:full_text.index(stop)]
                        break
                else:
                    continue
                break
            summary = full_text

        elif model_class in _CUSTOM_BACKENDS:
            formatted_prompt = _apply_instruction_template(raw_prompt, live_state=summary_state)
            from modules.text_generation import generate_reply_custom
            final_chunk = ""
            for chunk in generate_reply_custom(
                formatted_prompt, formatted_prompt, summary_state, _STOP_SEQS, is_chat=False
            ):
                final_chunk = chunk
            generated = (
                final_chunk[len(formatted_prompt):]
                if final_chunk.startswith(formatted_prompt)
                else final_chunk
            )
            for stop in _STOP_SEQS:
                if stop in generated:
                    generated = generated[:generated.index(stop)]
                    break
            summary = generated

        else:
            formatted_prompt = _apply_instruction_template(raw_prompt, live_state=summary_state)
            try:
                from modules.text_generation import generate_reply_HF
                final_chunk = ""
                for chunk in generate_reply_HF(
                    formatted_prompt, formatted_prompt, summary_state, _STOP_SEQS, is_chat=False
                ):
                    final_chunk = chunk
                generated = (
                    final_chunk[len(formatted_prompt):]
                    if final_chunk.startswith(formatted_prompt)
                    else final_chunk
                )
                for stop in _STOP_SEQS:
                    if stop in generated:
                        generated = generated[:generated.index(stop)]
                        break
                summary = generated

            except ModuleNotFoundError as _torch_err:
                if "torch" in str(_torch_err):
                    logger.warning(
                        "torch not found for model class %r — falling back to "
                        "generate_reply_custom.  Add this class name to "
                        "_CUSTOM_BACKENDS in summary.py to silence this warning.",
                        model_class,
                    )
                    from modules.text_generation import generate_reply_custom
                    formatted_prompt = _apply_instruction_template(raw_prompt, live_state=summary_state)
                    final_chunk = ""
                    for chunk in generate_reply_custom(
                        formatted_prompt, formatted_prompt, summary_state, _STOP_SEQS, is_chat=False
                    ):
                        final_chunk = chunk
                    generated = (
                        final_chunk[len(formatted_prompt):]
                        if final_chunk.startswith(formatted_prompt)
                        else final_chunk
                    )
                    for stop in _STOP_SEQS:
                        if stop in generated:
                            generated = generated[:generated.index(stop)]
                            break
                    summary = generated
                else:
                    raise

    except Exception as exc:
        _summary_status = f"❌ Summary error: {exc}"
        logger.exception("Auto-summary generation failed for %s / %s", char_stem, conv_id)
        return ""

    summary = summary.strip()

    import re as _re
    summary = _re.sub(
        r'<\|channel>thought\n.*?<channel\|>',
        '',
        summary,
        flags=_re.DOTALL,
    ).strip()

    if summary:
        _update_summary_entry(char_stem, conv_id, summary)
        with _summary_lock:
            prev = _summary_char_state.setdefault(csk, {})
            prev.update({"last_turn": current_turn_count})
        tok = _count_tokens(summary)
        _summary_status = (
            f"✅ [{char_stem} / {conv_id}] Summary updated ({mode_label}) — "
            f"{tok} tokens written, {current_turn_count} turns processed."
        )
    else:
        _summary_status = "⚠ Summary generation returned empty text — no update written."

    return summary


def should_summarise(state: dict) -> bool:
    char, conv_id = _conv_key(state)
    if not conv_id:
        return False

    history_internal = state.get("history", {}).get("internal", [])
    csk = _conv_state_key(char, conv_id)

    with _last_seen_history_lock:
        _last_seen_history[csk] = copy.deepcopy(state.get("history", {}))

    with _last_seen_state_lock:
        _last_seen_state[csk] = copy.deepcopy(state)

    _ensure_summary_lorebook(char, conv_id)

    interval = max(1, int(params.get("auto_summary_interval", 2000)))

    with _summary_lock:
        cs        = _summary_char_state.setdefault(csk, {"last_turn": 0})
        last_turn = cs.get("last_turn", 0)

        new_turns  = history_internal[last_turn:]
        new_tokens = sum(
            _count_tokens((u or "").strip() + "\n" + (a or "").strip())
            for u, a in new_turns
        )

        if params.get("auto_summary_include_char_card", False):
            char_card_text = state.get("context", "") or ""
            new_tokens += _count_tokens(char_card_text.strip())

        if new_tokens >= interval:
            return True

    return False


def run_auto_summary(state: dict) -> str:
    char, conv_id    = _conv_key(state)
    if not conv_id:
        return ""
    history_snapshot = copy.deepcopy(state.get("history", {}))
    return _run_summary_inline(history_snapshot, char, conv_id,
                               generation_state=state)


def force_summary() -> None:
    global _summary_status

    char_stem = _active_char_stem
    conv_id   = _active_conv_id

    if not conv_id:
        with _last_any_state_lock:
            fallback_state = _last_any_state
        if fallback_state:
            char_stem, conv_id = _conv_key(fallback_state)

    if not conv_id:
        _summary_status = (
            "⚠ No active conversation identified yet — "
            "send at least one message first."
        )
        return

    _ensure_summary_lorebook(char_stem, conv_id)

    csk = _conv_state_key(char_stem, conv_id)
    with _summary_lock:
        _summary_char_state[csk] = {"last_turn": 0}

    with _last_seen_history_lock:
        history_snapshot = copy.deepcopy(_last_seen_history.get(csk, {}))

    if not history_snapshot.get("internal"):
        with _last_any_state_lock:
            fallback_state = _last_any_state
        if fallback_state:
            candidate = fallback_state.get("history", {})
            candidate_char, candidate_id = _conv_key(fallback_state)
            if candidate_id == conv_id and candidate.get("internal"):
                history_snapshot = copy.deepcopy(candidate)

    if not history_snapshot.get("internal"):
        try:
            import json as _json
            import modules.shared as _shared

            mode = (_last_any_state or {}).get("mode", "chat")

            if mode == "instruct":
                log_path = (
                    _shared.user_data_dir / "logs" / "instruct"
                    / f"{conv_id}.json"
                )
            else:
                char_dir = _active_char_raw or char_stem
                log_path = (
                    _shared.user_data_dir / "logs" / "chat"
                    / char_dir / f"{conv_id}.json"
                )

            if log_path.exists():
                history_snapshot = _json.loads(
                    log_path.read_text(encoding="utf-8")
                )
                with _last_seen_history_lock:
                    _last_seen_history[csk] = copy.deepcopy(history_snapshot)
        except Exception:
            logger.debug("force_summary: could not load history from disk", exc_info=True)

    with _last_seen_state_lock:
        stored_state = copy.deepcopy(_last_seen_state.get(csk, {}))

    if not stored_state:
        with _last_any_state_lock:
            stored_state = copy.deepcopy(_last_any_state)

    _run_summary_inline(
        history_snapshot, char_stem, conv_id,
        force_full=True,
        generation_state=stored_state or None,
    )
