import logging

logger = logging.getLogger(__name__)


def _install() -> None:
    try:
        import modules.chat as _chat
    except ImportError:
        logger.warning(
            "lorebook._chat_hook: modules.chat not importable — "
            "instruct-mode lorebook sync will not work."
        )
        return

    if getattr(_chat, "_lorebook_hook_installed", False):
        return

    _chat._conversation_change_callbacks = []

    _chat._pending_conversation_change = None

    def register_conversation_change_callback(fn) -> None:
        _chat._conversation_change_callbacks.append(fn)
        pending = _chat._pending_conversation_change
        if pending:
            _chat._pending_conversation_change = None
            try:
                fn(pending["uid"], pending["char"], pending["mode"])
            except Exception:
                logger.debug(
                    "lorebook: replay pending conversation_change for %r raised", fn,
                    exc_info=True,
                )

    def _fire_conversation_change(unique_id: str, character: str, mode: str) -> None:
        if not _chat._conversation_change_callbacks:
            _chat._pending_conversation_change = {
                "uid":  unique_id or "",
                "char": character or "",
                "mode": mode or "chat",
            }
            return
        _chat._pending_conversation_change = None
        for cb in list(_chat._conversation_change_callbacks):
            try:
                cb(unique_id or "", character or "", mode or "chat")
            except Exception:
                logger.debug(
                    "lorebook: conversation_change callback %r raised", cb,
                    exc_info=True,
                )

    _chat.register_conversation_change_callback = register_conversation_change_callback
    _chat._fire_conversation_change            = _fire_conversation_change

    _orig_load_latest = _chat.load_latest_history

    def _patched_load_latest_history(state):
        result = _orig_load_latest(state)
        try:
            uid = result[1] if (result and len(result) > 1) else None
            if not uid or not isinstance(uid, str):
                uid = state.get("unique_id", "") or ""
            if uid:
                _chat._fire_conversation_change(
                    str(uid),
                    state.get("character_menu", "") or "",
                    state.get("mode", "chat") or "chat",
                )
        except Exception:
            logger.debug(
                "lorebook._chat_hook: load_latest_history fire failed",
                exc_info=True,
            )
        return result

    _chat.load_latest_history = _patched_load_latest_history

    _orig_uid_select = _chat.handle_unique_id_select

    def _patched_handle_unique_id_select(state):
        result = _orig_uid_select(state)
        try:
            uid = state.get("unique_id", "") or ""
            if not uid and result and len(result) > 1 and isinstance(result[1], str):
                uid = result[1]
            if uid:
                _chat._fire_conversation_change(
                    str(uid),
                    state.get("character_menu", "") or "",
                    state.get("mode", "chat") or "chat",
                )
        except Exception:
            logger.debug(
                "lorebook._chat_hook: handle_unique_id_select fire failed",
                exc_info=True,
            )
        return result

    _chat.handle_unique_id_select = _patched_handle_unique_id_select

    _orig_new_chat = _chat.handle_start_new_chat_click

    def _patched_handle_start_new_chat_click(state):
        result = _orig_new_chat(state)
        try:
            uid = state.get("unique_id", "") or ""
            if not uid:
                try:
                    import gradio as gr
                    update = result[2] if len(result) > 2 else None
                    if update and hasattr(update, "get"):
                        uid = str(update.get("value", "") or "")
                    elif isinstance(update, dict):
                        uid = str(update.get("value", "") or "")
                except Exception:
                    pass
            if uid:
                _chat._fire_conversation_change(
                    uid,
                    state.get("character_menu", "") or "",
                    state.get("mode", "chat") or "chat",
                )
        except Exception:
            logger.debug(
                "lorebook._chat_hook: handle_start_new_chat_click fire failed",
                exc_info=True,
            )
        return result

    _chat.handle_start_new_chat_click = _patched_handle_start_new_chat_click

    _orig_after_del = getattr(_chat, "load_history_after_deletion", None)
    if _orig_after_del is not None:

        def _patched_load_history_after_deletion(state, idx):
            result = _orig_after_del(state, idx)
            try:
                uid = state.get("unique_id", "") or ""
                if not uid and result and len(result) > 1:
                    candidate = result[1]
                    if isinstance(candidate, str):
                        uid = candidate
                if uid:
                    _chat._fire_conversation_change(
                        str(uid),
                        state.get("character_menu", "") or "",
                        state.get("mode", "chat") or "",
                    )
            except Exception:
                logger.debug(
                    "lorebook._chat_hook: load_history_after_deletion fire failed",
                    exc_info=True,
                )
            return result

        _chat.load_history_after_deletion = _patched_load_history_after_deletion

    _chat._lorebook_hook_installed = True
    logger.info(
        "lorebook._chat_hook: modules.chat patched — "
        "conversation-change callbacks active."
    )


_install()
