"""
_chat_hook.py  —  runtime patch for modules.chat
=================================================
Injects the conversation-change callback infrastructure into modules.chat
WITHOUT modifying chat.py on disk.

This module MUST be imported before Oobabooga builds its Gradio UI so that
the wrapped function objects are the ones Gradio captures.  The import in
script.py (which runs during extension loading, before UI construction) meets
that requirement.

What it does
------------
1. Adds to modules.chat:
     _conversation_change_callbacks   list
     register_conversation_change_callback(fn)
     _fire_conversation_change(uid, character, mode)
   …so that ui.py's Path-1 wiring
     (from modules.chat import register_conversation_change_callback)
   works without any change to chat.py.

2. Wraps four functions that switch the active conversation:
     load_latest_history       — startup, mode switch, character change
     handle_unique_id_select   — user clicks a different conversation
     handle_start_new_chat_click   — user starts a brand-new chat
     load_history_after_deletion   — conversation deleted, next one loaded

   Each wrapper fires _fire_conversation_change after the original returns,
   so _on_chat_switch_for_summary in ui.py is called immediately on every
   conversation change — no generation needed.
"""

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

    # ------------------------------------------------------------------
    # Guard: don't double-install if the hook is loaded more than once.
    # ------------------------------------------------------------------
    if getattr(_chat, "_lorebook_hook_installed", False):
        return

    # ==================================================================
    # 1. Inject callback infrastructure
    # ==================================================================
    _chat._conversation_change_callbacks = []

    def register_conversation_change_callback(fn) -> None:
        """Register *fn(unique_id, character, mode)* to be called on every
        conversation switch, including UI startup and mode/character changes.
        The function must not block; do slow work in a background thread."""
        _chat._conversation_change_callbacks.append(fn)

    def _fire_conversation_change(unique_id: str, character: str, mode: str) -> None:
        """Internal dispatcher — swallows all exceptions so a misbehaving
        callback can never crash the chat handler."""
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

    # ==================================================================
    # 2. Wrap load_latest_history
    #    Signature: (state) -> (history, unique_id)
    #    Covers: startup, mode switch, character change.
    # ==================================================================
    _orig_load_latest = _chat.load_latest_history

    def _patched_load_latest_history(state):
        result = _orig_load_latest(state)
        try:
            # result is (history, unique_id); unique_id may be None
            uid = result[1] if (result and len(result) > 1) else None
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

    # ==================================================================
    # 3. Wrap handle_unique_id_select
    #    Signature: (state) -> [history, html]
    #    state['unique_id'] already holds the selected conversation ID.
    #    Covers: user manually clicking a different conversation.
    # ==================================================================
    _orig_uid_select = _chat.handle_unique_id_select

    def _patched_handle_unique_id_select(state):
        result = _orig_uid_select(state)
        try:
            uid = state.get("unique_id", "") or ""
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

    # ==================================================================
    # 4. Wrap handle_start_new_chat_click
    #    Signature: (state) -> [history, html, past_chats_update]
    #    A brand-new chat gets a new unique_id; we read it from the
    #    state dict that Oobabooga mutates in-place inside start_new_chat,
    #    OR fall back to the first item in past_chats_update choices.
    # ==================================================================
    _orig_new_chat = _chat.handle_start_new_chat_click

    def _patched_handle_start_new_chat_click(state):
        result = _orig_new_chat(state)
        try:
            # Oobabooga puts the new unique_id in state after the call.
            uid = state.get("unique_id", "") or ""
            if not uid:
                # Fallback: past_chats_update is result[2]; its value field
                # is the newly selected id if available.
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

    # ==================================================================
    # 5. Wrap load_history_after_deletion
    #    Signature: (state, idx) -> various
    #    After a chat is deleted the next one is loaded; we need to sync.
    # ==================================================================
    _orig_after_del = getattr(_chat, "load_history_after_deletion", None)
    if _orig_after_del is not None:

        def _patched_load_history_after_deletion(state, idx):
            result = _orig_after_del(state, idx)
            try:
                uid = state.get("unique_id", "") or ""
                if not uid and result and len(result) > 1:
                    # Some Oobabooga versions return (history, unique_id, ...)
                    candidate = result[1]
                    if isinstance(candidate, str):
                        uid = candidate
                if uid:
                    _chat._fire_conversation_change(
                        str(uid),
                        state.get("character_menu", "") or "",
                        state.get("mode", "chat") or "chat",
                    )
            except Exception:
                logger.debug(
                    "lorebook._chat_hook: load_history_after_deletion fire failed",
                    exc_info=True,
                )
            return result

        _chat.load_history_after_deletion = _patched_load_history_after_deletion

    # Mark installed so a second import is a no-op.
    _chat._lorebook_hook_installed = True
    logger.info(
        "lorebook._chat_hook: modules.chat patched — "
        "conversation-change callbacks active."
    )


_install()
