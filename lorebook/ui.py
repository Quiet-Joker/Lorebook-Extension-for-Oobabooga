import gradio as gr

from .storage import get_lorebook_files
from .injection import _get_active_keys
from .ui_helpers import _build_stats_html, custom_css          # noqa: F401
from .ui_layout import build_layout
from .ui_handlers import (
    do_load, do_reload_lb, do_new_lb, do_save_lb_and_refresh, do_delete_lb_and_refresh,
    do_toggle_active,
    do_select_entry, do_new_entry, do_clone_entry, do_delete_entry, do_save_entry,
    do_filter_entries, do_test_match,
    do_overview,
    do_preview_refresh, do_preview_refresh_notebook,
    do_clear_history, do_clear_history_notebook,
    do_st_import_preview, do_st_import_confirm, do_st_import_cancel, do_st_export,
    set_activate, set_position_override, safe_int, set_param,
    do_select_summary_template, do_refresh_template_list,
    do_save_template, do_delete_template, do_force_summary,
)

_ENTRY_FIELDS = [
    "entry_comment", "entry_keys", "entry_sec_keys", "entry_content",
    "entry_selective_logic", "entry_inclusion_group",
    "entry_enabled", "entry_case", "entry_whole", "entry_regex",
    "entry_priority", "entry_position", "entry_scan_depth",
    "entry_probability", "entry_constant",
]


def ui():
    w = build_layout()

    def _ef(keys):
        return [w[k] for k in keys]

    w["lb_dropdown"].change(
        do_load, [w["lb_dropdown"]],
        [w["lb_name_input"], w["lb_desc_input"], w["lb_status"], w["entry_radio"]])
    w["lb_new_btn"].click(
        do_new_lb, [],
        [w["lb_name_input"], w["lb_desc_input"], w["lb_status"], w["entry_radio"]])
    w["lb_refresh_btn"].click(
        lambda: gr.update(choices=get_lorebook_files()), [], [w["lb_dropdown"]])
    w["lb_reload_btn"].click(
        do_reload_lb, [],
        [w["lb_name_input"], w["lb_desc_input"], w["lb_status"], w["entry_radio"]])
    w["lb_save_btn"].click(
        do_save_lb_and_refresh,
        [w["lb_name_input"], w["lb_desc_input"]],
        [w["lb_status"], w["lb_dropdown"], w["active_pills"], w["stats_html"]])
    w["lb_delete_btn"].click(
        do_delete_lb_and_refresh, [w["lb_dropdown"]],
        [w["lb_dropdown"], w["lb_status"], w["active_pills"], w["stats_html"],
         w["lb_name_input"], w["lb_desc_input"], w["entry_radio"]])

    w["active_pills"].change(
        do_toggle_active, [w["active_pills"]],
        [w["stats_html"], w["active_status"], w["active_pills"]])
    w["active_refresh_btn"].click(
        lambda: gr.update(choices=get_lorebook_files(), value=_get_active_keys()),
        [], [w["active_pills"]])

    w["entry_radio"].change(
        do_select_entry, [w["entry_radio"]], _ef(_ENTRY_FIELDS))
    w["entry_new_btn"].click(
        do_new_entry, [],
        [w["entry_radio"], w["entry_save_status"]] + _ef(_ENTRY_FIELDS))
    w["entry_clone_btn"].click(
        do_clone_entry, [w["entry_radio"]],
        [w["entry_radio"], w["entry_save_status"]] + _ef(_ENTRY_FIELDS))
    w["entry_delete_btn"].click(
        do_delete_entry, [w["entry_radio"]],
        [w["entry_radio"], w["entry_save_status"]] + _ef(_ENTRY_FIELDS))
    w["entry_save_btn"].click(
        do_save_entry,
        [w["entry_radio"]] + _ef(_ENTRY_FIELDS),
        [w["entry_save_status"], w["entry_radio"]])

    w["entry_filter"].change(
        do_filter_entries, [w["entry_filter"], w["entry_radio"]], [w["entry_radio"]])

    w["entry_test_btn"].click(
        do_test_match,
        [w["entry_radio"], w["entry_test_input"]],
        [w["entry_test_result"]])

    w["activate_cb"].change(        set_activate,                                           [w["activate_cb"]],          [w["stats_html"]])
    w["scan_depth_n"].change(       lambda x: safe_int("scan_depth", x),                   [w["scan_depth_n"]],         None)
    w["token_budget_n"].change(     lambda x: safe_int("token_budget", x),                 [w["token_budget_n"]],       None)
    w["constant_entries_cb"].change(lambda x: set_param("constant_entries", x),            [w["constant_entries_cb"]], None)
    w["recursive_scan_cb"].change(  lambda x: set_param("recursive_scan", x),              [w["recursive_scan_cb"]],   None)
    w["chat_only_scan_cb"].change(  lambda x: set_param("chat_only_scan", x),              [w["chat_only_scan_cb"]],   None)
    w["max_recursion_n"].change(    lambda x: safe_int("max_recursion_steps", x),          [w["max_recursion_n"]],     None)
    w["inj_prefix"].change(         lambda x: set_param("injection_prefix", x),            [w["inj_prefix"]],          None)
    w["inj_suffix"].change(         lambda x: set_param("injection_suffix", x),            [w["inj_suffix"]],          None)
    w["mid_gen_interrupt_cb"].change(lambda x: set_param("mid_gen_interrupt", x),          [w["mid_gen_interrupt_cb"]], None)
    w["max_interrupts_n"].change(   lambda x: safe_int("max_interrupts", x),               [w["max_interrupts_n"]],    None)
    w["position_override_cb"].change(set_position_override,                                [w["position_override_cb"]], [w["position_override_dd"]])
    w["position_override_dd"].change(lambda x: set_param("position_override_value", x),   [w["position_override_dd"]], None)

    w["overview_btn"].click(       do_overview,            [], [w["overview_box"]])
    w["stats_refresh_btn"].click(  lambda: gr.update(value=_build_stats_html()), [], [w["stats_html"]])

    w["preview_refresh_btn"].click(
        do_preview_refresh, [],
        [w["preview_box"], w["history_box"], w["budget_bar"]])
    w["notebook_preview_refresh_btn"].click(
        do_preview_refresh_notebook, [],
        [w["notebook_preview_box"], w["notebook_history_box"], w["budget_bar"]])

    w["history_clear_btn"].click(          do_clear_history,          [], [w["history_box"]])
    w["notebook_history_clear_btn"].click( do_clear_history_notebook, [], [w["notebook_history_box"]])

    w["st_import_btn"].click(
        do_st_import_preview,
        [w["st_import_file"]],
        [w["st_import_status"], w["st_import_preview"], w["st_import_pending"],
         w["st_import_confirm_btn"], w["st_import_cancel_btn"]])
    w["st_import_confirm_btn"].click(
        do_st_import_confirm,
        [w["st_import_pending"]],
        [w["st_import_status"], w["st_import_preview"], w["st_import_pending"],
         w["st_import_confirm_btn"], w["st_import_cancel_btn"],
         w["lb_dropdown"], w["active_pills"],
         w["lb_name_input"], w["lb_desc_input"], w["entry_radio"]])
    w["st_import_cancel_btn"].click(
        do_st_import_cancel, [],
        [w["st_import_status"], w["st_import_preview"], w["st_import_pending"],
         w["st_import_confirm_btn"], w["st_import_cancel_btn"]])
    w["st_export_btn"].click(
        do_st_export, [], [w["st_export_status"], w["st_export_file"]])

    w["auto_summary_enabled_cb"].change(
        lambda x: set_param("auto_summary_enabled", x),
        [w["auto_summary_enabled_cb"]], None)
    w["auto_summary_interval_n"].change(
        lambda x: safe_int("auto_summary_interval", x),
        [w["auto_summary_interval_n"]], None)
    w["auto_summary_max_new_tokens_n"].change(
        lambda x: safe_int("auto_summary_max_new_tokens", x),
        [w["auto_summary_max_new_tokens_n"]], None)
    w["auto_summary_history_turns_n"].change(
        lambda x: safe_int("auto_summary_history_turns", x),
        [w["auto_summary_history_turns_n"]], None)
    w["auto_summary_include_char_card_cb"].change(
        lambda x: set_param("auto_summary_include_char_card", x),
        [w["auto_summary_include_char_card_cb"]], None)
    w["auto_summary_template_dd"].change(
        do_select_summary_template,
        [w["auto_summary_template_dd"]],
        [w["auto_summary_full_prompt_tb"], w["auto_summary_delta_prompt_tb"],
         w["auto_summary_template_name_tb"]])
    w["auto_summary_template_refresh_btn"].click(
        do_refresh_template_list, [],
        [w["auto_summary_template_dd"]])
    w["auto_summary_template_save_btn"].click(
        do_save_template,
        [w["auto_summary_template_name_tb"], w["auto_summary_full_prompt_tb"],
         w["auto_summary_delta_prompt_tb"]],
        [w["auto_summary_template_status"], w["auto_summary_template_dd"]])
    w["auto_summary_template_delete_btn"].click(
        do_delete_template,
        [w["auto_summary_template_name_tb"]],
        [w["auto_summary_template_status"], w["auto_summary_template_dd"]])
    w["auto_summary_full_prompt_tb"].change(
        lambda x: set_param("auto_summary_full_prompt", x),
        [w["auto_summary_full_prompt_tb"]], None)
    w["auto_summary_delta_prompt_tb"].change(
        lambda x: set_param("auto_summary_delta_prompt", x),
        [w["auto_summary_delta_prompt_tb"]], None)
    w["auto_summary_force_btn"].click(
        do_force_summary, [], [w["auto_summary_status_md"]])

    # Wire summary entry switching to ooba's chat-switch event
    try:
        import modules.shared as _ms

        def _on_chat_switch_for_summary(uid, character, mode):
            try:
                from .summary import _conv_key, _ensure_summary_lorebook, _SUMMARIES_STEM, set_active_char_raw
                from .config import LOREBOOKS_DIR
                set_active_char_raw(character or "")
                if not (LOREBOOKS_DIR / f"{_SUMMARIES_STEM}.json").exists():
                    return
                mini_state = {
                    "unique_id":      uid      or "",
                    "character_menu": character or "",
                    "mode":           mode     or "chat",
                    "history":        {},
                }
                char, conv_id = _conv_key(mini_state)
                if conv_id:
                    _ensure_summary_lorebook(char, conv_id)
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "Summary chat-switch hook failed", exc_info=True)

        _ms.gradio['unique_id'].change(
            _on_chat_switch_for_summary,
            inputs=[
                _ms.gradio['unique_id'],
                _ms.gradio['character_menu'],
                _ms.gradio['mode'],
            ],
            outputs=[],
            show_progress=False,
        )
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "Could not wire summary chat-switch hook to unique_id.change",
            exc_info=True,
        )
