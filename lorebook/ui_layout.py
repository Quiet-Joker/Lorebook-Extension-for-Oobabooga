import gradio as gr

from .config import params
from .storage import get_lorebook_files
from .injection import _get_active_keys
from .ui_helpers import _build_stats_html, _build_history_html, _budget_bar_html


def build_layout() -> dict:
    with gr.Row():

        with gr.Column(scale=3):

            with gr.Row():
                lb_dropdown    = gr.Dropdown(choices=get_lorebook_files(), label="Lorebook file",
                                             elem_classes="slim-dropdown", scale=4, interactive=True,
                                             info="All .json files in the lorebooks/ folder.")
                lb_new_btn     = gr.Button("New",    elem_classes="refresh-button")
                lb_save_btn    = gr.Button("Save",   variant="primary", elem_classes="lb-btn-primary refresh-button")
                lb_delete_btn  = gr.Button("Delete", variant="stop",    elem_classes="lb-btn-danger refresh-button")
                lb_refresh_btn = gr.Button("🔄",      elem_classes="refresh-button", scale=0)
                lb_reload_btn  = gr.Button("↺ Reload", elem_classes="refresh-button", scale=0,
                                           tooltip="Reload the current lorebook from disk, discarding unsaved changes.")

            lb_name_input = gr.Textbox(label="Name", placeholder="My Fantasy World",
                                       info="Displayed in the active lorebooks tab.")
            lb_desc_input = gr.Textbox(label="Description",
                                       placeholder="Optional notes — the AI never reads this.")
            lb_status = gr.Markdown("*No lorebook open — select one above or click New.*")

            with gr.Accordion("Entries", open=False, elem_classes="tgw-accordion"):

                gr.HTML('<div style="margin-top:6px"></div>')

                entry_filter = gr.Textbox(label="Filter entries", placeholder="Search by name or keys…",
                                          info="Type to filter the dropdown below.",
                                          elem_classes="slim-input")

                with gr.Row():
                    entry_radio      = gr.Dropdown(choices=[], label="Entry",
                                                   elem_classes="slim-dropdown", interactive=True,
                                                   info="✅ enabled  🔵 constant  ❌ disabled", scale=4)
                    entry_new_btn    = gr.Button("+ Add",  variant="primary", elem_classes="lb-btn-primary refresh-button", scale=1)
                    entry_clone_btn  = gr.Button("Clone",  elem_classes="refresh-button", scale=1)
                    entry_delete_btn = gr.Button("Remove", variant="stop",    elem_classes="lb-btn-danger refresh-button",  scale=1)

                entry_comment = gr.Textbox(label="Entry name", placeholder="e.g. Dragon Lore",
                                           info="Label only — the AI never reads this.")

                with gr.Row():
                    entry_enabled  = gr.Checkbox(label="Enabled",             value=True)
                    entry_constant = gr.Checkbox(label="Constant (always on)", value=False)
                    entry_case     = gr.Checkbox(label="Case sensitive",       value=False)
                    entry_whole    = gr.Checkbox(label="Whole words only",     value=True)
                    entry_regex    = gr.Checkbox(label="Use regex",            value=False)

                with gr.Row():
                    entry_keys     = gr.Textbox(label="Trigger words",           placeholder="dragon, drake, wyrm", scale=2,
                                                info="Comma-separated. ANY one fires this entry.")
                    entry_sec_keys = gr.Textbox(label="Also require (optional)", placeholder="ancient, fire",       scale=2,
                                                info="Secondary keys evaluated against the logic below.")

                entry_selective_logic = gr.Dropdown(
                    choices=["AND ANY", "AND ALL", "NOT ANY", "NOT ALL"], value="AND ANY",
                    label="Secondary key logic", elem_classes="slim-dropdown",
                    info="AND ANY = at least one must match.  AND ALL = all must match.  NOT ANY = none may match.  NOT ALL = suppressed only when all match.")

                entry_content = gr.Textbox(label="Content", lines=5,
                                           placeholder="e.g. Aurelion is an ancient fire-breathing dragon...",
                                           info="This is what the AI reads when the entry fires.")

                with gr.Row():
                    entry_priority   = gr.Number(label="Priority",   value=10, step=1, minimum=0, scale=1,
                                                 info="Higher priority fires first.")
                    entry_scan_depth = gr.Number(label="Scan depth", value=0,  step=1, minimum=0, scale=1,
                                                 info="Messages back to scan for this entry.")

                entry_position = gr.Dropdown(
                    choices=[
                        ("After System Prompt",        "after_context"),
                        ("Before System Prompt",        "before_context"),
                        ("Between User and Assistant",  "before_reply"),
                        ("Notebook Mode",               "notebook"),
                    ],
                    value="after_context", label="Position", elem_classes="slim-dropdown",
                    info="after_context = below context (default).  before_context = above context.  "
                         "before_reply = injected just before the assistant's reply.  "
                         "notebook = text-completion mode only.")

                entry_probability     = gr.Slider(minimum=0, maximum=100, value=100, step=1, label="Trigger %",
                                                  info="Chance this entry fires when triggered.")
                entry_inclusion_group = gr.Textbox(label="Inclusion group", value="", placeholder="e.g. weather_mood",
                                                   info="Only the highest-priority entry in a group fires.")

                entry_save_btn    = gr.Button("Save entry", variant="primary", elem_classes="lb-btn-primary")
                entry_save_status = gr.Markdown("")

                with gr.Accordion("🔍 Test match", open=False, elem_classes="tgw-accordion"):
                    gr.Markdown("Paste sample text to instantly check whether this entry's keys would fire — "
                                "no generation needed.")
                    entry_test_input  = gr.Textbox(label="Sample text", lines=3,
                                                   placeholder="e.g. The ancient dragon Aurelion soared overhead…")
                    entry_test_btn    = gr.Button("Test", elem_classes="refresh-button")
                    entry_test_result = gr.Markdown("*Select an entry and paste text above.*")

        with gr.Column(scale=2):
            with gr.Tabs():

                with gr.Tab("Lorebooks"):
                    gr.Markdown("Check to turn on, uncheck to turn off. Multiple can be active at once.")
                    active_pills = gr.CheckboxGroup(choices=get_lorebook_files(), value=_get_active_keys(),
                                                    label="", show_label=False, elem_id="lb-active-pills")
                    active_refresh_btn = gr.Button("Refresh list", elem_classes="refresh-button")
                    active_status      = gr.Markdown("")

                with gr.Tab("Overview"):
                    stats_html        = gr.HTML(_build_stats_html())
                    stats_refresh_btn = gr.Button("Refresh stats", elem_classes="refresh-button")

                    with gr.Tabs():
                        with gr.Tab("Chat / Instruct"):
                            gr.Markdown("**Last injection** — entries in priority order.")
                            preview_box         = gr.Markdown("*No generation yet — send a message first.*")
                            preview_refresh_btn = gr.Button("Refresh injection", elem_classes="refresh-button")
                            gr.Markdown("**Session history**")
                            history_box       = gr.HTML(_build_history_html(notebook=False))
                            history_clear_btn = gr.Button("Clear history", elem_classes="refresh-button")

                        with gr.Tab("Notebook"):
                            gr.Markdown("**Last injection** — entries in priority order.")
                            notebook_preview_box         = gr.Markdown("*No generation yet — use the Notebook tab first.*")
                            notebook_preview_refresh_btn = gr.Button("Refresh injection", elem_classes="refresh-button")
                            gr.Markdown("**Session history**")
                            notebook_history_box       = gr.HTML(_build_history_html(notebook=True))
                            notebook_history_clear_btn = gr.Button("Clear history", elem_classes="refresh-button")

                with gr.Tab("All entries"):
                    gr.Markdown("Quick table of all entries in the currently open lorebook.")
                    overview_btn = gr.Button("Refresh", elem_classes="refresh-button")
                    overview_box = gr.Markdown("")

                with gr.Tab("Settings"):
                    settings_refs = _build_settings_tab()

                with gr.Tab("Import / Export"):
                    ie_refs = _build_import_export_tab()

                with gr.Tab("Auto Summary"):
                    auto_summary_refs = _build_auto_summary_tab()

    return dict(
        lb_dropdown=lb_dropdown, lb_new_btn=lb_new_btn, lb_save_btn=lb_save_btn,
        lb_delete_btn=lb_delete_btn, lb_refresh_btn=lb_refresh_btn, lb_reload_btn=lb_reload_btn,
        lb_name_input=lb_name_input, lb_desc_input=lb_desc_input, lb_status=lb_status,
        entry_filter=entry_filter,
        entry_radio=entry_radio, entry_new_btn=entry_new_btn,
        entry_clone_btn=entry_clone_btn, entry_delete_btn=entry_delete_btn,
        entry_comment=entry_comment, entry_enabled=entry_enabled, entry_constant=entry_constant,
        entry_case=entry_case, entry_whole=entry_whole, entry_regex=entry_regex,
        entry_keys=entry_keys, entry_sec_keys=entry_sec_keys,
        entry_selective_logic=entry_selective_logic, entry_content=entry_content,
        entry_priority=entry_priority, entry_scan_depth=entry_scan_depth,
        entry_position=entry_position, entry_probability=entry_probability,
        entry_inclusion_group=entry_inclusion_group,
        entry_save_btn=entry_save_btn, entry_save_status=entry_save_status,
        entry_test_input=entry_test_input, entry_test_btn=entry_test_btn,
        entry_test_result=entry_test_result,
        active_pills=active_pills, active_refresh_btn=active_refresh_btn, active_status=active_status,
        stats_html=stats_html, stats_refresh_btn=stats_refresh_btn,
        preview_box=preview_box, preview_refresh_btn=preview_refresh_btn,
        history_box=history_box, history_clear_btn=history_clear_btn,
        notebook_preview_box=notebook_preview_box,
        notebook_preview_refresh_btn=notebook_preview_refresh_btn,
        notebook_history_box=notebook_history_box,
        notebook_history_clear_btn=notebook_history_clear_btn,
        overview_btn=overview_btn, overview_box=overview_box,
        **settings_refs, **ie_refs, **auto_summary_refs,
    )


def _build_settings_tab() -> dict:
    gr.Markdown("These settings apply to all active lorebooks.")

    activate_cb = gr.Checkbox(label="Lorebook system active", value=params["activate"],
                              info="Master on/off switch.")

    with gr.Row():
        scan_depth_n   = gr.Number(label="Scan depth override", value=params["scan_depth"],   step=1,  minimum=-1,
                                   info="-1 = disabled, 0 = current message only.")
        token_budget_n = gr.Number(label="Token budget",        value=params["token_budget"], step=64, minimum=64,
                                   info="Max world info tokens to inject per turn.")

    budget_bar = gr.HTML(_budget_bar_html())

    with gr.Row():
        max_recursion_n = gr.Number(label="Max recursion steps", value=params["max_recursion_steps"],
                                    step=1, minimum=1, maximum=10,
                                    info="Caps recursive passes to prevent infinite loops.")

    with gr.Row():
        constant_entries_cb = gr.Checkbox(label="Inject constant entries", value=params["constant_entries"],
                                          info="Constant entries are injected every turn regardless of trigger words.")
        recursive_scan_cb   = gr.Checkbox(label="Recursive scanning",      value=params["recursive_scan"],
                                          info="Matched entries can trigger further entries via keywords in their content.")

    chat_only_scan_cb = gr.Checkbox(label="Only trigger words in chat", value=params["chat_only_scan"],
                                    info="Ignore the character persona when scanning — only actual chat messages fire entries.")

    inj_prefix = gr.Textbox(label="Injection prefix", value=params["injection_prefix"], lines=2,
                             info="Added before the world info block in the prompt.")
    inj_suffix = gr.Textbox(label="Injection suffix", value=params["injection_suffix"], lines=2,
                             info="Added after the world info block in the prompt.")

    gr.HTML('<div style="margin:14px 0 6px;padding:8px 12px;border-left:3px solid rgba(139,92,246,.7);'
            'background:rgba(139,92,246,.06);border-radius:0 6px 6px 0">'
            '<span style="font-weight:600;font-size:13px">Mid-generation interrupt</span>'
            '<span style="font-size:12px;color:var(--body-text-color-subdued)"> — pauses on new trigger words in model output, expands WI block, then resumes.</span></div>')
    mid_gen_interrupt_cb = gr.Checkbox(label="Enable mid-gen interrupt", value=params["mid_gen_interrupt"],
                                       info="Requires stream mode to be on in generation settings.")
    max_interrupts_n = gr.Number(label="Max interrupts per reply", value=params["max_interrupts"],
                                 step=1, minimum=1, maximum=100)

    gr.HTML('<div style="margin:14px 0 6px;padding:8px 12px;border-left:3px solid rgba(139,92,246,.7);'
            'background:rgba(139,92,246,.06);border-radius:0 6px 6px 0">'
            '<span style="font-weight:600;font-size:13px">Context position override</span>'
            '<span style="font-size:12px;color:var(--body-text-color-subdued)"> — force all entries to one position.</span></div>')
    position_override_cb = gr.Checkbox(label="Enable override", value=params["position_override_enabled"])
    position_override_dd = gr.Dropdown(
        choices=[("After System Prompt", "after_context"), ("Before System Prompt", "before_context"),
                 ("Between User and Assistant", "before_reply"), ("Notebook Mode", "notebook")],
        value=params["position_override_value"], label="Force all entries to",
        elem_classes="slim-dropdown", interactive=params["position_override_enabled"])

    return dict(
        activate_cb=activate_cb, scan_depth_n=scan_depth_n, token_budget_n=token_budget_n,
        budget_bar=budget_bar,
        max_recursion_n=max_recursion_n, constant_entries_cb=constant_entries_cb,
        recursive_scan_cb=recursive_scan_cb, chat_only_scan_cb=chat_only_scan_cb,
        inj_prefix=inj_prefix, inj_suffix=inj_suffix,
        mid_gen_interrupt_cb=mid_gen_interrupt_cb, max_interrupts_n=max_interrupts_n,
        position_override_cb=position_override_cb, position_override_dd=position_override_dd,
    )


def _build_import_export_tab() -> dict:
    gr.Markdown("#### Import from SillyTavern")
    gr.Markdown("Drop a SillyTavern world-info `.json` file below and click **Import**. "
                "A summary card will appear for you to review before anything is saved.")
    with gr.Row():
        st_import_file = gr.File(label="SillyTavern .json", file_types=[".json"], scale=3)
        st_import_btn  = gr.Button("Import", scale=1, variant="primary", elem_classes="lb-btn-primary")
    st_import_status  = gr.Markdown("")
    st_import_pending = gr.State(None)
    st_import_preview = gr.HTML("", visible=False)
    with gr.Row():
        st_import_confirm_btn = gr.Button("\u2705\u00a0Confirm Import", variant="primary",
                                          elem_classes="lb-btn-primary", visible=False)
        st_import_cancel_btn  = gr.Button("\u2716\u00a0Cancel",          variant="stop",
                                          elem_classes="lb-btn-danger",   visible=False)

    gr.Markdown("#### Export to SillyTavern")
    gr.Markdown("Export the currently open lorebook as a SillyTavern world-info file.")
    st_export_btn    = gr.Button("Export current lorebook", variant="secondary", elem_classes="refresh-button")
    st_export_file   = gr.File(label="Download", visible=False, interactive=False)
    st_export_status = gr.Markdown("")

    return dict(
        st_import_file=st_import_file, st_import_btn=st_import_btn, st_import_status=st_import_status,
        st_import_pending=st_import_pending, st_import_preview=st_import_preview,
        st_import_confirm_btn=st_import_confirm_btn, st_import_cancel_btn=st_import_cancel_btn,
        st_export_btn=st_export_btn, st_export_file=st_export_file, st_export_status=st_export_status,
    )


def _build_auto_summary_tab() -> dict:
    gr.Markdown(
        "Automatically generates a running story summary and injects it as a "
        "constant lorebook entry at high priority. The summary fires after a "
        "configurable number of **new story tokens** have been generated since "
        "the last summary — independent of the truncation window size."
    )

    auto_summary_enabled_cb = gr.Checkbox(
        label="Enable auto story summary",
        value=params.get("auto_summary_enabled", False),
        info="Master switch. When on, a summary is generated periodically during chat.",
    )

    with gr.Row():
        auto_summary_interval_n = gr.Number(
            label="Summary interval (tokens)",
            value=params.get("auto_summary_interval", 2000),
            step=100, minimum=1,
            info="New story tokens to accumulate since the last summary before triggering the next one.",
        )
        auto_summary_max_new_tokens_n = gr.Number(
            label="Max new tokens (summary)",
            value=params.get("auto_summary_max_new_tokens", 512),
            step=64, minimum=64,
            info="Maximum tokens the model may generate for the summary text.",
        )

    auto_summary_history_turns_n = gr.Number(
        label="History turns to include",
        value=params.get("auto_summary_history_turns", 40),
        step=1, minimum=1,
        info="Recent chat turns passed to the model when building the summary prompt.",
    )

    gr.HTML(
        '<div style="margin:14px 0 6px;padding:8px 12px;border-left:3px solid rgba(139,92,246,.7);'
        'background:rgba(139,92,246,.06);border-radius:0 6px 6px 0">'
        '<span style="font-weight:600;font-size:13px">Context counter scope</span>'
        '<span style="font-size:12px;color:var(--body-text-color-subdued)"> — '
        'choose what counts toward the summary interval.</span></div>'
    )
    auto_summary_include_char_card_cb = gr.Checkbox(
        label="Include character card in context counter",
        value=params.get("auto_summary_include_char_card", False),
        info=(
            "When enabled, the system prompt / character card tokens are added "
            "to the interval counter alongside the conversation turns. "
            "This makes the threshold reflect total context growth rather than "
            "chat turns alone — useful when the character card is large "
            "(1 000 + tokens) and you want the summary to fire based on "
            "everything the model sees, not just new dialogue."
        ),
    )

    with gr.Accordion("📑 Prompt templates", open=False, elem_classes="tgw-accordion"):
        gr.Markdown(
            "**Full summary prompt** — used for the very first summary and after *Force summary now*. "
            "Use `{conversation}` where the transcript should be inserted."
        )
        auto_summary_full_prompt_tb = gr.Textbox(
            label="Full prompt", lines=10,
            value=params.get("auto_summary_full_prompt", ""),
        )
        gr.Markdown(
            "**Delta summary prompt** — used for incremental updates. "
            "Use `{previous_summary}` and `{new_events}` as placeholders."
        )
        auto_summary_delta_prompt_tb = gr.Textbox(
            label="Delta prompt", lines=10,
            value=params.get("auto_summary_delta_prompt", ""),
        )

    gr.HTML(
        '<div style="margin:14px 0 6px;padding:8px 12px;border-left:3px solid rgba(139,92,246,.7);'
        'background:rgba(139,92,246,.06);border-radius:0 6px 6px 0">'
        '<span style="font-weight:600;font-size:13px">Manual trigger</span>'
        '<span style="font-size:12px;color:var(--body-text-color-subdued)"> — force a full re-summarise right now.</span></div>'
    )
    auto_summary_force_btn   = gr.Button("Force summary now", variant="secondary", elem_classes="refresh-button")
    auto_summary_status_md   = gr.Markdown("*No summary generated yet this session.*")

    return dict(
        auto_summary_enabled_cb=auto_summary_enabled_cb,
        auto_summary_interval_n=auto_summary_interval_n,
        auto_summary_max_new_tokens_n=auto_summary_max_new_tokens_n,
        auto_summary_history_turns_n=auto_summary_history_turns_n,
        auto_summary_include_char_card_cb=auto_summary_include_char_card_cb,
        auto_summary_full_prompt_tb=auto_summary_full_prompt_tb,
        auto_summary_delta_prompt_tb=auto_summary_delta_prompt_tb,
        auto_summary_force_btn=auto_summary_force_btn,
        auto_summary_status_md=auto_summary_status_md,
    )
