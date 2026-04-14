import re

from .config import params
from .state import _st
from .matching import _count_tokens


def _entry_label(e, i):
    uid = e.get("uid", i)
    comment = (e.get("comment", "") or f"Entry {uid}")[:32]
    status = "🔵" if e.get("constant") else ("✅" if e.get("enabled", True) else "❌")
    return f"{status} [{uid}] {comment}"


def _entry_choices():
    if not _st.current_lorebook:
        return []
    return [_entry_label(e, i) for i, e in enumerate(_st.current_lorebook.get("entries", []))]


def _uid_from_choice(choice):
    m = re.search(r'\[(\d+)\]', choice or "")
    return int(m.group(1)) if m else None


def _idx_from_uid(uid):
    if not _st.current_lorebook:
        return -1
    entries = _st.current_lorebook.get("entries", [])
    uid_map = {e.get("uid"): i for i, e in enumerate(entries)}
    return uid_map.get(uid, -1)


def _budget_bar_html():
    with _st.last_injection_lock:
        used = _st.last_injection_info.get("total_tokens", 0)
    budget = max(1, int(params.get("token_budget", 1024)))
    pct = min(100, round(used / budget * 100))
    color = "#22c55e" if pct < 70 else "#f59e0b" if pct < 90 else "#ef4444"
    return (
        f'<div style="margin-top:6px">'
        f'<div style="height:6px;border-radius:3px;background:rgba(0,0,0,.10);overflow:hidden">'
        f'<div style="height:100%;width:{pct}%;background:{color};transition:width .3s ease"></div>'
        f'</div>'
        f'<div style="font-size:11px;color:var(--body-text-color-subdued);margin-top:3px">'
        f'~{used}&thinsp;/&thinsp;{budget} tokens used last turn &nbsp;({pct}%)'
        f'</div>'
        f'</div>'
    )


def _build_stats_html():
    if not params.get("activate", True):
        return ('<div style="padding:10px 14px;border-radius:8px;font-size:13px;font-weight:500;'
                'background:rgba(239,68,68,.08);border:0.5px solid rgba(239,68,68,.4);'
                'color:var(--color-text-danger,#c0392b)">'
                '⚠ Lorebook system is <strong>OFF</strong> — enable it in Settings.</div>')

    with _st.last_injection_lock:
        info = dict(_st.last_injection_info)
    with _st.state_lock:
        active_lbs = list(_st.active_lorebooks.values())
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
    with _st.injection_history_lock:
        history = list(_st.injection_history_notebook if notebook else _st.injection_history)

    if not history:
        msg = "use the Notebook tab first" if notebook else "send a message first"
        return (f'<p style="color:var(--body-text-color-subdued);font-size:13px;padding:8px 4px">'
                f'No history yet — {msg}.</p>')

    _S = {
        "new":            ("🆕", "rgba(34,197,94,.14)",  "rgba(34,197,94,.55)",  "#15803d"),
        "repeat":         ("🔁", "rgba(99,102,241,.10)", "rgba(99,102,241,.40)", "var(--body-text-color-subdued)"),
        "returned":       ("↩️", "rgba(245,158,11,.13)", "rgba(245,158,11,.50)", "#b45309"),
        "budget_dropped": ("✂",  "rgba(251,146,60,.13)", "rgba(251,146,60,.50)", "#c2410c"),
    }

    cards = []
    for rec in history:
        entries        = rec["entries"]
        budget_dropped = rec.get("budget_dropped", [])
        dropped        = rec["dropped"]
        total          = len(entries)
        ints           = rec["interrupts"]

        int_note = f" &nbsp;·&nbsp; {ints} interrupt{'s' if ints != 1 else ''}" if ints else ""
        header = (
            f'<div style="padding:5px 10px;'
            f'background:var(--color-background-secondary,rgba(0,0,0,.05));'
            f'border-bottom:1px solid var(--border-color-primary,rgba(0,0,0,.08));'
            f'font-size:11px;font-weight:600;display:flex;justify-content:space-between;align-items:center">'
            f'<span>Turn {rec["turn"]}{int_note}</span>'
            f'<span style="font-weight:400;color:var(--body-text-color-subdued)">'
            f'~{rec["total_tokens"]} tokens &nbsp;·&nbsp; {rec["time"]}</span>'
            f'</div>'
        )

        row_html = []

        for i, e in enumerate(entries):
            if total == 1:
                pos = "#1"
            elif i == 0:
                pos = "#1 &nbsp;<span title='Lowest priority' style='opacity:.55;font-size:10px'>oldest</span>"
            elif i == total - 1:
                pos = f"#{total} &nbsp;<span title='Highest priority' style='opacity:.55;font-size:10px'>freshest</span>"
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
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued);white-space:nowrap;vertical-align:middle">{pos}</td>'
                f'<td style="padding:4px 8px;font-size:12px;vertical-align:middle">{e["label"]}</td>'
                f'<td style="padding:4px 8px;text-align:center;vertical-align:middle">{prio_badge}</td>'
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued);text-align:right;white-space:nowrap;vertical-align:middle">~{e["tokens"]} tok</td>'
                f'<td style="padding:4px 8px;text-align:right;vertical-align:middle">{badge}</td>'
                f'</tr>'
            )

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
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued);text-align:right;white-space:nowrap">~{e["tokens"]} tok</td>'
                f'<td style="padding:4px 8px;text-align:right">{badge}</td>'
                f'</tr>'
            )

        for lbl in dropped:
            row_html.append(
                f'<tr style="opacity:.40;border-bottom:1px solid var(--border-color-primary,rgba(0,0,0,.06))">'
                f'<td style="padding:4px 8px;font-size:11px;color:var(--body-text-color-subdued)">—</td>'
                f'<td style="padding:4px 8px;font-size:12px;text-decoration:line-through">{lbl}</td>'
                f'<td></td><td></td>'
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

    return '<div style="max-height:360px;overflow-y:auto;padding:2px 0">' + "".join(cards) + '</div>'


def _build_import_preview_html(lb, stats):
    name    = lb.get("name", "Unnamed")
    total   = stats.get("total", 0)
    n_const = stats.get("constant", 0)
    n_nokey = stats.get("no_keys", 0)

    pos_counts: dict = {}
    for e in lb.get("entries", []):
        p = e.get("position", "after_context")
        pos_counts[p] = pos_counts.get(p, 0) + 1
    _PLABELS = {"after_context": "after context", "before_context": "before context",
                "before_reply": "before reply",  "notebook": "notebook"}
    pos_str = ", ".join(
        f"{cnt}\u00d7\u00a0{_PLABELS.get(k, k)}"
        for k, cnt in sorted(pos_counts.items(), key=lambda x: -x[1])
    ) or "\u2014"

    def _row(label, value, warn=False):
        vc = "#b45309" if warn else "var(--color-text-primary,inherit)"
        return (
            f'<tr>'
            f'<td style="padding:4px 14px 4px 0;font-size:12px;'
            f'color:var(--body-text-color-subdued);white-space:nowrap">{label}</td>'
            f'<td style="padding:4px 0;font-size:12px;font-weight:500;color:{vc}">{value}</td>'
            f'</tr>'
        )

    rows = "".join([
        _row("Lorebook name",         f"<em>{name}</em>"),
        _row("Total entries",          total),
        _row("Constant (always-on \U0001f535)", n_const),
        _row("No trigger words",
             f"\u26a0\ufe0f {n_nokey} \u2014 will need keys to fire",
             warn=n_nokey > 0) if n_nokey else _row("No trigger words", "none"),
        _row("Position breakdown",     pos_str),
    ])

    return (
        '<div style="padding:10px 14px;border-radius:8px;'
        'border:1px solid rgba(139,92,246,.45);background:rgba(139,92,246,.06);margin:6px 0 2px">'
        '<div style="font-size:13px;font-weight:600;margin-bottom:8px">'
        '\U0001f4cb\u00a0Import preview \u2014 review then click <strong>Confirm&nbsp;Import</strong></div>'
        f'<table style="border-collapse:collapse">{rows}</table>'
        '</div>'
    )


def custom_css():
    return """
/* ── Buttons ── */
.lb-btn-primary{border-color:rgba(139,92,246,.8)!important;background:rgba(139,92,246,.15)!important}
.lb-btn-primary:hover{background:rgba(139,92,246,.28)!important;border-color:rgba(139,92,246,1)!important}
.lb-btn-danger{border-color:rgba(239,68,68,.7)!important;background:rgba(239,68,68,.08)!important}
.lb-btn-danger:hover{border-color:rgba(239,68,68,1)!important;background:rgba(239,68,68,.18)!important}

/* ── Active lorebook pills ── */
#lb-active-pills .wrap{display:flex!important;flex-wrap:wrap!important;gap:6px!important;padding:4px 0!important}
#lb-active-pills .wrap label{display:inline-flex!important;align-items:center!important;gap:5px!important;padding:3px 12px 3px 8px!important;border-radius:20px!important;border:1px solid var(--border-color-primary)!important;background:var(--button-secondary-background-fill)!important;font-size:12px!important;cursor:pointer!important;white-space:nowrap!important}
#lb-active-pills .wrap label:hover{border-color:rgba(139,92,246,.7)!important}
#lb-active-pills input[type=checkbox]{accent-color:#8b5cf6!important;width:13px!important;height:13px!important}
#lb-active-pills .gap{display:none!important}

/* ── Checkbox alignment ── */
#lorebook-tab .row > .checkbox-wrap,
#lorebook-tab .row > div > .checkbox-wrap{margin-top:auto!important}

/* ── Section label ── */
.lb-section-label p{font-size:12px!important;font-weight:600!important;color:var(--body-text-color-subdued)!important;margin:10px 0 2px!important}

/* ── Misc ── */
#lorebook-tab textarea{resize:vertical!important}
"""
