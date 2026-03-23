# Lorebook Extension for Oobabooga

A full-featured Lorebook / World Info extension for [oobabooga's text-generation-web-ui](https://github.com/oobabooga/text-generation-webui). Adds dynamic, keyword-triggered context injection across **instruct mode, chat mode, chat-instruct mode, and text completion (notebook) mode** — with a built-in GUI editor, SillyTavern import/export, mid-generation interrupts, recursive scanning, and persistent settings.

---
## Quick Start

1. Open the **Lorebook** tab and click **New** to create a lorebook.
2. Give it a name and click **Save**.
3. Open the **Entries** accordion and click **+ Add** to create your first entry.
4. Fill in **Trigger words** and **Content**, then click **Save entry**.
5. In the right panel's **Lorebooks** tab, check the checkbox next to your lorebook to activate it.
6. Start a generation — matching keywords in the conversation will inject your world info automatically.

---

## Installation

1. Clone or copy this repository into your `text-generation-webui/user_data/extensions/` folder:
   ```
   text-generation-webui
   └── user_data
       └── extensions
           ├── script.py
           └── lorebooks   ← created automatically on first run
   ```

2. Launch oobabooga with `--extensions lorebook` or enable it from the **Extensions** tab in the UI.

3. A **Lorebook** tab will appear in the interface.
- Mid-generation interrupt requires streaming mode to be active in generation settings.
- SillyTavern world-info `.json` files are directly importable without modification.
  
---

## Compatibility

- Tested with oobabooga's text-generation-web-ui.
- Tested by  importing silly tavern lorebooks.
  
---

## Features

### Generation Mode Support

- **Instruct mode** — injects world info into `custom_system_message`, the correct prompt key for instruct templates. Entries set to *Between User and Assistant* are placed immediately before the assistant turn header so the model reads them at maximum recency.
- **Chat / Chat-Instruct mode** — injects into `context`. Supports all four injection positions relative to the conversation structure.
- **Notebook / Text Completion mode** — entries set to *Notebook Mode* are appended directly to the end of the raw prompt text so the model continues generation with world info in scope.

---

### Lorebook Management

- Create, rename, save, and delete lorebook files through the UI — no manual file editing required.
- All lorebooks are stored as `.json` files in the extension's `lorebooks/` folder.
- **Multiple lorebooks can be active simultaneously.** Toggle any number of them on or off with checkbox pills; they stack and are all scanned during generation.
- Active lorebook state persists across restarts — the extension remembers which lorebooks you had enabled.
- User settings (scan depth, token budget, injection strings, etc.) are also persisted to disk and restored on load.

---

### Entry Editor

Each lorebook can contain any number of entries. Every entry has its own configurable fields:

| Field | Description |
|---|---|
| **Entry name** | Internal label. The AI never reads this. |
| **Trigger words** | Comma-separated primary keywords. Any one match fires the entry. |
| **Also require** | Optional secondary keywords evaluated against the logic setting below. |
| **Secondary key logic** | `AND ANY` · `AND ALL` · `NOT ANY` · `NOT ALL` |
| **Content** | The text injected into the prompt when the entry fires. |
| **Enabled** | Toggle the entry on or off without deleting it. |
| **Constant (always on)** | Entry always fires regardless of trigger words. |
| **Case sensitive** | Match trigger words with or without case sensitivity. |
| **Whole words only** | Require full word boundaries around matches (default on). |
| **Use regex** | Treat trigger words as regular expressions. |
| **Priority** | Higher-priority entries are injected first and win tie-breaks. |
| **Scan depth** | How many messages back to scan for this entry's keywords. Overrides the global setting per entry. |
| **Position** | *After System Prompt* or *Before System Prompt* (see Injection Positions). |
| **Trigger %** | Probability (0–100) that the entry fires when its keywords match. 100 = always, 50 = coin flip. |
| **Inclusion group** | Only the highest-priority entry within a group fires per turn. Useful for mutually exclusive lore (e.g. time of day, weather). |

---

### Injection Positions

Each entry can be placed at one of four positions in the prompt. A global **context position override** in Settings can force all entries to a single position, ignoring individual settings.

| Position | Where it lands |
|---|---|
| **After System Prompt** | Injected below the character context / system message (default). Stays fresh and close to the conversation. |
| **Before System Prompt** | Injected above the system message. Good for grounding world rules at the very top. |
| **Between User and Assistant** | Injected just before the assistant turn header. Maximum recency — the model reads it immediately before replying. Supports instruct, chat, and chat-instruct modes. |
| **Notebook Mode** | Appended to the end of the text completion prompt. Text completion mode only. |

---

### Scanning Engine

- **Keyword scanning** covers the current user message plus N messages of history (configurable scan depth).
- **Whole-word matching** uses lookaround assertions (`(?<!\w)` / `(?!\w)`) so punctuation-adjacent keys match correctly.
- **Regex support** per entry for advanced matching patterns.
- **Selective / secondary key logic** (`AND ANY`, `AND ALL`, `NOT ANY`, `NOT ALL`) for fine-grained control over when entries fire.
- **Scan depth = 0** means current message only; positive values scan that many messages back into history; the global override defaults to the per-entry setting when set.
- **Only trigger words in chat** mode — when enabled, the character persona/system context is stripped from the scan target so persona keywords never accidentally fire entries every turn. Only actual chat messages are scanned.

---

### Recursive Scanning

When enabled, the *content* of matched entries is itself scanned for keywords that could trigger additional entries. This allows chains of lore to activate automatically — mention a location, which triggers an entry about the city guard, which triggers an entry about the city's ruler.

- Configurable number of recursion steps (default 3, max 10) to prevent runaway chains.
- Inclusion group filtering and token budget limits are enforced at every recursion pass.

---

### Token Budget

- A global **token budget** caps the total world info injected per turn.
- Entries are trimmed to fit within the budget, with higher-priority entries kept first.
- **Chars per token** is configurable for accurate estimation across different tokenizers (default 4).
- The Overview tab shows the estimated token cost of the last injection.

---

### Mid-Generation Interrupt

An optional advanced feature that monitors the model's output token-by-token while it streams and injects new world info if the model mentions a keyword mid-reply.

- When a new trigger word appears in streamed output, generation is **paused**, the WI block is updated with the new entry, and generation **resumes** from where it stopped.
- **Max interrupts per reply** is configurable (default 3) to avoid excessive re-starts.
- Requires streaming to be enabled in generation settings.
- Inclusion group filtering and budget trimming are re-applied on each interrupt so the prompt stays coherent.

> **Note:** Mid-gen interrupt is a chat-mode feature. Notebook mode uses the initial scan only.

---

### Overview & Injection Preview

- **Overview tab** — shows a quick table of all entries in the currently open lorebook (name, keys, status, position, priority).
- **Stats panel** — live display of how many lorebooks are active, how many entries they contain, approximate tokens injected last turn, and how many mid-gen interrupts fired.
- **Last injection preview** — after each generation, shows exactly which entries fired and their estimated token cost. Useful for debugging unexpected lorebook behaviour.

---

### SillyTavern Import / Export

Full round-trip compatibility with SillyTavern world-info files.

**Import:**
- Drop any SillyTavern `.json` world-info file into the Import tab and click **Import**.
- Preserves: keys, secondary keys, selective logic, content, constant flag, enabled/disabled state, priority (`insertion_order`), scan depth, probability, inclusion group (`group`), case sensitivity, whole-word, and regex flags.
- Entries with no trigger words are flagged with a warning label so you can add keys before use.
- The editor populates immediately after import — no manual reload needed.

**Export:**
- Export any open lorebook back to a SillyTavern-compatible `.json` file for use in other tools.
- All fields round-trip losslessly. `before_reply` and `notebook` positions export as ST position 2 (AT_DEPTH), the closest semantic equivalent.

---

### Settings (persist across restarts)

| Setting | Default | Description |
|---|---|---|
| Lorebook system active | On | Master on/off switch. |
| Scan depth override | -1 (disabled) | Global message history scan depth. -1 = use per-entry setting. 0 = current message only. |
| Token budget | 1024 | Max tokens of world info per turn. |
| Chars per token | 4 | Tokenizer estimation factor. |
| Max recursion steps | 3 | Caps recursive scanning passes. |
| Inject constant entries | On | Always-on entries fire every turn. |
| Recursive scanning | On | Matched entry content is scanned for further keywords. |
| Only trigger words in chat | Off | Prevents persona text from firing entries. |
| Injection prefix | `\n[World Info]\n` | Wraps the injected block. |
| Injection suffix | `\n[/World Info]` | Closes the injected block. |
| Mid-gen interrupt | Off | Monitor streamed output for new triggers. |
| Max interrupts per reply | 3 | How many times a single reply can be interrupted. |
| Context position override | Off | Force all entries to one position. |
