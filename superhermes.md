# Superhermes

Observability, control words, and memory improvements built on top of the Hermes Agent core. These features work across all interfaces: CLI, TUI, messaging platforms, and API.

---

## /pi — Prompt Inspector

`/pi` shows a breakdown of every component in the system prompt sent to the model, including conversation messages, tool schemas, and server parameters. Each part is labeled with its origin (file-backed, hardcoded, or generated), size in bytes, and source path where applicable.

### Commands

| Command | Purpose |
|---------|---------|
| `/pi` | Show prompt overview table |
| `/pi all` | Show parts from all recent requests, not just the latest |
| `/pi <line>` | Drill into a specific part (human-readable) |
| `/style raw` | Switch drill-down to verbatim wire JSON |
| `/style human` | Switch drill-down back to decoded text (default) |
| `/redirect <path>` | Route /pi output to a file (appended on each line; target is re-opened per write, so deleting it mid-session is safe) |
| `/redirect off` | Stop redirecting; print to terminal again |

### Wire captures in /tmp

Every API call writes its outgoing body to the following files. These are ground-truth artifacts: useful for auditing what was actually sent, comparing against /pi's model of it, or debugging prompt-cache issues.

| File | Contents |
|------|----------|
| `/tmp/prompt.txt` | Exact bytes of the last outgoing request body (SDK-serialized) |
| `/tmp/prompt.json` | Human-readable JSON of the same body (indented) |
| `/tmp/prompt_part_<n>.txt` | Raw slice per conversation message (debug output, indexed by position) |

> Note: with multiple concurrent Hermes instances, each overwrites these files on its own calls. Correlate with `prompt_lines` timestamps for byte-exact analysis.

---

## Control words

Type a single word by itself to trigger a special behavior. Both are case-insensitive.

### `done`

Triggers a session wrap-up. The agent reads instructions from `~/.hermes/messages/DONE.md` (or a built-in default) and performs it: updating memory, saving skills, and checkpointing notes — without starting new work.

Intended for use just before shutting down the machine or ending a long session.

### `stop`

Emergency brake. Two behaviors:

- **Mid-turn**: aborts the in-flight model request (closes the stream connection to the model server) and ends the turn immediately. Running external tools are not killed.
- **At turn start** (before any tools have run): tool calls are blocked for that turn; the session remains ready for new instructions.

Intended use case: you see the agent about to run something dangerous, and you want it to halt before anything executes.

---

## Memory file restructure

The injected memory files were reorganized on 2026-10-02 to separate concerns cleanly:

| File | Purpose |
|------|---------|
| `~/.hermes/memories/MEMORY.md` | Scratch pad only. Temporary working state (in-flight tasks, transient notes). No permanent entries — the rule is enforced in RULES.md itself. |
| `~/.hermes/memories/RULES.md` | Standing rules that apply to every session regardless of task. Injected into the system prompt before MEMORY.md. Edit directly; takes effect on next restart. |
| `~/.hermes/memories/HISTORY.md` | Index of all projects worked on together. Keywords and pointers for each project, so context is recognized without loading details. |
| `~/.hermes/SOUL.md` | Persona file (unchanged from upstream). |

---

## Session Search Improvements (2026-10-03)

Clear documentation for using `~N` shorthand to recall previous sessions directly, avoiding unnecessary browsing steps.

### Problem

Previous implementation required agents to first browse for session IDs then query by explicit ID, leading to confusion and potential hallucination about what was actually called.

### Solution

Enhanced tool schema documentation in three places:

1. **Main tool description**: Now explicitly mentions using `~1` to recall the previous session
2. **session_id parameter**: Documents two usage patterns - read shape (with `~N` or concrete ID) and scroll shape (with around_message_id)
3. **read_head/read_tail parameters**: Both mention pairing with `~1` for their respective use cases (session start/end context)

### Usage Pattern

```python
# Recall previous session context directly
session_search(session_id='~1', read_tail=10)  # Where did we leave off?
session_search(session_id='~1', read_head=5)   # How did it start?
```

This makes the recall workflow explicit in the tool's own documentation, reducing reliance on memory/rules alone.

---

## Previous-Session Context Injection (2026-10-03)

When a session starts with `--continue`, the last exchange from the previous session is injected as context on the first API call, so the model retains recency even after compression or restart.

- Injected exactly once per session: only when no assistant response exists yet.
- Retrieves both user and assistant messages from the handoff (previous version retrieved assistant-only).
- Implemented in `agent/session_context_loader.py` / `agent/turn_context.py`.

### `--fresh` flag

`hermes --fresh` skips previous-session context injection entirely and starts clean. Use when the carried-over context is stale or harmful to the current task.

---

## /skills loaded — Skill Observability (2026-10-03)

`/skills loaded` lists every skill loaded in the current session via `skill_view`, with load timestamp and cumulative load count per (skill, file). Similar purpose to `/pi` but for skill usage.

Tracking records at both full-load and deduplication paths; deliberately not reset on context compression, since skills remain available in summarized context until the session ends.

---

## Transparent Tool-Call Display (2026-10-03)

Replaces the "friendly verb" renderer in CLI scrollback with a transparent format: real tool name, all arguments (with `path="."` resolved to absolute), and a result line — `✓ SUCCESS — <summary>` or `✗ ERROR: <message>`.

Example output:

```
⚡ search_files (3 args)  0.4s
    pattern=def get_cute_tool_message
    path=/home/hermes/workspace/hermes
    file_glob=*.py
    ✓ SUCCESS — 5 matches
```

Outcome summaries are tool-specific where derivable: match counts, exit codes, line/byte totals. Secret redaction unchanged. CLI-only; other interfaces keep the friendly labels.

Implementation: `get_transparent_tool_message()` in `agent/display.py`, called from `_on_tool_progress` in `hermes_cli/cli_stream_mixin.py`.

Status: deployed and live-verified (2060a1fab0).

---
