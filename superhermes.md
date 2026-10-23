# Superhermes

Observability, control words, steering/agent-control, and memory improvements built on top of the Hermes Agent core. These features work across all interfaces: CLI, TUI, messaging platforms, and API.

> **Last updated 2026-10-23.** Verify current state with `git log --oneline -1`.

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

The capture path is configurable: set `HERMES_PROMPT_CAPTURE` to any file path (the `.json` companion is derived by suffix swap). The hook fails open — capture errors never break a turn.

> Note: with multiple concurrent Hermes instances, each overwrites these files on its own calls.

---

## Smart Steering Mode

The `smart` busy input mode provides phase-aware steering behavior that improves user experience when redirecting the agent while it's working.

### How it works (UNIFIED DESIGN, 2026-10-23)

**Core principle**: ONE place owns steering delivery — both display boxes AND wire rows must match exactly.

When you type a new instruction while the agent is processing:

1. **During reasoning**: At newline boundaries, `handle_stream_event` renders input boxes and appends user wire rows with real accumulated_text (byte-identical to screen)
2. **During tool execution/argument streaming**: Steering deferred until current batch completes; boxes + wire rows appended at phase boundary
3. **Multiple steers**: Consecutive messages concatenate cleanly without visual noise or decorations

### Architecture

**Single owner**: `agent/stream_delivery.py::StreamDeliveryMixin.handle_stream_event(kind, text, title)` is the sole steering delivery mechanism. At phase boundaries it:
- Renders display boxes (INPUT/RESPONSE/TOOL_RESULT etc.)
- Appends wire rows to conversation history with ACTUAL streamed text (via `accumulated_text`)
- Guarantees display and prompt are IDENTICAL — no more drift between what you see vs what model receives

**Tagged observability**: Every append site is tagged (`append_message(tag=...)`), so debug logs show which code path wrote each message: `HSE-*` (steering boundary), `RED-*` (redirect), `CLI-USR` (user input), etc.

### Usage

Smart mode is enabled by default in the CLI:

```bash
/busy smart        # Smart steering (default, recommended)
/busy interrupt    # Immediate interruption mode
/busy queue        # Queue mode (waits for turn to complete)
/busy steer        # Steer mode (legacy behavior)
```

### Configuration

Set your preferred busy input mode in `~/.hermes/config.yaml`:

```yaml
display:
  busy_input_mode: smart  # Options: smart, interrupt, queue, steer
```

Or use the environment variable for gateway processes:

```bash
export HERMES_GATEWAY_BUSY_INPUT_MODE=smart
```

---

### `done`

Triggers a session wrap-up. The agent reads instructions from `~/.hermes/messages/DONE.md` (or a built-in default) and performs it: updating memory, saving skills, and checkpointing notes — without starting new work.

Intended for use just before shutting down the machine or ending a long session.

### `stop`

Emergency brake. Two behaviors:

- **Mid-turn**: aborts the in-flight model request (closes the stream connection to the model server) and ends the turn immediately. Running external tools are not killed.
- **At turn start** (before any tools have run): tool calls are blocked for that turn; the session remains ready for new instructions.

Intended use case: you see the agent about to run something dangerous, and you want it to halt before anything executes.

### Recognition rules

Both words are standalone-only: the stripped message must equal the word exactly (case-insensitive). Anything else — e.g. `done with step 3` or `stop the server` — passes through to the model untouched. For `done`, the injected wrap-up instruction is API-local: only that turn's model-facing copy carries it, so the stored transcript keeps your clean word.

---

## Memory file restructure

The injected memory files were reorganized on 2026-10-02 to separate concerns cleanly:

| File | Purpose |
|------|---------|
| `~/.hermes/memories/MEMORY.md` | Scratch pad only. Temporary working state (in-flight tasks, transient notes). No permanent entries — the rule is enforced in RULES.md itself. Written via the memory tool (`target='memory'`). |
| `~/.hermes/memories/USER.md` | Who the user is: persona, preferences, work style. Written via the memory tool (`target='user'`). Injected into the system prompt. |
| `~/.hermes/memories/RULES.md` | Standing rules that apply to every session regardless of task. Injected into the system prompt before MEMORY.md. Edit directly; takes effect on next restart. |
| `~/.hermes/memories/HISTORY.md` | Index of all projects worked on together. Keywords and pointers for each project, so context is recognized without loading details. |
| `~/.hermes/SOUL.md` | Persona file (unchanged from upstream). |

New vs upstream: `USER.md`, `RULES.md`, `HISTORY.md`. All are plain markdown — hand-editable; the agent routes writes through the memory tool / its own discipline per RULES.md.

### Explicit-control setting

Upstream's automatic `background_review` (a fork that periodically rewrites memory/skills on its own) is disabled here so those writes happen only when you ask, via `done`:

```yaml
auxiliary:
  background_review:
    enabled: false
```

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

## Previous-Session Context Injection (2026-10-03, corrected 2026-10-18)

On a new session, context from the previous session is injected on the first API call so the model keeps recency after restart or compression — automatically, no flag needed.

- **Session selection:** the most recently active previous session that has at least 10 substantive messages. Eligibility uses `substantive_message_count` — a SQL count of user/assistant messages with non-empty content computed from the message rows. It deliberately never reads the stored `sessions.message_count` counter, which also counts tool noise and can drift (a stale value once selected a session that really held only a handful of messages).
- **Content:** if the chosen session has at most 16 substantive messages, all of them are injected; otherwise its first 8 plus its last 8 — so both where the session started and how it ended carry over.
- Injected exactly once per session: only when no assistant response exists yet.
- Implemented in `agent/session_context_loader.py` / `agent/turn_context.py`; counts via `hermes_state_messages.substantive_message_count`.

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

## Input Hook — Agent Control Script

A user-configured script that runs after each assistant turn and can decide whether to auto-reply or hand control back to the human. This is the mechanism for external agent-control systems: a supervising script manages orchestration while the LLM handles the actual work.

### How it works

After every completed assistant turn, Hermes invokes your script with three arguments:

1. `reasoning` — the assistant's reasoning text (may be empty)
2. `content` — the assistant's final response text
3. `metadata_block` — a multi-line `key=value` block with: `context_used`, `context_size`, `total_messages`, `finish_reason`, `session_id`

Decision logic:
- **exit 0 + non-empty stdout** → the stdout is queued as the next user message (auto-reply; the agent continues)
- **non-zero exit OR empty stdout** → control returns to the human (normal prompt)

### Configuration

Set in `~/.hermes/config.yaml`:

```yaml
input_hook:
  script: /path/to/hook.sh   # Your control script (required to enable)
  timeout: 30                # Seconds before the script is killed (default 30, min 1)
```

Or override with the environment variable `HERMES_INPUT_HOOK` (takes precedence; handy for testing). If no script is configured, the feature is inert.

Implementation: `hermes_cli/input_hook.py`, wired from `hermes_cli/cli_input_hook_mixin.py`.

---

## Live Tool-Arg Streaming (Dim Box)

While the model generates a tool call's arguments, the raw JSON is streamed into a dim box (styled like the reasoning box) as it arrives. Useful for watching large payloads form (e.g. `write_file` content, long queries); harmless noise otherwise.

The box opens on the first chunk with a label like ` Tool args: write_file`, re-labels if a different call streams next, and closes when the tool actually starts running or at content/stream boundaries.

Implementation: `_on_tool_args_stream()` in `hermes_cli/cli_stream_mixin.py` (shares rendering with the reasoning box).

---

## True Wire-Byte Status Display

In quiet mode, the waiting indicator starts with a neutral placeholder and is updated with the **real** character count once the wire bytes are actually known (via the `on_prompt_sent` hook) — never a stale pre-flight estimate. This gives an accurate "Sent N chars" signal instead of a fabricated token guess.

Implementation: `agent/prompt_capture.py` (`set_on_prompt_sent`) + `agent/turn_iteration_prep.py`.

---

## Debug Instrumentation (2026-10-23)

CLI-only debug logging for tracing steering behavior and stream lifecycle. All instrumentation is tagged and exception-safe — it never breaks a turn.

### PROMPT_SEND

Logs when each API prompt is sent, showing message count, call role, and API mode. Two locations:

- **Non-stream**: `agent/relay_llm.py::execute()` — logs before each non-streaming provider call
- **Stream**: `agent/relay_llm.py::ManagedLlmStream.__init__()` — logs at stream initialization with a unique stream ID (0x...) for correlating with STREAM_CLOSE events

Format: `PROMPT_SEND [non-stream | 0x<id>] msgs=<count> role=<call_role> api_mode=<api_mode>`

### STREAM_CLOSE

Logs when streams are closed, with UNIQUE tags to identify the close site. Only logs once per stream (guarded by `_closed` flag) — multiple close callsites won't duplicate events.

Four distinct close locations:
- `SCR-F`: Finalizer close (normal completion path)
- `SCR-H`: Halt abort (mid-turn "stop" control word)
- `SCR-S`: Stream end (model sent finish_reason, stream naturally complete)
- `SCR-B`: Buffer drain (async cleanup after main loop exits)

Format: `STREAM_CLOSE <tag> 0x<id> [reason]` where reason varies by close type.

### Usage

Enable via environment variable or flag:

```bash
# Runtime flag
hermes --debug

# Or environment variable
export HERMES_DEBUG_LOG=1
```

Debug events log to `/tmp/hermes_debug.log` (configurable via `HERMES_DEBUG_LOG_FILE`). Use with grep/tail to trace steering boundary behavior and correlate prompt send → stream close sequences.

**Temporary**: This instrumentation is TEMPORARY — it's currently enabled to help debug the steering race condition (see improve/todo.txt "STEERING CRASH RACE"). Will be removed once that issue is resolved.

---
