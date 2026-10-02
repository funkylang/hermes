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

Procedural knowledge lives in skills (loaded only when relevant), not in these files. Memory has a strict character budget; overflow is handled by replacing stale entries, never by expanding the limit.
