"""Turn-level user-input control words: standalone whole-message commands detected before the
model call so they can shape how ONE turn runs without changing what is durably recorded.

Currently implements "done" (Superhermes): the user types exactly ``done`` as a message and we
tell the model to checkpoint its work (update memory/skills/notes, finish current activity)
without starting new tasks — the pre-shutdown use case. The instruction text itself is
user-editable via ``~/.hermes/messages/DONE.md``; a built-in default is used when that file is
absent or blank, so the feature works out of the box.

Also implements "halt": an emergency brake command that immediately blocks all NEW tool calls for
the current turn (does not interrupt running tools). The user types "halt" to prevent destructive
operations when they notice the agent is about to do something dangerous. Detection only sets a
flag; the actual gating happens in run_tool_round.py before execution.

Design notes:
- Recognition is standalone-only: the stripped message must equal the word exactly,
  case-insensitive. Anything else (e.g. "done with step 3" or "halt the server") passes through untouched.
- The prepended instruction is API-local: only this turn's model-facing copy of the user message
  carries it. ``persist_user_message`` is pinned to the original clean text so the durable
  transcript row stays exactly what the user typed (mirrors how voice-input prefixes are handled
  in hermes_cli/cli_chat_turn_mixin.py::_chat_run_agent).
- This is wired centrally in agent.conversation_loop._run_conversation_turn so it applies to
  every platform (CLI, TUI/Desktop gateway, ACP, API server) that funnels through
  run_conversation — not just the interactive CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# The single recognized control word, matched exactly after strip() + lower().
_DONE_WORD = "done"

# The second control word for emergency halting of tool calls.
_HALT_WORD = "halt"

# Name of the user-editable file under ~/.hermes/messages/. Uppercase to match the convention
# used by RULES.md / MEMORY.md / USER.md / HISTORY.md.
_DONE_FILE_NAME = "DONE.md"

# Fallback instruction used when ~/.hermes/messages/DONE.md is absent or blank. Written from
# the agent's own perspective (second person "you") since it is injected as a system-style
# notice addressed to the model, not shown to the user verbatim (the user sees their own typed
# word, "done").
DEFAULT_DONE_INSTRUCTION = (
    "[SYSTEM NOTICE — session wrap-up requested] The user has asked you to wrap up: the session "
    "or computer will be shut down soon. Before anything else, persist what is worth keeping: "
    "update your memory, skills and any relevant notes with the important facts, decisions or "
    "open threads from this conversation. You may RESUME finishing work you are already in the "
    "middle of, but do NOT start new tasks, new searches, or new discovery work. When finished, "
    "summarize briefly what you saved and confirm you are ready for the session to end."
)


def is_standalone_control_word(message: Any, word: str) -> bool:
    """True when ``message`` is exactly ``word`` after whitespace stripping, case-insensitive."""
    if not isinstance(message, str):
        return False
    return message.strip().lower() == word.lower()


def _read_control_word_file(filename: str, default_text: str) -> str:
    """Read a user-editable instruction file from ``~/.hermes/messages/``, falling back to
    ``default_text``.

    Failures (unreadable file, missing directory, permission error, etc.) log at warning and
    fall through to the default — this hook must never break a turn because of a bad config
    file or missing directory.
    """
    try:
        from hermes_constants import get_hermes_home

        path = Path(get_hermes_home()) / "messages" / filename
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception as exc:  # noqa: BLE001 — never let this raise into the turn loop
        logger.warning("Failed to read control-word file %s (using default): %s", filename, exc)
    return default_text


def is_standalone_halt_control_word(message: Any) -> bool:
    """True when ``message`` is exactly "halt" after whitespace stripping, case-insensitive."""
    return is_standalone_control_word(message, _HALT_WORD)


def apply_done_control_word(user_message: Any, persist_user_message: Optional[Any]) -> Tuple[Any, Optional[Any]]:
    """Detect a standalone "done" message and inject its wrap-up instruction for this turn only.

    Returns ``(user_message, persist_user_message)``, possibly modified: when the control word
    is recognized, ``user_message`` gets the prepended instruction (seen by the model, not
    persisted) and ``persist_user_message`` is pinned to the original clean text so the stored
    transcript row is exactly what the user typed. Non-matching input returns unchanged.
    """
    if not is_standalone_control_word(user_message, _DONE_WORD):
        return user_message, persist_user_message

    note = _read_control_word_file(_DONE_FILE_NAME, DEFAULT_DONE_INSTRUCTION)
    prefixed = f"{note}\n\n{user_message}"
    clean = user_message if persist_user_message is None else persist_user_message
    logger.info(
        "Standalone 'done' control word detected; wrap-up instruction injected for this turn only."
    )
    return prefixed, clean
