"""TEMP debug: log every live-transcript message append as a DEBUG box line (CLI only).

Instrumentation for the steering/display work — remove when done. Centralized so
append_message (and the direct tool-result appends) each need one call.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger("run_agent")

# One row per logical message; long text is abbreviated to fit a single terminal line:
# first N chars + "..." + last N chars, so both ends of the message stay visible.
_ABBREV_CHARS = 30


def debug_log_message_append(message: Mapping[str, Any], tag: str = "") -> None:
    """Print kind + abbreviated content for an appended transcript message.

    ASSISTANT messages use two lines (reasoning first, then content); all others one.
    When ``tag`` is given, every emitted line carries the ``[TAG] `` prefix so field
    descriptions stay attached to their source site. No-op outside the interactive CLI
    (tests, gateway, headless) and on any error — instrumentation must never break a turn."""
    try:
        from agent.conversation_loop import ACTIVE_CLI, LineKind, print_text_line

        if ACTIVE_CLI is None or not isinstance(message, dict):
            return
        prefix = f"[{tag}] " if tag else ""
        role = str(message.get("role", "?"))
        if role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            content = _abbrev(_content_text(message))
            print_text_line(LineKind.DEBUG, f"{prefix}ASSISTANT (reasoning): {_abbrev(reasoning)}")
            print_text_line(LineKind.DEBUG, f"{prefix}ASSISTANT: {content}")
        elif role == "tool":
            name = message.get("name") or ""
            print_text_line(LineKind.DEBUG, f"{prefix}TOOL {name}: {_abbrev(_content_text(message))}")
        else:
            print_text_line(LineKind.DEBUG, f"{prefix}{role.upper()}: {_abbrev(_content_text(message))}")
    except Exception as exc:  # noqa: BLE001 — debug path must never raise into the loop
        logger.debug("debug_log_message_append failed: %s", exc)


def _content_text(message: Mapping[str, Any]) -> str:
    """Flatten a message's content (str or content-block list) to displayable text."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                part = block.get("text") or block.get("thinking") or ""
                if part:
                    parts.append(str(part))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return "" if content is None else str(content)


def _abbrev(text: Any, chars: int = _ABBREV_CHARS) -> str:
    text = " ".join(str(text or "").split())  # single-line, collapsed whitespace
    if len(text) > chars * 2 + 3:  # abbreviated form is exactly chars*2+3 wide — no gain below that
        return text[:chars] + "..." + text[-chars:]
    return text
