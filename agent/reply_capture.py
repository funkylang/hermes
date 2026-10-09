"""Reply capture for debugging interruptions.

Stores the complete generated output + finish_reason to /tmp/reply_N.txt files
for post-hoc analysis of streaming issues and interruptions.

Configuration:
- HERMES_REPLY_CAPTURE env var sets the directory (default: /tmp)
- Each process gets its own numbered sequence
- First capture in each process clears stale reply_*.txt from previous sessions

All functions are fail-open: errors never block streaming.
"""

from __future__ import annotations

import glob
import os


# Reply capture for debugging interruptions: stores the complete generated output + finish_reason
_REPLY_COUNTER = 0
_REPLY_CAPTURE_ENV = "HERMES_REPLY_CAPTURE"
_REPLY_DEFAULT_DIR = "/tmp"


def reply_capture_dir() -> str:
    """Directory for reply captures (env override wins; read live, never cached).

    Defaults to /tmp (the live session's log location that downstream replay tooling
    reads). Parallel test runs must set HERMES_REPLY_CAPTURE to an isolated dir so their
    first-write cleanup glob cannot delete the live session's reply_*.txt files.
    """
    return os.environ.get(_REPLY_CAPTURE_ENV) or _REPLY_DEFAULT_DIR


def next_reply_capture_path() -> str:
    """Return a numbered path for the next reply capture (increments each call).

    On the very first call in this process, cleans up any leftover reply_*.txt files from
    previous sessions — within the capture dir only. The dir is configurable via
    HERMES_REPLY_CAPTURE so parallel test runs never touch the live session's logs.
    """
    global _REPLY_COUNTER
    base = reply_capture_dir()
    if _REPLY_COUNTER == 0:
        # First reply of this process — clear stale captures from previous runs (own dir only)
        try:
            for path in glob.glob(os.path.join(base, "reply_*.txt")):
                os.unlink(path)
        except Exception:
            pass
    _REPLY_COUNTER += 1
    return os.path.join(base, f"reply_{_REPLY_COUNTER}.txt")


def write_reply_capture(content: str | None, finish_reason: str,
                        reasoning: str | None = None,
                        tool_calls: list | None = None) -> None:
    """Write the complete wire response to /tmp/reply_N.txt for debugging."""
    try:
        parts: list[str] = []
        if reasoning:
            parts.append(f"== REASONING ==\n{reasoning}")
        if content:
            parts.append(f"== CONTENT ==\n{content}")
        if tool_calls:
            import json
            parts.append(f"== TOOL CALLS ==\n{json.dumps(tool_calls, indent=2)}")
        parts.append(f"[[finish_reason]] {finish_reason}")

        with open(next_reply_capture_path(), "w", encoding="utf-8") as f:
            f.write("\n\n".join(parts) + "\n")
    except Exception:
        pass  # Never block streaming for debugging
