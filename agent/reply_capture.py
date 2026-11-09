"""Reply capture for debugging interruptions.

Stores the complete generated output + finish_reason to ~/logs/reply_N.txt files
for post-hoc analysis of streaming issues and interruptions.

Configuration:
- HERMES_REPLY_CAPTURE env var sets the directory (overrides the default)
- Default target is ~/logs, gated by a ONE-SHOT startup check: if ~/logs does
  not exist at first use, reply logging is skipped for this whole process —
  it is never created automatically (Michael, 2026-11-09: multi-agent /tmp
  collision fix)
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

# One-shot startup gate: ~/logs must already exist; then logging is on for the
# whole process, off otherwise. Checked exactly once (Michael, 2026-11-09).
_logs_dir_checked = False
_logs_dir_ok = False


def _default_logs_dir_available() -> bool:
    """Check ~/logs exactly once per process; never create it."""
    global _logs_dir_checked, _logs_dir_ok
    if not _logs_dir_checked:
        _logs_dir_checked = True
        _logs_dir_ok = os.path.isdir(os.path.join(os.path.expanduser("~"), "logs"))
    return _logs_dir_ok


def reply_capture_dir():
    """Directory for reply captures (env override wins; read live, never cached).

    Defaults to ~/logs, but only when the one-shot startup check found it —
    otherwise returns None and logging is skipped for this process. Explicit
    env overrides are honoured as-is (fail-open on write if the dir is gone).
    """
    return os.environ.get(_REPLY_CAPTURE_ENV) or (
        os.path.join(os.path.expanduser("~"), "logs") if _default_logs_dir_available() else None
    )


def next_reply_capture_path():
    """Return a numbered path for the next reply capture (increments each call).

    On the very first call in this process, cleans up any leftover reply_*.txt files from
    previous sessions — within the capture dir only. Returns None when reply
    logging is disabled for this process (~/logs absent at startup).
    """
    global _REPLY_COUNTER
    base = reply_capture_dir()
    if base is None:
        return None
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
    """Write the complete wire response to reply_N.txt for debugging (skipped
    silently when ~/logs logging is disabled for this process)."""
    try:
        parts: list[str] = []
        if reasoning:
            parts.append(f"== REASONING ==\n{reasoning}")
        if content:
            parts.append(f"== CONTENT ==\n{content}")
        if tool_calls:
            import json
            
            # Convert SimpleNamespace objects to dicts for JSON serialization
            def _serialize_tool_calls(tcs):
                result = []
                for tc in tcs:
                    if isinstance(tc, dict):
                        result.append(tc)
                    elif hasattr(tc, 'id') and hasattr(tc, 'function'):
                        # SimpleNamespace or similar object
                        result.append({
                            'id': getattr(tc, 'id', None),
                            'type': getattr(tc, 'type', None),
                            'function': {
                                'name': getattr(getattr(tc, 'function', None), 'name', None),
                                'arguments': getattr(getattr(tc, 'function', None), 'arguments', None)
                            }
                        })
                    else:
                        # Fallback for unknown types
                        result.append(str(tc))
                return result
            
            parts.append(f"== TOOL CALLS ==\n{json.dumps(_serialize_tool_calls(tool_calls), indent=2)}")
        parts.append(f"[[finish_reason]] {finish_reason}")

        path = next_reply_capture_path()
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(parts) + "\n")
    except Exception as _exc_dbg:
        import sys as _sys_dbg2
        print(f"[DEBUG-write_reply_capture] ERROR: {_exc_dbg}", file=_sys_dbg2.stderr, flush=True)
        pass  # Never block streaming for debugging
