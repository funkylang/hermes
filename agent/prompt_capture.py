"""Wire-faithful prompt capture: write the request body bytes the SDK sends.

Rides an httpx ``request`` event hook on the agent's client (same pattern as
the served-model response capture in ``agent/served_model.py``). The hook
writes whatever the OpenAI SDK serialized — nothing we re-assemble — to a
configurable path, so /tmp/prompt.txt always holds the body of the LAST
prompt sent to the server. A human-readable copy (pretty-printed, same data)
is written alongside as <base>.json (/tmp/prompt.json by default).
Fail-open everywhere; observability never breaks a turn.
"""

from __future__ import annotations

import json
import glob
import logging
import os
from typing import Any, List

logger = logging.getLogger(__name__)

_CAPTURE_ENV = "HERMES_PROMPT_CAPTURE"
_DEFAULT_PATH = "/tmp/prompt.txt"
_HOOK_MARK = "_hermes_prompt_capture_hook"


def prompt_capture_path() -> str:
    """Current capture path (env override wins; read live, never cached)."""
    return os.environ.get(_CAPTURE_ENV) or _DEFAULT_PATH


def prompt_capture_json_path() -> str:
    """Readable-JSON companion of :func:`prompt_capture_path` (.txt -> .json)."""
    path = prompt_capture_path()
    if path.endswith(".txt"):
        return path[: -len(".txt")] + ".json"
    return path + ".json"


def _message_part_base() -> str:
    """Base for per-message part files (no index): /tmp/prompt.txt -> /tmp/prompt_part."""
    path = prompt_capture_path()
    if path.endswith(".txt"):
        return path[: -len(".txt")] + "_part"
    return path + "_part"


def _collect_message_slices(raw: bytes) -> List[str]:
    """Split the captured body into per-message raw-text slices (wire-faithful).

    Walks the messages array with ``json.JSONDecoder.raw_decode``: at each '{'
    that opens a message, raw_decode returns where that message ends in the
    text (nested braces handled), so ``text[idx:end]`` is the exact wire slice.
    The slices in order reassemble exactly the messages array content between
    the brackets.  Fail-open: returns [] on any problem; never raises.
    """
    slices: List[str] = []
    try:
        text = raw.decode("utf-8")
        kpos = text.find('"messages"')
        if kpos < 0:
            return slices
        apos = text.find(":", kpos + len('"messages"'))
        bpos = text.find("[", apos + 1) if apos >= 0 else -1
        if bpos < 0:
            return slices
        dec = json.JSONDecoder()
        idx = bpos + 1
        while True:
            c = text[idx]
            if c == "]":
                break
            if c != "{":
                # separator (comma/whitespace) between messages — step past it
                idx += 1
                continue
            _, end = dec.raw_decode(text, idx)
            slices.append(text[idx:end])
            idx = end
    except Exception:
        logger.debug("prompt capture message slices skipped", exc_info=True)
    return slices


def write_message_parts(slices: List[str]) -> None:
    """Debug output only: one raw-slice file per message (/tmp/prompt_part_<n>.txt).

    The in-memory store (:func:`set_message_parts`) is what /pi reads; these
    files exist for offline inspection. Stale parts from a previous larger
    capture are removed first. Fail-open: never raises.
    """
    try:
        if not slices:
            return
        base = _message_part_base()
        for f in glob.glob(glob.escape(base) + "_*.txt"):
            try:
                os.remove(f)
            except OSError:
                pass
        for i, s in enumerate(slices):
            with open(f"{base}_{i}.txt", "wb") as f:
                f.write(s.encode("utf-8"))
    except Exception:
        logger.debug("prompt capture message parts skipped", exc_info=True)


# In-memory copy of the parsed body of the LAST captured request. The /pi
# display reads this (instead of parsing files) for tool schemas and server
# parameters: same object that was dumped to prompt.json, held in-process.
_last_wire_body: Any = None


def set_wire_body(body: Any) -> None:
    """Store the parsed body of the last captured request (fail-open)."""
    global _last_wire_body
    try:
        _last_wire_body = body
    except Exception:
        logger.debug("wire body store skipped", exc_info=True)


def get_wire_body() -> Any:
    """Return the stored parsed body, or None if no capture happened yet."""
    return _last_wire_body


# Per-message raw-text slices of the LAST captured request (exact wire text).
# The /pi display builds its message parts from these.
_message_parts: List[str] = []


def set_message_parts(texts: List[str]) -> None:
    """Store the per-message raw-text slices of the last captured request."""
    global _message_parts
    try:
        _message_parts = list(texts)
    except Exception:
        logger.debug("message parts store skipped", exc_info=True)


def get_message_parts() -> List[str]:
    """Return stored per-message raw-text slices (empty if no capture yet)."""
    return list(_message_parts)


def install_prompt_capture(client: Any) -> None:
    """Register the request hook on *client*'s ``httpx`` transport (idempotent per client).

    Writes the full POST/PUT body bytes to :func:`prompt_capture_path` before
    each request leaves; other methods are ignored. Each distinct httpx client
    gets the hook once (delegated children and recovery paths build their own).
    """
    http_client = getattr(client, "_client", None)
    hooks = getattr(http_client, "event_hooks", None)
    if not isinstance(hooks, dict):
        return
    if any(getattr(h, _HOOK_MARK, False) for h in hooks.get("request", ())):
        return
    def _on_request(request: Any) -> None:
        try:
            if str(getattr(request, "method", "")).upper() not in ("POST", "PUT"):
                return
            read = getattr(request, "read", None)
            data = read() if callable(read) else None
            if isinstance(data, (bytes, bytearray)) and data:
                raw = bytes(data)
                with open(prompt_capture_path(), "wb") as f:
                    f.write(raw)
                # Per-message raw-text slices (exact wire text).
                try:
                    slices = _collect_message_slices(raw)
                    write_message_parts(slices)
                    set_message_parts(slices)
                except Exception:
                    logger.debug("prompt capture message parts skipped", exc_info=True)
                # Human-readable companion of the same body (.json, indent=2).
                try:
                    parsed = json.loads(raw)
                    with open(prompt_capture_json_path(), "w", encoding="utf-8") as jf:
                        json.dump(parsed, jf, indent=2, ensure_ascii=False)
                    # In-memory copy for the /pi display (tools, server params).
                    set_wire_body(parsed)
                except Exception:
                    logger.debug("prompt capture JSON companion skipped", exc_info=True)
        except Exception:
            logger.debug("prompt capture skipped", exc_info=True)

    setattr(_on_request, _HOOK_MARK, True)
    try:
        # httpx copies on assignment; rebuild the mapping instead of mutating the live list.
        new_hooks: dict = {**hooks, "request": [*hooks.get("request", ()), _on_request]}
        http_client.event_hooks = new_hooks
    except Exception:
        logger.debug("prompt capture hook install skipped", exc_info=True)
