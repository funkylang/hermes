"""Wire-faithful prompt capture: write the request body bytes the SDK sends.

Rides an httpx ``request`` event hook on the agent's client (same pattern as
the served-model response capture in ``agent/served_model.py``). The hook
writes whatever the OpenAI SDK serialized — nothing we re-assemble — to a
configurable path, so /tmp/prompt.txt always holds the body of the LAST
prompt sent to the server. Fail-open everywhere; observability never breaks
a turn.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CAPTURE_ENV = "HERMES_PROMPT_CAPTURE"
_DEFAULT_PATH = "/tmp/prompt.txt"
_HOOK_MARK = "_hermes_prompt_capture_hook"


def prompt_capture_path() -> str:
    """Current capture path (env override wins; read live, never cached)."""
    return os.environ.get(_CAPTURE_ENV) or _DEFAULT_PATH


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
                with open(prompt_capture_path(), "wb") as f:
                    f.write(bytes(data))
        except Exception:
            logger.debug("prompt capture skipped", exc_info=True)

    setattr(_on_request, _HOOK_MARK, True)
    try:
        # httpx copies on assignment; rebuild the mapping instead of mutating the live list.
        new_hooks: dict = {**hooks, "request": [*hooks.get("request", ()), _on_request]}
        http_client.event_hooks = new_hooks
    except Exception:
        logger.debug("prompt capture hook install skipped", exc_info=True)
