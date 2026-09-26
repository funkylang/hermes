"""Tests for agent/prompt_capture.py — wire-faithful prompt body capture.

The install target mirrors production: an openai-style SDK client whose
``_client`` is the httpx transport owner.
"""

from __future__ import annotations

import httpx
from typing import Any

from agent.prompt_capture import install_prompt_capture, prompt_capture_json_path, prompt_capture_path


def _sdk_with_hook() -> Any:
    """Fake openai-style client (has ``_client``) with the capture installed."""
    http_client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    class FakeSDK:
        _client: Any

    sdk = FakeSDK()
    sdk._client = http_client
    install_prompt_capture(sdk)
    return sdk


def test_captures_post_body_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", str(tmp_path / "prompt.txt"))
    sdk = _sdk_with_hook()

    sdk._client.post(
        "http://localhost/v1/chat/completions",
        content=b'{"messages": [{"role": "user", "content": "hi"}]}',
    )

    body = (tmp_path / "prompt.txt").read_bytes()
    assert body == b'{"messages": [{"role": "user", "content": "hi"}]}'


def test_ignores_non_post_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", str(tmp_path / "prompt.txt"))
    sdk = _sdk_with_hook()
    sdk._client.get("http://localhost/v1/models")

    assert not (tmp_path / "prompt.txt").exists()


def test_writes_last_request_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", str(tmp_path / "prompt.txt"))
    sdk = _sdk_with_hook()
    base = "http://localhost/v1/chat/completions"
    sdk._client.post(base, content=b"first")
    sdk._client.post(base, content=b"second")

    assert (tmp_path / "prompt.txt").read_bytes() == b"second"


def test_writes_json_companion_of_same_body(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", str(tmp_path / "prompt.txt"))
    sdk = _sdk_with_hook()
    body = b'{"messages": [{"role": "user", "content": "hi"}], "model": "m"}'
    sdk._client.post(
        "http://localhost/v1/chat/completions", content=body,
    )

    json_path = tmp_path / "prompt.json"
    assert json_path.exists()
    # The .json companion holds the SAME data as the captured body.
    assert json.loads(json_path.read_text(encoding="utf-8")) == json.loads(body)


def test_invalid_body_still_writes_txt_no_json(tmp_path, monkeypatch):
    # Fail-open: a non-JSON body still lands in prompt.txt; companion is skipped.
    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", str(tmp_path / "prompt.txt"))
    sdk = _sdk_with_hook()
    sdk._client.post("http://localhost/v1/chat/completions", content=b"second")

    assert (tmp_path / "prompt.txt").read_bytes() == b"second"
    assert not (tmp_path / "prompt.json").exists()


def test_json_path_derives_from_txt_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", str(tmp_path / "capture.txt"))
    assert prompt_capture_json_path() == str(tmp_path / "capture.json")


def test_hook_not_double_installed():
    sdk = _sdk_with_hook()
    install_prompt_capture(sdk)  # second call must be a no-op
    markers = [h for h in sdk._client.event_hooks["request"]
               if getattr(h, "_hermes_prompt_capture_hook", False)]
    assert len(markers) == 1


def test_missing_httpx_client_is_noop():
    class NoHttpx:
        pass
    install_prompt_capture(NoHttpx())  # must not raise


def test_path_resolves_from_env_live(tmp_path, monkeypatch):
    p1 = str(tmp_path / "a.txt")
    p2 = str(tmp_path / "b.txt")
    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", p1)
    assert prompt_capture_path() == p1
    monkeypatch.setenv("HERMES_PROMPT_CAPTURE", p2)
    assert prompt_capture_path() == p2
