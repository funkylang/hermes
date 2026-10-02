"""Tests for agent/prompt_manifest.py — prompt manifest instrumentation.

Covers the node/record model (single stored prompt; system parts published at
build time on the agent), byte totals computed from held text at render time,
render numbering + get_line resolution, and rendering of conversation messages
from the captured per-message raw-text slices (the exact wire text).
"""

import json

import pytest

from agent.prompt_manifest import (
    ComponentNode,
    MessageNode,
    PromptManifest,
    utf8_bytes,
)
from agent import prompt_capture


# Autouse fixture to reset the captured outgoing traffic before each test —
# render() may read it for tool schemas, server parameters and messages.
@pytest.fixture(autouse=True)
def reset_wire_body():
    prompt_capture.set_wire_body(None)
    prompt_capture.set_message_parts([])
    yield
    prompt_capture.set_wire_body(None)
    prompt_capture.set_message_parts([])


class FakeAgent:
    pass


def _make_agent() -> FakeAgent:
    return FakeAgent()


def _make_manifest() -> tuple:
    agent = _make_agent()
    PromptManifest._store_system_nodes(agent, [
        ComponentNode("core instructions", "CORE" * 10),
    ])
    return agent, PromptManifest()


def _slice(msg: dict) -> str:
    """One message as its wire slice (compact JSON — what the capture hook stores)."""
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


def test_render_numbers_lines_and_get_line_resolves_them():
    agent, manifest = _make_manifest()
    rec = manifest.record_send(agent)
    assert rec is not None
    manifest.fill_usage(rec.seq, prompt_tokens=100, completion_tokens=5,
                        cache_read_tokens=None, latency_ms=10.0)

    rendered = manifest.render()
    # Rows are numbered; the stats row is plain "label value" pairs.
    assert "prompt 100" in rendered

    # Simulate the captured outgoing traffic (what the hook stores).
    prompt_capture.set_wire_body({
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "CORE" * 10},
            {"role": "user", "content": "hello"},
        ],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
    })
    prompt_capture.set_message_parts([
        _slice({"role": "user", "content": "hello"}),
    ])
    rendered = manifest.render()

    # Order: system part, messages, server params, tool schemas.
    got = manifest.get_line(2)  # the user message line (1=sys part, then msgs)
    assert got is not None
    seq, text = got
    assert seq == rec.seq
    assert "role=user" in text
    assert "hello" in text


def test_only_last_prompt_is_kept():
    """MAX_RECORDS=1: each send replaces the previous; prompt numbers stay monotonic."""
    agent, manifest = _make_manifest()
    r1 = manifest.record_send(agent)
    r2 = manifest.record_send(agent)
    assert r1 is not None and r2 is not None
    assert len(manifest._records) == 1
    kept = manifest._records[-1]
    # It is the SECOND send, but its seq reflects its position in the stream.
    assert kept.seq == r2.seq > r1.seq
    rendered = manifest.render()
    assert f"Prompt #{r2.seq}" in rendered
    assert "Prompt #" + str(r1.seq) not in rendered


def test_message_raw_slice_printed_verbatim():
    """The held raw slice is what raw mode prints — nothing reassembled."""
    agent, manifest = _make_manifest()
    manifest.record_send(agent)
    big = "x" * 50000
    prompt_capture.set_message_parts([_slice({"role": "user", "content": big})])

    manifest.render()  # establishes the item numbers for get_line()
    got_msg = manifest.get_line(2, display_mode="raw")  # the user message line
    assert got_msg is not None
    text = got_msg[1]
    assert big in text
    # The exact wire slice appears verbatim (escaped nothing, dropped nothing).
    assert _slice({"role": "user", "content": big}) in text

    got_sys = manifest.get_line(1)  # system part — text intact
    assert got_sys is not None
    assert "CORE" * 10 in got_sys[1]


def test_message_bytes_counted_from_raw_slice():
    """Per-message bytes and the bytes_total row are derived from the raw slices."""
    agent, manifest = _make_manifest()
    manifest.record_send(agent)
    s1 = _slice({"role": "user", "content": "abcd"})
    s2 = _slice({"role": "assistant", "content": "wxyz"})
    prompt_capture.set_message_parts([s1, s2])

    rendered = manifest.render()
    # The message row carries the raw slice's byte count.
    assert f"{utf8_bytes(s1):,}" in rendered
    # bytes_total covers system part + all messages (no tools/params seeded).
    expected = utf8_bytes("CORE" * 10) + utf8_bytes(s1) + utf8_bytes(s2)
    assert f"bytes_total {expected:,}" in rendered


def test_record_send_never_raises_and_keeps_no_messages():
    agent, manifest = _make_manifest()
    rec = manifest.record_send(agent)
    assert rec is not None
    # Messages live in the capture store, never on the record.
    assert not hasattr(rec, "messages")
    assert rec.system_components  # system parts still snapshotted onto the record


def test_get_line_display_mode_raw_vs_human():
    """/style raw = verbatim wire slice; human = pretty-printed JSON of it."""
    agent, manifest = _make_manifest()
    msg = {"role": "tool", "name": "read_file",
           "content": "1|line one\n2|line two"}
    prompt_capture.set_message_parts([_slice(msg)])
    manifest.record_send(agent)
    rendered = manifest.render()

    # Find the tool-message line number (kind column = role).
    msg_line_no = None
    for line in rendered.splitlines():
        if line.split() and line.split()[1] == "tool:":
            msg_line_no = int(line.split()[0])
            break
    assert msg_line_no is not None

    seq_raw, text_raw = manifest.get_line(msg_line_no, display_mode="raw")
    assert text_raw is not None
    # raw: the exact slice as sent (escaped \n on one line).
    assert _slice(msg) in text_raw
    assert '"content": "1|line one' not in text_raw  # not pretty-printed

    seq_human, text_human = manifest.get_line(msg_line_no, display_mode="human")
    assert text_human is not None
    # human: JSON of the slice, pretty-printed (real line breaks between fields).
    assert json.dumps(msg, ensure_ascii=False, indent=2) in text_human


def test_get_line_tool_call_args_wire_form_raw():
    """raw mode renders tool-call arguments exactly as sent (escaped JSON string)."""
    agent, manifest = _make_manifest()
    args_str = json.dumps({"path": "a.txt"})
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"function": {"name": "read_file", "arguments": args_str}}]}
    prompt_capture.set_message_parts([_slice(msg)])
    manifest.record_send(agent)
    rendered = manifest.render()
    msg_line_no = None
    for line in rendered.splitlines():
        if line.split() and line.split()[1] == "assistant:":
            msg_line_no = int(line.split()[0])
            break
    assert msg_line_no is not None

    # raw: the exact wire slice — arguments stay the escaped JSON string.
    seq_raw, text_raw = manifest.get_line(msg_line_no, display_mode="raw")
    assert _slice(msg) in text_raw

    # human: the parsed slice, readable (tool name present).
    _, text_human = manifest.get_line(msg_line_no, display_mode="human")
    assert "read_file" in text_human


def test_wmessage_nodes_helper_reads_capture_store():
    """_wire_message_nodes maps stored slices to MessageNodes with raw_text only."""
    prompt_capture.set_message_parts(["{\"role\": \"user\"}", "not-json"])
    nodes = PromptManifest._wire_message_nodes()
    assert nodes == [MessageNode(raw_text='{"role": "user"}'),
                     MessageNode(raw_text="not-json")]
    # No other fields exist on the node.
    assert set(MessageNode(raw_text="x").__dict__.keys()) == {"raw_text"}


def test_message_role_and_parsed_derives_at_render_time():
    role, parsed = PromptManifest._message_role_and_parsed(
        '{"role": "assistant", "content": "hi"}')
    assert role == "assistant"
    assert parsed["content"] == "hi"

    # Unparseable slice: unknown role, empty dict (never raises).
    role, parsed = PromptManifest._message_role_and_parsed("garbage")
    assert role == "?"
    assert parsed == {}


def test_render_shows_source_provenance_for_tiers():
    """render() stays clean; numbering unchanged with sources on the node."""
    agent, manifest = _make_manifest_with_sources()
    rec = manifest.record_send(agent)
    assert rec is not None
    manifest.fill_usage(rec.seq, prompt_tokens=100, completion_tokens=5,
                        cache_read_tokens=None, latency_ms=10.0)

    rendered = manifest.render()
    # Overview stays clean (no source sub-lines); the part row exists.
    assert "SOUL.md" not in rendered

    # With captured traffic, numbering is: 1=sys, 2=user msg, 3=server params, 4=tool.
    prompt_capture.set_wire_body({
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "CORE" * 10},
            {"role": "user", "content": "hello"},
        ],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
    })
    prompt_capture.set_message_parts([_slice({"role": "user", "content": "hello"})])
    manifest.render()
    got = manifest.get_line(2)
    assert got is not None and "hello" in got[1]


def test_get_line_sys_shows_sources_header():
    """get_line for a system part (human mode) lists its sections; text stays intact."""
    agent, manifest = _make_manifest_with_sources()
    rec = manifest.record_send(agent)
    prompt_capture.set_message_parts([_slice({"role": "user", "content": "hi"})])
    manifest.render()
    got = manifest.get_line(1, display_mode="human")  # the system part line
    assert got is not None
    seq, text = got
    assert "Sections:" in text
    assert "SOUL.md" in text
    assert "CORE" * 10 in text


def test_get_line_sys_raw_mode_text_only():
    """Raw mode for a system part returns ONLY the verbatim text — no header or Sections."""
    agent, manifest = _make_manifest_with_sources()
    manifest.record_send(agent)
    prompt_capture.set_message_parts([_slice({"role": "user", "content": "hi"})])
    manifest.render()
    got = manifest.get_line(1, display_mode="raw")
    assert got is not None
    text = got[1]
    assert text == "CORE" * 10


def test_get_line_sys_sections_full_width_and_tilde_paths():
    """Section labels are never clipped and home paths render as ~/..."""
    agent, manifest = _make_manifest()
    long_label = "/home/hermes/workspace/some/very/deep/project/file.txt"
    short_label = "SOUL.md"
    PromptManifest._store_system_nodes(agent, [
        ComponentNode("tier", "TIERTEXTTT",
                      sources=((short_label, 5), (long_label, 5))),
    ])
    manifest.record_send(agent)
    prompt_capture.set_message_parts([_slice({"role": "user", "content": "x"})])
    manifest.render()
    got = manifest.get_line(1, display_mode="human")
    assert got is not None
    text = got[1]
    # Full long label preserved (column auto-widened, no clipping).
    assert long_label not in text  # display_path converted it to ~/...
    assert "~/workspace/some/very/deep/project/file.txt" in text
    assert "source" in text


def test_store_system_nodes_refreshes_on_rebuild():
    """A rebuild (fresh nodes at build time) is seen by the NEXT send only."""
    agent, manifest = _make_manifest_with_sources()
    r1 = manifest.record_send(agent)
    assert r1 is not None
    # Rebuild with different text + sources under the same description.
    PromptManifest._store_system_nodes(agent, [
        ComponentNode("core instructions", "CORE" * 20,
                      sources=(("SOUL.md", 80),)),
    ])
    r2 = manifest.record_send(agent)
    assert r2 is not None
    # The new send sees the rebuilt part.
    node = r2.system_components[0]
    assert node.text == "CORE" * 20
    assert node.sources == (("SOUL.md", 80),)


def _make_manifest_with_sources() -> tuple:
    agent = _make_agent()
    PromptManifest._store_system_nodes(agent, [
        ComponentNode("core instructions", "CORE" * 10,
                      sources=(("SOUL.md", 40), ("generated constant", 10))),
    ])
    return agent, PromptManifest()


def test_user_row_label_shows_content_prefix():
    """User rows use the start of the message as part label; others fall back.

    Contract (not a snapshot): the user's own words identify the row in the
    rendered overview and drill-down header.
    """
    agent, manifest = _make_manifest()
    long_q = "How do I refactor this module cleanly without breaking callers?"
    prompt_capture.set_message_parts([
        _slice({"role": "user", "content": long_q}),
        _slice({"role": "assistant", "content": "ok, here is a plan"}),
    ])
    manifest.record_send(agent)

    # And it flows through to the rendered overview + drill-down header.
    rendered = manifest.render()
    for l in rendered.splitlines():
        if "  assistant:" in l:
            assert long_q[:20] not in l
    # Find the user message row and its line number.
    user_line_no = None
    for l in rendered.splitlines():
        if "  user:" in l:
            user_line_no = int(l.split()[0])
            assert long_q[:20] in l
            break
    assert user_line_no is not None
    # The drill-down header carries the same label.
    header = manifest.get_line_header(user_line_no)
    assert header is not None and long_q[:20] in header


def test_get_line_header_for_message_rows():
    """get_line_header resolves wmsg rows with the same numbers as the overview."""
    agent, manifest = _make_manifest()
    msg = {"role": "user", "content": "hello there"}
    prompt_capture.set_message_parts([_slice(msg)])
    manifest.record_send(agent)
    rendered = manifest.render()
    line_no = None
    for l in rendered.splitlines():
        if "  user:" in l:
            line_no = int(l.split()[0])
            break
    assert line_no is not None
    header = manifest.get_line_header(line_no)
    assert header is not None
    assert "user:" in header
    assert f"{utf8_bytes(_slice(msg)):,}" in header
