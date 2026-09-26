"""Tests for agent/prompt_manifest.py — prompt manifest instrumentation.

Covers the node/record model (single stored prompt; system parts published at
build time on the agent), byte totals computed from held text at render time,
render numbering + get_line resolution, and JSON envelope decoding of tool-
result content for readable /pi drill-downs.
"""

import json

import pytest

from agent.prompt_manifest import (
    ComponentNode,
    PromptManifest,
    _content_to_str,
)
from agent import prompt_capture


# Autouse fixture to reset the wire body before each test — render() may
# read the module-level capture; tests that don't set one get a clean state.
@pytest.fixture(autouse=True)
def reset_wire_body():
    prompt_capture.set_wire_body(None)
    yield
    prompt_capture.set_wire_body(None)


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


def test_render_numbers_lines_and_get_line_resolves_them():
    agent, manifest = _make_manifest()
    rec = manifest.record_send(
        agent,
        [{"role": "user", "content": "hello"}],
        tools_for_api=[{"type": "function", "function": {"name": "terminal"}}],
    )
    assert rec is not None
    manifest.fill_usage(rec.seq, prompt_tokens=100, completion_tokens=5,
                        cache_read_tokens=None, latency_ms=10.0)

    rendered = manifest.render()
    # Rows are numbered; the stats row is plain "label value" pairs.
    assert "prompt 100" in rendered

    got = manifest.get_line(3)  # the user message line (1=sys part, 2=tool schemas)
    assert got is not None
    seq, text = got
    assert seq == rec.seq
    assert "role=user" in text
    assert "hello" in text


def test_only_last_prompt_is_kept():
    """MAX_RECORDS=1: each send replaces the previous; prompt numbers stay monotonic."""
    agent, manifest = _make_manifest()
    r1 = manifest.record_send(agent, [{"role": "user", "content": "first"}])
    r2 = manifest.record_send(agent, [{"role": "user", "content": "second"}])
    assert r1 is not None and r2 is not None
    assert len(manifest._records) == 1
    kept = manifest._records[-1]
    # It is the SECOND send, but its seq reflects its position in the stream.
    assert kept.seq == r2.seq > r1.seq
    rendered = manifest.render()
    assert f"Prompt #{r2.seq}" in rendered
    assert "Prompt #" + str(r1.seq) not in rendered


def test_message_text_stored_verbatim_no_eviction():
    # Drill-down text is never evicted: every stored message keeps its text.
    agent, manifest = _make_manifest()
    rec = manifest.record_send(agent, [{"role": "user", "content": "x" * 50000}])
    assert rec is not None
    assert rec.messages[0].content == "x" * 50000

    manifest.render()  # establishes the item numbers for get_line()
    got_msg = manifest.get_line(2)  # the user message line
    assert got_msg is not None
    assert "x" * 50000 in got_msg[1]

    got_sys = manifest.get_line(1)  # system part — text intact
    assert got_sys is not None
    assert "CORE" * 10 in got_sys[1]


def test_byte_totals_computed_from_held_text():
    """Totals are derived from node text at render time, never stored."""
    agent, manifest = _make_manifest()
    rec = manifest.record_send(agent, [{"role": "user", "content": "abcd"}])
    assert rec is not None
    # Totals reflect the wire sizes of all components.
    assert manifest._record_totals(rec) == (len("CORE" * 10) + len("abcd"))


def test_record_send_never_raises_on_odd_inputs():
    agent, manifest = _make_manifest()
    # None input yields an empty record (never raises).
    rec = manifest.record_send(agent, None)
    assert rec is not None
    assert len(rec.messages) == 0

    # Non-dict entries are skipped; dicts with a role become zero-byte nodes.
    rec = manifest.record_send(agent, ["not a message", {"role": "user"}])
    assert rec is not None
    assert len(rec.messages) == 1
    assert PromptManifest._message_bytes(rec.messages[0]) == 0


def test_content_to_str_unwraps_json_envelope_tool_result():
    """read_file/search_files style content arrives JSON-encoded; /pi must print readable text."""
    raw = json.dumps({
        "content": "1|line one\n2|line two",
        "total_lines": 2,
    }, ensure_ascii=False)
    assert _content_to_str(raw) == (
        "content:\n  1|line one\n  2|line two\ntotal_lines: 2"
    )


def test_content_to_str_renders_all_fields_equally():
    """No preference between fields: every key renders, in envelope order."""
    raw = json.dumps({
        "status": "success",
        "output": "stdout here\nnext line",
        "exit_code": 0,
    }, ensure_ascii=False)
    assert _content_to_str(raw) == (
        "status: success\n"
        "output:\n  stdout here\n  next line\n"
        "exit_code: 0"
    )


def test_content_to_str_leaves_plain_text_untouched():
    plain = 'Just a normal sentence with {braces} inside.'
    assert _content_to_str(plain) == plain

    non_json_brackets = '{"unterminated": true'
    assert _content_to_str(non_json_brackets) == non_json_brackets


def test_content_to_str_no_field_preference():
    raw = json.dumps({"output": "stdout here\nnext line", "exit_code": 0})
    assert _content_to_str(raw) == (
        "output:\n  stdout here\n  next line\nexit_code: 0"
    )


def test_record_send_stores_decoded_content_for_drilldown():
    agent, manifest = _make_manifest()
    raw = json.dumps({"content": "A\nB"})
    rec = manifest.record_send(
        agent, [{"role": "tool", "name": "read_file", "content": raw}])
    assert rec is not None
    m = rec.messages[0]
    # Human mode shows decoded text with the field name label.
    assert m.content == "content:\n  A\n  B"
    # Bytes are the wire length.
    assert PromptManifest._message_bytes(m) == len(raw)


def test_get_line_display_mode_raw_vs_human():
    """/style raw = verbatim wire form; human = decoded readable text."""
    agent, manifest = _make_manifest()
    raw = json.dumps({"content": "1|line one\n2|line two"})
    rec = manifest.record_send(agent, [{"role": "tool", "name": "read_file", "content": raw}])
    assert rec is not None
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
    seq_human, text_human = manifest.get_line(msg_line_no, display_mode="human")
    assert text_human is not None
    # raw: JSON as sent (escaped \n), no decoding.
    assert json.dumps({"content": "1|line one\n2|line two"}) in text_raw
    assert "\\n" in text_raw  # escaped newline visible = pure wire form
    # human: decoded readable text with real line breaks.
    assert "1|line one" in text_human and "2|line two" in text_human
    assert "\n  2|line two" in text_human  # real line break, not escaped


def test_get_line_tool_call_args_wire_form_raw():
    """raw mode renders tool-call arguments exactly as sent (compact JSON string)."""
    agent, manifest = _make_manifest()
    args_str = json.dumps({"path": "a.txt"})
    rec = manifest.record_send(agent, [
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "read_file", "arguments": args_str}}]},
    ])
    assert rec is not None
    rendered = manifest.render()
    msg_line_no = None
    for line in rendered.splitlines():
        if line.split() and line.split()[1] == "assistant:":
            msg_line_no = int(line.split()[0])
            break
    assert msg_line_no is not None

    # raw: args string preserved as sent.
    seq_raw, text_raw = manifest.get_line(msg_line_no, display_mode="raw")
    assert "read_file" in text_raw and args_str in text_raw

    # human: decoded readable tool_calls text.
    _, text_human = manifest.get_line(msg_line_no, display_mode="human")
    assert "read_file" in text_human


def test_render_shows_source_provenance_for_tiers():
    """render() stays clean; numbering unchanged with sources on the node."""
    agent, manifest = _make_manifest_with_sources()
    rec = manifest.record_send(
        agent,
        [{"role": "user", "content": "hello"}],
        tools_for_api=[{"type": "function", "function": {"name": "terminal"}}],
    )
    assert rec is not None
    manifest.fill_usage(rec.seq, prompt_tokens=100, completion_tokens=5,
                        cache_read_tokens=None, latency_ms=10.0)

    rendered = manifest.render()
    # Overview stays clean (no source sub-lines); the part row exists.
    assert "SOUL.md" not in rendered
    # Numbering still: 1=sys, 2=tool schemas, 3=user message.
    got = manifest.get_line(3)
    assert got is not None and "hello" in got[1]


def test_get_line_sys_shows_sources_header():
    """get_line for a system part (human mode) lists its sections; text stays intact."""
    agent, manifest = _make_manifest_with_sources()
    rec = manifest.record_send(
        agent,
        [{"role": "user", "content": "hi"}],
        tools_for_api=[{"type": "function", "function": {"name": "terminal"}}],
    )
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
    manifest.record_send(
        agent,
        [{"role": "user", "content": "hi"}],
        tools_for_api=[{"type": "function", "function": {"name": "terminal"}}],
    )
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
    rec = manifest.record_send(agent, [{"role": "user", "content": "x"}])
    assert rec is not None
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
    r1 = manifest.record_send(agent, [{"role": "user", "content": "first"}])
    assert r1 is not None
    # Rebuild with different text + sources under the same description.
    PromptManifest._store_system_nodes(agent, [
        ComponentNode("core instructions", "CORE" * 20,
                      sources=(("SOUL.md", 80),)),
    ])
    r2 = manifest.record_send(agent, [{"role": "user", "content": "second"}])
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


def test_record_send_captures_reasoning_content():
    """Echo-back providers send reasoning_content on the wire; it must be counted + stored."""
    agent, manifest = _make_manifest()
    rec = manifest.record_send(agent, [
        {"role": "assistant", "content": "the answer",
         "reasoning_content": "step 1\nstep 2"},
    ])
    assert rec is not None
    m = rec.messages[0]
    assert m.reasoning_text == "step 1\nstep 2"
    # Reasoning rides the wire, so it is part of the counted total.
    tot = manifest._record_totals(rec)
    assert tot == (len("CORE" * 10) + len("the answer") + len("step 1\nstep 2"))


def test_record_send_tags_origin_by_this_run_boundary():
    """Messages before the this-run user row are 'resumed'; at/after it are 'this run'."""
    agent, manifest = _make_manifest()
    msgs = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
    ]
    rec = manifest.record_send(agent, msgs, this_run_start_idx=2)
    assert rec is not None
    origins = [m.origin for m in rec.messages]
    assert origins == [
        "resumed from session DB",
        "resumed from session DB",
        "this run",
    ]


def test_render_omits_origin_and_reasoning_tags():
    """render() shows plain rows — origin and reasoning live in /pi N only."""
    agent, manifest = _make_manifest()
    rec = manifest.record_send(agent, [
        {"role": "assistant", "content": "old answer", "reasoning_content": "rrr"},
        {"role": "user", "content": "current question"},
    ], this_run_start_idx=1)
    assert rec is not None
    rendered = manifest.render()
    # Overview table rows only: numbered lines (footer prose mentions the tags'
    # words and must not count). The tags are gone from those rows.
    msg_rows = [l for l in rendered.splitlines() if l[:2].strip().isdigit()]
    asst_row = next(l for l in msg_rows if "assistant" in l)
    user_row = next(l for l in msg_rows if "user" in l)
    assert "resumed from session DB" not in asst_row
    assert "reasoning" not in asst_row
    assert "this run" not in user_row


def test_user_message_part_label_shows_content_prefix():
    """User rows use the start of the message as part label; others keep the role.

    Contract (not a snapshot): the user's own words identify the row, while
    non-user roles must NOT leak content into the part column — so the
    relationship between role and label is what's asserted.
    """
    agent, manifest = _make_manifest()
    long_q = "How do I refactor this module cleanly without breaking callers?"
    rec = manifest.record_send(agent, [
        {"role": "user", "content": long_q},
        {"role": "assistant", "content": long_q},  # same text, different role
    ])
    assert rec is not None

    # Direct label contract: user -> content prefix; assistant -> role name.
    user_label = PromptManifest._message_part_label(rec.messages[0])
    asst_label = PromptManifest._message_part_label(rec.messages[1])
    assert user_label.startswith(long_q[:20])
    assert asst_label == "assistant"

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


def test_get_line_shows_reasoning_and_origin_block():
    """/pi <part no.> drill-down prints the reasoning text + origin for a message."""
    agent, manifest = _make_manifest()
    rec = manifest.record_send(agent, [
        {"role": "assistant", "content": "the answer",
         "reasoning_content": "thinking steps here"},
    ], this_run_start_idx=0)
    assert rec is not None
    # Line 1 = sys part, 2 = the assistant message (no tools sent).
    manifest.render()
    got = manifest.get_line(2)
    assert got is not None
    seq, text = got
    assert "role=assistant" in text
    assert "origin: this run" in text
    assert "reasoning (sent on the wire" in text
    assert "thinking steps here" in text
