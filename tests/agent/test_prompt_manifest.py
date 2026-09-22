"""Tests for agent/prompt_manifest.py — prompt manifest instrumentation.

Covers the node/record model (single stored prompt, canonical tier/schema
text), render numbering + get_line resolution, and JSON envelope decoding of
tool-result content for readable /pi drill-downs.
"""

import json

import pytest

from agent.prompt_manifest import (
    ComponentNode,
    PromptManifest,
    _content_to_str,
)


class FakeAgent:
    pass


def _make_manifest() -> PromptManifest:
    manifest = PromptManifest()
    manifest.set_system_components(
        [ComponentNode("core instructions", len("CORE" * 10))],
        texts=["CORE" * 10],
    )
    return manifest


def test_render_numbers_lines_and_get_line_resolves_them():
    manifest = _make_manifest()
    rec = manifest.record_send(
        [{"role": "user", "content": "hello"}],
        tools_for_api=[{"type": "function", "function": {"name": "terminal"}}],
    )
    assert rec is not None
    manifest.fill_usage(rec.seq, prompt_tokens=100, completion_tokens=5,
                        cache_read_tokens=None, latency_ms=10.0)

    rendered = manifest.render()
    # Rows are numbered; the stats row is plain "label value" pairs.
    assert "prompt 100" in rendered
    assert "chars/tok" in rendered

    got = manifest.get_line(3)  # the user message line (1=sys tier, 2=tool schemas)
    assert got is not None
    seq, text = got
    assert seq == rec.seq
    assert "role=user" in text
    assert "hello" in text


def test_only_last_prompt_is_kept():
    """MAX_RECORDS=1: each send replaces the previous; prompt numbers stay monotonic."""
    manifest = _make_manifest()
    r1 = manifest.record_send([{"role": "user", "content": "first"}])
    r2 = manifest.record_send([{"role": "user", "content": "second"}])
    assert r1 is not None and r2 is not None
    assert len(manifest._records) == 1
    kept = manifest._records[-1]
    # It is the SECOND send, but its seq reflects its position in the stream.
    assert kept.seq == r2.seq > r1.seq
    rendered = manifest.render()
    assert f"Prompt #{r2.seq}" in rendered
    assert "Prompt #" + str(r1.seq) not in rendered


def test_message_text_stored_verbatim_no_eviction():
    # Drill-down text is never evicted: every stored message keeps its text,
    # and tier text is canonical (stored once). 1-2 MB per agent is fine.
    manifest = _make_manifest()
    rec = manifest.record_send([{"role": "user", "content": "x" * 50000}])
    assert rec is not None
    assert rec.messages[0].content == "x" * 50000

    manifest.render()  # establishes the item numbers for get_line()
    got_msg = manifest.get_line(2)  # the user message line
    assert got_msg is not None
    assert "x" * 50000 in got_msg[1]

    got_sys = manifest.get_line(1)  # system tier — canonical text intact
    assert got_sys is not None
    assert "CORE" * 10 in got_sys[1]


def test_char_counts_reflect_wire_sizes():
    manifest = _make_manifest()
    rec = manifest.record_send([{"role": "user", "content": "abcd"}])
    assert rec is not None
    # Totals reflect the wire sizes of all components.
    assert manifest._record_totals(rec) == (len("CORE" * 10) + len("abcd"))


def test_record_send_never_raises_on_odd_inputs():
    manifest = PromptManifest()
    # None input yields an empty record (never raises).
    rec = manifest.record_send(None)
    assert rec is not None
    assert len(rec.messages) == 0

    # Non-dict entries are skipped; dicts with a role become zero-char nodes.
    rec = manifest.record_send(["not a message", {"role": "user"}])
    assert rec is not None
    assert len(rec.messages) == 1
    assert rec.messages[0].chars == 0


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
    manifest = PromptManifest()
    raw = json.dumps({"content": "A\nB"})
    rec = manifest.record_send(
        [{"role": "tool", "name": "read_file", "content": raw}])
    assert rec is not None
    m = rec.messages[0]
    # Human mode shows decoded text with the field name label.
    assert m.content == "content:\n  A\n  B"
    # Char count is still the wire length.
    assert m.chars == len(raw)


def test_get_line_display_mode_off_vs_human():
    """/json off = pure wire JSON; human = decoded readable text."""
    manifest = _make_manifest()
    raw = json.dumps({"content": "1|line one\n2|line two"})
    rec = manifest.record_send([{"role": "tool", "name": "read_file", "content": raw}])
    assert rec is not None
    rendered = manifest.render()

    # Find the message line number.
    msg_line_no = None
    for line in rendered.splitlines():
        if "[msg]" in line:
            msg_line_no = int(line.split()[0])
            break
    assert msg_line_no is not None

    seq_off, text_off = manifest.get_line(msg_line_no, display_mode="off")
    assert text_off is not None
    seq_human, text_human = manifest.get_line(msg_line_no, display_mode="human")
    assert text_human is not None
    # off: raw JSON as sent (escaped \n), no decoding.
    assert json.dumps({"content": "1|line one\n2|line two"}) in text_off
    assert "\\n" in text_off  # escaped newline visible = pure wire form
    # human: decoded readable text with real line breaks.
    assert "1|line one" in text_human and "2|line two" in text_human
    assert "\n  2|line two" in text_human  # real line break, not escaped


def test_get_line_tool_call_args_wire_form_off():
    """off mode renders tool-call arguments exactly as sent (compact JSON string)."""
    manifest = _make_manifest()
    args_str = json.dumps({"path": "a.txt"})
    rec = manifest.record_send([
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "read_file", "arguments": args_str}}]},
    ])
    assert rec is not None
    rendered = manifest.render()
    msg_line_no = None
    for line in rendered.splitlines():
        if "[msg]" in line:
            msg_line_no = int(line.split()[0])
            break
    assert msg_line_no is not None

    # off: raw args string preserved as sent.
    seq_off, text_off = manifest.get_line(msg_line_no, display_mode="off")
    assert "read_file" in text_off and args_str in text_off

    # human: decoded readable tool_calls text.
    _, text_human = manifest.get_line(msg_line_no, display_mode="human")
    assert "read_file" in text_human


def test_render_shows_source_provenance_for_tiers():
    """render() lists per-block sources under each system tier; numbering unchanged."""
    manifest = _make_manifest_with_sources()
    rec = manifest.record_send(
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
    """get_line for a system tier lists its sources; text stays intact."""
    manifest = _make_manifest_with_sources()
    rec = manifest.record_send(
        [{"role": "user", "content": "hi"}],
        tools_for_api=[{"type": "function", "function": {"name": "terminal"}}],
    )
    manifest.render()
    got = manifest.get_line(1)  # the system tier line
    assert got is not None
    seq, text = got
    assert "Sources:" in text
    assert "SOUL.md" in text
    assert "CORE" * 10 in text


def _make_manifest_with_sources() -> PromptManifest:
    manifest = PromptManifest()
    manifest.set_system_components(
        [ComponentNode("core instructions", len("CORE" * 10),
                       "CORE" * 10,
                       sources=(("SOUL.md", 40), ("generated constant", 10)))],
        texts=["CORE" * 10],
    )
    return manifest


def test_set_system_components_preserves_sources_on_rebuild():
    """A rebuild (production shape: fresh nodes) refreshes stored sources."""
    manifest = _make_manifest_with_sources()
    # Rebuild with different sources under the same description.
    manifest.set_system_components(
        [ComponentNode("core instructions", len("CORE" * 20),
                       "CORE" * 20,
                       sources=(("SOUL.md", 80),))],
        texts=["CORE" * 20],
    )
    node = manifest._current_system[0]
    assert node.chars == len("CORE" * 20)
    assert node.sources == (("SOUL.md", 80),)


def test_record_send_captures_reasoning_content():
    """Echo-back providers send reasoning_content on the wire; it must be counted + stored."""
    manifest = _make_manifest()
    rec = manifest.record_send([
        {"role": "assistant", "content": "the answer",
         "reasoning_content": "step 1\nstep 2"},
    ])
    assert rec is not None
    m = rec.messages[0]
    assert m.reasoning_chars == len("step 1\nstep 2")
    assert m.reasoning_text == "step 1\nstep 2"
    # Reasoning rides the wire, so it is part of the counted total.
    tot = manifest._record_totals(rec)
    assert tot == (len("CORE" * 10) + len("the answer") + len("step 1\nstep 2"))


def test_record_send_tags_origin_by_this_run_boundary():
    """Messages before the this-run user row are 'resumed'; at/after it are 'this run'."""
    manifest = _make_manifest()
    msgs = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
    ]
    rec = manifest.record_send(msgs, this_run_start_idx=2)
    assert rec is not None
    origins = [m.origin for m in rec.messages]
    assert origins == [
        "resumed from session DB",
        "resumed from session DB",
        "this run",
    ]


def test_render_omits_origin_and_reasoning_tags():
    """render() shows plain rows — origin and reasoning live in /pi N only."""
    manifest = _make_manifest()
    rec = manifest.record_send([
        {"role": "assistant", "content": "old answer", "reasoning_content": "rrr"},
        {"role": "user", "content": "current question"},
    ], this_run_start_idx=1)
    assert rec is not None
    rendered = manifest.render()
    # The tags are gone from the overview rows.
    asst_line = next(l for l in rendered.splitlines() if "[msg] assistant" in l)
    user_line = next(l for l in rendered.splitlines() if "[msg] user" in l)
    assert "resumed from session DB" not in asst_line
    assert "reasoning" not in asst_line
    assert "this run" not in user_line


def test_get_line_shows_reasoning_and_origin_block():
    """/pi <part no.> drill-down prints the reasoning text + origin for a message."""
    manifest = _make_manifest()
    rec = manifest.record_send([
        {"role": "assistant", "content": "the answer",
         "reasoning_content": "thinking steps here"},
    ], this_run_start_idx=0)
    assert rec is not None
    # Line 1 = sys tier, 2 = the assistant message (no tools sent).
    manifest.render()
    got = manifest.get_line(2)
    assert got is not None
    seq, text = got
    assert "role=assistant" in text
    assert "origin: this run" in text
    assert "reasoning (sent on the wire" in text
    assert "thinking steps here" in text
