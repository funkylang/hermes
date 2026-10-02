"""Per-agent prompt manifest: what each API call's prompt was made of.

Tracks the prompt components of every send (system-prompt parts, tool
schemas, conversation messages) by holding their text; byte totals are
computed from that text at render time.  Tokens are NEVER estimated per
component — provider usage (prompt/completion/cache tokens) is recorded per
prompt at the send points, so the manifest always shows the real token total
of each call alongside its byte breakdown.

Model:
  - A ``PromptManifest`` lives on one agent (``agent._prompt_manifest``).
  - System-prompt parts are published onto the agent once per prompt build
    (``_store_system_nodes``) and snapshotted onto every send until a rebuild
    replaces them.
  - Each API call calls ``record_send``, which snapshots system parts + send
    metadata into a single ``PromptRecord`` and returns it. Conversation
    messages are NOT held on the record: they come from prompt_capture's
    in-memory store (raw slices of the actual captured request body) and are
    read at render time, like tool schemas and server parameters.
  - After the response arrives, ``fill_usage`` stamps that record's provider
    token counts (matched by seq).

In-memory only (per process).  Only the LAST prompt per agent is kept; each
new send replaces the previous record.

Read side: ``render()`` numbers every content line; ``get_line(n)`` resolves a
number from the last render to that component's full text.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.path_display import display_path


# Matches absolute POSIX paths (e.g., /home/hermes/foo) embedded in a string.
_ABS_PATH_RE = re.compile(r"(?<![\w])(/[\w./-]+)")


def _shorten_paths_in_text(text: str) -> str:
    """Shorten every absolute path substring via display_path (in-place).

    display_path only shortens text that STARTS with the home prefix, but our
    /pi section labels often embed a path inside parentheses or after a name
    (e.g., "skill foo (/home/hermes/.hermes/skills)"). This walks the label and
    shortens each embedded absolute path so all of them render as ~/... in one
    pass.
    """
    return _ABS_PATH_RE.sub(lambda m: display_path(m.group(0)), text)


def utf8_bytes(text: str) -> int:
    """UTF-8 byte length of *text* — what /pi counts for every prompt part."""
    return len(text.encode("utf-8"))


def format_table(header: List[Tuple[str, str]], rows: List[List[str]],
                 show_header: bool = True) -> List[str]:
    """Render a column-aligned text table with DYNAMIC column widths.

    *header* is ``(name, alignment)`` pairs — alignment ``"l"`` (left) or
    ``"r"`` (right, for numbers).  Each *rows* entry must match the header
    length.

    Every column is exactly as wide as its widest cell across header and
    body — no fixed widths, nothing clipped.  Used by /pi's overview,
    drill-down tables, and single-row headers so column alignment behaves
    the same everywhere.  Pass ``show_header=False`` to render only the rows
    (still computed with the header's alignments).
    """
    if not header:
        return []
    cols = len(header)

    # Per-column max width across header + all rows (single-line cells).
    col_w = [0] * cols
    if show_header:
        for i, (name, _al) in enumerate(header):
            col_w[i] = max(col_w[i], len(str(name)))
    for row in rows:
        if len(row) != cols:
            raise ValueError(f"row width {len(row)} != header width {cols}")
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(str(cell)))

    def _fmt(cell: str, i: int, align: str) -> str:
        s = str(cell)
        return s.rjust(col_w[i]) if align == "r" else s.ljust(col_w[i])

    out: List[str] = []
    if show_header:
        out.append("  ".join(_fmt(name, i, al) for i, (name, al) in enumerate(header)).rstrip())
    for row in rows:
        out.append("  ".join(_fmt(cell, i, header[i][1]) for i, cell in enumerate(row)).rstrip())
    return out


# ── Nodes ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComponentNode:
    """One named prompt component; its UTF-8 byte count is computed from text.

    ``text`` holds the actual content (empty string for non-textual markers).

    ``sources`` lists per-block provenance ``(label, bytes)`` in assembly
    order — e.g. which files (SOUL.md, AGENTS.md) or generated constants
    make up a part — so /pi can show where the text was loaded from.

    ``annotation`` is optional human-mode-only explanatory text: for parts
    built from hardcoded string literals at runtime it says how/when the
    text is constructed. Never shown in raw mode, never part of the prompt.
    """

    description: str
    text: str = ""
    sources: Tuple[Tuple[str, int], ...] = ()
    annotation: str = ""
    kind: str = "generated"  # what this part is: file | generated | hardcoded | summary | user | assistant | tool | ...


@dataclass(frozen=True)
class MessageNode:
    """One conversation message part, held as its exact wire text.

    ``raw_text`` is the message's raw slice of the captured request body (the
    text between its braces on the wire) — nothing decoded, nothing reassembled.
    Role/label are derived from it at render time; sizes are computed from it.
    """

    raw_text: str = ""


def _wire_tool_call_text(msg: Dict[str, Any]) -> str:
    """tool_calls in wire form: each call's arguments EXACTLY as sent.

    On the wire, tool-call arguments are compact JSON *strings*; this renders
    them verbatim (one call per line) so /style raw shows pure wire data.
    """
    calls = msg.get("tool_calls") or []
    if not calls:
        return ""
    lines = []
    for tc in calls:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments", "")
        lines.append(f"{fn.get('name', '?')}: {args}")
    return "\n".join(lines)


# ── Records ──────────────────────────────────────────────────────────────


@dataclass
class PromptRecord:
    """What one API call's prompt consisted of, plus its provider token counts.

    System-part text lives on the ComponentNodes referenced here. Message parts
    are NOT held on the record: they come from prompt_capture's in-memory store
    (the raw slices of the actual request body) and are read at render time —
    same pattern as tool schemas and server parameters.
    """

    seq: int
    built_at: float = field(default_factory=time.time)
    system_components: List[ComponentNode] = field(default_factory=list)  # refs to canonical nodes
    label: str = ""
    # Filled after the API response lands (None until then):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


# ── Manifest ─────────────────────────────────────────────────────────────


class PromptManifest:
    """Ordered, per-agent record of prompt components and their token totals.

    Thread-safe: the turn loop, stream reader, background reviews and the
    read-side slash command all touch it from different threads; every
    mutation runs under one lock.  Never raises across the public API —
    instrumentation must not break a turn.
    """

    MAX_RECORDS = 1            # only the LAST prompt per agent is kept

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[PromptRecord] = []
        self._next_seq = 1     # monotonic prompt number for the "Prompt #N" header
        # line number (last render) -> (record_seq, kind, index), for get_line()
        # kind in {sys, wmsg, wtool, wenv, wtail}; index = item position
        self._line_to_seq: Dict[int, Tuple[int, str, int]] = {}
        # Last conversation-history entry at /pi call time (live tail); used by
        # the wtail drill-down. Set by render(); never part of any send path.
        self._live_tail: Any = None

    # -- write side (attachment sites) ----------------------------------

    @staticmethod
    def _store_system_nodes(agent: Any, nodes: List[ComponentNode]) -> None:
        """Attach the current system-part nodes to the agent for record_send.

        Called from ``build_system_prompt`` after the blocks are assembled
        (one node per text block). The nodes reference the same strings the
        build already produced — no extra copy; each send snapshots them onto
        its record, so a later rebuild never rewrites history.
        """
        try:
            agent._pm_system_nodes = list(nodes or [])
        except Exception:
            pass  # observability must never break prompt build

    def record_send(self, agent: Any, label: str = "") -> Optional[PromptRecord]:
        """Snapshot one send into a new PromptRecord.

        Records the system parts attached to the agent and the send metadata;
        conversation messages are NOT snapshotted here — /pi reads them at
        render time from prompt_capture's in-memory store (the raw slices of
        the actual captured request body).  Returns the record so callers can
        later :meth:`fill_usage` it; the record is already stored. Never raises.

        Only the LAST prompt per agent is kept (MAX_RECORDS = 1): each send
        replaces the previous record.
        """
        try:
            with self._lock:
                record = PromptRecord(
                    seq=self._next_seq,
                    built_at=time.time(),
                    system_components=list(getattr(agent, "_pm_system_nodes", None) or []),
                    label=label or "",
                )
                self._records.append(record)
                if len(self._records) > self.MAX_RECORDS:
                    del self._records[:len(self._records) - self.MAX_RECORDS]
                self._next_seq += 1
                return record
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("prompt_manifest.record_send failed", exc_info=True)
            return None

    def fill_usage(self, seq: int, *, prompt_tokens: Optional[int],
                   completion_tokens: Optional[int], cache_read_tokens: Optional[int],
                   latency_ms: Optional[float]) -> None:
        """Stamp a stored record with provider-reported token counts. No-op if unknown seq."""
        try:
            with self._lock:
                for rec in reversed(self._records):
                    if rec.seq == seq:
                        rec.prompt_tokens = prompt_tokens
                        rec.completion_tokens = completion_tokens
                        rec.cache_read_tokens = cache_read_tokens
                        rec.latency_ms = latency_ms
                        return
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("prompt_manifest.fill_usage failed", exc_info=True)

    # -- sizes ------------------------------------------------------------

    @staticmethod
    def _component_bytes(comp: Optional[ComponentNode]) -> int:
        """UTF-8 bytes of a component's held text (0 for None/empty)."""
        return utf8_bytes(comp.text) if comp and comp.text else 0

    @staticmethod
    def _message_bytes(node: MessageNode) -> int:
        """UTF-8 bytes of one message part (its held raw wire text)."""
        return utf8_bytes(node.raw_text) if node and node.raw_text else 0

    # -- wire-message parts -------------------------------------------------
    # Conversation messages are rendered from the captured request body: the
    # per-message raw-text slices prompt_capture stores in memory (exact wire
    # text). Read at display time, like tool schemas and server parameters.

    @staticmethod
    def _wire_message_nodes() -> List[MessageNode]:
        """MessageNode list built from the captured in-memory raw slices."""
        try:
            from agent.prompt_capture import get_message_parts
            return [MessageNode(raw_text=t) for t in get_message_parts() if isinstance(t, str)]
        except Exception:
            return []

    @staticmethod
    def _message_role_and_parsed(raw_text: str) -> Tuple[str, Any]:
        """(role, parsed message dict) derived from one raw slice at render time."""
        try:
            obj = json.loads(raw_text)
        except Exception:
            obj = None
        if not isinstance(obj, dict):
            return "?", {}
        role = obj.get("role")
        return (str(role) if role else "?"), obj

    # -- drill-down -------------------------------------------------------

    def get_line(self, line_no: int, display_mode: str = "human") -> Optional[Tuple[int, str]]:
        """Resolve a numbered output line (from /pi) to prompt text.

        Returns ``(seq, full_text)`` or None if unknown.  Numbers refer to the
        lines shown by the most recent ``render()`` call — so run /pi first,
        then use its numbers.  Walks the same line ordering ``render()`` uses.

        ``display_mode`` (set via /style): "human" (default) decodes tool-result
        JSON into readable text with real line breaks and shows each system
        part's header + source sections; "raw" shows message contents exactly as
        sent — compact/escaped where it is JSON on the wire, plain text otherwise.
        """
        try:
            with self._lock:
                entry = self._line_to_seq.get(line_no)
                records = list(self._records)

            if entry is None:
                return None
            seq, kind, idx = entry
            rec = next((r for r in records if r.seq == seq), None)
            if not rec:
                return None

            if kind == "sys":
                comp = rec.system_components[idx]
                t = comp.text or ""
                # Raw: just the text of this whole part, no metadata. Human:
                # header + explanation + source provenance for composite parts
                # (context-file block: per file; skills index: per skill).
                if display_mode == "raw":
                    return seq, t if t else "  text not available (not captured)"
                lines = [f"[{self._cell(comp.description)} ({utf8_bytes(t):,} bytes)]"]
                # A single source row repeating the part's own name says nothing
                # new — show the Sections table only for genuinely composite parts.
                base = self._cell(comp.description)
                has_inner_sources = bool(
                    comp.sources and not (
                        len(comp.sources) == 1 and _shorten_paths_in_text(comp.sources[0][0]) == base))
                if comp.annotation:
                    for ln in textwrap.wrap(comp.annotation, width=80, initial_indent="  ",
                                            subsequent_indent="  "):
                        lines.append(ln)
                if has_inner_sources:
                    lines.append("  Sections:")
                    # Shorten embedded absolute paths in-place; format_table
                    # sizes the columns from what is actually displayed.
                    shown = [(self._cell(_shorten_paths_in_text(lbl)), chars)
                             for lbl, chars in comp.sources]
                    header = [("source", "l"), ("bytes", "r")]
                    rows = [[label, f"{s_chars:,}"] for label, s_chars in shown]
                    lines.extend("    " + ln for ln in format_table(header, rows))
                if not t:
                    return seq, "\n".join(lines) + "\n  text not available (not captured)"
                return seq, "\n".join(lines) + f"\n\n{t}"

            # Wire-truth parts: resolved from the captured wire body and the
            # captured per-message raw slices at call time.
            try:
                from agent.prompt_capture import get_wire_body
                wb = get_wire_body() or {}
            except Exception:
                wb = {}
            msg_nodes = self._wire_message_nodes()
            if kind == "wtool":
                wtools = wb.get("tools") or []
                if idx >= len(wtools):
                    return None
                t = wtools[idx]
                size = self._wire_bytes(t)
                form = "raw (as sent)" if display_mode != "human" else "human-readable JSON"
                return seq, (f"[tool schema] {self._tool_name(t)} ({size:,} bytes, {form})]\n\n"
                             + self._wire_text(t, display_mode))
            if kind == "wenv":
                wenv = {k: v for k, v in wb.items() if k not in ("messages", "tools")}
                size = self._wire_bytes(wenv)
                form = "raw (as sent)" if display_mode != "human" else "human-readable JSON"
                return seq, (f"[top-level params] ({size:,} bytes, {form})]\n\n"
                             + self._wire_text(wenv, display_mode))

            # LIVE TAIL: rendered at /pi call time; never sent, so no wire form exists.
            if kind == "wtail":
                tail_item = getattr(self, "_live_tail", None)
                if not isinstance(tail_item, dict):
                    return None
                clean = self._clean_tail_entry(tail_item)
                size = self._msg_body_bytes(clean)
                form = "raw" if display_mode != "human" else "human-readable JSON"
                body = (self._tail_message_text(clean)
                        if display_mode == "raw"
                        else json.dumps(clean, ensure_ascii=False, indent=2))
                return seq, (f"[live tail] role={tail_item.get('role', '?')} ({size:,} bytes, {form})]\n\n"
                             + body)

            # Conversation messages: raw slices of the captured request body.
            if kind == "wmsg":
                if idx >= len(msg_nodes):
                    return None
                node = msg_nodes[idx]
                raw = node.raw_text
                role, parsed = self._message_role_and_parsed(raw)
                size = utf8_bytes(raw)
                form = "raw (as sent)" if display_mode != "human" else "human-readable JSON"
                body = (raw if display_mode != "human"
                        else json.dumps(parsed, ensure_ascii=False, indent=2))
                return seq, f"[message] role={role} ({size:,} bytes, {form})]\n\n{body}"
            return None
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("prompt_manifest.get_line failed", exc_info=True)
            return None

    @staticmethod
    def _tool_name(t: Any) -> str:
        """Tool name of one tools-array entry (e.g. "tool: browser_exec")."""
        try:
            fn = (t.get("function") or {}) if isinstance(t, dict) else {}
            return f"tool: {fn.get('name', '?')}"
        except Exception:
            return "(tool)"

    @staticmethod
    def _wire_bytes(obj: Any) -> int:
        """UTF-8 bytes of one wire part (compact JSON form, as the SDK sends it)."""
        try:
            s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return utf8_bytes(str(obj))
        return utf8_bytes(s)

    @staticmethod
    def _wire_text(obj: Any, mode: str) -> str:
        """Drill-down text of one wire part.

        "raw": compact JSON as sent; "human": pretty-printed for reading.
        """
        try:
            if mode == "human":
                return json.dumps(obj, ensure_ascii=False, indent=2)
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(obj)

    @staticmethod
    def _wire_msg_label(m: Any) -> str:
        """Label for a wire message row: content prefix, or tool-call names.

        Content rows get their first chars on one line; assistant rows that
        only carry tool_calls (empty content) get the called tool names —
        nothing is shown as bare "?" when something descriptive exists.
        """
        if not isinstance(m, dict):
            return ""
        content = m.get("content")
        if isinstance(content, str) and content:
            label = PromptManifest._cell(content)
            return (label[:37] + "...") if len(label) > 40 else label
        calls = m.get("tool_calls") or []
        names = []
        for tc in calls:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            if name:
                names.append(str(name))
        if names:
            label = "call tool: " + ", ".join(names)
            return (label[:37] + "...") if len(label) > 40 else label
        return ""

    # -- message renderers --------------------------------------------------
    # Three distinct bodies, three distinct sources:
    #  - captured messages (wmsg): raw slices of the request body held in
    #    prompt_capture's in-memory store — printed verbatim.
    #  - tool schemas (wtool) + server parameters (wenv): read from the same
    #    in-memory wire body at display time.
    #  - live tail (wtail): never sent, so no wire form exists — rendered at
    #    /pi call time: reasoning_content in <reasoning> tags first, then the
    #    content as plain text.

    @staticmethod
    def _tail_message_text(clean: Dict[str, Any]) -> str:
        """Raw body of the live tail row (never sent — rendered, not read)."""
        parts: List[str] = []
        reasoning = clean.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            parts.append(f"<reasoning>\n{reasoning}\n</reasoning>")
        content = clean.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        tool_text = _wire_tool_call_text(clean)
        if tool_text:
            parts.append(tool_text)
        return "\n".join(parts)

    @classmethod
    def _msg_body_bytes(cls, obj: Any) -> int:
        """Bytes of the live-tail row: exactly what its raw body prints."""
        return utf8_bytes(cls._tail_message_text(obj))

    # -- display (slash command) -----------------------------------------

    def render(self, agent: Any = None, live_tail: Any = None) -> str:
        """Tabular overview of the manifest's single stored prompt.

        Renders system parts from the stored PromptRecord nodes and conversation
        messages / tool schemas / server parameters from prompt_capture's
        in-memory wire stores (the captured request body); appends the live tail
        (last entry of conversation_history) at the end when present.

        One numbered line per part, aligned columns; pass a number to
        ``get_line`` (via /pi <no>) for that part's full text.
        """
        try:
            with self._lock:
                records = list(self._records)

            if not records:
                return "Prompt manifest: no API calls recorded in this session yet."

            out: List[str] = []
            line_map: Dict[int, Tuple[int, str, int]] = {}
            n = 0
            for rec in records:
                # Read the captured wire body + message slices ONCE per render;
                # all record rows below share them (at most one record anyway).
                try:
                    from agent.prompt_capture import get_wire_body
                    wb = get_wire_body() or {}
                except Exception:
                    wb = {}
                msg_nodes = self._wire_message_nodes()
                wtools = [t for t in (wb.get("tools") or []) if isinstance(t, dict)]
                wenv = {k: v for k, v in wb.items() if k not in ("messages", "tools")}
                live_tail_clean = self._clean_tail_entry(live_tail) if isinstance(live_tail, dict) else None

                # The parts shown determine the total bytes.
                sys_bytes = sum(self._component_bytes(c) for c in rec.system_components)
                msg_bytes = sum(self._message_bytes(node) for node in msg_nodes)
                tools_bytes = (sum(self._wire_bytes(t) for t in wtools)
                               + (self._wire_bytes(wenv) if wenv else 0))
                tail_bytes = (self._msg_body_bytes(live_tail_clean) if live_tail_clean else 0)
                tot_bytes = sys_bytes + msg_bytes + tools_bytes + tail_bytes
                tok = rec.prompt_tokens

                when = time.strftime("%H:%M:%S", time.localtime(rec.built_at))
                out.append(f"Prompt #{rec.seq}  ({when})")
                if rec.label:
                    out.append(f"{rec.label}")

                # Call stats: metadata, not content - never numbered.
                # One row, "label value" pairs; plain numbers (no column padding).
                lat_s = f"{rec.latency_ms / 1000:.1f}s" if rec.latency_ms else "-"
                out.append(
                    "prompt " + ("-" if not tok else format(tok, ","))
                    + "  completion " + format(rec.completion_tokens or 0, ",")
                    + "  cache_read " + format(rec.cache_read_tokens or 0, ",")
                    + f"  latency {lat_s}"
                    + "  bytes_total " + format(tot_bytes, ",")
                )
                # Numbered content rows — dynamic column widths via format_table
                # (longest label sets the width; nothing is clipped).
                header = [("#", "r"), ("kind", "l"), ("description", "l"), ("bytes", "r")]
                rows: List[List[str]] = []

                for i, comp in enumerate(rec.system_components):
                    n += 1
                    line_map[n] = (rec.seq, "sys", i)
                    rows.append([str(n), comp.kind + ":", self._cell(comp.description),
                                 f"{self._component_bytes(comp):,}"])

                # Conversation messages: raw slices of the captured request body.
                for i, node in enumerate(msg_nodes):
                    role, parsed = self._message_role_and_parsed(node.raw_text)
                    n += 1
                    line_map[n] = (rec.seq, "wmsg", i)
                    rows.append([str(n), role + ":", self._wire_msg_label(parsed),
                                 f"{self._message_bytes(node):,}"])

                # Server parameters + tool schemas from the CAPTURED wire body
                # (in-memory copy of the SDK's own serialized request — same
                # object dumped to prompt.json), in wire key order.
                if wenv:
                    n += 1
                    line_map[n] = (rec.seq, "wenv", 0)
                    rows.append([str(n), "generated:", "server parameters",
                                 f"{PromptManifest._wire_bytes(wenv):,}"])
                for i, t in enumerate(wtools):
                    n += 1
                    line_map[n] = (rec.seq, "wtool", i)
                    rows.append([str(n), "hardcoded:", self._tool_name(t),
                                 f"{PromptManifest._wire_bytes(t):,}"])

                # LIVE TAIL: the last conversation-history entry — the one
                # piece this turn added that no API call sent yet (the
                # final reply, or a tool result). Appended after everything;
                # always shown when present.
                if live_tail_clean:
                    self._live_tail = live_tail_clean
                    n += 1
                    line_map[n] = (rec.seq, "wtail", 0)
                    rows.append([str(n), str(live_tail.get("role")) + ":",
                                 self._wire_msg_label(live_tail_clean),
                                 f"{self._msg_body_bytes(live_tail_clean):,}"])

                out.extend(format_table(header, rows))

                out.append("")

            with self._lock:
                self._line_to_seq = line_map
            rendered = "\n".join(out).rstrip()
            return (rendered + "\n\nUse /pi <part no.> for the full text and sources of a part.")
        except Exception:
            return "Prompt manifest: render failed (see logs)."



    @staticmethod
    def _clean_tail_entry(entry: Any) -> Dict[str, Any]:
        """One conversation-history entry reduced to its wire-relevant fields.

        History rows carry local bookkeeping (timestamps, DB flags); only the
        payload that an API call would send is shown. reasoning_content is
        preferred over 'reasoning' (the field the sanitizer emits for the wire).

        Trailing whitespace is stripped from content: it originates in the raw
        model output but is removed before display (final_response .strip()),
        so showing it here would misrepresent what the user saw.
        """
        if not isinstance(entry, dict):
            return {}
        out: Dict[str, Any] = {}
        role = entry.get("role")
        if role:
            out["role"] = role
        content = entry.get("content")
        if content is not None:
            out["content"] = content.rstrip() if isinstance(content, str) else content
        reasoning = entry.get("reasoning_content") or entry.get("reasoning")
        if reasoning:
            out["reasoning_content"] = reasoning
        for key in ("tool_calls", "name", "tool_call_id"):
            if entry.get(key) is not None:
                out[key] = entry[key]
        return out

    @staticmethod
    def _cell(value) -> str:
        """Single-line cell text (multi-line input folded onto one line)."""
        return " ".join(str(value or "").split())

    def get_line_header(self, line_no: int) -> Optional[str]:
        """Tabular header line for one numbered part from the last render.

        Shows kind, description and chars — the same numbers as the /pi
        overview row, with roomier columns.
        """
        try:
            with self._lock:
                entry = self._line_to_seq.get(line_no)
                records = list(self._records)
            if entry is None:
                return None
            seq, kind, idx = entry
            rec = next((r for r in records if r.seq == seq), None)
            if not rec:
                return None

            # Single-row table through format_table (dynamic widths, same
            # rules as the overview); only alignments matter here.
            if kind == "sys":
                comp = rec.system_components[idx]
                desc = self._cell(comp.description) or "(text not captured)"
                row = [str(line_no), f"{comp.kind}:", desc, f"{self._component_bytes(comp):,}"]
            elif kind == "wtool":
                try:
                    from agent.prompt_capture import get_wire_body
                    wb = get_wire_body() or {}
                except Exception:
                    wb = {}
                wtools = wb.get("tools") or []
                if idx >= len(wtools):
                    return None
                t = wtools[idx]
                row = [str(line_no), "hardcoded:", self._tool_name(t),
                       f"{self._wire_bytes(t):,}"]
            elif kind == "wenv":
                try:
                    from agent.prompt_capture import get_wire_body
                    wb = get_wire_body() or {}
                except Exception:
                    wb = {}
                wenv = {k: v for k, v in wb.items() if k not in ("messages", "tools")}
                row = [str(line_no), "generated:", "server parameters",
                       f"{self._wire_bytes(wenv):,}"]
            elif kind == "wtail":
                # Live tail: rendered at /pi call time; never sent, no wire form.
                tail_item = getattr(self, "_live_tail", None)
                if not isinstance(tail_item, dict):
                    return None
                row = [str(line_no), str(tail_item.get("role")) + ":",
                       self._wire_msg_label(tail_item),
                       f"{self._msg_body_bytes(tail_item):,}"]
            elif kind == "wmsg":
                nodes = self._wire_message_nodes()
                if idx >= len(nodes):
                    return None
                node = nodes[idx]
                role, parsed = self._message_role_and_parsed(node.raw_text)
                row = [str(line_no), role + ":", self._wire_msg_label(parsed),
                       f"{self._message_bytes(node):,}"]
            else:
                return None
            header = [("#", "r"), ("kind", "l"), ("description", "l"), ("bytes", "r")]
            return format_table(header, [row], show_header=False)[0]
        except Exception:
            return None


def get_or_create_manifest(agent: Any) -> PromptManifest:
    """Return the agent's manifest, creating it on first use."""
    m = getattr(agent, "_prompt_manifest", None)
    if not isinstance(m, PromptManifest):
        m = PromptManifest()
        try:
            agent._prompt_manifest = m
        except Exception:
            pass
    return m
