"""Per-agent prompt manifest: what each API call's prompt was made of.

Tracks prompt components by character count at assembly time (system-prompt
tiers, tool schemas, conversation messages).  Tokens are NEVER estimated per
component — provider usage (prompt/completion/cache tokens) is recorded per
prompt at the send points, so the manifest always shows the real token total
of each call alongside its char-based breakdown.

Model:
  - A ``PromptManifest`` lives on one agent (``agent._prompt_manifest``).
  - System-prompt tiers are stored ONCE per agent via ``set_system_components``
    at build time and referenced by every subsequent send until a rebuild.
  - Each API call calls ``record_send`` with the wire messages + tool schemas;
    that snapshots those plus the current system components into a
    ``PromptRecord`` and returns it.
  - After the response arrives, ``fill_usage`` stamps that record's provider
    token counts.

In-memory only (per process).  Only the LAST prompt per agent is kept; each
new send replaces the previous record.  Prompt TEXT for drill-down is stored
once, not per record: system-prompt tiers and tool schemas are session-stable
(rebuilt only on compression), so the manifest keeps one canonical copy of
each; records reference it by id and keep char counts only.  Conversation-
message text is stored verbatim per record (single record kept ⇒ bounded by
the context window; no eviction).

Read side: ``render()`` numbers every content line; ``get_line(n)`` resolves a
number from the last render to that component's full text.
"""

from __future__ import annotations

import json
import logging
import re
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


# ── Nodes ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComponentNode:
    """One named prompt component. Char count; no token estimation.

    Drill-down text is stored on the node itself (``text``); records hold
    a reference to it, so identical tiers share one text copy across every
    record and across agents that register the same shared id.

    ``sources`` lists per-block provenance ``(label, chars)`` in assembly
    order — e.g. which files (SOUL.md, AGENTS.md) or generated constants
    make up a tier — so /pi can show where the text was loaded from.
    """

    description: str
    chars: int
    text: str = ""
    sources: Tuple[Tuple[str, int], ...] = ()


@dataclass
class MessageNode:
    """One conversation message attached to the prompt."""

    role: str
    chars: int
    tool_chars: int = 0   # char cost of tool_calls on an assistant message
    content: str = ""     # readable (decoded) message text (for drill-down)
    tool_text: str = ""   # tool_calls as readable text (for drill-down)
    raw_content: str = ""  # wire form exactly as sent (no decoding)
    raw_tool_text: str = ""  # tool_calls JSON exactly as sent
    origin: str = ""      # "resumed from session DB" or "this run"
    reasoning_chars: int = 0  # char cost of reasoning_content on the wire
    reasoning_text: str = ""  # reasoning exactly as sent (echo-back providers)


def _message_content_chars(content: Any) -> int:
    """Character count of an OpenAI-format message's content (string or parts)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, str):
                total += len(part)
            elif isinstance(part, dict):
                total += len(str(part.get("text", "")))
        return total
    return 0


def _maybe_decode_json_text(value: Any) -> Any:
    """Unwrap a JSON-encoded string into its native form when it looks like one.

    Several tools (read_file, search_files, terminal) return their result as a
    JSON *string* in the wire message content, so drill-downs would otherwise
    show ``\\n`` escapes on one long line instead of readable text.  When the
    payload parses as a JSON dict, every field is rendered equally (nothing
    dropped, nothing preferred); unparseable input passes through untouched.
    Char counts are computed separately and stay faithful to the wire form.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    # Cheap pre-filter: must start and end with matching brackets.
    if len(stripped) < 2 or not (
        (stripped[0] == "{" and stripped[-1] == "}")
        or (stripped[0] == "[" and stripped[-1] == "]")
    ):
        return value
    try:
        decoded = json.loads(stripped)
    except (ValueError, RecursionError):
        return value

    if isinstance(decoded, dict):
        # All fields treated equally, in envelope order: a "key:" line, then
        # the value. Values containing newlines (or pretty-printed JSON) start
        # on the next line; each of their lines is indented by two spaces.
        # Nothing is dropped and no field is preferred over another.
        blocks = []
        for key, inner in decoded.items():
            body = inner if isinstance(inner, str) else json.dumps(
                inner, ensure_ascii=False, indent=2)
            if "\n" in body:
                indented = "\n".join("  " + l for l in body.split("\n"))
                blocks.append(f"{key}:\n{indented}")
            else:
                blocks.append(f"{key}: {body}")
        return "\n".join(blocks) if blocks else value
    if isinstance(decoded, list):
        parts = [str(item.get("text", item)) for item in decoded if isinstance(item, dict)]
        return "\n\n".join(parts) if parts else json.dumps(decoded, ensure_ascii=False, indent=2)
    # JSON scalars/numbers-as-strings (e.g. "12345"): keep as-is — decoding adds nothing.
    return value


def _content_to_str(content: Any) -> str:
    """Readable single-string form of an OpenAI-format message's content."""
    if isinstance(content, str):
        return _maybe_decode_json_text(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(_maybe_decode_json_text(part))
            elif isinstance(part, dict):
                text = part.get("text", "")
                parts.append(_maybe_decode_json_text(text) if isinstance(text, str) else str(text))
        return "\n\n".join(p for p in parts if p)
    return str(content or "")


def _wire_content_to_str(content: Any) -> str:
    """Wire-form single-string content (no JSON decoding — as sent)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "\n\n".join(p for p in parts if p)
    return str(content or "")


def _wire_tool_call_text(msg: Dict[str, Any]) -> str:
    """tool_calls in wire form: each call's arguments EXACTLY as sent.

    On the wire, tool-call arguments are compact JSON *strings*; this renders
    them verbatim (one call per line) so /json off shows pure wire data.
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


def _tool_call_chars(msg: Dict[str, Any]) -> int:
    """Character cost of tool calls attached to an assistant message."""
    total = 0
    for tc in msg.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        total += len(str(fn.get("name", ""))) + len(str(fn.get("arguments", "")))
    return total


def _tool_call_text(msg: Dict[str, Any]) -> str:
    """Readable form of tool_calls attached to an assistant message."""
    calls = msg.get("tool_calls") or []
    if not calls:
        return ""
    blocks = []
    for tc in calls:
        fn = (tc or {}).get("function") or {}
        raw_args = fn.get("arguments", "")
        # Wire arguments arrive as a JSON *string* with escaped \n; parse it so
        # the drill-down prints real line breaks instead of escape sequences.
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                args = raw_args
        else:
            args = raw_args

        def _pretty(v: Any, ind: int = 0) -> str:
            pad = "  " * ind
            if isinstance(v, dict):
                if not v:
                    return "{}"
                items = [
                    f"{pad}  {json.dumps(k, ensure_ascii=False)}: {_pretty(x, ind + 1)}"
                    for k, x in v.items()
                ]
                return "{\n" + ",\n".join(items) + f"\n{pad}}}"
            if isinstance(v, list):
                if not v:
                    return "[]"
                items = [f"{pad}  {_pretty(x, ind + 1)}" for x in v]
                return "[\n" + ",\n".join(items) + f"\n{pad}]"
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)

        try:
            args_str = _pretty(args)
        except Exception:
            args_str = str(args)
        blocks.append(f"{fn.get('name', '?')}:\n{args_str}")
    return "\n".join(blocks)


def _wire_message_node(msg: Any) -> Optional[MessageNode]:
    """Build a MessageNode from one wire message dict, or None if not a dict."""
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    if role == "system":
        # The system prompt is tracked separately as tier components; skipping it
        # here avoids double counting.
        return None
    if not role:
        role = "?"
    content = msg.get("content")
    raw_reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
    if not isinstance(raw_reasoning, str):
        raw_reasoning = str(raw_reasoning)
    return MessageNode(
        role=str(role),
        chars=_message_content_chars(content),
        tool_chars=_tool_call_chars(msg),
        content=_content_to_str(content) if content is not None else "",
        raw_content=content if isinstance(content, str) else _wire_content_to_str(content)
                   if content is not None else "",
        tool_text=_tool_call_text(msg),
        raw_tool_text=_wire_tool_call_text(msg),
        reasoning_chars=len(raw_reasoning),
        reasoning_text=raw_reasoning,
    )


# ── Records ──────────────────────────────────────────────────────────────


@dataclass
class PromptRecord:
    """What one API call's prompt consisted of, plus its provider token counts.

    System tiers and tool schemas carry only char counts here; their drill-down
    text lives on the canonical objects (``ComponentNode.text`` / the manifest's
    tool-schema store) so it is stored once per agent, never per record.
    """

    seq: int
    built_at: float = field(default_factory=time.time)
    system_components: List[ComponentNode] = field(default_factory=list)  # refs to canonical nodes
    messages: List[MessageNode] = field(default_factory=list)             # in send order (no system)
    tool_schemas_chars: int = 0
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
        self._shared_components: Dict[str, ComponentNode] = {}
        # Canonical system-tier nodes (cid -> node); replaced wholesale on each
        # build so drill-down always shows the CURRENT tier text.
        self._system_by_cid: Dict[str, ComponentNode] = {}
        # Persistent system-component list (set at build, referenced by sends):
        self._current_system: List[ComponentNode] = []
        # Canonical tool-schema drill-down text (stored once; refreshed only
        # when the serialized payload actually changes).
        self._tools_raw: str = ""
        self._tools_pretty: str = ""
        # line number (last render) -> (record_seq, kind, index), for get_line()
        # kind in {stats_tokens, stats_ratio, sys, tools, msg}; index = item pos
        self._line_to_seq: Dict[int, Tuple[int, str, int]] = {}

    # -- shared component registry (subagent inheritance) ---------------

    def register_shared(self, cid: str, node: ComponentNode) -> ComponentNode:
        """Register a component as shared under id *cid*.

        If *cid* is already known the stored canonical node wins (callers then
        reference it instead of their own copy); otherwise *node* is stored.
        Identical content built by several agents thus appears once, while each
        manifest keeps its own ordered reference list.
        """
        with self._lock:
            existing = self._shared_components.get(cid)
            if existing is not None:
                return existing
            self._shared_components[cid] = node
            return node

    # -- write side (attachment sites) ----------------------------------

    def set_system_components(self, components: List[ComponentNode],
                              texts: Optional[List[str]] = None) -> None:
        """Store the system prompt's ordered tier components for this agent.

        Called from ``build_system_prompt`` after the tiers are assembled.
        Blank tiers are dropped; *texts* (same order) become the canonical
        drill-down text, stored ONCE here (attached to the nodes) and shared
        by every record until a rebuild replaces them — not copied per send.
        Per-tier provenance rides on each node's ``sources``; a rebuild passes
        fresh nodes, so sources never go stale.
        """
        with self._lock:
            comps = list(components or [])
            keep = [i for i, c in enumerate(comps) if c and c.chars > 0]
            kept_nodes = []
            for i in keep:
                node = comps[i]
                text = texts[i] if isinstance(texts, list) and i < len(texts) else ""
                node = ComponentNode(node.description, node.chars,
                                     text or node.text, tuple(node.sources))
                self._system_by_cid[f"sys|{node.description}"] = node
                kept_nodes.append(node)
            self._current_system = kept_nodes

    def _tool_schema_chars(self, tools: Any) -> Tuple[int, str]:
        """(char count, wire form) of the tool-schema payload (compact JSON, as sent)."""
        if not tools:
            return 0, ""
        try:
            raw = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            # Fallback: rough per-tool size so a non-serializable schema never
            # zeroes the manifest silently.
            total = 0
            for t in tools or []:
                fn = (t or {}).get("function", {})
                total += len(str(fn.get("name", ""))) + len(str(fn.get("description", "")))
                total += len(str(fn.get("parameters", "")))
            return total, str(tools)[:8192]
        return len(raw), raw

    def _tool_schema_text(self, tools: Any) -> str:
        """Pretty JSON of the tool schemas (human drill-down); raw is as sent."""
        if not tools:
            return ""
        try:
            return json.dumps(tools, ensure_ascii=False, indent=2)
        except Exception:
            return str(tools)[:8192]

    def record_send(self, api_messages: Any, tools_for_api: Any = None,
                    label: str = "", this_run_start_idx: Optional[int] = None) -> Optional[PromptRecord]:
        """Snapshot the prompt about to be sent into a new PromptRecord.

        Pass ``api_messages`` in its final wire form (post-sanitization) and the
        tools list being sent.  Returns the record so callers can later
        :meth:`fill_usage` it; the record is already stored.  Never raises.

        Only the LAST prompt per agent is kept (MAX_RECORDS = 1): each send
        replaces the previous record.

        ``this_run_start_idx`` (index into *api_messages*, usually the current
        turn's user row) tags every message before it as "resumed from session
        DB" and the rest as "this run".
        """
        try:
            with self._lock:
                messages = [n for n in (_wire_message_node(m) for m in (api_messages or [])) if n]
                if this_run_start_idx is not None:
                    for i, node in enumerate(messages):
                        node.origin = "resumed from session DB" if i < this_run_start_idx else "this run"
                tool_chars, tool_raw = self._tool_schema_chars(tools_for_api)
                # Canonical tool-schema text is stored once and refreshed only
                # when the serialized payload actually changes.
                if tool_raw != self._tools_raw:
                    self._tools_raw = tool_raw
                    self._tools_pretty = self._tool_schema_text(tools_for_api)
                record = PromptRecord(
                    seq=self._next_seq,
                    built_at=time.time(),
                    system_components=list(self._current_system),
                    messages=messages,
                    tool_schemas_chars=tool_chars,
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
    def _record_totals(rec: PromptRecord) -> int:
        return (sum(c.chars for c in rec.system_components)
                + sum(m.chars + m.tool_chars + m.reasoning_chars for m in rec.messages)
                + rec.tool_schemas_chars)

    # -- drill-down -------------------------------------------------------

    def get_line(self, line_no: int, display_mode: str = "off") -> Optional[Tuple[int, str]]:
        """Resolve a numbered output line (from /pi) to prompt text.

        Returns ``(seq, full_text)`` or None if unknown.  Numbers refer to the
        lines shown by the most recent ``render()`` call — so run /pi first,
        then use its numbers.  Walks the same line ordering ``render()`` uses.

        ``display_mode`` (set via /json): "off" (default) shows message contents
        in their raw wire form (exactly as sent) and system parts as pure text;
        "human" decodes tool-result JSON into readable text with real line breaks
        and shows each system part's header + source sections.
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
                tok = rec.prompt_tokens
                tot = self._record_totals(rec)
                # Raw: just the text of this whole part, no metadata. Human:
                # header + source provenance (which files/constants make up this tier).
                if display_mode == "off":
                    return seq, t if t else "  text not available (not captured)"
                lines = [f"[{self._cell(comp.description)} ("
                         f"{len(t):,} chars,  {self._tok_est(len(t), tot, tok):>8} tok)]"]
                if comp.sources:
                    lines.append("  Sections:")
                    # Shorten embedded absolute paths in-place; compute the
                    # column from the DISPLAYED (shortened) labels so nothing
                    # is clipped.
                    shown = [(self._cell(_shorten_paths_in_text(lbl)), chars)
                             for lbl, chars in comp.sources]
                    width = max([8] + [len(l) for l, _ in shown]) + 2
                    lines.append(f"    {'source':<{width}} {'chars':>9} {'~tok':>7}")
                    for label, s_chars in shown:
                        lines.append(
                            f"    {label[:width]:<{width}} "
                            f"{s_chars:>9,} {self._tok_est(s_chars, tot, tok):>7}")
                if not t:
                    return seq, "\n".join(lines) + "\n  text not available (not captured)"
                return seq, "\n".join(lines) + f"\n\n{t}"

            if kind == "tools":
                with self._lock:
                    raw = self._tools_raw
                    pretty = self._tools_pretty
                t = self._content_for_mode(pretty, raw, display_mode)
                if not t:
                    return seq, "[tool schemas] text not available (not captured)"
                tok = rec.prompt_tokens
                tot = self._record_totals(rec)
                form = "wire form" if display_mode != "human" else "pretty-printed JSON"
                return seq, (f"[tool schemas ("
                             f"{len(t):,} chars,  {self._tok_est(len(t), tot, tok):>8} tok, {form})]\n\n{t}")

            m = rec.messages[idx]
            msg_size = m.chars + m.tool_chars + m.reasoning_chars
            tok = rec.prompt_tokens
            tot = self._record_totals(rec)
            parts = [f"[message] role={m.role} ("
                     f"{msg_size:,} chars,  {self._tok_est(msg_size, tot, tok):>8} tok)"]
            if m.origin:
                parts.append(f"origin: {m.origin}")
            content = self._content_for_mode(m.content, m.raw_content, display_mode)
            parts.append(content or "(no content)")
            reasoning = (m.reasoning_text or "").strip()
            if reasoning:
                parts.append(f"reasoning (sent on the wire, {m.reasoning_chars:,} chars):\n{reasoning}")
            tool_text = self._content_for_mode(m.tool_text, m.raw_tool_text, display_mode)
            if tool_text:
                parts.append(f"tool_calls:\n{tool_text}")
            return seq, "\n".join(parts)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("prompt_manifest.get_line failed", exc_info=True)
            return None

    @staticmethod
    def _content_for_mode(readable: str, raw: str, mode: str) -> str:
        """Pick the drill-down text form for one payload based on display mode.

        "off": raw wire form exactly as sent; "human": decoded readable form.
        """
        if mode == "human":
            return readable or raw
        return raw or readable

    # -- display (slash command) -----------------------------------------

    def render(self) -> str:
        """Tabular overview of the manifest's single stored prompt.

        One numbered line per part, aligned columns; no details here.  Pass a
        number to ``get_line`` (via /pi <no>) for that part's full text,
        sources and token counts.
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
                sys_chars = sum(c.chars for c in rec.system_components)
                msg_chars = sum(m.chars + m.tool_chars + m.reasoning_chars for m in rec.messages)
                tot_chars = self._record_totals(rec)
                tok = rec.prompt_tokens

                when = time.strftime("%H:%M:%S", time.localtime(rec.built_at))
                out.append(f"Prompt #{rec.seq}  ({when})")
                if rec.label:
                    out.append(f"{rec.label}")

                # Call stats: metadata, not content - never numbered.
                # One row, "label value" pairs; plain numbers (no column padding).
                lat_s = f"{rec.latency_ms / 1000:.1f}s" if rec.latency_ms else "-"
                ratio = (f"{tot_chars / tok:.2f}" if tok else "-")
                out.append(
                    "prompt " + ("-" if not tok else format(tok, ","))
                    + "  completion " + format(rec.completion_tokens or 0, ",")
                    + "  cache_read " + format(rec.cache_read_tokens or 0, ",")
                    + f"  latency {lat_s}"
                    + "  chars_total " + format(tot_chars, ",")
                    + f"  chars/tok {ratio}"
                )
                # Column header for the numbered rows below.
                out.append(f"{'':>6}  {'kind':<5} {'part':<40} {'chars':>9} {'tok*':>8}")

                for i, comp in enumerate(rec.system_components):
                    n += 1
                    line_map[n] = (rec.seq, "sys", i)
                    share = f"{100.0 * comp.chars / sys_chars:>5.1f}%" if sys_chars else ""
                    out.append(
                        f"{n:>6}  {'[sys]':<5} {self._cell(comp.description)[:40]:<40}"
                        f" {comp.chars:>9,} {self._tok_est(comp.chars, tot_chars, tok):>8}"
                        f"{share:>7}"
                    )

                if rec.tool_schemas_chars:
                    n += 1
                    line_map[n] = (rec.seq, "tools", 0)
                    out.append(
                        f"{n:>6}  {'[sys]':<5} {'(tool schemas)':<40}"
                        f" {rec.tool_schemas_chars:>9,}"
                        f" {self._tok_est(rec.tool_schemas_chars, tot_chars, tok):>8}"
                    )

                for i, m in enumerate(rec.messages):
                    size = m.chars + m.tool_chars + m.reasoning_chars
                    n += 1
                    line_map[n] = (rec.seq, "msg", i)
                    share = f"{100.0 * size / msg_chars:>5.1f}%" if msg_chars else ""
                    # Tags removed - see user feedback on display clutter
                    out.append(
                        f"{n:>6}  {'[msg]':<5} {self._cell(m.role)[:40]:<40}"
                        f" {size:>9,} {self._tok_est(size, tot_chars, tok):>8}"
                        f"{share:>7}"
                    )

                out.append("")

            with self._lock:
                self._line_to_seq = line_map
            tail = "\n".join(out).rstrip()
            return (tail + "\n\n*tok* proportional estimate from the prompt's real token total. "
                       "Use /pi <part no.> for full text, sources and tokens of a part.")
        except Exception:
            return "Prompt manifest: render failed (see logs)."



    @staticmethod
    def _tok_est(part_chars: int, total_chars: int, tok):
        """Proportional token estimate for one part, from the prompt's real tokens."""
        if tok and total_chars:
            return f"{int(part_chars * tok / total_chars):,}"
        return "-"

    @staticmethod
    def _cell(value) -> str:
        """Single-line cell text (multi-line input folded onto one line)."""
        return " ".join(str(value or "").split())

    def get_line_header(self, line_no: int) -> Optional[str]:
        """Tabular header line for one numbered part from the last render.

        Shows kind, description, chars and proportional token estimate - the
        same numbers as the /pi overview row, with roomier columns.
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
            tok = rec.prompt_tokens
            tot = self._record_totals(rec)

            if kind == "sys":
                comp = rec.system_components[idx]
                desc = self._cell(comp.description) or "(text not captured)"
                return (f"{line_no:>6}  {'[sys]':<5} {desc:<48}"
                        f" {comp.chars:>9,}"
                        f"{self._tok_est(comp.chars, tot, tok):>8}")

            if kind == "tools":
                return (f"{line_no:>6}  {'[sys]':<5} {'(tool schemas)':<48}"
                        f" {rec.tool_schemas_chars:>9,}"
                        f"{self._tok_est(rec.tool_schemas_chars, tot, tok):>8}")

            m = rec.messages[idx]
            size = m.chars + m.tool_chars + m.reasoning_chars
            tags = [t for t in (m.origin,
                                f"reasoning {m.reasoning_chars:,} chars" if m.reasoning_chars else None)
                    if t]
            tag_s = ("  [" + ", ".join(tags) + "]") if tags else ""
            return (f"{line_no:>6}  {'[msg]':<5} {self._cell(m.role):<48}"
                    f" {size:>9,}"
                    f"{self._tok_est(size, tot, tok):>8}{tag_s}")
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
