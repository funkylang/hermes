"""Streaming output, reasoning preview, tool progress callbacks, and busy-command spinner for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import textwrap
import time

from contextlib import contextmanager
from pathlib import Path
from rich.markup import escape as _escape

from agent.i18n import t
from agent.think_scrubber import THINK_CLOSE_TAGS, THINK_OPEN_TAGS

# Model-generated reasoning tags: suppressed during streaming (they'd display as raw XML;
# the agent strips them from final_response too) unless show_reasoning routes them to the box.
_OPEN_TAGS = THINK_OPEN_TAGS
_CLOSE_TAGS = THINK_CLOSE_TAGS
_MAX_CLOSE_TAG_LEN = max(len(t) for t in _CLOSE_TAGS)

# Ordered (prefix, catalog key) rows for _slow_command_status — first match wins; the key is
# resolved through t() at call time so the active language applies.
_SLOW_COMMAND_STATUS = (
    ("/skills search", "cli.stream.busy_skills_search"), ("/skills browse", "cli.stream.busy_skills_browse"),
    ("/skills inspect", "cli.stream.busy_skills_inspect"), ("/skills install", "cli.stream.busy_skills_install"),
    ("/skills", "cli.stream.busy_skills"), ("/browser", "cli.stream.busy_browser"))
_SLOW_COMMAND_STATUS_EXACT = {
    "/reload-mcp": "cli.stream.busy_reload_mcp",
    "/reload-skills": "cli.stream.busy_reload_skills",
    "/reload_skills": "cli.stream.busy_reload_skills"}


def _thinking_prefix() -> str:
    """``  [thinking] `` — the reasoning-preview label; also measured for wrap width."""
    return f"  {t('cli.stream.thinking_label')} "


def _terminal_columns(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


def _wrap_box_line(line: str, width: int) -> list[str]:
    """Word-wrap a box content line to the interior width (no-op when it fits).

    Fits-in-width lines pass through untouched so code indentation and exact spacing are
    preserved. Over-width lines wrap at word boundaries; the first physical line keeps the
    original leading indent. Tokens longer than a full physical line are hard-split."""
    if not line or len(line) <= width:
        return [line]
    m = re.match(r"[ \t]+", line)
    indent = m.group(0) if m else ""
    out, cur = [], ""

    def emit(word_part: str) -> None:
        """Place word_part on the current physical line or open a new one."""
        nonlocal cur
        # Hard-split pieces that exceed even an empty physical line.
        while len(word_part) > width - (len(indent) if not out and not cur else 0):
            room = width - len(cur) - (1 if cur else 0)
            if room < 4:
                out.append(cur)
                cur = ""
                room = width - (len(indent) if not out else 0)
            out.append((cur + " " if cur else "") + word_part[:room])
            cur = ""                          # that content is committed to `out` now
            word_part = word_part[room:]
        if not cur:
            cur = (indent + word_part) if (not out and indent) else word_part
        elif len(cur) + 1 + len(word_part) <= width:
            cur += " " + word_part
        else:
            out.append(cur)
            cur = word_part

    for word in line.strip().split():
        emit(word)
    if cur:
        out.append(cur)
    return [ln.rstrip() for ln in out]


class CLIStreamMixin:
    """Streaming output, reasoning preview, tool progress callbacks, and busy-command spinner for the interactive CLI"""

    def _on_thinking(self, text: str) -> None:
        """Called by agent when thinking starts/stops. Updates TUI spinner."""
        if getattr(getattr(self, "agent", None), "_mute_notification_reply", False):
            return
        from agent.warning_notifications import DiagnosticText, render_notification
        def show():
            if not text:
                self._flush_reasoning_preview(force=True)
            self._spinner_text = text or ""
            self._tool_start_time = 0.0  # clear tool timer when switching to thinking
            self._invalidate()
        render_notification(show, platform="cli", diagnostic=isinstance(text, DiagnosticText),
                            user_config=getattr(getattr(self, "agent", None), "_notification_config", None))

    def _on_notice(self, notice) -> None:
        """Queue an out-of-band AgentNotice for rendering at the next clean boundary.

        Notices fire mid-turn (cold-start seed, per-turn _capture_credits); printing immediately
        races the stream and buries the line behind the prompt. Flushed by _flush_credit_notices()
        after run_conversation returns. Fail-soft.
        """
        try:
            text = getattr(notice, "text", "") or ""
            if getattr(getattr(self, "agent", None), "_mute_notification_reply", False):
                return
            if not text:
                return
            level = getattr(notice, "level", "info") or "info"
            from agent.warning_notifications import is_diagnostic_notice, render_notification
            def queue_notice():
                if not hasattr(self, "_pending_credit_notices"):
                    self._pending_credit_notices = []
                self._pending_credit_notices.append((level, text))
            render_notification(queue_notice, platform="cli", diagnostic=is_diagnostic_notice(notice),
                                user_config=getattr(getattr(self, "agent", None), "_notification_config", None))
        except Exception:
            pass

    def _flush_credit_notices(self) -> None:
        """Print queued credit notices as level-colored lines at turn end (after
        run_conversation) where _cprint paints cleanly above the prompt."""
        from cli import _DIM, _RST, _cprint
        try:
            pending = getattr(self, "_pending_credit_notices", None)
            if not pending:
                return
            self._pending_credit_notices = []
            colors = {"error": "\033[31m", "warn": "\033[33m", "success": "\033[32m", "info": _DIM}
            for level, text in pending:
                _cprint(f"  {colors.get(level, _DIM)}{text}{_RST}")
        except Exception:
            pass

    def _on_notice_clear(self, key: str) -> None:
        """No-op for the REPL (lines are printed, no persistent slot to wipe); kept so the
        agent's clear callback is bound symmetrically with the show callback."""
        return

    def _current_reasoning_callback(self):
        """Return the active reasoning display callback for the current mode."""
        if self.show_reasoning and self.streaming_enabled:
            return self._stream_reasoning_delta
        if self.verbose and not self.show_reasoning:
            return self._on_reasoning
        return None

    def _emit_reasoning_preview(self, reasoning_text: str) -> None:
        """Render a buffered reasoning preview as a single [thinking] block."""
        from cli import _DIM, _RST, _cprint
        preview_text = reasoning_text.strip()
        if not preview_text:
            return
        wrap_width = max(30, _terminal_columns() - len(_thinking_prefix()) - 2)
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n+", preview_text.replace("\r\n", "\n")):
            compact = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
            if compact:
                paragraphs.append(textwrap.fill(compact, width=wrap_width))
        preview_text = "\n".join(paragraphs)
        if not preview_text:
            return
        if self.verbose:
            _cprint(f"{_DIM}{_thinking_prefix()}{preview_text}{_RST}")
            return
        lines = preview_text.splitlines()
        if len(lines) > 5:
            preview = "\n".join(lines[:5]) + f"\n  {t('cli.stream.thinking_more_lines', count=len(lines) - 5)}"
        else:
            preview = preview_text
        _cprint(f"{_DIM}{_thinking_prefix()}{preview}{_RST}")

    def _flush_reasoning_preview(self, *, force: bool = False) -> None:
        """Flush buffered reasoning text at natural boundaries.

        Some providers stream reasoning in tiny word/punctuation chunks; buffering keeps the
        preview path from printing one `[thinking]` line per token.
        """
        buf = getattr(self, "_reasoning_preview_buf", "")
        if not buf:
            return
        target_width = max(40, _terminal_columns() - len(_thinking_prefix()) - 4)
        flush_text = ""
        if force:
            flush_text, buf = buf, ""
        else:
            line_break = buf.rfind("\n")
            min_newline_flush = max(16, target_width // 3)
            if line_break != -1 and (
                line_break >= min_newline_flush
                or buf.endswith(("\n\n", ".\n", "!\n", "?\n", ":\n"))):
                flush_text, buf = buf[: line_break + 1], buf[line_break + 1 :]
            elif len(buf) >= target_width:
                search_start = max(20, target_width // 2)
                search_end = min(
                    len(buf), max(target_width + (target_width // 3), target_width + 8))
                cut = max(
                    buf.rfind(b, search_start, search_end)
                    for b in (" ", "\t", ".", "!", "?", ",", ";", ":"))
                if cut != -1:
                    flush_text, buf = buf[: cut + 1], buf[cut + 1 :]

        self._reasoning_preview_buf = buf.lstrip() if flush_text else buf
        if flush_text:
            self._emit_reasoning_preview(flush_text)

    def _format_submitted_user_message_preview(self, user_input: str) -> str:
        """Format the submitted user-message scrollback preview (content only)."""
        lines = user_input.split("\n")
        if len(lines) <= 1:
            return f"[bold]{_escape(user_input)}[/]"

        first_lines = max(1, int(getattr(self, "user_message_preview_first_lines", 2)))
        last_lines = max(0, int(getattr(self, "user_message_preview_last_lines", 2)))
        head = lines[:first_lines]
        tail_count = min(last_lines, max(0, len(lines) - len(head)))
        tail = lines[-tail_count:] if tail_count else []
        hidden_middle_count = len(lines) - len(head) - len(tail)
        if hidden_middle_count < 0:
            hidden_middle_count = 0
            tail = []

        preview_lines = [f"[bold]{_escape(line)}[/]" for line in head]
        if hidden_middle_count > 0:
            key = "cli.stream.preview_more_lines_one" if hidden_middle_count == 1 else "cli.stream.preview_more_lines_other"
            preview_lines.append(f"[dim]{_escape(t(key, count=hidden_middle_count))}[/]")
        preview_lines.extend(f"[bold]{_escape(line)}[/]" for line in tail)
        return "\n".join(preview_lines)

    def _expand_paste_references(self, text: str | None) -> str:
        """Expand [Pasted text #N -> file] placeholders into file contents."""
        from cli import logger
        if not isinstance(text, str) or "[Pasted text #" not in text:
            return text or ""
        paste_ref_re = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')

        def _expand_ref(match):
            path = Path(match.group(1))
            # try/except rather than path.exists(): the paste file may be deleted between
            # check and read (TOCTOU), silently dropping the input.
            try:
                # See #17666.
                return path.read_text(encoding="utf-8-sig")
            except (OSError, IOError):
                logger.warning("Paste file gone or unreadable, returning placeholder: %s", path)
                return match.group(0)

        return paste_ref_re.sub(_expand_ref, text)

    def _user_header_line(self) -> str:
        """Divider-with-label used as the top border of a user message box."""
        from cli import datetime, _accent_hex
        label = t("cli.stream.user_label")
        ts = datetime.now().strftime(getattr(self, "timestamp_format", "%H:%M")) \
            if getattr(self, "show_timestamps", False) else None
        inner = f" {label} {ts} " if ts else f" {label} "
        total_width = 40  # matches the plain divider width used elsewhere
        left = 5
        right = max(0, total_width - left - len(inner))
        return f"[{_accent_hex()}]{'─' * left}{inner}{'─' * right}[/]"

    def _print_user_message_preview(self, user_input: str) -> None:
        """Render a user message using the normal chat scrollback style."""
        from cli import ChatConsole, _accent_hex
        from tools.process_registry_notifications import TimelineNotification
        if isinstance(user_input, TimelineNotification):
            from agent.warning_notifications import render_notification
            render_notification(lambda: ChatConsole().print(f"[dim]◈ {_escape(user_input.display_text)}[/dim]"),
                                platform="cli", diagnostic=user_input.notification_category == "diagnostic")
            return
        ChatConsole().print(self._user_header_line())
        text = str(user_input or "")
        ChatConsole().print(self._format_submitted_user_message_preview(text))

    def _stream_reasoning_delta(self, text: str) -> None:
        """Stream reasoning tokens into a dim box above the response.

        DISABLED 2026-10-19: unified streaming via handle_stream_event now renders
        reasoning directly; this old path was causing double display. Kept as no-op
        for callback registration compatibility.
        """
        if not text:
            return
        # Old rendering commented out - new system in stream_delivery.handle_stream_event

    def _process_dim_box_text(self, text: str) -> None:
        """Buffer dim box text (reasoning/tool args) and print complete lines with word-wrapping."""
        from cli import _DIM, _RST, _cprint
        from wcwidth import wcswidth

        box_width = self._scrollback_box_width()

        def print_one_line(text):
            """Print text that fits in box_width, return the remainder."""
            within_text = False
            width = 0
            last_space_position = 0

            for i, ch in enumerate(text):
                if ch == ' ':
                    width += 1
                    if within_text:
                        last_space_position = i
                elif ch == '\t':
                    width += 8
                else:
                    within_text = True
                    width += wcswidth(ch)

                # Check after adding char - does it exceed box_width?
                if width > box_width:
                    if last_space_position:
                        # Soft break
                        _cprint(f"{_DIM}{text[:last_space_position]}{_RST}")
                        return text[last_space_position+1:]
                    else:
                        # Hard break
                        _cprint(f"{_DIM}{text[:i]}{_RST}")
                        return text[i:]

            # Text fits - print all of it
            _cprint(f"{_DIM}{text}{_RST}")
            return ""

        # Initialize _line_buf if it doesn't exist yet
        if not hasattr(self, "_line_buf"):
            self._line_buf = ""

        # Extract a line from the buffer for processing.
        self._line_buf += text
        if "\n" in self._line_buf:
            # complete line (newline arrived)
            line, self._line_buf = self._line_buf.split("\n", 1)
            if line == "":
                _cprint("")
            else:
                while line:
                    line = print_one_line(line)
        else:
            line = self._line_buf
            width = wcswidth(line)
            if width > box_width:
                # print all but the last partial line
                while width > box_width:
                    line = print_one_line(line)
                    width = wcswidth(line)
                self._line_buf = line

    def _agent_status_print(self, *args, **kwargs) -> None:
        """``agent._print_fn`` for the interactive CLI: agent status lines (subagent completion ``✓ [set n · i/N]``,
        background-process notices, spinner ``print_above`` text) arrive from other threads at any moment. While
        a response or reasoning box is being streamed they are HELD and released at the box footer, so a line
        never lands between two paragraphs of the reply. Outside a box they print immediately."""
        from cli import _cprint
        text = kwargs.get("sep", " ").join(str(a) for a in args)
        if getattr(self, "_stream_box_live", False) or getattr(self, "_reasoning_box_opened", False):
            self._held_status_lines = getattr(self, "_held_status_lines", []) + [text]
            return
        _cprint(text)

    def _release_held_status_lines(self) -> None:
        """Print status lines held while a box was open (called right after a box footer)."""
        from cli import _cprint
        held, self._held_status_lines = getattr(self, "_held_status_lines", []), []
        for line in held:
            _cprint(line)

    def _close_reasoning_box(self) -> None:
        """Close the live reasoning box if it's open (renders the buffered reasoning tail)."""
        from cli import _DIM, _RST, _cprint, flush_inline_output_history
        if not getattr(self, "_reasoning_box_opened", False):
            return
        # Print any remaining text in the line buffer (partial line that didn't get printed)
        # Strip trailing newlines: a steering-forced close right after a token ending in \n\n
        # otherwise flushes a bare newline here, rendering a blank line inside the box.
        line_buf = getattr(self, "_line_buf", "").rstrip("\n")
        if line_buf:
            _cprint(f"{_DIM}{line_buf}{_RST}")
            self._line_buf = ""
        w = self._scrollback_box_width()
        # Print the box footer
        _cprint(f"{_DIM}└{'─' * (w - 2)}┘{_RST}")
        flush_inline_output_history()  # commit inline fragments as finished lines for replay
        self._reasoning_box_opened = False
        if not getattr(self, "_stream_box_live", False):
            self._release_held_status_lines()

    def _stream_delta(self, text) -> None:
        """Line-buffered streaming callback for real-time token rendering.

        DISABLED 2026-10-19: unified streaming via handle_stream_event now renders
        response content directly; this old path was causing double display.
        Kept as no-op for callback registration compatibility.
        
        ``None`` = intermediate turn boundary (tools about to run): flush boxes and reset state.
        """
        if text is None:
            # Flush boxes on turn boundaries (still needed for state management)
            self._flush_stream()
            self._reset_stream_state()
            return
        # Old rendering commented out - new system in stream_delivery.handle_stream_event
        return  # Early return to prevent double rendering

    def _emit_stream_line(self, printed_line: str, wrap: bool = True) -> None:
        """Print one response line with the skin's true-color text escape (if any).

        ``wrap=True`` word-wraps over-wide lines to the box interior so they never run
        past the right border. Table rows pass wrap=False: they are re-aligned as a block
        by _flush_stream_table_buf and must keep their column geometry."""
        from cli import _RST, _STREAM_PAD, _cprint
        _tc = getattr(self, "_stream_text_ansi", "")
        if not wrap:
            lines = [printed_line]
        else:
            lines = _wrap_box_line(printed_line, self._scrollback_box_width())
        for ln in lines:
            _cprint(
                f"{_STREAM_PAD}{_tc}{ln}{_RST}" if _tc else f"{_STREAM_PAD}{ln}")

    def _flush_stream_table_buf(self) -> None:
        """Emit the held table block re-aligned as a whole. Cell-level markdown is stripped FIRST
        so the realigner pads to the final visible width, not the marker-decorated width."""
        from cli import (
            _strip_markdown_syntax, _terminal_width_for_streaming, realign_markdown_tables)
        buf = self._stream_table_buf
        self._stream_table_buf = []
        self._in_stream_table = False
        if not buf:
            return
        joined = "\n".join(buf)
        if self.final_response_markdown == "strip":
            joined = _strip_markdown_syntax(joined)
        block = realign_markdown_tables(joined, _terminal_width_for_streaming())
        for ln in block.split("\n"):
            self._emit_stream_line(ln, wrap=False)

    def _emit_stream_text(self, text: str) -> None:
        """Emit filtered text to the streaming display."""
        from agent.markdown_tables import is_table_divider, looks_like_table_row
        from cli import (
            HermesCLI, _ACCENT, _RST, _STREAM_PARTIAL_PREVIEW_LEN, _cprint, _strip_markdown_syntax, datetime)
        if not text:
            return
        # Close a still-open reasoning box on the first content token so the answer streams
        # token-by-token; _close_reasoning_box renders the reasoning tail first, so ordering holds.
        self._close_reasoning_box()
        self._close_tool_args_box()  # arguments fully generated: close any live args box

        # Open the response box header on the very first visible text
        if not self._stream_box_opened:
            text = text.lstrip("\n")
            if not text:
                return
            self._stream_box_opened = True
            self._stream_box_live = True  # header drawn; cleared at the footer
            try:
                from hermes_cli.skin_engine import get_active_skin
                _skin = get_active_skin()
                label = _skin.get_branding("response_label", "☤ Hermes")
                _text_hex = _skin.get_color("banner_text", "#FFF8DC")
            except Exception:
                label = "☤ Hermes"
                _text_hex = "#FFF8DC"
            try:  # true-color escape so streamed text matches the Rich Panel appearance
                _r, _g, _b = (int(_text_hex[i:i + 2], 16) for i in (1, 3, 5))
                self._stream_text_ansi = f"\033[38;2;{_r};{_g};{_b}m"
            except (ValueError, IndexError):
                self._stream_text_ansi = ""
            if self.show_timestamps:
                label = f"{label}{datetime.now().strftime(getattr(self, 'timestamp_format', '%H:%M'))} "
            w = self._scrollback_box_width()
            fill = w - 2 - HermesCLI._status_bar_display_width(label)
            _cprint(f"\n{_ACCENT}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}")

        # Turn-level record of what actually reached the screen; survives _reset_stream_state at
        # tool-call boundaries so an interrupted reply isn't re-rendered as a Panel (#65666).
        self._streamed_text_this_turn = getattr(self, "_streamed_text_this_turn", "") + text
        self._stream_buf += text
        while "\n" in self._stream_buf:
            line, self._stream_buf = self._stream_buf.split("\n", 1)
            # Table rows are held and re-padded as a block once it ends (already-printed rows
            # can't be re-aligned), so a table appears in one batch when the block closes.
            if self._in_stream_table:
                if looks_like_table_row(line) or is_table_divider(line):
                    self._stream_table_buf.append(line)
                    continue
                self._flush_stream_table_buf()
            elif looks_like_table_row(line):
                self._stream_table_buf.append(line)
                self._in_stream_table = True
                continue
            if self.final_response_markdown == "strip":
                line = _strip_markdown_syntax(line)
            self._emit_stream_line(line)

        # Partial lines are emitted ONLY at real newlines (no hard-wrapping — the terminal
        # soft-wraps, so highlight-copy yields the original text). For TTFT perception, mirror
        # the tail of a long unfinished paragraph into the status-bar spinner.
        if (
            self._stream_buf
            and not self._in_stream_table
            and not self._stream_buf.lstrip().startswith("|")
            and len(self._stream_buf) >= 80):
            preview = self._stream_buf[-int(_STREAM_PARTIAL_PREVIEW_LEN):]
            cut = preview.find(" ")
            if 0 < cut < len(preview) - 1:
                preview = preview[cut + 1:]
            try:
                self._spinner_text = f"… {preview}"
                self._invalidate()
            except Exception:
                pass

    def _flush_stream(self) -> None:
        """Emit any remaining partial line from the stream buffer and close the box."""
        from agent.markdown_tables import is_table_divider, looks_like_table_row
        from cli import _ACCENT, _RST, _cprint, _strip_markdown_syntax
        # Still inside a "reasoning block" at end-of-stream = false positive (the model
        # mentioned a tag in prose and never closed it): recover the buffer as regular text.
        if getattr(self, "_in_reasoning_block", False) and getattr(self, "_stream_prefilt", ""):
            self._in_reasoning_block = False
            self._emit_stream_text(self._stream_prefilt)
            self._stream_prefilt = ""
        self._close_reasoning_box()  # in case no content tokens arrived
        self._close_tool_args_box()  # end-of-stream: close any live args box
        # A trailing partial table row joins the table buffer so the whole block is re-aligned
        # together (else the final row prints under-padded).
        if (
            self._stream_buf
            and getattr(self, "_in_stream_table", False)
            and (looks_like_table_row(self._stream_buf) or is_table_divider(self._stream_buf))):
            self._stream_table_buf.append(self._stream_buf)
            self._stream_buf = ""
        if getattr(self, "_stream_table_buf", None):
            self._flush_stream_table_buf()
        if self._stream_buf:
            line = _strip_markdown_syntax(self._stream_buf) if self.final_response_markdown == "strip" else self._stream_buf
            self._emit_stream_line(line)
            self._stream_buf = ""
        if self._stream_box_opened and getattr(self, "_stream_box_live", False):
            w = self._scrollback_box_width()
            _cprint(f"{_ACCENT}╰{'─' * (w - 2)}╯{_RST}")
        self._stream_box_live = False
        self._release_held_status_lines()

    def _reset_stream_state(self) -> None:
        """Reset streaming state before each agent invocation."""
        self._stream_buf = ""
        self._stream_started = False
        self._stream_box_opened = False
        self._stream_text_ansi = ""
        self._stream_prefilt = ""
        self._in_reasoning_block = False
        self._stream_last_was_newline = True
        self._reasoning_box_opened = False
        self._reasoning_preview_buf = ""
        self._tool_args_box_opened = False
        self._tool_args_box_tool = None
        # A batch cancelled/errored before any tool.started would otherwise mute the next turn's line.
        self.__dict__.pop("_tool_gen_announced", None)
        self._stream_table_buf = []
        self._in_stream_table = False
        self._stream_box_live = False

    def _slow_command_status(self, command: str) -> str:
        """Return a user-facing status message for slower slash commands."""
        cmd_lower = command.lower().strip()
        exact = _SLOW_COMMAND_STATUS_EXACT.get(cmd_lower)
        if exact:
            return t(exact)
        for prefix, key in _SLOW_COMMAND_STATUS:
            if cmd_lower.startswith(prefix):
                return t(key)
        return t("cli.stream.processing_command")

    def _command_spinner_frame(self) -> str:
        """Return the current spinner frame for slow slash commands."""
        from cli import _COMMAND_SPINNER_FRAMES
        return _COMMAND_SPINNER_FRAMES[int(time.monotonic() * 10) % len(_COMMAND_SPINNER_FRAMES)]

    @contextmanager
    def _busy_command(self, status: str, *, blocks_input: bool = True):
        """Expose a temporary busy state in the TUI while a slash command runs.

        Most sync slash commands reserve the composer (their completion changes session state);
        manual compression is safe to draft through (queued input runs against compacted history).
        """
        from cli import _cprint
        previous_blocks_input = getattr(self, "_command_blocks_input", False)
        self._command_running = True
        self._command_blocks_input = blocks_input
        self._command_status = status
        self._invalidate(min_interval=0.0)
        try:
            _cprint(f"⏳ {status}")
            yield
        finally:
            self._command_running = False
            self._command_blocks_input = previous_blocks_input
            self._command_status = ""
            self._invalidate(min_interval=0.0)

    def _preprocess_images_with_vision(self, text: str, images: list, *, announce: bool = True) -> str:
        """Describe attached images via the auxiliary vision model and prepend the descriptions
        to the user's text (works with non-vision models; same approach as the gateway). The
        local path is included so the agent can re-examine via ``vision_analyze``."""
        from cli import _DIM, _RST, _cprint
        import asyncio as _asyncio
        from agent.warning_notifications import render_notification
        from tools.vision_tools import vision_analyze_tool
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information.")
        enriched_parts = []
        for img_path in images:
            if not img_path.exists():
                continue
            size_kb = img_path.stat().st_size // 1024
            if announce:
                _cprint(f"  {_DIM}{t('cli.stream.vision_analyzing', name=img_path.name, size_kb=size_kb)}{_RST}")
            try:
                result_json = _asyncio.run(
                    vision_analyze_tool(image_url=str(img_path), user_prompt=analysis_prompt))
                result = json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    enriched_parts.append(
                        f"[The user attached an image. Here's what it contains:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {img_path}]")
                    if announce:
                        _cprint(f"  {_DIM}{t('cli.stream.vision_analyzed')}{_RST}")
                else:
                    enriched_parts.append(
                        f"[The user attached an image but it couldn't be analyzed. "
                        f"You can try examining it with vision_analyze using "
                        f"image_url: {img_path}]")
                    if announce:
                        render_notification(lambda: _cprint(f"  {_DIM}{t('cli.stream.vision_failed')}{_RST}"),
                                            platform="cli", user_config=getattr(getattr(self, "agent", None), "_notification_config", None))
            except Exception as e:
                enriched_parts.append(
                    f"[The user attached an image but analysis failed ({e}). "
                    f"You can try examining it with vision_analyze using "
                    f"image_url: {img_path}]")
                if announce:
                    render_notification(lambda: _cprint(f"  {_DIM}{t('cli.stream.vision_error')}{_RST}"),
                                        platform="cli", user_config=getattr(getattr(self, "agent", None), "_notification_config", None))

        # Vision descriptions first, then the user's original text
        user_text = text if isinstance(text, str) and text else ""
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            return f"{prefix}\n\n{user_text}" if user_text else prefix
        return user_text or "What do you see in this image?"

    def _console_print(self, *args, **kwargs):
        """Print through the active command-safe console (prompt_toolkit-safe Rich once the
        TUI is live, else the plain console)."""
        from cli import ChatConsole
        console = ChatConsole() if getattr(self, "_app", None) else self.console
        console.print(*args, **kwargs)

    def _on_tool_gen_start(self, tool_name: str) -> None:
        """Model began generating tool-call arguments: close open boxes once, then print a status
        line so a large payload (e.g. 45 KB write_file) doesn't look like a frozen screen.

        Fires once per tool CALL, so a batch of parallel calls to the same tool printed the same
        line N times (#10478); repeats within one generation batch are coalesced. The set is
        cleared when a tool actually starts (``tool.started``), i.e. on the next batch."""
        if getattr(self, '_stream_box_opened', False):
            self._flush_stream()
            self._stream_box_opened = False
        self._close_reasoning_box()
        self._close_tool_args_box()
        # Old "preparing" status line REMOVED 2026-10-20: the agent-owns-display box engine
        # (stream_delivery._fire_tool_args_stream → print_header_line TOOL_CALL) now renders
        # the per-tool header itself ("🔍 using {tool}"), so printing it here doubled the line.
        # Kept as no-op for callback registration + the box state management above.
        return
        # announced = self.__dict__.setdefault("_tool_gen_announced", set())
        # if tool_name in announced:
        #     return
        # announced.add(tool_name)
        # from agent.display import bridge_generating_phrase, get_tool_emoji
        # what = bridge_generating_phrase(tool_name) or tool_name
        # _cprint(f"  ┊ {t('cli.stream.tool_preparing', emoji=get_tool_emoji(tool_name, default='⚡'), what=what)}")

    def _on_tool_args_stream(self, tool_name: str, text_chunk: str) -> None:
        """Stream raw tool-call argument fragments into a dim box (reasoning-box style).

        DISABLED 2026-10-19: unified streaming via handle_stream_event now renders
        tool args directly; this old path was causing double display. Kept as no-op
        for callback registration compatibility.
        
        The JSON being generated is shown as it arrives — meaningful for big payloads
        (write_file content, long queries), honest noise otherwise. Box opens on the first
        chunk, re-labels if a different call streams next, closes when the tool actually
        starts or content/stream boundaries hit. Text rendering is shared with the
        reasoning box via _process_dim_box_text.
        """
        if not text_chunk:
            return
        # Old rendering commented out - new system in stream_delivery.handle_stream_event

    def _close_tool_args_box(self) -> None:
        """Close the live tool-args box if open (renders the buffered argument tail)."""
        from cli import _DIM, _RST, _cprint
        if not getattr(self, "_tool_args_box_opened", False):
            return
        # Print any remaining text in the line buffer (partial line that didn't get printed)
        line_buf = getattr(self, "_line_buf", "")
        if line_buf:
            _cprint(f"{_DIM}{line_buf}{_RST}")
            self._line_buf = ""
        w = self._scrollback_box_width()
        _cprint(f"{_DIM}└{'─' * (w - 2)}┘{_RST}")
        self._tool_args_box_opened = False
        self._tool_args_box_tool = None

    def _on_tool_progress(self, event_type: str, function_name: str = None, preview: str = None, function_args: dict = None, **kwargs):
        """Tool lifecycle events (tool.started / tool.completed / reasoning.* / moa.*).

        Drives the TUI spinner (tool.started stamps the elapsed timer); in "all"/"new"/"verbose"
        progress modes tool.completed also commits a stacked scrollback line (tool history).
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint, _hermes_home
        # MoA reference outputs (display-only events from the MoA facade): render each answer
        # as a labelled thinking-style block BEFORE the aggregator acts.
        if event_type == "moa.reference":
            label = function_name or t("cli.stream.reference_noun")
            text = preview or ""
            idx = kwargs.get("moa_index")
            count = kwargs.get("moa_count")
            header = (t("cli.stream.reference_header_indexed", index=idx, count=count, label=label) if idx and count
                      else t("cli.stream.reference_header", label=label))
            try:
                self._flush_reasoning_preview(force=True)
            except Exception:
                pass
            _cprint(f"  {_DIM}┊ ◇ {header}{_RST}")
            try:
                self._emit_reasoning_preview(text)
            except Exception:
                if text.strip():
                    _cprint(f"  {_DIM}{text.strip()}{_RST}")
            self._invalidate()
            return
        if event_type == "moa.aggregating":
            agg = function_name or ""
            self._spinner_text = t("cli.stream.aggregating_detail", detail=agg) if agg else t("cli.stream.aggregating")
            self._invalidate()
            return

        # Feed the pet: tools mean "running"; a failed tool latches the turn to end on a sulk.
        if event_type == "tool.started":
            self._pet_reasoning = False
            self.__dict__.pop("_tool_gen_announced", None)
            self._close_tool_args_box()  # args done streaming: tool is running now
        elif event_type == "tool.completed" and kwargs.get("is_error"):
            self._pet_turn_error = True
        elif event_type and event_type.startswith("reasoning"):
            self._pet_reasoning = True

        if event_type == "tool.completed":
            self._tool_start_time = 0.0
            self._turn_summary_record(
                function_name, kwargs.get("result"), kwargs.get("is_error", False))
            # Focus view: count the hidden scrollback line for the post-turn recovery report.
            if getattr(self, "_focus_view_enabled", False):
                try:
                    self._note_focus_hidden_line(function_name or "")
                except Exception:
                    pass
            # "verbose" must commit the same line as "all": non-streaming calls (MoA aggregator,
            # copilot-acp) never emit the "preparing" line, so nothing else builds history.
            if function_name and self.tool_progress_mode in {"new", "all", "verbose"}:
                # Pop stored args from tool.started (kept alive: bookkeeping still feeds turn summaries).
                stored = self._pending_tool_info.get(function_name)
                if stored:
                    stored.pop(0)
                if stored is not None and not stored:
                    del self._pending_tool_info[function_name]
                # "new" mode: skip consecutive repeats of the same tool
                if self.tool_progress_mode == "new" and function_name == self._last_scrollback_tool:
                    self._invalidate()
                    return
                self._last_scrollback_tool = function_name
                # One-time /verbose hint on the first long tool in the noisiest mode; latched
                # on self and persisted to config.yaml.
                try:
                    if (
                        not getattr(self, "_long_tool_hint_fired", False)
                        and self.tool_progress_mode == "all"
                        and kwargs.get("duration", 0.0) >= 30.0):
                        from agent.onboarding import (
                            TOOL_PROGRESS_FLAG, is_seen, mark_seen, tool_progress_hint_cli)
                        if not is_seen(CLI_CONFIG, TOOL_PROGRESS_FLAG):
                            self._long_tool_hint_fired = True
                            _cprint(f"  {_DIM}{tool_progress_hint_cli()}{_RST}")
                            mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
                            CLI_CONFIG.setdefault("onboarding", {}).setdefault("seen", {})[TOOL_PROGRESS_FLAG] = True
                except Exception:
                    pass
            self._invalidate()
            return
        if event_type != "tool.started":
            return
        if function_name and not function_name.startswith("_"):
            from agent.display import get_tool_preview_max_len, tool_row_emoji
            label = preview or function_name
            _pl = get_tool_preview_max_len()
            if _pl > 0 and len(label) > _pl:
                label = label[:_pl - 3] + "..."
            self._spinner_text = f"{tool_row_emoji(function_name, function_args)} {label}"
            self._tool_start_time = time.monotonic()
            # Store args for stacked scrollback line on completion
            self._pending_tool_info.setdefault(function_name, []).append(
                function_args if function_args is not None else {})
            self._invalidate()

    def _on_tool_start(self, tool_call_id: str, function_name: str, function_args: dict):
        """Capture local before-state for write-capable tools."""
        from cli import logger
        try:
            from agent.display import capture_local_edit_snapshot
            snapshot = capture_local_edit_snapshot(function_name, function_args)
            if snapshot is not None:
                self._pending_edit_snapshots[tool_call_id] = snapshot
        except Exception:
            logger.debug("Edit snapshot capture failed for %s", function_name, exc_info=True)

    def _on_tool_complete(self, tool_call_id: str, function_name: str, function_args: dict, function_result: str):
        """Render file edits with inline diff after write-capable tools complete."""
        from cli import _cprint, logger

        # Route the review diff through our DIFF LineKind box (agent-owns-display).
        # _emit_inline_diff sends the locale header line first, then each diff line;
        # we replace that first header with a styled DIFF header and box the rest.
        diff_box = {"header_drawn": False}

        def diff_print_fn(line: str) -> None:
            try:
                from agent.conversation_loop import ACTIVE_CLI, LineKind, print_header_line, print_text_line
                if ACTIVE_CLI is not None:
                    if not diff_box["header_drawn"]:
                        diff_box["header_drawn"] = True
                        print_header_line(LineKind.DIFF)  # draws "Diff" rule + stamps header_kind
                        return
                    print_text_line(LineKind.DIFF, line)
                    return
            except Exception:
                pass
            _cprint(line)

        # A background delegate_task re-enters as a fresh turn when done; say so once so the
        # idle prompt doesn't read as "nothing happened".
        if function_name == "delegate_task":
            try:
                parsed = json.loads(function_result) if isinstance(function_result, str) else (function_result or {})
            except Exception:
                parsed = {}
            if isinstance(parsed, dict) and parsed.get("status") == "dispatched" and parsed.get("mode") == "background":
                n = parsed.get("count") or 1
                key = "cli.stream.background_running_one" if n == 1 else "cli.stream.background_running_other"
                try:
                    _cprint(f"\033[2m{t(key, count=n)}\033[0m")
                except Exception:
                    pass
        snapshot = self._pending_edit_snapshots.pop(tool_call_id, None)
        try:
            from agent.display import render_edit_diff_with_delta
            render_edit_diff_with_delta(
                function_name, function_result, function_args=function_args, snapshot=snapshot,
                print_fn=diff_print_fn)
        except Exception:
            logger.debug("Edit diff preview failed for %s", function_name, exc_info=True)
