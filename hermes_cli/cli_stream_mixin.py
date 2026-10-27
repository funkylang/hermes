"""Streaming output, reasoning preview, tool progress callbacks, and busy-command spinner for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).
"""

from __future__ import annotations

import json
import re
import shutil
import textwrap
import time

from contextlib import contextmanager
from pathlib import Path
from rich.markup import escape as _escape

from agent.i18n import t

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


    def _agent_status_print(self, *args, **kwargs) -> None:
        """``agent._print_fn`` for the interactive CLI: agent status lines (subagent completion ``✓ [set n · i/N]``,
        background-process notices, spinner ``print_above`` text) arrive from other threads at any moment."""
        from cli import _cprint
        text = kwargs.get("sep", " ").join(str(a) for a in args)
        _cprint(text)







    def _stream_delta(self, text) -> None:
        """Stream consumer marker for the provider loops (``_has_stream_consumers`` gates
        ``on_text_delta`` wiring). Display itself flows through
        ``stream_delivery.handle_stream_event`` (the print_* funnel); this method renders nothing.
        """

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
