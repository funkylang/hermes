"""System-prompt assembly for :class:`AIAgent`.

Built once per session and reused across turns (only context compression
triggers a rebuild) so the upstream prefix cache stays warm.  Three tiers are
joined with ``\\n\\n``: ``stable`` (identity, guidance, env hints, coding brief,
platform hints), ``context`` (workspace snapshot, caller ``system_message``,
context files) and ``volatile`` (skills index, memory, USER.md, external memory
provider, timestamp line).  See ``references/system-prompt-invariant.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.delegation_context import owned_kanban_task
from agent.prompt_builder import (
    ASYNC_HANDOFF_GUIDANCE, DEFAULT_AGENT_IDENTITY, EXECUTION_GUIDANCE_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE, HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS, KANBAN_GUIDANCE,
    PARALLEL_TOOL_CALL_GUIDANCE, PLATFORM_HINTS, SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE, STEER_CHANNEL_NOTE, TASK_COMPLETION_GUIDANCE, TELEGRAM_RICH_MESSAGES_HINT,
    TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS, drain_truncation_warnings,
)
from agent import prompt_builder as _pb
from agent.path_display import display_path
from agent.prompt_manifest import utf8_bytes
from agent.prompt_provenance import (
    HARDCODED_CONSTANTS,
    context_file_labels_for_block,
    memories_file_block,
    skills_index_sources,
)
from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd
from hermes_constants import get_default_hermes_root, get_hermes_home
from utils import is_truthy_value

logger = logging.getLogger(__name__)
_PLUGIN_SECTION_FRAME_RE = re.compile(
    r"^## Plugin Context: (?P<id>[a-z0-9][a-z0-9._-]{0,127})\n<!-- hermes-plugin-section-chars:(?P<chars>[0-9]{1,4}) -->\n\n",
    re.MULTILINE,
)
_GATE_WORDS = {**dict.fromkeys(("true", "always", "yes", "on"), True), **dict.fromkeys(("false", "never", "no", "off"), False)}


def _model_gate(setting: Any, model: Optional[str], default_models) -> bool:
    """Resolve a config gate: True/"true"-ish -> on, False/"false"-ish -> off,
    list -> case-insensitive model-substring match, anything else ("auto") ->
    match against *default_models*."""
    if setting is True or setting is False:
        return setting
    if isinstance(setting, str) and setting.lower() in _GATE_WORDS:
        return _GATE_WORDS[setting.lower()]
    model_lower = (model or "").lower()
    if isinstance(setting, list):
        return any(p.lower() in model_lower for p in setting if isinstance(p, str))
    return any(p in model_lower for p in default_models)


def _resolve_platform_hint(agent: Any, platform_key: str, default_hint: str) -> str:
    """Apply the ``platform_hints.<platform>`` config override: ``replace``
    substitutes the default, ``append`` adds text (a bare string is shorthand
    for append). Malformed entries fall back to the unmodified default so bad
    config can never break prompt assembly or leak across platforms."""
    overrides = getattr(agent, "_platform_hint_overrides", None)
    spec = overrides.get(platform_key) if platform_key and isinstance(overrides, dict) else None
    if isinstance(spec, str):
        spec = {"append": spec}
    if not isinstance(spec, dict):
        return default_hint
    replace_text, append_text = (v.strip() if isinstance(v, str) else "" for v in (spec.get("replace"), spec.get("append")))
    base = replace_text or default_hint
    return f"{base}\n\n{append_text}".strip() if append_text else base


_TUI_EMBEDDED_PANE_CLARIFIER = (
    " You're in its embedded terminal pane, beside the GUI chat — the user can "
    "select your output (Option-drag on macOS, Shift-drag elsewhere) and press "
    "Cmd/Ctrl+L to send it to the chat composer."
)


def _tui_embedded_pane_clarifier(hint: str) -> str:
    """Append the desktop embedded-terminal clarifier when ``HERMES_DESKTOP_TERMINAL``
    is set (only the desktop's TUI PTY, never the chat backend). Idempotent."""
    if not hint or _TUI_EMBEDDED_PANE_CLARIFIER in hint or not is_truthy_value(os.getenv("HERMES_DESKTOP_TERMINAL")):
        return hint
    return hint + _TUI_EMBEDDED_PANE_CLARIFIER


def _plugin_session_info(agent: Any) -> Dict[str, str]:
    """Return immutable-at-render-time metadata exposed to prompt sections."""
    try:
        cwd = str(resolve_context_cwd() or "")
    except Exception:
        cwd = ""
    info = {k: str(getattr(agent, k, None) or "") for k in ("session_id", "model", "provider", "platform")}
    info.update(profile_name=_active_profile_name(agent, _ambient_plugin_profile_name), cwd=cwd)
    return info


def _ambient_plugin_profile_name() -> str:
    from hermes_cli.profiles import get_active_profile_name
    return str(get_active_profile_name() or "default")


def _active_profile_name(agent: Any, ambient) -> str:
    """Profile name from the agent's OWN home, else *ambient()*; "default" on any
    failure. Ambient resolution misreports on threads that lost the HERMES_HOME
    ContextVar, which is why the agent's home is preferred."""
    try:
        home = _agent_home(agent)
        return _profile_name_for_home(home) if home is not None else ambient()
    except Exception:
        return "default"


def _frozen_plugin_prompt_sections(agent: Any) -> tuple:
    """Render plugin sections once per session and freeze them on the agent.
    A restored ``_cached_system_prompt`` is parsed instead of re-running plugin
    code; a render that raises at a rebuild boundary keeps the previous bytes
    (stashed by ``invalidate_system_prompt``) instead of silently vanishing."""
    if hasattr(agent, "_plugin_system_prompt_sections_snapshot"):
        return agent._plugin_system_prompt_sections_snapshot
    stored_prompt = getattr(agent, "_cached_system_prompt", None)
    if isinstance(stored_prompt, str) and stored_prompt:
        rendered = _restore_plugin_prompt_sections(stored_prompt)
    else:
        try:
            from hermes_cli.plugins import render_system_prompt_sections
            rendered = tuple(render_system_prompt_sections(_plugin_session_info(agent)))
        except Exception as exc:
            rendered = getattr(agent, "_plugin_system_prompt_sections_previous", None)
            if rendered:
                logger.warning("Plugin system prompt sections failed to re-render (%s); keeping the previous frozen sections", exc)
            else:
                logger.warning("Plugin system prompt sections could not be rendered: %s", exc)
                rendered = ()
    agent._plugin_system_prompt_sections_snapshot = rendered
    return rendered


def _restore_plugin_prompt_sections(prompt: str) -> tuple:
    """Recover frozen section bytes from the persisted full prompt.  Only the
    exact canonical container emitted by core is accepted — user/project text
    may resemble a frame."""
    from hermes_cli.plugins import (
        MAX_SYSTEM_PROMPT_SECTION_CHARS, PLUGIN_SECTIONS_END, PLUGIN_SECTIONS_START,
        RenderedPluginSystemPromptSection, format_system_prompt_sections,
    )
    start = prompt.rfind(PLUGIN_SECTIONS_START)
    end = prompt.find(PLUGIN_SECTIONS_END, start + len(PLUGIN_SECTIONS_START)) if start >= 0 else -1
    if end < 0:
        return ()
    after_end = end + len(PLUGIN_SECTIONS_END)
    if not prompt[after_end:].startswith("\n\nConversation started:"):
        return ()
    framed = prompt[start:after_end]
    restored = []
    for match in _PLUGIN_SECTION_FRAME_RE.finditer(framed):
        content_len = int(match.group("chars"))
        content = framed[match.end() : match.end() + content_len]
        if content_len > MAX_SYSTEM_PROMPT_SECTION_CHARS or len(content) != content_len:
            continue
        restored.append(RenderedPluginSystemPromptSection(id=match.group("id"), content=content,
                                                          position="after_memory", plugin="persisted-prompt"))
    return tuple(restored) if format_system_prompt_sections(restored) == framed else ()


def restore_plugin_prompt_sections(agent: Any, prompt: str) -> None:
    """Seed a resumed agent's frozen snapshot from persisted prompt bytes."""
    agent._plugin_system_prompt_sections_snapshot = _restore_plugin_prompt_sections(prompt)


def _plugin_section_blocks(sections: tuple, position: str) -> List[str]:
    from hermes_cli.plugins import format_system_prompt_sections
    block = format_system_prompt_sections([s for s in sections if s.position == position])
    return [block] if block else []


def _session_start_like(agent: Any, now: Any) -> Any:
    """Best-known conversation start time, or ``now`` as a fallback.
    ``Conversation started:`` must be byte-stable across rebuilds (compression,
    resume, fresh gateway turns), so prefer immutable sources in order: the
    lineage-root session id's embedded stamp (compaction rotates ids, each with
    its own mint time), the current session id's stamp, ``agent.session_start``,
    then ``now``.  Stamps are box-local wall-clock: attach that zone first, then
    convert to ``now``'s zone so the date matches the per-turn clock.

    0. the LINEAGE-ROOT session id's embedded timestamp — compaction can rotate the session id, and each
    rotated id embeds its OWN mint time, so after months of compactions rung 1 alone would quietly re-birth
    the conversation at its latest rotation. Walking to the lineage root (same walk as
    ``_conversation_root_id``) recovers the ORIGINAL birth stamp — a Bot Mode forever-chat keeps knowing
    when it was first born, across every compaction (maintainer-directed, #98426); 1. the timestamp embedded
    in ``session_id`` (``YYYYMMDD_HHMMSS_...``) — immutable for the life of the session, so the line is
    byte-stable across every rebuild boundary (preserving prefix-cache KV); 2. 3. ``now`` (initial/legacy
    build without either).
    """
    from datetime import datetime
    def _to_display_tz(dt: Any) -> Any:
        if dt.tzinfo is None:
            try:
                # The offset in force at dt, not today's: a stamp from the other DST half differs by an hour.
                dt = dt.astimezone()
            except (ValueError, OSError):
                pass
        if getattr(now, "tzinfo", None) is not None and dt.tzinfo is not None:
            try:
                dt = dt.astimezone(now.tzinfo)
            except (ValueError, OSError):
                pass
        return dt
    session_id = getattr(agent, "session_id", None)
    db = getattr(agent, "_session_db", None)
    try:
        root_id = db.get_conversation_root(session_id) if db is not None and isinstance(session_id, str) and session_id else None
    except Exception:
        root_id = None
    for candidate in (root_id, session_id):
        m = re.match(r"^(\d{8})_(\d{6})", candidate) if isinstance(candidate, str) else None
        if m:
            try:
                return _to_display_tz(datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S"))
            except ValueError:
                pass
    session_start = getattr(agent, "session_start", None)
    return _to_display_tz(session_start) if hasattr(session_start, "astimezone") else now


def _agent_home(agent: Any) -> Optional[Path]:
    """The agent's OWN profile home, or None to use ambient resolution.
    A bound HERMES_HOME ContextVar override wins (the gateway multiplexes
    profiles over one shared session DB and binds the home per turn); else the
    parent of ``_session_db.db_path`` — ground truth on threads that lost the
    ContextVar, where ambient resolution would leak the launch profile.

    1. Surfaces that multiplex several profiles over ONE shared session DB (the messaging gateway:
    ``gateway/run.py`` hands every agent the launch-home ``state.db`` and binds the profile home per turn
    via ``_profile_runtime_scope`` + ``copy_context``) would otherwise have the db-derived launch home STOMP
    the correctly-bound profile — inverting the leak this helper exists to fix (found by @kshitijk4poor's
    post-merge probe on #86313). 2. Fallback: the home containing the agent's ``_session_db.db_path``
    (``<home>/state.db``) — ground truth on threads that lost the ContextVar (ContextVars don't propagate
    into ``threading.Thread``), where the unbound build previously fell back to the launch home and leaked
    the default profile's skills/identity into a bot prompt.
    """
    try:
        from hermes_constants import get_hermes_home_override
        override = get_hermes_home_override()
        if override:
            return Path(override)
    except Exception:
        pass
    try:
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        return Path(db_path).parent if db_path else None
    except Exception:
        return None


def _agent_skills_dir(agent: Any) -> Optional[Path]:
    """The agent's own ``<home>/skills`` dir, or None to use ambient home."""
    home = _agent_home(agent)
    return home / "skills" if home is not None else None


def _profile_name_for_home(home: Path) -> str:
    """``<root>/profiles/X`` -> ``"X"``; anything else -> ``"default"``.
    Uses ``get_default_hermes_root()`` (NOT ``get_hermes_home()``): on a bound
    profile session the ambient home IS the profile dir, so every profile
    would misreport as "default"."""
    try:
        from hermes_constants import get_default_hermes_root
        rel = home.resolve().relative_to((get_default_hermes_root() / "profiles").resolve())
        return rel.parts[0] if rel.parts else "default"
    except (ValueError, OSError):
        return "default"


def _tool_guidance_block(agent: Any) -> Optional[str]:
    """Tool-aware behavioral guidance, injected only when the tools are loaded."""
    names = agent.valid_tool_names
    # With both memory stores disabled no store is built, so the full guidance
    # would steer the model at a tool that always answers "Memory is not
    # available"; with only USER.md enabled the narrower block is used.
    memory_guidance = None
    if "memory" in names:
        memory_guidance = _pb.build_memory_guidance(
            getattr(agent, "_memory_enabled", True),
            getattr(agent, "_user_profile_enabled", True),
            skill_manage_available="skill_manage" in names,
        )
    # Kanban lifecycle: resolved once at __init__ (_kanban_worker_guidance);
    # fallback paths must also limit task protocol guidance to dispatcher workers.
    _kanban_guidance = getattr(agent, "_kanban_worker_guidance", None)
    if _kanban_guidance is None and "kanban_show" in names and owned_kanban_task():
        _kanban_guidance = KANBAN_GUIDANCE
    tool_guidance = [
        memory_guidance,
        SESSION_SEARCH_GUIDANCE if "session_search" in names else None,
        SKILLS_GUIDANCE if "skill_manage" in names else None,
        _kanban_guidance,
    ]
    return " ".join(g for g in tool_guidance if g) or None


def _skills_prompt(agent: Any) -> str:
    """Skills index (empty without skills tools).  Focus mode demotes non-coding
    categories to names-only — never hidden, every name stays visible."""
    if not any(name in agent.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage']):
        return ""
    import model_tools
    avail_toolsets = {model_tools.get_toolset_for_tool(tool_name) for tool_name in agent.valid_tool_names} - {None, ""}
    try:
        from agent.coding_context import coding_compact_skill_categories
        _compact_cats = coding_compact_skill_categories(platform=agent.platform, cwd=resolve_context_cwd())
    except Exception:
        _compact_cats = frozenset()
    return _pb.build_skills_system_prompt(available_tools=agent.valid_tool_names, available_toolsets=avail_toolsets,
                                         compact_categories=_compact_cats or None, skills_dir_override=_agent_skills_dir(agent))


def _auto_load_parts(agent: Any) -> List[Tuple[str, str]]:
    """``skills.auto_load`` text as ``(text, source_label)`` — one pair for the
    whole join.

    Resolved once per agent lifecycle (config, skill files and
    HERMES_IGNORE_RULES read on the first build only) so the prompt stays
    byte-stable across model switches, compression and static-prefix
    restoration. The CLI may have pre-resolved the same result
    (``cli._auto_load_skills_result`` copied onto the agent); both paths use
    ``build_auto_load_prompt``.

    The label names the loaded skills — no text re-parsing (skill blocks
    contain blank lines, so the join cannot be undone safely).

    Same gate as ``_skills_prompt``: nothing without the skills toolset, and
    nothing for agents that skip context files (delegate children, curator /
    review forks, gateway hygiene agents) — pinned skills are operator
    guidance for the user's session, not payload for every internal fork."""
    if getattr(agent, "skip_context_files", False) or not any(
            name in agent.valid_tool_names for name in ("skills_list", "skill_view", "skill_manage")):
        return []
    if not getattr(agent, "_auto_load_skills_resolved", False):
        result = ("", [], [])
        try:
            if not is_truthy_value(os.environ.get("HERMES_IGNORE_RULES")):
                from agent.skill_commands import build_auto_load_prompt
                result = build_auto_load_prompt(task_id=getattr(agent, "session_id", None), home_override=_agent_home(agent))
            if result[2]:
                logger.warning("skills.auto_load: skill(s) not found or disabled, skipped: %s", ", ".join(result[2]))
        except Exception:
            logger.debug("skills.auto_load: injection skipped", exc_info=True)  # config errors never block session start
        agent._auto_load_skills_result = result
        agent._auto_load_skills_resolved = True
    prompt = agent._auto_load_skills_result[0]
    if not prompt:
        return []
    # One part, one label — no text re-parsing (blocks contain blank lines, so
    # the join cannot be undone safely). The name list answers "which skills
    # are auto-loaded"; per-block splits would be metadata-only and are
    # deferred until they matter.
    loaded_names = list(agent._auto_load_skills_result[1] or [])
    names = ", ".join(loaded_names)
    label = f"skills.auto_load config ({names})" if names else "skills.auto_load config"
    return [(prompt, label)]


def _skills_index_sources(agent: Any, index_text: str) -> List[Tuple[str, int]]:
    """Per-skill provenance for the rendered skills index."""
    return skills_index_sources(index_text)


def _bot_mode_parts(agent: Any) -> List[str]:
    """Bot Mode teammate protocol — only in a bot's canonical "Bot Chat" session.
    Marks the prompt timeless (the volatile date line is dropped) since a birth
    date pinned in a months-long session is misinformation."""
    parts: List[str] = []
    try:
        from tools.bot_mode_probe import BOT_CHAT_TITLE, epoch_line, get_bot_mode_protocol_section
        _title = str(getattr(agent, "_session_title_hint", "") or "").strip()
        if not _title:
            _sdb = getattr(agent, "_session_db", None)
            _sid = getattr(agent, "session_id", None)
            _title = str((_sdb.get_session_title(_sid) if (_sdb and _sid) else None) or "").strip()
        _bot_section = get_bot_mode_protocol_section(_agent_home(agent)) if _title == BOT_CHAT_TITLE else None
        if _bot_section:
            parts.append(_bot_section)
            # Capability epoch lets the restore path rebuild ONCE per
            # user-initiated capability change in an eternal session.
            parts.append(epoch_line(_agent_home(agent)))
            agent._bot_chat_timeless_prompt = True
    except Exception:
        pass
    return parts


def _ambient_file_safety_profile_name() -> str:
    from agent.file_safety import _resolve_active_profile_name
    return _resolve_active_profile_name()


def _active_profile_line(agent: Any) -> str:
    """Name the running profile so the agent doesn't conflate ``~/.hermes/skills``
    (default) with ``~/.hermes/profiles/<active>/skills``.  Resolved from the
    agent's OWN home first (a build thread that lost the ContextVar would
    otherwise print "default" for a bot profile)."""
    _agent_home_path = _agent_home(agent)
    active_profile = _active_profile_name(agent, _ambient_file_safety_profile_name)
    if active_profile == "default":
        # With an explicit agent home, the default profile's data lives at the
        # ROOT (get_hermes_home() on a bound profile session is the PROFILE dir).
        # Without one, keep the ambient (patchable) resolution byte-identical.
        _root_str = str(get_default_hermes_root() if _agent_home_path is not None else get_hermes_home())
        return (
            "Active Hermes profile: default. Other profiles (if any) live "
            "under " + _root_str + "/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )
    # A non-default name is only returned when the resolved home is ALREADY
    # <root>/profiles/<name>, so the profile home is the session home itself.
    profile_home = str(_agent_home_path) if _agent_home_path is not None else str(get_hermes_home())
    # A non-default name is only ever returned when the resolved home is ALREADY <root>/profiles/<name> —
    # that is exactly how both _profile_name_for_home() and _resolve_active_profile_name() derive it. So the
    # profile home is the session home itself; appending /profiles/<name> again doubled it (#72894). The
    # default profile's data sits at the ROOT (get_default_hermes_root()), which in ambient profile mode is
    # NOT get_hermes_home().
    default_root = get_default_hermes_root()
    return (
        f"Active Hermes profile: {active_profile}. This session reads "
        f"and writes {profile_home}/. The default "
        f"profile's data lives at {default_root}/skills/, {default_root}/plugins/, "
        f"{default_root}/cron/, {default_root}/memories/ — those belong to a "
        f"different session run from a different shell. Do NOT modify "
        f"another profile's skills/plugins/cron/memories unless the user "
        f"explicitly directs you to."
    )


def _default_platform_hint(platform_key: str) -> str:
    """Built-in hint, else the plugin adapter's ``platform_hint``, else ``""``."""
    hint = PLATFORM_HINTS.get(platform_key, "")
    if not hint and platform_key:
        try:
            from gateway.platform_registry import platform_registry
            _entry = platform_registry.get(platform_key)
            hint = (_entry and _entry.platform_hint) or ""
        except Exception:
            pass
    if platform_key == "telegram" and hint and _telegram_rich_messages_enabled():
        hint = hint.rstrip() + " " + TELEGRAM_RICH_MESSAGES_HINT
    return hint


def _cron_delivery_hint(agent: Any) -> str:
    """The destination channel's hint (default + its ``platform_hints`` override) for a cron agent.

    A cron agent runs as platform ``cron`` but its final response lands on the job's ``deliver``
    channel, so without this the model never learns that MEDIA: tags become Slack/Telegram
    attachments or that tables do not render there — and a user's ``platform_hints.slack.append``
    never reached scheduled jobs at all. The scheduler publishes the primary auto-deliver target
    into the session ContextVar before the agent runs (same seam ``send_message`` routes by).
    """
    from gateway.session_context import get_session_env
    deliver_key = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").lower().strip()
    if not deliver_key or deliver_key == "cron":
        return ""
    hint = _resolve_platform_hint(agent, deliver_key, _default_platform_hint(deliver_key))
    return f"Delivery destination ({deliver_key}): {hint}" if hint else ""


def platform_hint(agent: Any) -> str:
    """Built-in/plugin platform hint + Telegram rich-messages opt-in + config
    override + desktop TUI clarifier; cron agents also carry their delivery channel's hint."""
    platform_key = (agent.platform or "").lower().strip()
    _effective_hint = _resolve_platform_hint(agent, platform_key, _default_platform_hint(platform_key))
    if platform_key == "tui" and _effective_hint:
        _effective_hint = _tui_embedded_pane_clarifier(_effective_hint)
    if platform_key == "cron":
        _delivery = _cron_delivery_hint(agent)
        if _delivery:
            _effective_hint = f"{_effective_hint}\n\n{_delivery}".strip()
    return _effective_hint


def _telegram_rich_messages_enabled() -> bool:
    """``rich_messages`` from the Telegram ``extra`` config; same precedence the
    adapter uses (top-level ``platforms.telegram.extra`` overrides
    ``gateway.platforms.telegram.extra`` at the leaf). False on any read failure."""
    try:
        from hermes_cli.config import load_config_readonly
        _cfg = load_config_readonly()
        _gw = (((_cfg.get("gateway") or {}).get("platforms") or {}).get("telegram") or {}).get("extra")
        _top = ((_cfg.get("platforms") or {}).get("telegram") or {}).get("extra")
        merged = {**(_gw if isinstance(_gw, dict) else {}), **(_top if isinstance(_top, dict) else {})}
        return bool(merged.get("rich_messages"))
    except Exception:
        return False


def _zone_bits(now: Any, tz: Any) -> List[str]:
    """IANA key, abbreviation (if different) and UTC offset — all constant for
    the day, so the byte-stable date line stays cacheable."""
    _iana = getattr(tz, "key", None)
    from hermes_time import safe_strftime
    _abbrev = safe_strftime(now, "%Z")
    _offset = safe_strftime(now, "%z")  # '-0400' -> 'UTC-04:00'
    bits = [_iana] if _iana else []
    if _abbrev and _abbrev != _iana:
        bits.append(_abbrev)
    if _offset:
        bits.append(f"UTC{_offset[:3]}:{_offset[3:]}")
    return bits


def _timestamp_line(agent: Any) -> str:
    """Date-only so the prompt is byte-stable for the day; zone + offset so
    tools needn't guess EST vs EDT. Long-lived sessions get an "as of" line on
    rebuild days (the cache prefix is already invalidated at that boundary)."""
    from hermes_time import get_timezone as _hermes_tz, now as _hermes_now, safe_strftime
    now = _hermes_now()
    _bits = _zone_bits(now, _hermes_tz())
    _zone_suffix = f" ({', '.join(_bits)})" if _bits else ""
    _start = _session_start_like(agent, now)
    timestamp_line = f"Conversation started: {safe_strftime(_start, '%A, %B %d, %Y')}{_zone_suffix}"
    # Second line (maintainer design, salvaging #96224's anchor): long-lived sessions — Bot Mode
    # forever-chats, messenger channels people never close — span many days and many compactions. A lone
    # birth date leads the model to believe it is still living in that old day. The prompt is rebuilt at
    # every compaction boundary, so stamp the rebuild day too: 'started' stays anchored and byte-stable, 'as
    # of' refreshes exactly when the cache prefix is already being invalidated (compaction), so the added
    # line costs no extra cache churn. Same-day sessions skip the second line entirely — nothing to correct,
    # and the single-line shape stays byte-identical for the day (prefix-cache safe).
    if now.strftime("%Y%m%d") != _start.strftime("%Y%m%d"):
        timestamp_line += (f"\nToday's date (as of the last context rebuild): {safe_strftime(now, '%A, %B %d, %Y')} "
                           "— trust this over the start date for what day it is now; query tools for exact time.")
    if getattr(agent, "_bot_chat_timeless_prompt", False):
        timestamp_line = f"Timezone: {', '.join(_bits)}" if _bits else ""
    trailer = (("Session ID", agent.session_id if agent.pass_session_id else None), ("Model", agent.model),
               ("Provider", agent.provider), ("Platform", agent.platform))
    return timestamp_line + "".join(f"\n{label}: {value}" for label, value in trailer if value)


def _memories_file_block(agent: Any, filename: str, title: str) -> Optional[Tuple[str, str]]:
    """A hand-edited memories/ context file (RULES.md, HISTORY.md), injected before MEMORY.md."""
    return memories_file_block(agent, filename, title)


def _memory_blocks(agent: Any) -> List[Tuple[str, str]]:
    """Built-in memory/USER.md blocks plus the external provider block as
    ``(text, label)`` pairs in assembly order (absent kinds yield nothing), gated
    on the same check ``inject_memory_provider_tools`` uses so we never advertise
    tools the toolset config gated off. Single code path keeps text and its
    provenance label in lockstep — the label is derived from what actually got
    appended, never from a parallel re-check."""
    blocks: List[Tuple[str, str]] = []
    # Hand-edited standing files first (RULES -> HISTORY -> MEMORY/USER order),
    # so stable index content precedes the churning scratch pad.
    for filename, title in (("RULES.md", "RULES (general standing rules)"),
                            ("HISTORY.md", "HISTORY (project index)")):
        block = _memories_file_block(agent, filename, title)
        if block:
            blocks.append(block)
    if agent._memory_store:
        for enabled, kind in ((agent._memory_enabled, "memory"), (agent._user_profile_enabled, "user")):
            block = agent._memory_store.format_for_system_prompt(kind) if enabled else None
            if block:
                # Label from the block's own banner — robust across memory-store
                # variants; matches on the stable header prefix so wording
                # drift inside the parentheses cannot break the mapping.
                if "MEMORY (your personal notes)" in block:
                    label = "~/.hermes/memories/MEMORY.md"
                elif "USER PROFILE" in block:
                    label = "~/.hermes/memories/USER.md"
                else:
                    label = f"{kind} memory block"
                blocks.append((block, label))
    # External memory provider system prompt block (additive to built-in). Gated on the same check
    # ``inject_memory_provider_tools`` uses so we never advertise provider tools that the agent's toolset
    # configuration has already gated off (#81014).
    if agent._memory_manager:
        try:
            from agent.memory_manager import memory_provider_tools_exposed as _mem_exposed
        except Exception:
            _mem_exposed = None
        if _mem_exposed is None or _mem_exposed(agent):
            try:
                _ext_mem_block = agent._memory_manager.build_system_prompt()
            except Exception:
                _ext_mem_block = None
            if _ext_mem_block:
                blocks.append((_ext_mem_block, "external memory provider"))
    return blocks


def _identity_parts(agent: Any, ctx_len: Optional[int]) -> Tuple[List[str], bool]:
    """SOUL.md (primary identity; cron keeps the persona while skipping cwd
    instructions, scoped to the agent's OWN home) or the default identity.
    Returns ``(parts, soul_loaded)``."""
    wants_soul = agent.load_soul_identity or not agent.skip_context_files
    _soul_content = _pb.load_soul_md(ctx_len, home_override=_agent_home(agent)) if wants_soul else None
    return ([_soul_content], True) if _soul_content else ([DEFAULT_AGENT_IDENTITY], False)


def _guidance_blocks(agent: Any) -> List[Tuple[Optional[str], str]]:
    """Universal + tool-aware + model-gated guidance blocks as ``(text,
    source_label)`` pairs in assembly order, each gated by its config.yaml key.
    Labels ride the same conditionals as their texts — a block that is not
    appended never produces a label."""
    out: List[Tuple[str, str]] = []
    if agent.valid_tool_names:
        for flag, text, label in (
            ("_task_completion_guidance", TASK_COMPLETION_GUIDANCE, "task completion guidance"),
            ("_parallel_tool_call_guidance", PARALLEL_TOOL_CALL_GUIDANCE, "parallel tool call guidance"),
        ):
            if getattr(agent, flag, True):
                out.append((text, label))
    # None/empty entries are dropped by _join_tier
    out.append((_tool_guidance_block(agent), "tool-aware behavioral guidance"))
    if not agent.valid_tool_names:
        return out
    # Steering only lands inside tool results, so only reachable with tools.
    out.append((STEER_CHANNEL_NOTE, "steering channel note"))
    # agent.tool_use_enforcement / agent.execution_guidance: "auto" (default)
    # matches the hardcoded model lists; true/false force; a list gives custom
    # model-name substrings.  Execution guidance is an independent gate so
    # DeepSeek/Kimi/Qwen-class models get it even with enforcement off.
    if _model_gate(agent._tool_use_enforcement, agent.model, TOOL_USE_ENFORCEMENT_MODELS):
        out.append((TOOL_USE_ENFORCEMENT_GUIDANCE, "tool-use enforcement guidance"))
        if any(g in (agent.model or "").lower() for g in ("gemini", "gemma")):
            out.append((GOOGLE_MODEL_OPERATIONAL_GUIDANCE, "google model operational guidance"))
    if _model_gate(getattr(agent, "_execution_guidance", "auto"), agent.model, EXECUTION_GUIDANCE_MODELS):
        from agent.prompt_builder import execution_guidance_text
        out.append((execution_guidance_text(agent.valid_tool_names), "execution discipline guidance"))
    # delegate_task background delivery is intentionally between turns. Put this after the generic persistence
    # blocks so their "keep working" rule cannot turn the required yield into no-op/polling activity.
    if "delegate_task" in agent.valid_tool_names:
        out.append((ASYNC_HANDOFF_GUIDANCE, "async handoff guidance (generated constant)"))
    return out


def _alibaba_identity_part(agent: Any) -> List[str]:
    """Alibaba Coding Plan always reports "glm-4.7" as the model name; inject
    the real identity so the agent can answer correctly."""
    if agent.provider != "alibaba":
        return []
    _model_short = agent.model.rsplit("/", 1)[-1]
    return [
        f"You are powered by the model named {_model_short}. "
        f"The exact model ID is {agent.model}. "
        f"When asked what model you are, always answer based on this information, "
        f"not on any model name returned by the API."
    ]


def _workspace_pin_key() -> str:
    """The directory the workspace probe inspects, which is also the prompt's ``Current working
    directory``: a build with no cwd bound (launch dir) and a later one binding that same dir
    (TUI ``/compress``) are one workspace, not two."""
    try:
        return str(resolve_context_cwd() or resolve_agent_cwd())
    except OSError:  # deleted cwd
        return ""


def _persisted_workspace_block(prompt: str, key: str) -> Optional[str]:
    """The workspace snapshot inside ``prompt`` taken for ``key`` (its ``- Root:`` is ``key`` or an
    ancestor); "" when the prompt has none; None when it has one for another root."""
    from agent.coding_context import WORKSPACE_BLOCK_HEADER
    head = f"\n\n{WORKSPACE_BLOCK_HEADER}\n- Root: "
    start = prompt.find(head)
    if start < 0:
        return ""
    cwd = Path(key).resolve()
    while start >= 0:
        block = prompt[start + 2:].split("\n\n", 1)[0]
        root = Path(block.split("\n", 2)[1][len("- Root: "):]).resolve()
        if root == cwd or root in cwd.parents:
            return block
        start = prompt.find(head, start + 2)
    return None


def _session_prompt(agent: Any) -> Optional[str]:
    """Prompt bytes this session already sends: the cached copy, else its persisted row."""
    cached = getattr(agent, "_cached_system_prompt", None)
    if isinstance(cached, str) and cached:
        return cached
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if db is None or not isinstance(session_id, str) or not session_id:
        return None
    try:
        row = db.get_session(session_id)
    except Exception:
        logger.debug("workspace snapshot: session row read failed (session=%s)", session_id, exc_info=True)
        return None
    prompt = row.get("system_prompt") if isinstance(row, dict) else None
    return prompt if isinstance(prompt, str) and prompt else None


def _seed_workspace_pin(agent: Any, key: str) -> None:
    """Pin the snapshot the session's existing prompt already carries.  An agent that did not
    build those bytes (resumed, or a fresh gateway/TUI agent whose first act is ``/compress``)
    would otherwise re-probe git at its first rebuild and rewrite the prompt for any repo that
    moved since session start.  Only a snapshot provably taken in this cwd is adopted."""
    from agent.surface_switch import runtime_host_value
    prompt = _session_prompt(agent)
    if not prompt:
        return
    stored_cwd = runtime_host_value(prompt, "Current working directory")
    if stored_cwd and stored_cwd != key:
        return
    block = _persisted_workspace_block(prompt, key)
    # Only a real snapshot is adopted: a prompt without one (built on a surface without the
    # coding posture, or with tools off) leaves the pin open so this build captures one.
    if block:
        agent._frozen_workspace_snapshot = (key, block)


def _coding_parts(agent: Any) -> Tuple[List[str], List[str], List[str]]:
    """``(prefix, workspace, trailing)`` coding-posture blocks; all empty
    without tools or when probing fails (it must never block prompt build).

    The workspace block is a live git probe after project context, ahead of the whole
    volatile band; re-probing at the compaction rebuild re-emits different bytes for any
    repo that moved and defeats the keep-prompt fast path.  So the bytes are pinned per
    session on the agent, keyed by the probed cwd (a gateway serves many cwds), seeded from
    the session's existing prompt when this agent did not build it, and replayed on
    rebuilds; ``reset_session_state`` drops the pin at a session boundary.
    """
    try:
        from agent.coding_context import coding_system_prompt_parts
        if not agent.valid_tool_names:
            return [], [], []
        cwd = resolve_context_cwd()
        cwd_key = _workspace_pin_key()
        if getattr(agent, "_frozen_workspace_snapshot", None) is None:
            _seed_workspace_pin(agent, cwd_key)
        pinned = getattr(agent, "_frozen_workspace_snapshot", None)
        # "" is a real pinned value (no workspace here) — only a cwd mismatch re-probes.
        replay = pinned[1] if pinned is not None and pinned[0] == cwd_key else None
        parts = coding_system_prompt_parts(platform=agent.platform, cwd=cwd, model=agent.model,
                                           valid_tool_names=agent.valid_tool_names, workspace_block=replay)
        if replay is None:
            agent._frozen_workspace_snapshot = (cwd_key, parts[1][0] if parts[1] else "")
        return parts
    except Exception:
        pass
    return [], [], []


def _post_workspace_blocks(agent: Any) -> List[Tuple[Optional[str], str]]:
    """Blocks that follow the worktree-specific context (environment probe,
    bot-mode protocol, profile line, platform hint) as ``(text, source_label)``
    pairs — the label is assigned where its text is appended, so it cannot
    drift from the gates."""
    blocks: List[Tuple[Optional[str], str]] = []
    if getattr(agent, "_environment_probe", True):
        try:
            from tools.env_probe import get_environment_probe_line
            probe = get_environment_probe_line()
            if probe:
                blocks.append((probe, "runtime environment probe"))
        except Exception:
            pass  # Probe failure must never block prompt build.
    if getattr(agent, "_bot_mode_protocol", True):
        _bot_parts = _bot_mode_parts(agent)
        for i, p in enumerate(_bot_parts):
            blocks.append((p, "bot-mode capability epoch" if i > 0 and _bot_parts[0] else "bot-mode protocol section"))
    blocks.append((platform_hint(agent), "platform hint"))
    return blocks


def _context_files_part(agent: Any, ctx_len: Optional[int], soul_loaded: bool) -> List[str]:
    """Project context files (AGENTS.md etc.) for the context tier. TERMINAL_CWD
    when set (gateway); None lets discovery fall back to the launch dir.  The
    install-tree fallback is only legitimate for cli/tui where the launch dir
    IS the user's shell cwd. Desktop launch artifacts skip the session cwd but
    still honor the profile-scoped TERMINAL_CWD; without one, the fallback guard
    can reject Hermes's bundled AGENTS.md."""
    if agent.skip_context_files:
        return []
    launch_artifact = getattr(agent, "_context_cwd_is_launch_artifact", False)
    cwd = resolve_context_cwd(include_session_override=not launch_artifact)
    return [_pb.build_context_files_prompt(
        cwd=cwd, skip_soul=soul_loaded, context_length=ctx_len,
        allow_install_tree_fallback=agent.platform in ("cli", "tui"), home_override=_agent_home(agent))]


def _join_tier(parts: Sequence[Optional[str]]) -> str:
    """Join non-empty parts; None/blank entries are dropped."""
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _context_file_labels_for_block(agent: Any, block_text: str) -> List[Tuple[str, int]]:
    """Per-file provenance for one context-files block as ``(label, chars)``."""
    return context_file_labels_for_block(agent, block_text)


# (HARDCODED_CONSTANTS now imported from prompt_provenance.py)


def _part_kind_for_label(label: str) -> str:
    """Classify what a system-prompt part IS, from its assembly label.

    Returns one of ``hardcoded | file | generated | summary``. Assigned at the
    point text is appended so the /pi kind column is data, not decoration.
    (To be refined as more parts get honest kinds; kept simple on purpose.)
    """
    if label in HARDCODED_CONSTANTS or "hardcoded" in label:
        return "hardcoded"
    # Real file-backed content: memory files, SOUL.md, skills, AGENTS/CLAUDE, etc.
    if (label.startswith("~/.hermes/memories/") or "memories/MEMORY.md" in label
            or "memories/USER.md" in label or label == "~/.hermes/SOUL.md"
            or label in ("SOUL.md", "MEMORY.md", "USER.md")
            or label.startswith("skill ") or "SKILL.md" in label
            or label.endswith((".md", ".MD"))):
        return "file"
    # Runtime-assembled from several sources (context-file bundle, skills index).
    if label in ("project context files", "skills index") or label.startswith("_ctx_block_"):
        return "summary"
    return "generated"


def build_system_prompt_parts(agent: Any, system_message: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the system prompt as three ordered cache tiers: ``stable`` (identity,
    guidance and the coding brief), ``context`` (caller ``system_message``, project
    context files, workspace snapshot and remaining workspace guidance) and
    ``volatile`` (skills index, memory, user profile, external memory block,
    timestamp line, runtime environment hints).  Worktree-dependent blocks follow project context so a
    shared context file can remain in the longest common prefix across worktrees.
    Never re-rendered mid-session.

    Also returns provenance metadata: ``_stable_sources`` / ``_context_sources`` /
    ``_volatile_sources`` hold ``(label, bytes)`` tuples in assembly order — one entry
    per text that actually lands in that tier (a context-file block yields one entry
    per file inside it). This is the source of the ``/pi`` prompt-part breakdown.

    Per-block texts and labels are published on the agent as an observability
    side-channel (``agent._system_prompt_blocks``) in full assembly order; the
    manifest publisher turns them into one part per block. The return dict stays
    exactly three string tiers — consumers iterate ``parts.values()``.
    """
    # Model context window scales the context-file caps; stable per conversation.
    _cc_len = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    _ctx_len = _cc_len if isinstance(_cc_len, int) and _cc_len > 0 else None

    def _sources_from(parts_: List[Optional[str]], labels: List[str]) -> Tuple[Tuple[str, int], ...]:
        """(label, bytes) for every non-blank part; mirrors _join_tier's drop rule."""
        return tuple(
            (labels[i], utf8_bytes(p)) for i, p in enumerate(parts_)
            if p and p.strip() and i < len(labels)
        )

    # ── Stable tier ────────────────────────────────────────────────
    # ``stable_sources`` holds LABELS only — lockstep with ``stable_parts`` by index.
    stable_parts: List[Optional[str]] = []
    stable_sources: List[str] = []
    _identity_texts, _soul_loaded = _identity_parts(agent, _ctx_len)
    if _soul_loaded:
        # Real file path (same home resolution the loader uses), displayed as
        # ~/... where possible — consistent with the memory-file labels.
        _home = _agent_home(agent) or get_hermes_home()
        _identity_label = display_path(str(_home / "SOUL.md"))
    else:
        _identity_label = "default identity"
    for t in _identity_texts:
        stable_parts.append(t)
        stable_sources.append(_identity_label)

    # The skill_view() pointer dangles without skill tools OR without the
    # hermes-agent skill installed, so the variant is chosen after the skills
    # index is built; this slot holds its position.
    _help_guidance_slot = len(stable_parts)
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS)
    stable_sources.append("hermes-agent help guidance")
    # Guidance blocks: one code path supplies texts and labels in lockstep.
    for text, label in _guidance_blocks(agent):
        stable_parts.append(text)
        stable_sources.append(label)

    skills_prompt = _skills_prompt(agent)
    # Skill-pointer variant requires BOTH skill_view AND the hermes-agent skill
    # in the rendered index (pure string check — inherits the index's stability).
    if "skill_view" in (agent.valid_tool_names or set()) and "- hermes-agent:" in skills_prompt:
        stable_parts[_help_guidance_slot] = HERMES_AGENT_HELP_GUIDANCE

    for t in _alibaba_identity_part(agent):
        stable_parts.append(t)
        stable_sources.append("model identity (Alibaba provider override)")
    # Pinned skills are per-agent constants (resolved once), so they live in the stable prefix.
    for t, label in _auto_load_parts(agent):
        stable_parts.append(t)
        stable_sources.append(label)
    # Coding posture: the operating brief stays in the stable prefix. The
    # environment block contains the current cwd/backend and belongs after
    # project context, not ahead of a large shared AGENTS.md block.
    environment_hints = _pb.build_environment_hints()
    coding_prefix_parts, coding_workspace_parts, coding_trailing_parts = _coding_parts(agent)
    for t in coding_prefix_parts:
        stable_parts.append(t)
        stable_sources.append("coding brief (stable portion)")
    # Post-workspace blocks are appended to whichever tier claims them below;
    # their labels ride along in lockstep.
    post_blocks = _post_workspace_blocks(agent)

    # ── Context tier (project/worktree-dependent, may change between sessions) ──
    # Parallel label lists, index-locked to their parts. The context-file block
    # is ONE part whose provenance decomposes per file (see metadata below).
    context_parts: List[Optional[str]] = []
    context_part_labels: List[str] = []
    _ctx_file_blocks: List[str] = []  # non-blank blocks from _context_files_part, in order
    # ephemeral_system_prompt is injected at API-call time only, never cached.
    if system_message is not None:
        context_parts.append(system_message)
        context_part_labels.append("caller-provided system message")
    for block in _context_files_part(agent, _ctx_len, _soul_loaded):
        if not (block and block.strip()):
            continue
        context_parts.append(block)
        _ctx_file_blocks.append(block)
        # Placeholder index into the per-file provenance lists built below.
        context_part_labels.append(f"_ctx_block_{len(_ctx_file_blocks) - 1}")
    # Per-file provenance from the loader instrumentation (prompt_builder's
    # _record_context_file); the loaders know their own sizes — no parsing.
    # Shape: [(label, chars), ...] in load order; empty list = no files loaded.
    _ctx_file_provenance = [
        (_e["path"], _e["bytes"]) for _e in _pb.drain_context_file_provenance()
    ] or []
    if coding_workspace_parts:
        for t in coding_workspace_parts:
            context_parts.append(t)
            context_part_labels.append("git workspace snapshot")
        for t in coding_trailing_parts:
            context_parts.append(t)
            context_part_labels.append("coding brief (trailing portion)")
        for text, label in post_blocks:
            context_parts.append(text)
            context_part_labels.append(label)
    else:
        # Preserve the stable placement for non-workspace sessions; there is no
        # worktree snapshot whose later position would improve their prefix.
        for t in coding_trailing_parts:
            stable_parts.append(t)
            stable_sources.append("coding brief (trailing portion)")
        for text, label in post_blocks:
            stable_parts.append(text)
            stable_sources.append(label)

    # ── Volatile tier (most likely to differ on a rebuild; kept last so the stable prefix stays reusable) ──
    # Skills are runtime-mutable, so the index leads the volatile band: on a longest-prefix
    # backend an unchanged index stays inside the reused prefix; a changed one re-prefills from here.
    # (text, label) pairs throughout — blank entries (e.g. no skills index) drop out of BOTH,
    # so text and labels never drift apart.
    volatile_parts: List[Optional[str]] = []
    volatile_sources: List[str] = []
    if skills_prompt:
        # The index is ONE part so _join_tier keeps the tier byte-identical to
        # the upstream shape; its per-skill provenance decomposes at metadata
        # assembly below (placeholder pattern, same as context-file blocks).
        volatile_parts.append(skills_prompt)
        volatile_sources.append("_skills_index")
    for text, label in _memory_blocks(agent):
        volatile_parts.append(text)
        volatile_sources.append(label)
    # Plugin sections are confined to one coarse anchor in the volatile tail so
    # a resumed process can reconstruct the stable prefix without re-running plugins.
    for t in _plugin_section_blocks(_frozen_plugin_prompt_sections(agent), "after_memory"):
        volatile_parts.append(t)
        volatile_sources.append("plugin system prompt sections")
    # The profile line names this home's path, so it rides in the volatile tier: the stable
    # prefix then stays byte-identical across every profile (and home) on the host.
    # (upstream f29163388c; re-added after a rebase conflict dropped it.)
    volatile_parts.append(_active_profile_line(agent))
    volatile_sources.append("active profile line")
    _ts_text = _timestamp_line(agent)
    volatile_parts.append(_ts_text)
    volatile_sources.append("conversation timestamp")
    # Keep the renderer-owned runtime anchor after all user/plugin prose so quoted
    # host examples cannot shadow it during persisted-prompt validation.
    if environment_hints:
        # Embedder hints are prose too; reserve the delimiter for the renderer.
        environment_hints = environment_hints.replace(_pb.RUNTIME_ENVIRONMENT_HEADING, "> " + _pb.RUNTIME_ENVIRONMENT_HEADING)
        volatile_parts.append(f"{_pb.RUNTIME_ENVIRONMENT_HEADING}\n\n{environment_hints}\n\n{_pb.RUNTIME_ENVIRONMENT_END}")
        volatile_sources.append("runtime environment")

    # ── Provenance metadata (internal; never part of the prompt) ──
    context_sources: List[Tuple[str, int]] = []
    for i, p in enumerate(context_parts):
        if not (p and p.strip()):
            continue
        label = context_part_labels[i]
        if label.startswith("_ctx_block_"):
            # Context-file block: one provenance entry per file, straight
            # from the loader instrumentation (no parsing of the block).
            context_sources.extend(_ctx_file_provenance or [(label, utf8_bytes(p))])
        else:
            context_sources.append((label, utf8_bytes(p)))

    volatile_source_entries: List[Tuple[str, int]] = []
    for (label, chars) in _sources_from(volatile_parts, volatile_sources):
        if label == "_skills_index":
            # Skills-index part: one provenance entry per listed skill.
            sk_src = _skills_index_sources(agent, skills_prompt)
            if sk_src:
                volatile_source_entries.extend(sk_src)
                continue
            # Unparseable index: fall back to one coarse entry.
            volatile_source_entries.append(("skills index", chars))
        else:
            volatile_source_entries.append((label, chars))

    try:
        agent._system_prompt_sources = {
            "stable": list(_sources_from(stable_parts, stable_sources)),
            "context": list(context_sources),
            "volatile": volatile_source_entries,
        }
    except Exception:
        pass  # observability must never break prompt build

    # Per-block publish for the /pi manifest: one part per text block instead of
    # three tier bundles (every hardcoded constant, every memory file, ...). The
    # entry texts are exactly the parts the tiers join (same drop rule), so the
    # published parts rejoin to the byte-identical prompt. Each entry is
    # (text, label, annotation, sources, kind); annotation is human-mode-only explanatory text for parts built
    # from hardcoded string literals at runtime; sources = the per-block share of
    # the tier provenance (composite blocks: one row per inner file/skill).
    #
    # Index-based (never zip): some paths append to a parts list more often than
    # its label list — an index miss must fall back, never drop a text.
    try:
        # (text, label, annotation, sources, kind); kind = what this part IS
        # (hardcoded | file | generated | summary), assigned where the text is appended.
        blocks: List[Tuple[str, str, str, List[Tuple[str, int]], str]] = []

        # Stable: 1:1 labels, EXCEPT the non-workspace path appends
        # coding_trailing_parts without labels — fall back by position.
        for i, p in enumerate(stable_parts):
            if not (p and p.strip()):
                continue
            lbl = stable_sources[i] if i < len(stable_sources) else "coding brief (trailing portion)"
            ann = ("Static constant in prompt_builder.py; the skill_view pointer "
                   "variant is swapped in after the skills index renders."
                   if lbl == "hermes-agent help guidance" else "")
            kind = _part_kind_for_label(lbl)
            blocks.append((p, lbl, ann, [(lbl, utf8_bytes(p))], kind))
        # Context: 1:1 labels by construction.
        for i, p in enumerate(context_parts):
            if not (p and p.strip()):
                continue
            lbl = context_part_labels[i] if i < len(context_part_labels) else "context section"
            ann = ""
            srcs = [(lbl, utf8_bytes(p))]
            kind = "generated"
            if lbl.startswith("_ctx_block_"):
                lbl = "project context files"
                ann = ("Built by build_context_files_prompt() from the AGENTS.md/"
                       "CLAUDE.md files discovered in the working directory; one "
                       "section per file (listed below in human mode).")
                srcs = list(_ctx_file_provenance) or [(lbl, utf8_bytes(p))]
                kind = "summary"
            blocks.append((p, lbl, ann, srcs, kind))
        # Volatile: 1:1 labels by construction.
        _vol_annots = {
            "_skills_index": ("Rendered by build_skills_system_prompt() from the active "
                              "skills index; one line per listed skill."),
            "conversation timestamp": ("Built by _timestamp_line() from the session id's embedded "
                                                   "timestamp plus model/provider/platform values."),
            "runtime environment": ("Probed by build_environment_hints() at build time: current "
                                                "host, user, working directory and backend."),
        }
        for i, p in enumerate(volatile_parts):
            if not (p and p.strip()):
                continue
            lbl = volatile_sources[i] if i < len(volatile_sources) else "runtime section"
            srcs = [(lbl, utf8_bytes(p))]
            kind = "summary" if lbl == "_skills_index" else _part_kind_for_label(lbl)
            if lbl == "_skills_index":
                sk_src = _skills_index_sources(agent, skills_prompt)
                if sk_src:
                    srcs = sk_src
            blocks.append((p, "skills index" if lbl == "_skills_index" else lbl,
                           _vol_annots.get(lbl, ""), srcs, kind))
        agent._system_prompt_blocks = blocks
    except Exception:
        pass  # observability must never break prompt build

    return {"stable": _join_tier(stable_parts), "context": _join_tier(context_parts), "volatile": _join_tier(volatile_parts)}


def build_system_prompt(agent: Any, system_message: Optional[str] = None) -> str:
    """Assemble the full prompt; cached on ``agent._cached_system_prompt`` and
    only rebuilt after compression.  Tiers are ordered stable -> context ->
    volatile so implicit longest-prefix caches keep the unchanged scaffold."""
    parts = build_system_prompt_parts(agent, system_message=system_message)
    agent._cached_system_prompt_static = parts["stable"]
    # Prompt manifest (observability): publish ONE part per text block instead of
    # three tier bundles. The blocks ride on the agent from build_system_prompt_parts;
    # their texts are exactly the tier inputs, so the published parts rejoin to the
    # byte-identical prompt. Provenance sub-rows (per-file / per-skill shares) ride on
    # each node as sources and show in the human-mode drill-down.
    try:
        from agent.prompt_manifest import ComponentNode, get_or_create_manifest as _pm_get
        _pm_m = _pm_get(agent)
        _comps = [ComponentNode(label, text, tuple(srcs), annotation=ann, kind=kind)
                  for (text, label, ann, srcs, kind) in (getattr(agent, "_system_prompt_blocks", None) or [])
                  if text and text.strip()]
        if not _comps:
            # Fail open to the previous tier-granular shape.
            sources = getattr(agent, "_system_prompt_sources", {}) or {}
            _tiers = ((("stable", "System Prompt (Stable)"), parts["stable"]),
                      (("context", "System Prompt (Context)"), parts["context"]),
                      (("volatile", "System Prompt (Volatile)"), parts["volatile"]))
            _comps = [ComponentNode(desc, text, tuple(sources.get(key, [])))
                      for (key, desc), text in _tiers if text and text.strip()]
        _pm_m._store_system_nodes(agent, [c for c in _comps if c.text.strip()])
    except Exception:
        pass  # observability must never break prompt build
    # Surface context-file truncation warnings in chat, not only in logs.
    for warning in drain_truncation_warnings():
        agent._emit_diagnostic_status(warning)
    return "\n\n".join(p for p in (parts["stable"], parts["context"], parts["volatile"]) if p)


def invalidate_system_prompt(agent: Any) -> None:
    """Force a rebuild on the next turn (after compression): reload memory from
    disk and clear the frozen plugin snapshot (previous bytes stashed as the
    fail-open fallback) so plugins re-render at the same boundary.

    Called after context compression events. Also reloads memory from disk so the rebuilt prompt captures
    any writes from this session, and clears the frozen plugin-section snapshot so plugins re-render at the
    same boundary (maintainer-directed, #95681 arc): a plugin section is just another prompt block carrying
    state — freezing it while memory, skills, and guidance refresh would recreate the stale-block disease
    inside plugin-land. The previous bytes are stashed so a plugin whose render RAISES falls back to its
    last good section instead of vanishing (fail-open guard, not a freeze).
    """
    agent._cached_system_prompt = None
    agent._cached_system_prompt_static = None
    if hasattr(agent, "_plugin_system_prompt_sections_snapshot"):
        agent._plugin_system_prompt_sections_previous = agent._plugin_system_prompt_sections_snapshot
        del agent._plugin_system_prompt_sections_snapshot
    if agent._memory_store:
        agent._memory_store.load_from_disk()


def reconstruct_static_prefix(agent: Any, system_message: Optional[str] = None, *, log_label: str = "restore") -> None:
    """Reconstruct ``_cached_system_prompt_static`` for a stored prompt.
    Only the full prompt is persisted, so restore / keep-prompt compression /
    mid-turn failover to a cache-on provider must rebuild the stable tier to
    regain the ``[static, volatile]`` layout.  The rebuilt tier is used ONLY
    when the stored prompt literally starts with it; otherwise static stays
    None and the stored bytes are sent untouched.  A failed rebuild is memoized
    per stored prompt so the retry-loop hot path doesn't redo the file I/O."""
    stored = getattr(agent, "_cached_system_prompt", None)
    existing = getattr(agent, "_cached_system_prompt_static", None)
    if (
        not getattr(agent, "_use_prompt_caching", False)
        or not isinstance(stored, str) or not stored
        or (isinstance(existing, str) and existing and stored.startswith(existing))
        or getattr(agent, "_static_rebuild_failed_for", None) == stored
    ):
        return
    try:
        static = build_system_prompt_parts(agent, system_message=system_message)["stable"]
        if static and stored.startswith(static):
            agent._cached_system_prompt_static = static
            agent._static_rebuild_failed_for = None
            return
    except Exception:
        logger.debug("static system-prefix reconstruction failed on %s", log_label, exc_info=True)
    agent._cached_system_prompt_static = None
    agent._static_rebuild_failed_for = stored


def format_tools_for_system_message(agent: Any) -> str:
    """JSON tool definitions in the trajectory format."""
    if not agent.tools:
        return "[]"
    return json.dumps([{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                        "parameters": t["function"].get("parameters", {}),
                        "required": None}  # Match the format in the example
                       for t in agent.tools], ensure_ascii=False)


__all__ = ["build_system_prompt_parts", "build_system_prompt", "invalidate_system_prompt",
           "platform_hint", "restore_plugin_prompt_sections", "format_tools_for_system_message"]
