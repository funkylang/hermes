"""Prompt provenance and Superhermes memory injection.

Contains logic for /pi observability (per-component labels) and RULES.md/HISTORY.md
injection, separated from system_prompt.py to minimize upstream merge conflicts.

All functions here are best-effort: they must never break prompt build.
On error, fall back to empty/None so the prompt still assembles.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import from our own modules (safe—no circular dependency)
from agent.path_display import display_path

logger = logging.getLogger(__name__)


def _utf8_bytes(text: str) -> int:
    """UTF-8 byte length of *text* — what /pi counts for every prompt part."""
    return len(text.encode("utf-8"))


def rules_block() -> Optional[Tuple[str, str]]:
    """The shared standing-rules directory as one ``(block, label)`` pair.

    Every agent's ``<HERMES_HOME>/rules`` is a symlink to the user-owned shared
    directory (``/home/agents/rules``): one source of truth for all agents.
    Files are injected in filename order — numbered prefixes encode the order;
    adding a file lands it in every agent's next session with no code change.

    Absent or empty yields None, so prompt bytes are untouched until rules
    exist (same contract as ``memories_file_block``). The label is the resolved
    real path — it names the shared origin for /pi.
    """
    try:
        from hermes_constants import get_hermes_home
        rules_dir = Path(get_hermes_home()) / "rules"
        if not rules_dir.is_dir():
            return None
        parts: List[str] = []
        for path in sorted(rules_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                content = path.read_text(encoding="utf-8-sig").strip()
            except OSError as exc:
                logger.warning("rules file unreadable, skipped: %s (%s)", path, exc)
                continue
            if content:
                parts.append(content)
        if not parts:
            return None
        sep = "═" * 46
        block = f"{sep}\nRULES (shared standing rules)\n{sep}\n" + "\n\n".join(parts)
        return (block, display_path(str(rules_dir.resolve())))
    except Exception:
        logger.debug("shared rules block skipped", exc_info=True)
        return None


def shared_soul_block() -> Optional[Tuple[str, str]]:
    """The shared Superhermes soul — one identity statement for the agent family.

    ``<HERMES_HOME>/soul`` is a symlink to the user-owned shared file
    (``/home/agents/soul``). Its text is injected after the rules block and
    before all per-agent content, byte-identical for every agent — part of
    the longest shared prefix on the server.

    Absent or empty yields None — nothing appended, prompt bytes untouched.
    The label carries the resolved real path for /pi provenance.
    """
    try:
        from hermes_constants import get_hermes_home
        soul_path = Path(get_hermes_home()) / "soul"
        if not soul_path.is_file():
            return None
        content = soul_path.read_text(encoding="utf-8-sig").strip()
        return (content, display_path(str(soul_path.resolve()))) if content else None
    except Exception:
        logger.debug("shared soul block skipped", exc_info=True)
        return None


# Per-skill provenance for the rendered skills index (regex-based splitting)
_SKILL_INDEX_ENTRY_RE = re.compile(r"(?m)^    - ([^\s:：][^:：]*?)[:：]")


def skills_index_sources(index_text: str) -> List[Tuple[str, int]]:
    """Per-skill provenance for the rendered skills index.

    The index is a deterministic list of ``  - <name>: <desc>`` lines; splitting
    on those lines yields one entry per listed skill with its real line size.
    Labels are the plain skill names — the directory is constant and adds no info.

    Falls back to a coarse "skills index" label when no entries are found.
    """
    if not index_text or not index_text.strip():
        return []
    matches = list(_SKILL_INDEX_ENTRY_RE.finditer(index_text))
    if not matches:
        return [("skills index", _utf8_bytes(index_text))]
    out: List[Tuple[str, int]] = []
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(index_text)
        name = m.group(1).strip()
        out.append((f"skill {name}", _utf8_bytes(index_text[m.start():body_end])))
    return out


# Context-files provenance: per-file labels via regex splitting on ## headers
_CONTEXT_SECTION_RE = re.compile(r"(?m)^## (.+?)\n\n")


def context_file_labels_for_block(agent: Any, block_text: str) -> List[Tuple[str, int]]:
    """Per-file provenance for one context-files block as ``(label, chars)``.

    The block is a concatenation of ``## <label>`` sections (one per file, see
    ``build_context_files_prompt``); splitting on its own headers yields the
    real per-file sizes. Labels resolve to full paths through the same discovery
    walk the builder uses; on any failure the relative label stands in.

    Provenance is best-effort and must never break prompt build.
    """
    if not block_text or not block_text.strip():
        return []
    path_by_label: Dict[str, str] = {}
    try:
        from agent.context_file_sources import context_file_sources_for_agent
        for e in context_file_sources_for_agent(agent):
            if e.get("loaded"):
                path_by_label[str(e["label"])] = str(e["path"])
    except Exception:
        pass  # labels stay as the section header (relative path)
    # Only headers that the builder wrote as file boundaries (known labels from
    # ``context_file_sources_for_agent``) split the block; any other ``## ...`` line is an
    # internal section header of the file and belongs to its source, not a new one.
    out: List[Tuple[str, int]] = []
    matches = [m for m in _CONTEXT_SECTION_RE.finditer(block_text) if m.group(1).strip() in path_by_label]
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(block_text)
        label = m.group(1).strip()
        out.append((path_by_label[label], body_end - m.start()))
    return out


# Static constants whose text never changes within the prompt (no builder, no loader);
# their parts are published with kind = "hardcoded".
HARDCODED_CONSTANTS = {
    "default identity",
    "hermes-agent help guidance",
    "task completion guidance",
    "parallel tool call guidance",
    "steering channel note",
    "tool-use enforcement guidance",
    "execution discipline guidance",
    "google model operational guidance",
}


def memories_file_block(agent: Any, filename: str, title: str) -> Optional[Tuple[str, str]]:
    """A hand-edited memories/ context file (HISTORY.md), injected
    before MEMORY.md.

    No store/entry model and no API: the file itself is the source of truth
    (edited by hand), read directly from the profile-scoped memories dir.

    Absent or empty yields nothing, so prompt bytes are untouched until it has
    content. The label carries its real path for /pi provenance.
    """
    try:
        from tools.memory_tool import get_memory_dir
        path = get_memory_dir() / filename
        if not path.is_file():
            return None
        content = path.read_text(encoding="utf-8-sig").strip()
        if not content:
            return None
        sep = "═" * 46
        block = f"{sep}\n{title}\n{sep}\n{content}"
        return (block, display_path(str(path)))
    except Exception:
        return None
