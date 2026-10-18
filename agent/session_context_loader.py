"""Load context from previous session to inject at the start of new conversations."""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Safety cap on raw rows scanned per candidate session while hunting for
# user/assistant messages with real content (tails are often tool noise).
_MAX_SCAN_ROWS = 500
_PAGE_SIZE = 50


def get_previous_session_context(
    session_db: Any,
    current_session_id: str,
    min_messages_required: int = 10,
    max_messages_to_return: int = 16,
) -> List[Dict[str, Any]]:
    """Get context messages from the most recent previous session.

    Session selection uses ``substantive_message_count`` (computed from the
    message rows via SQL), never the denormalized ``sessions.message_count``
    counter — that stored field counts tool noise and can drift, which once
    made a session with only 4 real messages pass the eligibility filter.

    Content: when a selected session has at most ``max_messages_to_return``
    substantive messages, all of them are returned; otherwise the first half
    plus the last half (e.g. 8 + 8 for 16), so both where the session started
    and how it ended carry over.

    Uses the SessionDB public API only (list_sessions_rich /
    substantive_message_count / get_messages).

    Args:
        session_db: The SessionDB instance (can be None)
        current_session_id: Current session ID to exclude
        min_messages_required: Minimum substantive message count for eligibility
        max_messages_to_return: Cap on returned messages (splits half/half when exceeded)

    Returns:
        List of message dicts with role and content, in chronological order (oldest first).
    """
    if not session_db or not current_session_id:
        return []

    try:
        # Candidate shortlist ordered by most recently active. Deliberately no
        # message-count filter in SQL (see docstring); eligibility is checked
        # per candidate from the actual rows.
        sessions = session_db.list_sessions_rich(
            order_by_last_active=True,
            limit=8,
        )

        for session in sessions:
            prev_session_id = session.get("id")
            if not prev_session_id or prev_session_id == current_session_id:
                continue

            # Eligibility from the actual rows, not the stored counter.
            try:
                substantive = session_db.substantive_message_count(prev_session_id)
            except Exception as exc:
                logger.warning(
                    "substantive_message_count failed for %s: %s", prev_session_id[:8], exc
                )
                continue
            if substantive < min_messages_required:
                continue

            context_messages = _collect_context_messages(
                session_db, prev_session_id, substantive, max_messages_to_return
            )
            if not context_messages:
                continue

            logger.debug(
                "Previous session context loaded: session=%s messages=%d",
                prev_session_id[:8], len(context_messages),
            )
            return context_messages

        return []
    except Exception as exc:
        # Visible by default: a silent failure here means context injection
        # is silently dead, which is hard to notice otherwise.
        logger.warning("Failed to load previous session context: %s", exc)
        return []


def _is_substantive(msg: Dict[str, Any]) -> bool:
    """User/assistant row with real (non-blank) content."""
    role = msg.get("role")
    content = msg.get("content") or ""
    return role in ("user", "assistant") and bool(content.strip())


def _collect_context_messages(
    session_db: Any,
    prev_session_id: str,
    substantive: int,
    max_messages_to_return: int,
) -> List[Dict[str, Any]]:
    """Collect context messages from one selected session, oldest first.

    All substantive messages when they fit in the cap; otherwise the first
    ``cap // 2`` from the start plus the last ``cap // 2`` from the end (no
    overlap possible: half + half <= cap < substantive).
    """
    if substantive <= max_messages_to_return:
        head = _scan_substantive(session_db, prev_session_id, max_messages_to_return,
                                 latest=False)
        return head

    half = max_messages_to_return // 2
    head = _scan_substantive(session_db, prev_session_id, half, latest=False)
    # Newest-first scan: reverse so oldest-of-the-tail comes first.
    tail = _scan_substantive(session_db, prev_session_id, half, latest=True)[::-1]
    return head + tail


def _scan_substantive(
    session_db: Any,
    session_id: str,
    needed: int,
    latest: bool,
) -> List[Dict[str, Any]]:
    """Collect up to *needed* substantive messages from one end of a session.

    latest=False scans from the start (oldest first, returned oldest first).
    latest=True scans back from the end (returned newest first). Stops at the
    other end of the history or the scan cap, whichever comes first.
    """
    collected: List[Dict[str, Any]] = []
    offset = 0
    while len(collected) < needed and offset < _MAX_SCAN_ROWS:
        rows = session_db.get_messages(
            session_id,
            limit=_PAGE_SIZE,
            offset=offset,
            latest=latest,
        )
        if not rows:
            break

        if latest:
            # Page is chronological; walk newest-first.
            for msg in reversed(rows):
                if _is_substantive(msg):
                    collected.append({"role": msg["role"], "content": msg["content"]})
                    if len(collected) >= needed:
                        break
        else:
            # Page is already oldest-first.
            for msg in rows:
                if _is_substantive(msg):
                    collected.append({"role": msg["role"], "content": msg["content"]})
                    if len(collected) >= needed:
                        break

        # A short page is the end of history at this direction: nothing further.
        if len(rows) < _PAGE_SIZE:
            break

        offset += _PAGE_SIZE
    return collected
