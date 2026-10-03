"""Load context from previous session to inject at the start of new conversations."""
import logging
from typing import List, Dict, Any

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
    """Get the last N substantive messages from the most recent previous session.

    Uses the SessionDB public API (list_sessions_rich / get_messages), never
    internal connections.

    Args:
        session_db: The SessionDB instance (can be None)
        current_session_id: Current session ID to exclude
        min_messages_required: Minimum message count for a session to be considered
        max_messages_to_return: Maximum number of messages to return

    Returns:
        List of message dicts with role and content, in chronological order (oldest first).
    """
    if not session_db or not current_session_id:
        return []

    try:
        sessions = session_db.list_sessions_rich(
            order_by_last_active=True, limit=20
        )

        for session in sessions:
            prev_session_id = session.get("id")
            if not prev_session_id or prev_session_id == current_session_id:
                continue

            # Check that this session has enough messages to be useful context
            if (session.get("message_count") or 0) < min_messages_required:
                continue

            # Walk backwards from the newest rows, collecting user/assistant
            # messages that carry real content (raw tails are often tool calls
            # or empty assistant turns). get_messages(latest=True, offset=N)
            # pages back from the newest and returns each page chronologically.
            context_messages = []
            offset = 0
            while (
                len(context_messages) < max_messages_to_return
                and offset < _MAX_SCAN_ROWS
            ):
                rows = session_db.get_messages(
                    prev_session_id,
                    limit=_PAGE_SIZE,
                    offset=offset,
                    latest=True,
                )
                if not rows:
                    break

                for msg in reversed(rows):  # newest page row first => newest context msg first
                    role = msg.get("role")
                    content = msg.get("content") or ""
                    if role in ("user", "assistant") and content.strip():
                        context_messages.append(
                            {"role": role, "content": content}
                        )
                        if len(context_messages) >= max_messages_to_return:
                            break

                # A short page is the start of history: nothing older to fetch.
                if len(rows) < _PAGE_SIZE:
                    break

                offset += _PAGE_SIZE

            if not context_messages:
                continue

            # Reverse so the oldest selected message comes first.
            context_messages.reverse()

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
