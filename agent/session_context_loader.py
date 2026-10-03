"""Load context from previous session to inject at the start of new conversations."""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def get_previous_session_context(
    session_db: Any,
    current_session_id: str,
    min_messages_required: int = 10,
    max_messages_to_return: int = 16,
) -> List[Dict[str, Any]]:
    """Get the last N messages from the most recent previous session.

    Args:
        session_db: The SessionDatabase instance (can be None)
        current_session_id: Current session ID to exclude
        min_messages_required: Minimum message count for a session to be considered
        max_messages_to_return: Maximum number of messages to return

    Returns:
        List of message dicts with role and content, in chronological order (oldest first).
    """
    if not session_db or not current_session_id:
        return []

    try:
        db = session_db._db
        # Find the most recent previous session by looking at sessions table
        rows = db.execute("""
            SELECT id FROM sessions
            WHERE id != ?
            ORDER BY last_activity_at DESC
            LIMIT 10
        """, (current_session_id,)).fetchall()

        for row in rows:
            prev_session_id = row[0]

            # Check that this session has enough messages to be useful context
            count_row = db.execute("""
                SELECT COUNT(*) FROM messages
                WHERE session_id = ? AND active = 1
            """, (prev_session_id,)).fetchone()

            if not count_row or count_row[0] < min_messages_required:
                continue

            # Get the last N user/assistant messages with content in chronological order
            # Using id DESC + reverse = most recent messages, oldest-first output.
            msgs = db.execute("""
                SELECT role, content, reasoning_content
                FROM messages
                WHERE session_id = ? AND active = 1
                AND role IN ('user', 'assistant')
                AND content IS NOT NULL AND LENGTH(content) > 0
                ORDER BY id DESC
                LIMIT ?
            """, (prev_session_id, max_messages_to_return)).fetchall()

            if not msgs:
                continue

            # Reverse so oldest message is first
            msgs = list(reversed(msgs))

            context_messages = []
            for role, content, reasoning in msgs:
                msg = {"role": role, "content": content}
                if reasoning and reasoning.strip():
                    msg["reasoning_content"] = reasoning
                context_messages.append(msg)

            logger.debug(
                "Previous session context loaded: session=%s messages=%d",
                prev_session_id[:8], len(context_messages)
            )
            return context_messages

        return []
    except Exception as exc:
        logger.debug("Failed to load previous session context: %s", exc)
        return []
