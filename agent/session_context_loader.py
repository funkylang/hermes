"""Fetch context from previous sessions for automatic injection into new conversations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_previous_session_context(
    session_db: Any,
    current_session_id: str,
    min_messages_required: int = 10,
    max_messages_to_return: int = 16,
) -> List[Dict[str, Any]]:
    """Fetch messages from the most recent eligible previous session.

    Args:
        session_db: The SessionDB instance (can be None)
        current_session_id: ID of the current session to exclude
        min_messages_required: Minimum message count for a session to be eligible
        max_messages_to_return: Maximum number of messages to return from that session

    Returns:
        List of message dicts in API format (role, content) or empty list if no eligible session
    """
    if not session_db:
        return []

    try:
        # Query recent sessions, ordered by last_active DESC (most recent first)
        # Exclude the current session
        sessions = session_db.list_sessions_rich(
            order_by_last_active=True,
            limit=20,  # Look through up to 20 most recent sessions
        )

        for session in sessions:
            session_id = session.get('id')
            if not session_id or session_id == current_session_id:
                continue

            message_count = session.get('message_count', 0)
            if message_count < min_messages_required:
                # Keep looking - try the next older session
                continue

            # Found an eligible session, fetch its recent messages
            # Use latest=True to get most recent messages first
            messages = session_db.get_messages(
                session_id=session_id,
                limit=max_messages_to_return,
                latest=True,
            )

            if not messages:
                continue

            # Convert to API format (only user and assistant roles)
            api_messages = []
            for msg in reversed(messages):  # Reverse back to chronological order
                role = msg.get('role')
                content = msg.get('content') or ''
                if role in ('user', 'assistant') and content.strip():
                    # Strip reasoning_content, finish_reason and other non-wire fields
                    api_msg = {
                        'role': role,
                        'content': content
                    }
                    api_messages.append(api_msg)

            # Only return if we found some valid messages
            if api_messages:
                logger.debug(
                    "Previous session context loaded: session=%s, messages=%d",
                    session_id, len(api_messages),
                )
                return api_messages

        # No eligible session found with sufficient messages
        return []

    except Exception as exc:
        logger.warning("Failed to fetch previous session context: %s", exc)
        return []
