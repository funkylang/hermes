"""Tests for automatic previous session context injection."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agent.session_context_loader import get_previous_session_context


class TestGetPreviousSessionContext:
    """Test the session context loader function."""

    def test_returns_empty_when_no_session_db(self):
        """Should return empty list when no session DB provided."""
        result = get_previous_session_context(None, "current-session")
        assert result == []

    def test_returns_empty_when_no_sessions_found(self):
        """Should return empty list when no sessions exist."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = []
        mock_db.get_messages.return_value = []

        result = get_previous_session_context(mock_db, "current-session")
        assert result == []

    def test_skips_current_session(self):
        """Should skip the current session and find previous."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            {'id': 'session-2', 'message_count': 15},  # This is current (filtered later)
            {'id': 'session-1', 'message_count': 20},  # This should be found
        ]
        mock_db.get_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = get_previous_session_context(mock_db, "session-2")
        # session-2 is current (passed as parameter), so we should get messages from session-1
        assert len(result) == 2
        mock_db.get_messages.assert_called_once_with(
            session_id='session-1',
            limit=16,
            latest=True,
        )

    def test_respects_min_message_count(self):
        """Should skip sessions with too few messages."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            {'id': 'session-2', 'message_count': 15},  # Current
            {'id': 'session-1', 'message_count': 5},    # Too few (below min 10)
            {'id': 'session-0', 'message_count': 12},   # Eligible
        ]
        mock_db.get_messages.return_value = [
            {"role": "user", "content": "Test"},
        ]

        result = get_previous_session_context(
            mock_db,
            "session-2",
            min_messages_required=10,
        )

        # Should skip session-1 (only 5 msgs) and use session-0
        mock_db.get_messages.assert_called_once_with(
            session_id='session-0',
            limit=16,
            latest=True,
        )
        assert len(result) == 1

    def test_filters_only_user_assistant_messages(self):
        """Should only include user and assistant roles."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            {'id': 'session-1', 'message_count': 15},  # Current
            {'id': 'session-0', 'message_count': 20},
        ]
        mock_db.get_messages.return_value = [
            {"role": "user", "content": "User msg"},
            {"role": "assistant", "content": "Assistant msg"},
            {"role": "tool", "content": "Tool msg"},  # Should be filtered
            {"role": "system", "content": "System msg"},  # Should be filtered
        ]

        result = get_previous_session_context(mock_db, "session-1")
        assert len(result) == 2
        roles = [msg['role'] for msg in result]
        assert 'tool' not in roles
        assert 'system' not in roles

    def test_handles_empty_content_gracefully(self):
        """Should skip messages with empty content."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            {'id': 'session-1', 'message_count': 15},  # Current
            {'id': 'session-0', 'message_count': 20},
        ]
        mock_db.get_messages.return_value = [
            {"role": "user", "content": ""},  # Empty - skip
            {"role": "assistant", "content": "Valid"},
            {"role": "user", "content": "   "},  # Whitespace only - skip
        ]

        result = get_previous_session_context(mock_db, "session-1")
        assert len(result) == 1
        assert result[0]['content'] == "Valid"

    def test_returns_messages_in_chronological_order(self):
        """Messages should be returned in chronological order (oldest first)."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            {'id': 'session-1', 'message_count': 15},  # Current
            {'id': 'session-0', 'message_count': 20},
        ]
        # get_messages with latest=True returns most recent first (reversed order)
        mock_db.get_messages.return_value = [
            {"role": "assistant", "content": "Newer msg"},
            {"role": "user", "content": "Older msg"},
        ]

        result = get_previous_session_context(mock_db, "session-1")
        # Should be reversed back to chronological: oldest first
        assert len(result) == 2
        assert result[0]['content'] == "Older msg"
        assert result[1]['content'] == "Newer msg"

    def test_handles_db_errors_gracefully(self):
        """Should not crash on database errors."""
        mock_db = Mock()
        mock_db.list_sessions_rich.side_effect = Exception("DB error")

        # Should return empty list, not raise
        result = get_previous_session_context(mock_db, "session-1")
        assert result == []
