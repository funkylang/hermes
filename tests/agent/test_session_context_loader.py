"""Tests for automatic previous session context injection."""

import pytest
from unittest.mock import Mock, MagicMock
from agent.session_context_loader import get_previous_session_context


class TestGetPreviousSessionContext:
    """Test the session context loader function."""

    def test_returns_empty_when_no_session_db(self):
        """Should return empty list when no session DB provided."""
        result = get_previous_session_context(None, "current-session")
        assert result == []

    def test_returns_empty_when_no_previous_sessions(self):
        """Should return empty list when only current session exists."""
        mock_db = Mock()
        # Query returns nothing (only current session exists)
        mock_db._db.execute.return_value.fetchall.return_value = []
        
        result = get_previous_session_context(mock_db, "current-session")
        assert result == []

    def test_returns_empty_when_no_session_id(self):
        """Should return empty list when no current session ID."""
        mock_db = Mock()
        
        result = get_previous_session_context(mock_db, "")
        assert result == []

    def test_skips_sessions_with_too_few_messages(self):
        """Should skip sessions that don't have enough messages."""
        mock_db = Mock()
        
        # First query: find previous sessions
        sessions_result = Mock()
        sessions_result.fetchall.return_value = [('session-1',)]
        
        # Second query: count messages (returns 5, below min of 10)
        count_result = Mock()
        count_result.fetchone.return_value = (5,)
        
        # Third and later queries should not be reached due to insufficient messages
        mock_db._db.execute.side_effect = [
            sessions_result,
            count_result,
        ]
        
        result = get_previous_session_context(mock_db, "current-session")
        assert result == []

    def test_includes_both_user_and_assistant_messages(self):
        """Should include messages from both roles."""
        mock_db = Mock()
        
        # First query: find previous sessions
        sessions_result = Mock()
        sessions_result.fetchall.return_value = [('session-1',)]
        
        # Second query: count messages (returns 15, meets minimum)
        count_result = Mock()
        count_result.fetchone.return_value = (15,)
        
        # Third query: get the actual messages
        messages_result = Mock()
        messages_result.fetchall.return_value = [
            ('assistant', 'Assistant response', None),
            ('user', 'User question', None),
        ]
        
        mock_db._db.execute.side_effect = [
            sessions_result,
            count_result,
            messages_result,
        ]
        
        result = get_previous_session_context(mock_db, "current-session")
        
        # Should be reversed to chronological order (oldest first)
        assert len(result) == 2
        roles = [msg['role'] for msg in result]
        # After reversal: user comes first, assistant second
        assert 'user' in roles
        assert 'assistant' in roles

    def test_handles_db_errors_gracefully(self):
        """Should not crash on database errors."""
        mock_db = Mock()
        mock_db._db.execute.side_effect = Exception("DB error")
        
        # Should return empty list, not raise
        result = get_previous_session_context(mock_db, "current-session")
        assert result == []

    def test_includes_reasoning_content_when_available(self):
        """Should include reasoning_content when present."""
        mock_db = Mock()
        
        sessions_result = Mock()
        sessions_result.fetchall.return_value = [('session-1',)]
        
        count_result = Mock()
        count_result.fetchone.return_value = (20,)
        
        messages_result = Mock()
        messages_result.fetchall.return_value = [
            ('assistant', 'Response with reasoning', 'Extended thinking process'),
            ('user', 'Question', None),
        ]
        
        mock_db._db.execute.side_effect = [
            sessions_result,
            count_result,
            messages_result,
        ]
        
        result = get_previous_session_context(mock_db, "current-session")
        
        # Find the assistant message and check for reasoning_content
        assistant_msgs = [m for m in result if m['role'] == 'assistant']
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get('reasoning_content') == 'Extended thinking process'
