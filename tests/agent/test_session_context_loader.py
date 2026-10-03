"""Tests for automatic previous session context injection.

NOTE: the loader talks to SessionDB through its PUBLIC API only
(``list_sessions_rich`` / ``get_messages``). Mock those methods — never
private attributes like ``_db`` or ``_conn``. A mock that mirrors a private
attribute can pass while the real class exposes something else (that is how
a broken loader shipped undetected on 2026-10-03).
"""

from unittest.mock import Mock

import pytest

from agent.session_context_loader import get_previous_session_context


def _session(sid, message_count):
    return {"id": sid, "message_count": message_count}


def _msg(role, content):
    return {"role": role, "content": content}


class TestGetPreviousSessionContext:
    """Test the session context loader function."""

    def test_returns_empty_when_no_session_db(self):
        result = get_previous_session_context(None, "current-session")
        assert result == []

    def test_returns_empty_when_no_session_id(self):
        mock_db = Mock()
        result = get_previous_session_context(mock_db, "")
        assert result == []
        mock_db.list_sessions_rich.assert_not_called()

    def test_returns_empty_when_no_eligible_sessions(self):
        mock_db = Mock()
        # Only the current session exists.
        mock_db.list_sessions_rich.return_value = [
            _session("current-session", 50),
        ]
        result = get_previous_session_context(mock_db, "current-session")
        assert result == []

    def test_skips_sessions_with_too_few_messages(self):
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            _session("small-session", 5),  # below min_messages_required=10
            _session("good-session", 30),
        ]
        mock_db.get_messages.return_value = [
            _msg("user", "hello"),
            _msg("assistant", "hi there"),
        ]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 2
        # Must have read the good session, not the small one.
        mock_db.get_messages.assert_called_once_with(
            "good-session", limit=50, offset=0, latest=True
        )

    def test_includes_both_user_and_assistant_messages(self):
        """Both roles are kept; tool rows and empty content are dropped."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [_session("s1", 20)]
        # get_messages returns the page in chronological order (latest=True).
        mock_db.get_messages.return_value = [
            _msg("tool", "ls -la output"),
            _msg("assistant", ""),  # empty content dropped
            _msg("user", "User question"),
            _msg("assistant", "Assistant response"),
        ]

        result = get_previous_session_context(mock_db, "current-session")

        assert [m["role"] for m in result] == ["user", "assistant"]
        assert [m["content"] for m in result] == [
            "User question",
            "Assistant response",
        ]

    def test_paginates_back_until_enough_messages(self):
        """Tails are often tool noise: the loader pages backwards to reach 16."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [_session("s1", 300)]

        # First page (offset 0): only 2 usable of 50 rows.
        page_one = [
            _msg("tool", f"tool output {i}") if i % 2 else _msg("assistant", "")
            for i in range(49)
        ] + [_msg("user", "u1"), _msg("assistant", "a1")]
        # Second page (offset 50): more usable rows.
        page_two = [
            _msg("user", f"u{i}") if i % 2 else _msg("assistant", f"a{i}")
            for i in range(50)
        ]
        mock_db.get_messages.side_effect = [page_one, page_two]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 16
        # Oldest selected message comes first: it is from the second (older) page.
        # Collected newest-first: 2 from page_one, then 14 from page_two walking
        # its tail backwards; after the final reverse the head is page_two[36].
        assert result[0]["content"] == "a36"
        # And the two from the newer page must be at the very end.
        assert result[-2:] == [{"role": "user", "content": "u1"},
                               {"role": "assistant", "content": "a1"}]
        assert mock_db.get_messages.call_args_list[-1][1]["offset"] == 50

    def test_uses_most_recent_eligible_session_first(self):
        mock_db = Mock()
        # list_sessions_rich(order_by_last_active=True) => newest first.
        mock_db.list_sessions_rich.return_value = [
            _session("newer", 25),
            _session("older", 40),
        ]
        mock_db.get_messages.return_value = [_msg("user", "recent hello")]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 1
        mock_db.list_sessions_rich.assert_called_once_with(
            order_by_last_active=True, limit=20
        )
        # Only the newer session should be probed.
        mock_db.get_messages.assert_called_once()
        assert mock_db.get_messages.call_args_list[0][0][0] == "newer"

    def test_handles_db_errors_gracefully(self):
        mock_db = Mock()
        mock_db.list_sessions_rich.side_effect = Exception("DB error")

        result = get_previous_session_context(mock_db, "current-session")
        assert result == []
