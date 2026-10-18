"""Tests for automatic previous session context injection.

NOTE: the loader talks to SessionDB through its PUBLIC API only
(``list_sessions_rich`` / ``substantive_message_count`` / ``get_messages``).
Mock those methods — never private attributes like ``_db`` or ``_conn``. A
mock that mirrors a private attribute can pass while the real class exposes
something else (that is how a broken loader shipped undetected on 2026-10-03).
"""

from unittest.mock import Mock

import pytest

from agent.session_context_loader import get_previous_session_context


def _session(sid, message_count):
    # message_count simulates the (potentially stale) stored counter; it must
    # not influence selection — eligibility comes from substantive_message_count.
    return {"id": sid, "message_count": message_count}


def _msg(role, content):
    return {"role": role, "content": content}


PAGE = 50  # loader page size (agent.session_context_loader._PAGE_SIZE)


def _noise_page(n_substantive=0, substantive_first=True, prefix="n"):
    """A full 50-row page: some substantive user/assistant rows, rest tool noise.

    substantive_first=True puts them at rows[0:n]; False puts them at the end
    (rows[PAGE-n:PAGE]) so a latest=True reverse-scan reaches them first."""
    rows = [_msg("tool", f"{prefix}-tool-{i}") for i in range(PAGE)]
    for i in range(n_substantive):
        pos = i if substantive_first else (PAGE - n_substantive + i)
        rows[pos] = _msg("user" if i % 2 == 0 else "assistant", f"{prefix}{i}")
    return rows


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

    def test_selects_by_substantive_count_not_stored_counter(self):
        """Regression: stale stored counter must never drive selection.

        A session whose stored message_count is 30 but which really holds only
        4 substantive messages must be skipped in favour of a later candidate
        that has enough real messages."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            _session("fake-big", 30),   # stored counter lies high
            _session("real-eligible", 5),  # stored counter lies low
        ]
        mock_db.substantive_message_count.side_effect = {
            "fake-big": 4,      # really too small -> skip
            "real-eligible": 12,  # eligible
        }.get
        mock_db.get_messages.return_value = [
            _msg("user", f"m{i}") for i in range(12)
        ]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 12
        # The lying session was probed but must not be the one read from.
        mock_db.get_messages.assert_called_once()
        assert mock_db.get_messages.call_args_list[0][0][0] == "real-eligible"

    def test_queries_shortlist_without_stored_counter_filter(self):
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = []
        get_previous_session_context(mock_db, "current-session")
        mock_db.list_sessions_rich.assert_called_once_with(
            order_by_last_active=True, limit=8
        )

    def test_returns_empty_when_no_eligible_sessions(self):
        mock_db = Mock()
        # Only the current session exists.
        mock_db.list_sessions_rich.return_value = [
            _session("current-session", 50),
        ]
        result = get_previous_session_context(mock_db, "current-session")
        assert result == []
        mock_db.substantive_message_count.assert_not_called()

    def test_returns_all_when_substantive_at_or_below_cap(self):
        """12 substantive messages (<= 16 cap) -> all 12 returned."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [_session("s1", 20)]
        mock_db.substantive_message_count.return_value = 12
        # One short page: all rows, chronological, a bit of noise.
        rows = [_msg("tool", "noise")] + [
            _msg("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(12)
        ]
        mock_db.get_messages.return_value = rows

        result = get_previous_session_context(mock_db, "current-session")

        assert [m["content"] for m in result] == [f"m{i}" for i in range(12)]
        # Single forward scan (all-messages path), nothing fetched from the tail.
        assert mock_db.get_messages.call_count == 1
        assert mock_db.get_messages.call_args_list[0][1]["latest"] is False

    def test_returns_first_half_and_last_half_when_over_cap(self):
        """30 substantive (> 16) -> first 8 + last 8, chronological."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [_session("s1", 200)]
        mock_db.substantive_message_count.return_value = 30

        head_page = _noise_page(n_substantive=8, substantive_first=True, prefix="h")
        tail_page = _noise_page(n_substantive=8, substantive_first=False, prefix="t")
        # First call scans from the start (head), second back from the end (tail).
        mock_db.get_messages.side_effect = [head_page, tail_page]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 16
        assert [m["content"] for m in result[:8]] == [f"h{i}" for i in range(8)]
        assert [m["content"] for m in result[8:]] == [f"t{i}" for i in range(8)]
        # Head call from the start, tail call from the end.
        assert mock_db.get_messages.call_args_list[0][1]["latest"] is False
        assert mock_db.get_messages.call_args_list[1][1]["latest"] is True

    def test_scans_forward_past_leading_noise_for_head(self):
        """Tool noise at session start must not eat the head budget."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [_session("s1", 200)]
        mock_db.substantive_message_count.return_value = 30

        # Page 1: only 3 substantive (indices 0-2); page 2 fills the remaining 5.
        head_page_one = _noise_page(n_substantive=3, substantive_first=True, prefix="h")
        head_page_two = _noise_page(n_substantive=8, substantive_first=True, prefix="H")
        tail_page = _noise_page(n_substantive=8, substantive_first=False, prefix="t")
        mock_db.get_messages.side_effect = [head_page_one, head_page_two, tail_page]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 16
        # First 3 from page one, next 5 from the start of page two.
        assert [m["content"] for m in result[:8]] == \
            ["h0", "h1", "h2", "H0", "H1", "H2", "H3", "H4"]
        assert [m["content"] for m in result[8:]] == [f"t{i}" for i in range(8)]
        # Three fetches: two head pages, one tail page.
        assert mock_db.get_messages.call_count == 3

    def test_uses_most_recent_eligible_session_first(self):
        mock_db = Mock()
        # list_sessions_rich(order_by_last_active=True) => newest first.
        mock_db.list_sessions_rich.return_value = [
            _session("newer", 25),
            _session("older", 40),
        ]
        mock_db.substantive_message_count.return_value = 12
        mock_db.get_messages.return_value = [
            _msg("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(12)
        ]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 12
        # Only the newer (first) session is probed.
        mock_db.substantive_message_count.assert_called_once_with("newer")

    def test_substantive_count_error_skips_candidate(self):
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [
            _session("broken", 30),
            _session("ok", 30),
        ]
        mock_db.substantive_message_count.side_effect = [
            RuntimeError("db hiccup"), 12,
        ]
        mock_db.get_messages.return_value = [
            _msg("user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(12)
        ]

        result = get_previous_session_context(mock_db, "current-session")

        assert len(result) == 12
        mock_db.get_messages.assert_called_once()
        assert mock_db.get_messages.call_args_list[0][0][0] == "ok"

    def test_includes_both_user_and_assistant_only(self):
        """Tool rows and empty content are dropped."""
        mock_db = Mock()
        mock_db.list_sessions_rich.return_value = [_session("s1", 20)]
        mock_db.substantive_message_count.return_value = 2
        # Fewer than 10 substantive -> ineligible.
        result = get_previous_session_context(mock_db, "current-session")
        assert result == []

    def test_handles_db_errors_gracefully(self):
        mock_db = Mock()
        mock_db.list_sessions_rich.side_effect = Exception("DB error")

        result = get_previous_session_context(mock_db, "current-session")
        assert result == []
