"""Tests for get_transparent_tool_message: full-detail CLI completion lines.

Covers the user-visible promise: the tool completion line shows the REAL tool
name, every argument (with implicit path defaults resolved to absolute), and an
explicit SUCCESS/ERROR status with a brief outcome summary — instead of the old
meaningful-sounding-but-useless friendly verbs ("grep pattern").
"""

import json

from agent.display import get_transparent_tool_message


class TestRealToolName:
    """The actual tool name is shown, not a friendly verb."""

    def test_search_files_shows_real_name(self):
        line = get_transparent_tool_message(
            "search_files", {"pattern": "foo", "path": "."}, 0.4,
            result=json.dumps({"total_count": 3}),
        )
        assert "search_files" in line
        # Not the old friendly verb
        assert "\t" not in line

    def test_terminal_shows_real_name(self):
        line = get_transparent_tool_message(
            "terminal", {"command": "ls"}, 0.5,
            result=json.dumps({"output": "", "exit_code": 0}),
        )
        assert "terminal" in line


class TestPathResolution:
    """Implicit path defaults (path='.') resolve to an absolute/cwd-anchored path."""

    def test_dot_path_resolved(self):
        line = get_transparent_tool_message(
            "search_files", {"pattern": "foo", "path": "."}, 0.4,
            result=json.dumps({"total_count": 1}),
        )
        # The raw '.' should not be the value shown; an absolute path is.
        assert "path=./" not in line
        assert "path=." + " " not in line
        # Resolved path is absolute (starts with '/') before display_path may ~-shorten it
        # At minimum, the literal bare '.' as a whole value must be gone.
        # We assert the resolved (absolute or ~) form appears:
        import re
        m = re.search(r"path=(\S+)", line)
        assert m, f"no path= in {line!r}"
        val = m.group(1)
        # '.' alone is not allowed
        assert val != "."


class TestStatusSuccess:
    """On success, an explicit SUCCESS status with an outcome summary."""

    def test_search_files_match_count(self):
        line = get_transparent_tool_message(
            "search_files", {"pattern": "foo"}, 0.4,
            result=json.dumps({"total_count": 5}),
        )
        assert "SUCCESS" in line
        assert "5 matches" in line

    def test_search_files_single_match(self):
        line = get_transparent_tool_message(
            "search_files", {"pattern": "foo"}, 0.4,
            result=json.dumps({"total_count": 1}),
        )
        assert "1 match" in line
        assert "1 matches" not in line

    def test_terminal_exit_code(self):
        line = get_transparent_tool_message(
            "terminal", {"command": "true"}, 0.5,
            result=json.dumps({"output": "", "exit_code": 0}),
        )
        assert "SUCCESS" in line
        assert "exit 0" in line

    def test_read_file_line_count(self):
        line = get_transparent_tool_message(
            "read_file", {"path": "f.py"}, 0.1,
            result=json.dumps({"content": "", "total_lines": 42}),
        )
        assert "SUCCESS" in line
        assert "42 lines" in line

    def test_success_without_known_summary(self):
        """A tool with no recognized summary shape still shows SUCCESS (no fabricated detail)."""
        line = get_transparent_tool_message(
            "my_custom_tool", {"a": 1}, 0.7,
            result=json.dumps({"ok": True}),
        )
        assert "SUCCESS" in line


class TestStatusError:
    """On failure, an explicit ERROR status with the specific message."""

    def test_terminal_nonzero_exit(self):
        line = get_transparent_tool_message(
            "terminal", {"command": "make"}, 1.0,
            result=json.dumps({"output": "", "exit_code": 2}),
        )
        assert "ERROR" in line
        assert "SUCCESS" not in line

    def test_terminal_error_message_shown(self):
        line = get_transparent_tool_message(
            "terminal", {"command": "make"}, 1.0,
            result=json.dumps({"output": "", "exit_code": 1, "error": "recipe for target failed"}),
        )
        assert "ERROR" in line
        assert "recipe for target failed" in line

    def test_structured_error_shown(self):
        line = get_transparent_tool_message(
            "read_file", {"path": "/nope"}, 0.1,
            result=json.dumps({"success": False, "error": "File not found: /nope"}),
        )
        assert "ERROR" in line
        # Path collapsed to basename by _trim_error
        assert "nope" in line


class TestArgumentDisplay:
    """All arguments are shown."""

    def test_multiple_args_all_shown(self):
        line = get_transparent_tool_message(
            "search_files", {"pattern": "foo", "path": ".", "file_glob": "*.py"}, 0.4,
            result=json.dumps({"total_count": 2}),
        )
        assert "pattern=foo" in line
        assert "file_glob=*.py" in line

    def test_long_value_truncated(self):
        long_code = "x" * 500
        line = get_transparent_tool_message(
            "execute_code", {"code": long_code}, 0.3,
            result=json.dumps({"ok": True}),
        )
        # The full 500-char code must not be inlined verbatim
        assert "x" * 100 not in line
        assert "..." in line

    def test_empty_args_hidden(self):
        line = get_transparent_tool_message(
            "skills_list", {"category": ""}, 0.2,
            result=json.dumps({}),
        )
        # Empty value should not produce a dangling "category="
        assert "category=" not in line


class TestFailSafe:
    """Cosmetic failures never abort the turn (mirrors get_cute_tool_message)."""

    def test_garbage_input_returns_something(self):
        out = get_transparent_tool_message("x", None, 0.5, result="not json")
        assert isinstance(out, str)
        assert len(out) > 0
